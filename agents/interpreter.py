from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)

def interpret_query(query: str) -> dict:
    today = datetime.date.today().isoformat()

    prompt = f"""
You are a trading query interpreter. Parse the user's query into structured JSON. 

Always extract buy and sell conditions as logical groups with 'and' / 'or' logic.
Use the following schema for each condition:
- logic: 'and' | 'or'
- conditions: list of objects like:
  - "indicator": "RSI", "operator": ">", "value": 70 
  - "indicator": "MACD", "operator": "between", "min": 40, "max": 60 
  - "indicator": "Price", "operator": ">", "value": 15, "value_type": "percent"   (if query says 15% profit)
  - "indicator": "Price", "operator": "<", "value": 10, "value_type": "percent"   (if query says 10% stop-loss)

Example:
Input: "Buy when MACD is positive and RSI between 40 and 60, sell when MACD is negative or 15% profit"
Output:
{{
  "buy_condition": {{
    "logic": "and",
    "conditions": [
      {{"indicator": "MACD", "operator": ">", "value": 0}},
      {{"indicator": "RSI", "operator": "between", "min": 40, "max": 60}}
    ]
  }},
  "sell_condition": {{
    "logic": "or",
    "conditions": [
      {{"indicator": "MACD", "operator": "<", "value": 0}},
      {{"indicator": "Price", "operator": ">", "value": 15, "value_type": "percent"}}
    ]
  }}
}}

Please return a JSON object containing the following:
- "ticker": Identify the Company names given in the query. Do not return ticker symbols. Just return the company names as mentioned in the query. 
  + Identify and normalize company names into their official names as listed on stock exchanges. 
  + If there are multiple companies mentioned, return a list of names.
  + Expand abbreviations (e.g., L&T → Larsen & Toubro, HDFC → Housing Development Finance Corporation). 
  + If the user adds a country suffix like "India", drop it when returning the canonical name.

- "strategy": The name of the strategy (e.g., RSI)

- "buy_condition": A dictionary with 'buy' conditions for the strategy
- "sell_condition": A dictionary with 'sell' conditions for the strategy

⚠️ VERY IMPORTANT: 
If the user specifies a strategy but does not provide explicit buy/sell rules, you must insert **default rules** for that strategy.
Default rule mapping:
  - RSI → Buy: RSI < 30, Sell: RSI > 70
  - MACD → Buy: MACD > Signal, Sell: MACD < Signal
  - SMA Crossover (short vs long) → Buy: SMA(short) > SMA(long), Sell: SMA(short) < SMA(long)
  - Bollinger Bands → Buy: Price < Lower Band, Sell: Price > Upper Band
If strategy is unknown and no rules are given, leave both conditions empty but do NOT invent unrelated indicators.

- "start_date": Use today's date={today} for reference for date calculations. This should be a date value. Interpret and calculate based on query what is the start date of range for which data is needed.
- "end_date": Use today's date={today} for reference for date calculations. This should be a date value. Interpret and calculate based on query what is the end date of range for which data is needed.

- IMP: Calculation and setting of start_date and end_date is very important. Do not just output the string explained above. Calculate the dates. 
  For example, if query contains "past 2 years" then get start_date=(today - 2 years), end_date=(today). 
  If query asks for "today" then start_date=today's date and end_date=today's date.

- When user says "for 3+ days", "consecutive days", or similar, convert that to a structured condition with keys:
  - indicator
  - comparison
  - value (if applicable)
  - duration_days: number of days
  - duration_type: "consecutive" or "non-consecutive"

- Before generating code or interpreting conditions, verify that all comparison operators correctly reflect the query's intended logic.
  For example, if the query says "stock trading below 20-day MA for X days", then the condition must express:
  current price < 20-day MA
  not the opposite (20-day MA < current price).
  Similarly, if a condition specifies consecutive duration, ensure the logic is applied correctly across all required days.

To summarize:
- Confirm that any "indicator vs current price" comparisons align with the natural language phrase in the query.
- Double-check that the operator (< or >) is not inverted.
- Ensure that the duration constraint (e.g., 3 consecutive days) is implemented on the correct comparison direction.
- If buy/sell conditions are missing but strategy is known, use default rules. 
- If strategy is unknown and no rules are given, leave conditions empty (do NOT hallucinate).
- When user specifies returns or stop-loss in percentage terms (e.g., "15% profit", "10% stop-loss"), include a "value_type": "percent" field in the condition JSON. 
  Default to absolute price (no value_type) if no percentage is mentioned.

The user has provided the following backtest query:

{query}
"""


    messages = [
        SystemMessage(
            content=(
                "You are an AI that extracts structured info from trading queries. And also interprets and calculates the time period (start_date, end_date) given in query and generates start_date and end_date accordingly.\n"
                "Respond with JSON with keys: ticker, strategy, buy_condition, sell_condition, start_date, end_date."
            )
        ),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    return {"intent": response.content}