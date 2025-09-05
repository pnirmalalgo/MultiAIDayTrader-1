from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv
import json
import datetime
import re

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

def extract_json_from_cot(output: str):
    try:
        # Try to extract JSON block using regex (greedy match for {...})
        match = re.search(r"{[\s\S]*}", output)
        if not match:
            raise ValueError("No JSON block found in output.")
        
        json_str = match.group(0)

        parsed = json.loads(json_str)
        return parsed

    except Exception as e:
        print("Failed to extract JSON from CoT:", str(e))
        return {}
    
def interpreter_with_cot(user_query):
    today = datetime.date.today().isoformat()
    prompt = f"""
    You are a trading query interpreter.
    1. First, write your reasoning step by step under "Thoughts:".
    2. Then, write the structured query in strict JSON format under "Structured Query:".
    3. Only include the JSON under "Structured Query:", no explanations there.

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
  + IMPORTANT: Always return the ticker as an actual string or list of strings. For example: "ticker": ["TCS.NS"]

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

The user has provided the following backtest query: {user_query}
    """
    
    # Call the LLM
    response = llm.invoke([HumanMessage(content=prompt)])

    print("DEBUG — Raw LLM output before parsing:", response)

    # Extract text output
    text_output = response.content
    print("Interpreter output (raw):", text_output)

    # Optional: Try to split thoughts from structured query (for UI/logging)
    if "Structured Query:" in text_output:
        thoughts_text, _ = text_output.split("Structured Query:", 1)
    else:
        thoughts_text = text_output

    # Use robust extractor to get JSON
    structured_query_dict = extract_json_from_cot(text_output)

    return thoughts_text.strip(), structured_query_dict