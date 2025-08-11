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
    Always extract buy and sell conditions as logical groups with 'and / 'or logic.
    Use the following schema for each condition:
    - logic: 'and' | 'or
    - conditions: list of objects like:
    - "indicator": "RSI", "operator": ">", "value": 70 
    - "indicator": "MACD", "operator": "between", "min": 40, "max": 60 
    Example:
    Input: "Buy when MACD is positive and RSI between 40 and 60, sell when MACD is negative or RSI > 80"
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
      {{"indicator": "RSI", "operator": ">", "value": 80}}
    ]
    }}
    }}
    
    Please return a JSON object containing the following:
    - "ticker": The stock ticker symbol not the company name (e.g., TCS)
    - "ticker": if there are multiple company/stock names mentioned then get the ticker name of all and return ticker names in a list.
    - "strategy": The name of the strategy (e.g., RSI)
    - "buy_condition": A dictionary with 'buy' conditions for the strategy (e.g., buy: RSI: <25, sell: RSI: >75)
    - "sell_condition": A dictionary with 'sell' conditions for the strategy (e.g., buy: RSI: <25, sell: RSI: >75)
    - "start_date": Use today's date={today} for reference for date calculations. This should be a date value. Interpret and calculate based on query what is the start date of range for which data is needed.
    - "end_date": Use today's date={today} for reference for date calculations. This should be a date value. Interpret and calculate based on query what is the end date of range for which data is needed.
    - "IMP: Calculation and setting of start_date and end_date is very important. do not just output the string explained above. calculate the dates. for eg. if query contains past 2 years then get start_date=(find exact date before 2 years), end_date=(find exact date of today). if query asks for today then start_date=today's date and end_date=today's date.
    - "When user says "for 3+ days", "consecutive days", or similar, convert that to a structured condition with keys:
    - indicator
    - comparison
    - value (if applicable)
    - duration_days: number of days
    - duration_type: "consecutive" or "non-consecutive"
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