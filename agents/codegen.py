from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key)

def generate_code(intent_json: str) -> dict:
    parsed = json.loads(intent_json)
    ticker = parsed.get("ticker")
    strategy = parsed.get("strategy_description", "")
    buy_condition = parsed.get("buy_condition", "")
    sell_condition = parsed.get("sell_condition", "")
    date_range = parsed.get("date_range", "")

    prompt = (
        f"Write Python code to backtest the following strategy on stock '{ticker}'.\n"
        f"Strategy: {strategy}\n"
        f"Buy when: {buy_condition}\n"
        f"Sell when: {sell_condition}\n"
        f"Date Range: {date_range}\n"
        f"Use pandas, yfinance, and any common backtesting technique. Print final return and plot the trades."
    )

    messages = [
        SystemMessage(content="You are a Python coder that writes backtesting scripts."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    print("Generated code: ",response.content)
    return {"code": response.content}