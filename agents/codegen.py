from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2,
                 openai_api_key=api_key)

def generate_code(intent_json: str) -> dict:
    messages = [
        SystemMessage(content="You are a Python developer that writes backtesting code for trading strategies using pandas and yfinance. Only return code, no explanation."),
        HumanMessage(content=f"Write code for the following trading strategy:\n{intent_json}")
    ]
    response = llm.invoke(messages)
    return {"code": response.content}