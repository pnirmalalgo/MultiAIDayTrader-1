from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key)

def interpret_query(user_query: str) -> dict:
    messages = [
        SystemMessage(content="You are a stock strategy interpreter. Extract structured JSON from natural language queries to define a backtesting strategy. Include fields: stock, strategy, buy_condition, sell_condition, period."),
        HumanMessage(content=user_query)
    ]
    response = llm.invoke(messages)
    return {"intent": response.content}