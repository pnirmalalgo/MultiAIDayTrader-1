from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key)

def interpret_query(user_input: str) -> dict:
    messages = [
        SystemMessage(
            content=(
                "You are an AI that extracts structured info from trading queries.\n"
                "Respond with JSON with keys: company_name, strategy_description, date_range, buy_condition, sell_condition."
            )
        ),
        HumanMessage(content=user_input)
    ]
    response = llm.invoke(messages)
    return {"intent": response.content}