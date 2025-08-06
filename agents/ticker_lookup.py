from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)


def resolve_ticker(intent_json: str) -> dict:
    parsed = json.loads(intent_json)
    company_name = parsed.get("company_name")

    if not company_name:
        return {"intent": intent_json}

    messages = [
        SystemMessage(
            content="You are a stock market assistant. Given a company name, return the stock ticker (including exchange suffix like .NS or .NYSE if known). Only return the ticker string."
        ),
        HumanMessage(content=company_name)
    ]

    response = llm.invoke(messages)
    parsed["ticker"] = response.content.strip().upper()

    return {"intent": json.dumps(parsed)}