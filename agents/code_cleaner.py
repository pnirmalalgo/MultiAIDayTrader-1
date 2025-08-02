from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key)

def clean_code(code: str) -> dict:
    messages = [
        SystemMessage(content="You are a Python code formatter. Clean and organize the following code, add any missing imports, and remove any unnecessary comments."),
        HumanMessage(content=code)
    ]
    response = llm.invoke(messages)
    return {"clean_code": response.content}