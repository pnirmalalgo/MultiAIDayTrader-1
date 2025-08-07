from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)

def clean_code(code: str) -> dict:
    
    messages = [
        SystemMessage(content="You are a Python code formatter. " \
        "Clean and organize the following code, add any missing imports" \
        "Important: Do not add '''python to the code. It breaks the code"
        "Remove all comments. Only executable python code needed."
        "The code should only contain executable code."
        "Add necessary code to ignore warnings."),
        HumanMessage(content=code)
    ]
    
    response = llm.invoke(messages)
    print("Code cleaned using LLM:", response.content)
    return {"clean_code": response.content}