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

def interpret_query(query: str, context=None) -> tuple:
    if context is None:
        context = {}
    context.setdefault("logs", [])

    today = datetime.date.today().isoformat()

    prompt = f"""
You are a trading query interpreter.
First, think step by step about what the query is asking, the strategy, defaults, and any missing rules. 
Then, output ONLY valid JSON that matches the schema below.

... [keep your full prompt exactly as-is, omitted here for brevity] ...

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
    print("Interpreter response:", response.content)

    # --- Append CoT reasoning log ---
    reasoning_text = "CoT reasoning for interpreting the query"  # optionally, you can extract from response.metadata or response.content
    context["logs"].append({
        "step": "interpreter",
        "cot": reasoning_text
    })

    # Return both intent and logs
    return {
        "intent": response.content,  # existing JSON string
        "logs": context["logs"]
    }
