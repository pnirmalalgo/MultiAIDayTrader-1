# interpreter.py

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
        match = re.search(r"{[\s\S]*}", output)
        if not match:
            raise ValueError("No JSON block found in output.")
        json_str = match.group(0)
        parsed = json.loads(json_str)
        return parsed
    except Exception as e:
        print("Failed to extract JSON from CoT:", str(e))
        return {}

def interpret_query_mcp(input: dict) -> dict:
    user_query = input.get("query", "").strip()

    if not user_query:
        return {
            "thought": "No query provided in input.",
            "action": "AskClarification",
            "action_input": "Can you please provide a trading query?",
        }

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
  - "indicator": "Price", "operator": "take_profit", "value": 15, "value_type": "percent"
  - "indicator": "Price", "operator": "stop_loss", "value": 10, "value_type": "percent"

### Important rules for stop-loss and take-profit ###
- If a sell condition is percentage-based ("value_type": "percent"):
    - Always calculate relative to the last buy entry price, not daily close.
    - Encode explicitly as:
        {{ "indicator": "Price", "operator": "take_profit", "value": X, "value_type": "percent" }}
        {{ "indicator": "Price", "operator": "stop_loss", "value": Y, "value_type": "percent" }}
    - Do NOT encode these as "<" or ">" comparisons.
    - Evaluate only when currently in a position.
    - Combine with other sell conditions using logical OR.
    - Reset entry price when position is closed.
- This ensures each trade has its own thresholds applied individually.

Example:
Input: "Buy when MACD is positive and RSI between 40 and 60, sell when MACD is negative or 15% profit or 5% stop-loss"
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
      {{"indicator": "Price", "operator": "take_profit", "value": 15, "value_type": "percent"}},
      {{"indicator": "Price", "operator": "stop_loss", "value": 5, "value_type": "percent"}}
    ]
  }}
}}

Please return a JSON object containing:
- "ticker": canonical company names (expand abbreviations, drop country suffixes), as string or list.
- "strategy": strategy name (e.g., RSI)
- "buy_condition": dictionary of buy conditions
- "sell_condition": dictionary of sell conditions
- "start_date" and "end_date": calculate actual dates based on query (e.g., "past 2 years")
- For consecutive duration conditions ("3+ days", "consecutive days"):
    - include keys: indicator, comparison, value, duration_days, duration_type
- Include "value_type": "percent" only for percent-based profit/loss conditions; default to absolute price otherwise.

⚠️ Important:
- Use default rules for known strategies if no explicit buy/sell conditions are provided.
- Ensure all operator directions (< or >) reflect natural language intent.
- Ensure stop-loss / take-profit is explicitly labeled with operator "stop_loss" or "take_profit", not "<" or ">".
- This JSON will be used by the code generator to implement conditional sell logic relative to trade entry price.
- If the user adds stop-loss or take-profit sell conditions for a known strategy (e.g., RSI, MACD), merge them with the strategy's default sell conditions using logical OR. Do NOT replace the default sell rules.

The user has provided the following backtest query: {user_query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("DEBUG — Raw LLM output before parsing:", response)

    text_output = response.content
    if "Structured Query:" in text_output:
        thoughts_text, _ = text_output.split("Structured Query:", 1)
    else:
        thoughts_text = text_output

    structured_query_dict = extract_json_from_cot(text_output)

    # Handle bad or missing parse
    if not structured_query_dict:
        return {
            "thought": "I could not extract a structured query from the input.",
            "action": "AskClarification",
            "action_input": "I couldn't interpret the query. Could you rephrase it?",
        }

    return {
        "thought": thoughts_text.strip(),
        "question": "Is this interpretation correct?",
        "action": "StructuredQueryReady",
        "action_input": structured_query_dict
    }
