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
    user_cot = input.get("cot", "").strip()

    if not user_query:
        return {
            "thought": "No query provided in input.",
            "action": "AskClarification",
            "action_input": "Can you please provide a trading query?",
        }

    today = datetime.date.today().isoformat()

    cot_hint = f"\nThe user also provided these reasoning notes (CoT): {user_cot}\n" if user_cot else ""

    prompt = f"""
You are a trading query interpreter.

{cot_hint}
...


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
- For crossover or event-based conditions, use:
+   {{"event": {{"indicator1": "<string>", "operator": "cross_above" | "cross_below", "indicator2": "<string>"}}

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
###IMPORTANT###
- Apply strategy default rules (e.g., RSI: buy<30, sell>70) **only if the user query does NOT specify explicit buy or sell thresholds**.
- Do NOT add or merge stop-loss / take-profit or any default sell conditions unless the user explicitly mentions them in the query.
- If the user specifies any sell conditions, use exactly those conditions and do not add any additional default rules.
- Only merge default stop-loss / take-profit conditions with user sell conditions **if the user explicitly requested them**.

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
"ticker": canonical company names (expand abbreviations, drop country suffixes), as string or list.
        Note:If the user mentions "Nifty 50" or "Nifty 50 shares only", expand it into a list of all current Nifty 50 constituent tickers (not the ETF like SETFNIF50.NS).
        If the user mentions "Nifty Junior" or "Nifty Next 50", expand it into a list of all current Nifty Next 50 constituent tickers.
        Never map Nifty 50 or Nifty Junior to a single ETF ticker. Always expand them to the underlying stocks.
- "strategy": strategy name (e.g., RSI)
- "buy_condition": dictionary of buy conditions
- "sell_condition": dictionary of sell conditions
+- "additional_buy_condition": (optional) dictionary of conditions for averaging down or pyramiding
+- "entry_condition": (optional) dictionary if the entry must be gated by a crossover event before buys are valid
- "start_date" and "end_date": calculate actual dates based on query and take today's date={today} as reference if relative.
- For consecutive duration conditions ("3+ days", "consecutive days"):
    - include keys: indicator, comparison, value, duration_days, duration_type
- Include "value_type": "percent" only for percent-based profit/loss conditions; default to absolute price otherwise.

- If a buy, additional buy, or sell condition is percentage-based (value_type: "percent"):
    - If it applies relative to the last executed trade, include "relative_to": "entry_price" in the JSON.
    - Do NOT encode this as "<" or ">" comparisons directly.
    - Example:
      {{ "indicator": "Price", "operator": "take_profit", "value": 15, "value_type": "percent", "relative_to": "entry_price" }}
      {{ "indicator": "Price", "operator": "stop_loss", "value": 5, "value_type": "percent", "relative_to": "entry_price" }}
      {{ "indicator": "Price", "operator": "<", "value": 10, "value_type": "percent", "relative_to": "entry_price" }}  # for additional buys
    - For averaging down or pyramiding (additional_buy_condition), if the user specifies a percentage drop (e.g., -10%, -20%) from the first buy price, always include "relative_to": "entry_price" in the structured query JSON.
    - Always populate "relative_to": "entry_price" for any percent-based condition that depends on a previous trade entry price.  
    - Only use absolute prices (no "relative_to") if the value is a direct price, not a percentage.

    ######Stepwise Trend Rule#####:
- If the user query mentions multiple EMAs or other moving averages in ordered relation (e.g., "20 EMA below 50 EMA and 50 EMA below 250 EMA") or asks for negative/positive crossovers:
    - Include a "stepwise_trend" key in the structured JSON.
    - The schema is:
        "stepwise_trend_buy": {{
            "indicators": ["EMA_20", "EMA_50", "EMA_250"],
            "direction": "negative",
            "action": "buy"
        }},
        "stepwise_trend_sell": {{
            "indicators": ["EMA_20", "EMA_50", "EMA_250"],
            "direction": "positive",
            "action": "sell"
        }}
- Only include this key if such a trend-based instruction exists in the query.
- Extract the EMA/MA numbers from the query text dynamically; do not hardcode EMA names.
- The translator agent will use this array to generate chained crossover conditions.

- If the user query mentions buying on a negative EMA/MA trend (e.g., 20 EMA < 50 EMA < 250 EMA) or similar stepwise trend:
    - Always generate the sell condition as the exact reversal (positive trend):
        - Example: sell_condition triggers when 20 EMA > 50 EMA > 250 EMA
    - Do NOT use negative trend operators in sell_condition.
    - Include this in the JSON either as 'stepwise_trend_sell' or in 'sell_condition' explicitly.
    - Ensure your sell_condition reflects a trend reversal relative to the buy_condition.

Important: Any sell condition based on a previous trend-based buy **must always represent a reversal**.  
Do NOT reuse the same trend direction as the buy. For example, if buying occurs on a negative EMA crossover, the sell must occur on the corresponding positive EMA crossover.

- If a trend-based EMA/MA condition triggers a buy (negative trend), the corresponding sell condition must be the reversal (positive trend) and should be included as 'stepwise_trend_sell'.
    - Example:
        "stepwise_trend_buy": {{ "indicators": ["EMA_20","EMA_50","EMA_250"], "direction":"negative", "action":"buy" }},
        "stepwise_trend_sell": {{ "indicators": ["EMA_20","EMA_50","EMA_250"], "direction":"positive", "action":"sell" }}
- Do not output negative trend operators in 'sell_condition' if the buy was on a negative trend.

- Both keys follow the same schema with indicators, direction, and action.


- "remarks": "If there is any additional context or instructions from the user, include them here. Do not include ticker information here."

⚠️ Important:
- Use default rules for known strategies if no explicit buy/sell conditions are provided.
- Ensure all operator directions (< or >) reflect natural language intent.
- Ensure stop-loss / take-profit is explicitly labeled with operator "stop_loss" or "take_profit", not "<" or ">".
- This JSON will be used by the code generator to implement conditional sell logic relative to trade entry price.
- If the user adds stop-loss or take-profit sell conditions for a known strategy (e.g., RSI, MACD), merge them with the strategy's default sell conditions using logical OR. Do NOT replace the default sell rules.

The user has provided the following backtest query: {user_query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("DEBUG — Raw LLM output before parsing:", response.content)

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
    
    # ✅ If ticker field exists but is a strategy name, clear it
    if structured_query_dict.get("ticker") in ["any", "Moving Average Crossover", "RSI", "MACD", "Bollinger Bands", "Mean Reversion", "Momentum", "Cross Over", "Crossover", "Death Cross", "Golden Cross"]:
        structured_query_dict["ticker"] = []

    if not structured_query_dict.get("ticker"):
        return {
            "thought": thoughts_text.strip(),
            "action": "AskClarification",
            "action_input": {
                "question": "Could you clarify which stock/ticker you're referring to?",
                "structured_query": structured_query_dict
            }
        }
    #  Missing start_date
    if not structured_query_dict.get("start_date"):
        return {
            "thought": thoughts_text.strip(),
            "action": "AskClarification",
            "action_input": {
                "question": "Could you clarify the start date for this backtest?",
                "structured_query": structured_query_dict
            }
        }

    #  Missing end_date
    if not structured_query_dict.get("end_date"):
        return {
            "thought": thoughts_text.strip(),
            "action": "AskClarification",
            "action_input": {
                "question": "Could you clarify the end date for this backtest?",
                "structured_query": structured_query_dict
            }
        }
    
    structured_query_dict.setdefault("additional_buy_condition", {})
    structured_query_dict.setdefault("entry_condition", {})
    #happy path
    return {
        "thought": thoughts_text.strip(),
        "question": "Is this interpretation correct?",
        "action": "StructuredQueryReady",
        "action_input": structured_query_dict
    }
