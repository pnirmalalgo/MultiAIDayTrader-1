# translator_agent.py
from langchain_community.chat_models import ChatOpenAI
import json
from datetime import datetime
from typing import Dict, Any
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

TRANSLATOR_PROMPT = """
You are the TRANSLATOR agent in a multi-agent trading system.

INPUT:
- structured_query JSON from Interpreter (may include a "remarks" key or user-edited CoT inside structured_query["remarks"]).
- remarks: freeform string (optional) — the same value is provided again in the LLM input for clarity.

OUTPUT:
- A single JSON object (translator_instructions) that strictly follows the schema described below. **Output JSON only. No explanations, no extra text.**

--- REQUIRED SCHEMA KEYS (must appear in the output JSON) ---
- human_summary: short one-line explanation of user intent.
- indicators: list of dicts {name, class, params, output_column}
- date_filter: {apply: true/false, start_date, end_date}
- buy_spec: {conditions: [ ... ] }  // fully expanded conditions; see CONDITION FORMAT
- sell_spec: {conditions: [ ... ] } // fully expanded conditions
- trade_management: {entry_tracking: true/false, stop_loss_pct: number|null, take_profit_pct: number|null, reentry_rule: string|null}
- duration_handling: {type: "consecutive"/"non-consecutive", days: int} or null
- code_tasks: ordered list of granular, executable steps for CodeGen (must mention crossover detection, position checks, stop-loss/take-profit only if requested, portfolio updates, final forced-sell handling, plots)
- required_files: list of filenames (use placeholders {ticker}, {curr_time_stamp})
- plots: list of plot descriptors {type, y, axis: "primary"|"secondary", description}
- safety_checks: list of assertions/tests to run
- indicators_to_plot: list of indicator names (must include Close Price plus any indicator used in rules)
- remarks: string (echo any freeform instructions that do not fit other fields)

--- CONDITION FORMAT (must use this canonical form) ---
Each condition is a dictionary. Use one of these operators depending on intent:

1) Crossover conditions (MUST be encoded as cross operators):
   {
     "indicator": "<name>",              # e.g., "RSI" or "SMA_50"
     "operator": "crosses_above",        # OR "crosses_below"
     "threshold": <number>               # e.g., 30 or 70; for cross between series, include "other_indicator":"SMA_200"
     // Optional for clarity:
     "previous_condition": "<relation>"  # e.g., "<30" or "<=30" (useful for human-read crossovers)
   }

   Example: "RSI moves back above 30 after dropping below 30" =>
   {"indicator":"RSI", "operator":"crosses_above", "threshold":30, "previous_condition":"<30"}

   For cross between two series (SMA50 crosses SMA200) use:
   {"indicator":"SMA_50", "operator":"crosses_above", "other_indicator":"SMA_200"}

2) Inequality / price formulas (for stop-loss / take-profit):
   {
     "indicator": "Price",
     "operator": "<=" | ">=" | "<" | ">",
     "formula": "entry_price * (1 - 0.05)"   # string formula allowed; preserve entry_price variable
   }

3) Multi-day / persistence:
   {
     "indicator": "<name>",
     "operator": ">",
     "threshold": <number>,
     "duration": {"type":"consecutive"|"non-consecutive", "days": <int>}
   }

4) Composite conditions:
   - If multiple atomic conditions must be combined with AND/OR, place them in the list and the code_tasks must explicitly state how to combine them (e.g., evaluate all conditions and apply logical AND).

--- RULES (strict, follow exactly) ---
1. **Always encode crossovers explicitly** as `crosses_above` / `crosses_below`. Never convert "moves back above 30 after being below 30" to a plain `>` inequality. Use the CONDITION FORMAT for crossovers.

2. **Trade management**
   - Always include `entry_tracking: true`.
   - Only include stop_loss_pct / take_profit_pct when the user explicitly requested them.
   - If a safeguard/re-entry rule is present in the user's query or remarks, populate `trade_management.reentry_rule` with an explicit short instruction (e.g., "After a sell, do not re-enter until RSI first exits oversold/overbought and then forms a fresh crossover above 30").

3. **Code tasks**
   - Provide a granular, ordered list of steps the CodeGen must implement (e.g., precompute indicators, implement crossover detection using shift(1), evaluate buy/sell only when position==0/1, use trades list for entry_price, update portfolio after trades, force-close final open position, save plots).
   - For every crossover step include the exact pandas-style hint: e.g., `cond = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)` and mention `shift(1)` explicitly.
   - For re-entry safeguard include an explicit state pattern and reset condition. Example step: "After a Sell, set `waiting_for_reset=True`; only set `waiting_for_reset=False` when RSI < 30 (for buy re-entry) OR RSI between 30 and 70 as appropriate; only allow new buys when `waiting_for_reset==False` and crossover condition occurs."

4. **Plots**
   - Provide `plots` list with entries for price (axis primary) and oscillators (axis secondary). Include `indicators_to_plot` listing all required indicators.
   - In `code_tasks` require CodeGen to plot buys/sells using the `trades` list only (not indicator boolean masks).

5. **Safety checks**
   - Include checks for non-empty DataFrame, sorted dates, indicator columns presence, alternating trades, no sell without buy, portfolio_series length equals df.index, final forced-sell sets `portfolio_series.iloc[-1] = cash`.

6. **Merging translator remarks / user-edited CoT**
   - If `structured_query["remarks"]` exists and is non-empty, incorporate it into `remarks` output and reflect any additional constraints mentioned there (e.g., "must use crossover detection" or "no intraday fills") by updating buy_spec / sell_spec / trade_management accordingly.

7. **Examples (required — translator must use the same JSON shape)**

User text:
"Buy when RSI moves back above 30 after having dropped below 30; sell when RSI > 70; stoploss 5%; target 15%; after any exit wait until RSI first exits oversold/overbought before re-entry."

Translator JSON (excerpt):
{
  "human_summary": "RSI recovery strategy: buy on RSI crossing above 30 after being below 30; sell on RSI>70 or SL -5% or TP +15%; enforce re-entry safeguard.",
  "indicators": [{"name":"RSI", "class":"oscillator", "params":{"window":14}, "output_column":"RSI"}],
  "date_filter": {"apply": true, "start_date": "2023-09-11", "end_date":"2025-09-11"},
  "buy_spec": {
    "conditions": [
      {"indicator":"RSI", "operator":"crosses_above", "threshold":30, "previous_condition":"<30"}
    ]
  },
  "sell_spec": {
    "conditions": [
      {"indicator":"RSI", "operator":">", "threshold":70},
      {"indicator":"Price", "operator":"<=", "formula":"entry_price * (1 - 0.05)"},
      {"indicator":"Price", "operator":">=", "formula":"entry_price * (1 + 0.15)"}
    ]
  },
  "trade_management": {
    "entry_tracking": true,
    "stop_loss_pct": 5,
    "take_profit_pct": 15,
    "reentry_rule": "After a sell, set waiting_for_reset=True; reset when RSI < 30 and only allow next buy on fresh crossover above 30."
  },
  "code_tasks": [
    "Precompute RSI: df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()",
    "Define crossover detection: buy_cond = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)",
    "Implement state variables: position, entry_price, cash, shares, waiting_for_reset",
    "Only evaluate buy when position==0 and waiting_for_reset==False and buy_cond is True",
    "Only evaluate sell when position==1 and (rsi>70 OR price<=entry_price*0.95 OR price>=entry_price*1.15)",
    "When sell occurs set waiting_for_reset=True; reset waiting_for_reset per reentry_rule",
    "Use trades list for entry_price: entry_price = trades[-1][2]",
    "Update portfolio_value = cash + shares * current_price AFTER buy/sell logic for each day",
    "If last trade is a Buy, force-close at df['Close'].iloc[-1] and update portfolio_series.iloc[-1] = cash",
    "Plot strategy and portfolio as separate HTML files using trades list for markers."
  ],
  "plots": [
    {"type":"price","y":"Close","axis":"primary","description":"Close price"},
    {"type":"indicator","name":"RSI","y":"RSI","axis":"secondary","description":"RSI (14)"}
  ],
  "indicators_to_plot": ["Close Price","RSI"],
  "safety_checks": ["non-empty df", "indicators present", "buy/sell alternation enforced", "portfolio_series length matches df.index", "final forced-sell updates portfolio_series"]
}

--- FINAL INSTRUCTIONS ---
- Output **ONLY** valid JSON (no markdown, no explanation).  
- Ensure all required schema keys are present (if a field is not applicable set it to null or an empty list as appropriate).  
- Use numeric types for numeric fields, strings for formulas, and boolean where required.  
- Keep code_tasks granular and explicitly mention pandas `.shift(1)` crossover patterns and the re-entry state machine when a re-entry rule is present.
- If the structured_query is ambiguous about crossovers vs inequality, prefer encoding as a crossover when language implies a *move back above/ below* threshold.
- If the query explicitly requests no risk management, set stop_loss_pct and take_profit_pct to null and do not add those conditions.

"""


def translator_mcp(structured_query: dict) -> dict:
    remarks = structured_query.get("remarks", "")

    llm_input = {
        "structured_query": structured_query,
        "remarks": remarks
    }

    messages = [
        SystemMessage(content=TRANSLATOR_PROMPT),
        HumanMessage(content=json.dumps(llm_input, indent=2))
    ]

    response = llm(messages)
    content = response.content.strip()

    print("DEBUG — Raw LLM output from translator:", content)

    try:
        translator_json = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Translator output not valid JSON: {content}")

    # Ensure code_tasks exists and is always a list
    if "code_tasks" not in translator_json or not isinstance(translator_json["code_tasks"], list):
        translator_json["code_tasks"] = []

    return translator_json
