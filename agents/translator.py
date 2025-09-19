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
- condition_code_snippets: { buy: [list of pandas conditions as strings], sell: [list of pandas conditions as strings] }
- code_tasks: ordered list of granular, executable steps for CodeGen (must mention crossover detection, position checks, stop-loss/take-profit only if requested, portfolio updates, final forced-sell handling, plots)
    The code_tasks list must cover all of: 
        - indicator precomputation, 
        - crossover detection with shift(1), 
        - position checks, 
        - stop-loss/take-profit if present, 
        - re-entry safeguard, 
        - portfolio update after trades, 
        - final forced sell, 
        - metrics computation, 
        - plotting with trades list, 
        - results saving. 
        Do not skip any.
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

   - Apply the same “wait-for-reset” logic for any indicator-based strategy, not just RSI. 
   - CodeGen must create a boolean flag that blocks new trades until the indicator first leaves the threshold region and a new crossover occurs.


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

IMPORTANT: Never translate "moves back above" or "moves back below" into > or <. 
Always encode them as crosses_above or crosses_below with shift(1). 

--- RULES (strict, follow exactly) ---

GOLDEN RULE: Every crossover-based strategy must include a re-entry safeguard. 
- After a sell, block re-entry until indicator exits the zone (e.g., RSI < 30). 
- Only allow new entry on the next fresh crossover. 
- Always generate condition_code_snippets from buy_spec and sell_spec.
- These must be copy-paste ready pandas conditions using shift(1) for crossovers.
- CodeGen will use these directly.

1. **Always encode crossovers explicitly** as `crosses_above` / `crosses_below`. Never convert "moves back above 30 after being below 30" to a plain `>` inequality. Use the CONDITION FORMAT for crossovers.

2. **Trade management**
   - Always include `entry_tracking: true`.
   - Only include stop_loss_pct / take_profit_pct when the user explicitly requested them.
   - If a safeguard/re-entry rule is present in the user's query or remarks, populate `trade_management.reentry_rule` with an explicit short instruction (e.g., "After a sell, do not re-enter until RSI first exits oversold/overbought and then forms a fresh crossover above 30").
    - Always convert oversold/overbought thresholds (like RSI < 30 or RSI > 70) into crossover-based conditions.
    - Buy condition must be encoded as: RSI crosses above 30 after having been below 30.
    - Sell condition must be encoded as: RSI crosses below 70 after having been above 70.
    - Re-entry safeguard is mandatory: after a trade is closed, set a state variable (e.g., waiting_for_reset = True).
    - New trades are only allowed when the indicator exits the prior zone and generates a fresh crossover.
    - Explicitly include this re-entry rule in the output JSON (under trade_management.reentry_rule) and in code_tasks.

Re-entry rule:
- After a sell, do not open a new position immediately if RSI is still in the oversold/overbought zone.
- A new buy is only valid when RSI crosses back above 30 after having been below 30.
- A new sell (for short strategies) is only valid when RSI crosses back below 70 after having been above 70.

Validation rule — indicators ↔ conditions (MANDATORY):
- For every condition in buy_spec.conditions and sell_spec.conditions, the referenced "indicator" must appear in the top-level "indicators" list, and that indicators[] entry must contain an "output_column" that will be the DataFrame column name used in code (e.g. "RSI").
- Conversely, every entry in the "indicators" list must appear in "indicators_to_plot".
- If this validation fails, the translator must set an "errors" key in its JSON containing a short list of validation error strings and set "code_tasks": [] so CodeGen will request clarification.
- Example translator behavior: If a buy_spec refers to "RSI" but "indicators" does not include RSI, then return errors:["buy_spec uses RSI but indicators[] missing RSI"].

3. **Code tasks**
   - Provide a granular, ordered list of steps the CodeGen must implement (e.g., precompute indicators, implement crossover detection using shift(1), evaluate buy/sell only when position==0/1, use trades list for entry_price, update portfolio after trades, force-close final open position, save plots).
   - For every crossover step include the exact pandas-style hint: e.g., `cond = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)` and mention `shift(1)` explicitly.
   - For re-entry safeguard include an explicit state pattern and reset condition. Example step: "After a Sell, set `waiting_for_reset=True`; only set `waiting_for_reset=False` when RSI < 30 (for buy re-entry) OR RSI between 30 and 70 as appropriate; only allow new buys when `waiting_for_reset==False` and crossover condition occurs."
    - When multiple tickers are provided, the code must:
        1. Run the backtest for each ticker individually (generating trades, portfolio series, and plots).
        2. Collect key performance metrics (cumulative return, annualized return, volatility, max drawdown) for each ticker into a single results list.
        3. At the end, build a comparison DataFrame (one row per ticker) and save it as a single HTML file (e.g. rsi_comparison_results.html).
        4. Generate a combined equity curve plot overlaying portfolio values of all tickers for direct performance comparison.
        5. Optionally generate a price comparison chart across tickers for reference.

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
    "reentry_rule": "After any trade exit, do not re-enter until the relevant indicator exits the oversold/overbought zone and a fresh crossover occurs. Implement as a boolean state reset per crossover type."},
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

- Consistency Rules:
  - Every indicator used in buy_spec or sell_spec must also be listed in indicators[] with the correct output_column.
  - Every indicator in indicators[] must also appear in indicators_to_plot.
  - This ensures no indicator is missing between specs, calculations, and plots.


--- FINAL INSTRUCTIONS ---
- Output **ONLY** valid JSON (no markdown, no explanation).  
- Ensure all required schema keys are present (if a field is not applicable set it to null or an empty list as appropriate).  
- Use numeric types for numeric fields, strings for formulas, and boolean where required.  
- Keep code_tasks granular and explicitly mention pandas `.shift(1)` crossover patterns and the re-entry state machine when a re-entry rule is present.
- If the structured_query is ambiguous about crossovers vs inequality, prefer encoding as a crossover when language implies a *move back above/ below* threshold.
- If the query explicitly requests no risk management, set stop_loss_pct and take_profit_pct to null and do not add those conditions.
For any generated strategy:
- Always implement a re-entry safeguard using the following pattern.
- Insert the actual buy signal logic into <buy_signal>.
- Insert the actual sell signal logic into <sell_signal>.
- Always include `and not waiting_for_reset` in buy conditions.
- After a sell, set waiting_for_reset = True.
- Reset waiting_for_reset only when the relevant indicator fully exits the trigger zone.

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
