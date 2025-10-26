# translator_agent.py
from langchain_community.chat_models import ChatOpenAI
import json
from datetime import datetime
from typing import Dict, Any
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

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
"trade_management": {
  "entry_tracking": true,
  "stop_loss_pct": null,
  "take_profit_pct": null,
  "reentry_rule": "After any trade exit, do not re-enter until the relevant indicator exits the oversold/overbought zone and a fresh crossover occurs."
}

- "features": list of all features that are present in the user's query.  
  To populate this list:
  1. Include "additional buying" if `additional_buy_condition` is present and non-empty.
  2. Include "entry gating" if `entry_condition` is present.
  3. Include "multi-level buying" if stepwise trends are detected (e.g., "20 EMA < 50 EMA < 250 EMA").
  4. Include "negative crossover" if any buy_condition or additional_buy_condition uses cross_below events.
  5. Include "positive crossover" if any buy_condition or additional_buy_condition uses cross_above events.
  6. Include "stop-loss" if any sell_condition specifies stop_loss or percentage-based loss.
  7. Include "take-profit" if any sell_condition specifies take_profit or percentage-based gain.
  8. Include any named technical indicators explicitly mentioned in buy/sell/additional conditions (e.g., RSI, MACD, SMA_50, EMA_233).
- The goal is to make `features` a complete list of **all strategy elements and indicators that influence buy/sell decisions** in the query.

# Additional buy handling:
- If the user mentions buying additional shares while already holding (e.g., "buy again if price closes below EMA_233 but EMA_55 > EMA_233"):
  - Populate translator_instructions["additional_buy_condition"] with a dictionary using the same operator format as normal crossovers:
    {"indicator":"Close","operator":"<","other_indicator":"EMA_233"}
  - Always include code_tasks to:
      1. Evaluate additional_buy_condition after normal buy logic.
      2. Buy maximum possible shares using available cash if condition is True.
      3. Update trades list, cash, shares, and portfolio_series before sell evaluation.
      4. Do not allow additional buys while waiting_for_reset==True.

- duration_handling: {type: "consecutive"/"non-consecutive", days: int} or null
- condition_code_snippets: { buy: [list of pandas conditions as strings], sell: [list of pandas conditions as strings] }
--- PORTFOLIO-LEVEL REQUIREMENTS (NEW SCHEMA KEYS) ---

    The translator_instructions JSON must include these additional keys:

    - **portfolio_config**: {
        "allocation_per_ticker": 10000,
        "risk_free_rate": 6.5,
        "combine_method": "sum"  // how to aggregate ticker portfolios
    }

    - **required_metrics**: {
        "ticker_level": ["cumulative_return", "annualized_return", "volatility", "max_drawdown"],
        "portfolio_level": ["cumulative_return", "annualized_return", "volatility", "max_drawdown", "sharpe_ratio", "gain_to_loss_ratio"],
        "trade_level": ["total_trades", "winning_trades", "losing_trades", "win_rate", "avg_win", "avg_loss", "profit_factor"]
    }

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
- **required_files** (update to include new files):
  - "{ticker}_strategy_plot_{curr_time_stamp}.html"
  - "{ticker}_portfolio_value_{curr_time_stamp}.html"
  - "trading_results.html"
  - "portfolio_summary_{curr_time_stamp}.html"  // NEW
  - "trade_analysis_{curr_time_stamp}.html"     // NEW
  - "portfolio_equity_curve_{curr_time_stamp}.html"  // NEW
- plots: list of plot descriptors {type, y, axis: "primary"|"secondary", description}
- safety_checks: list of assertions/tests to run
- indicators_to_plot: list of indicator names. MUST always include "Close Price" plus every indicator referenced in buy_spec.conditions and sell_spec.conditions. 
    - This list cannot be empty. 
    - Automatically populate it from the top-level indicators[] entries. 
    - If additional_buy_condition or stepwise_trend is present, include those indicators as well.
    - Do NOT leave it blank or omit any indicator used in rules or plots.

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

   # EMA trend-based crossovers:
        - If the query mentions multiple EMAs (e.g., EMA_55, EMA_233), encode crossovers between EMAs as:
        {"indicator":"EMA_55","operator":"crosses_above","other_indicator":"EMA_233"} 
        or {"indicator":"EMA_55","operator":"crosses_below","other_indicator":"EMA_233"}.
        - Do not convert these into numeric thresholds; always use cross_above / cross_below.
        - Label these as positive_crossover or negative_crossover in the features list.


   - Apply the same “wait-for-reset” logic for any indicator-based strategy, not just RSI. 
   - CodeGen must create a boolean flag that blocks new trades until the indicator first leaves the threshold region and a new crossover occurs.


2) Inequality / price formulas (for stop-loss / take-profit):
   {
     "indicator": "Close",
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

   - When strategy requires ordering of multiple series (e.g., Close > EMA_20 > EMA_50 > EMA_250), encode it as a strict chained inequality
        - DO NOT apply full ordering if any atomic condition is an explicit crossover (crosses_above / crosses_below). 
        - Only enforce chained inequalities when the query explicitly specifies multi-level stepwise trends. 
        - Always wrap composite conditions in parentheses and use & / | consistently.

    - Always use & and | operators consistently in pandas conditions. 
        - Never mix & / | with 'and' / 'or'. 
        - Wrap composite conditions in parentheses to avoid operator precedence issues.

IMPORTANT: Never translate "moves back above" or "moves back below" into > or <. 
Always encode them as crosses_above or crosses_below with shift(1). 

--- RULES (strict, follow exactly) ---

GOLDEN RULE: Every crossover-based strategy must include a re-entry safeguard. 

- After a sell, block re-entry until indicator exits the zone (e.g., RSI < 30). 
# Re-entry safeguards for trend strategies:
- For EMA or MA crossover strategies, after a sell:
    1. Set waiting_for_reset = True
    2. Reset waiting_for_reset only when trend reverses:
       - For positive crossover buy: EMA_55 < EMA_233
       - For negative crossover short/sell: EMA_55 > EMA_233
- Do not allow additional buys while waiting_for_reset==True.

- Only allow new entry on the next fresh crossover. 
- Always generate condition_code_snippets from buy_spec and sell_spec.
- These must be copy-paste ready pandas conditions using shift(1) for crossovers.
- CodeGen will use these directly.

IMPORTANT: Ensure that indicators_to_plot is always filled. 
- Include all indicators used in buy_spec and sell_spec conditions.
- Include all indicators listed under indicators[]. 
- Always include "Close Price" even if not referenced in conditions.
- If the strategy uses multiple indicators, list all of them explicitly.

1. **Always encode crossovers explicitly** as `crosses_above` / `crosses_below`. Never convert "moves back above 30 after being below 30" to a plain `>` inequality. Use the CONDITION FORMAT for crossovers.

2. **Trade management**
   - Always include "entry_tracking": true.
    - If the user explicitly mentions stop-loss, set stop_loss_pct to the numeric value. Otherwise set it to null.
    - If the user explicitly mentions take-profit (target), set take_profit_pct to the numeric value. Otherwise set it to null.
    - Do NOT invent or assume default stop-loss/take-profit values. If not present in the query, these must be null.
     - If the query does not mention stop-loss or take-profit, you MUST set both stop_loss_pct and take_profit_pct to null and you MUST NOT add them in sell_spec.conditions.
    - Do NOT assume default values. Do NOT merge with strategy defaults unless explicitly stated.
     - Always convert oversold/overbought thresholds (like RSI < 30 or RSI > 70) into crossover-based conditions.
    - Buy condition must be encoded as: RSI crosses above 30 after having been below 30.
    - Sell condition must be encoded as: RSI crosses below 70 after having been above 70.
    - New trades are only allowed when the indicator exits the prior zone and generates a fresh crossover.
    - Explicitly include this re-entry rule in the output JSON (under trade_management.reentry_rule) and in code_tasks.

Re-entry rule:
- After a sell, do not open a new position immediately if RSI is still in the oversold/overbought zone.
- A new buy is only valid when RSI crosses back above 30 after having been below 30.
- A new sell (for short strategies) is only valid when RSI crosses back below 70 after having been above 70.
“Always enforce re-entry safeguard. After a sell, set waiting_for_reset=True. Reset it only when the indicator exits the trigger zone (below oversold or above overbought), then allow fresh crossover buy.”

--- STEPWISE TREND RULE ---
- If structured_query includes a "stepwise_trend" key:
    {
        "indicators": ["EMA_20","EMA_50","EMA_250"],
        "direction": "positive"|"negative",
        "action": "buy"|"sell"
    }
- The translator must:
    1. Generate a chained inequality condition between the indicators according to the direction:
        - "negative": indicator[i] < indicator[i+1]
        - "positive": indicator[i] > indicator[i+1]
    2. Add this condition to the appropriate spec (buy_spec or sell_spec) as a dictionary with:
        - "indicator": first indicator in the list
        - "operator": "chained_trend"
        - "formula": the pandas-style chained inequality string
    3. Add the corresponding pandas-ready string in condition_code_snippets["buy"] or ["sell"]
    4. Include a code_task describing evaluation of the stepwise trend before the trade

Validation rule — indicators ↔ conditions (MANDATORY):
- For every condition in buy_spec.conditions and sell_spec.conditions, the referenced "indicator" must appear in the top-level "indicators" list, and that indicators[] entry must contain an "output_column" that will be the DataFrame column name used in code (e.g. "RSI").
- Conversely, every entry in the "indicators" list must appear in "indicators_to_plot".
- If this validation fails, the translator must set an "errors" key in its JSON containing a short list of validation error strings and set "code_tasks": [] so CodeGen will request clarification.
- Example translator behavior: If a buy_spec refers to "RSI" but "indicators" does not include RSI, then return errors:["buy_spec uses RSI but indicators[] missing RSI"].

3. **Code tasks**
   - Provide a granular, ordered list of steps the CodeGen must implement (e.g., precompute indicators, implement crossover detection using shift(1), evaluate buy/sell only when position==0/1, use trades list for entry_price, update portfolio after trades, force-close final open position, save plots).
   - For every crossover step include the exact pandas-style hint: e.g., `cond = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)` and mention `shift(1)` explicitly.
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
      {"indicator":"Close", "operator":"<=", "formula":"entry_price * (1 - 0.05)"},
      {"indicator":"Close", "operator":">=", "formula":"entry_price * (1 + 0.15)"}
    ]
  },
  "trade_management": {
    "entry_tracking": true,
    "stop_loss_pct": 5,
    "take_profit_pct": 15,
    - "reentry_rule": must be adapted to the strategy:
    • For oscillator-based strategies (RSI, Stoch, etc.), use oversold/overbought exit before re-entry.  
    • For EMA/MAs or trend-based strategies, simply enforce: "After any trade exit, set waiting_for_reset=True. Do not allow new entry until a fresh valid crossover occurs."
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
        -  Evaluate additional_buy_condition if position > 0 (already holding). Buy additional shares if condition is True.
        - Update cash, shares, and append a ('Buy', date, price) entry to trades.
        - Portfolio value must be updated after any normal or additional buys and before sell evaluation.
    "Evaluate stepwise trend (chained_trend) as defined in translator_instructions for buy/sell conditions",
        "- Evaluate additional_buy_condition if position > 0 (already holding). Buy additional shares if condition is True.",
        "- Update cash, shares, and append a ('Buy', date, price) entry to trades.",
        "- Portfolio value must be updated after any normal or additional buys and before sell evaluation.",    
    "Plot strategy and portfolio as separate HTML files using trades list for markers."
  ],
  "plots": [
    {"type":"price","y":"Close","axis":"primary","description":"Close price"},
    {"type":"indicator","name":"RSI","y":"RSI","axis":"secondary","description":"RSI (14)"}
  ],
  "indicators_to_plot": ["Close Price","RSI"],
  "safety_checks": ["non-empty df", "indicators present", "buy/sell alternation enforced", "portfolio_series length matches df.index", "final forced-sell updates portfolio_series"]
}

User text:
"Buy when EMA_55 crosses above EMA_233; sell when EMA_55 crosses below EMA_233; buy additional shares if price closes below EMA_233 while EMA_55 > EMA_233."

Translator JSON:
{
  "human_summary": "EMA crossover strategy with additional buy if price drops below EMA_233 but trend positive.",
  "indicators": [
    {"name":"EMA_55","class":"trend","params":{"window":55},"output_column":"EMA_55"},
    {"name":"EMA_233","class":"trend","params":{"window":233},"output_column":"EMA_233"}
  ],
  "buy_spec": {
    "conditions":[
      {"indicator":"EMA_55","operator":"crosses_above","other_indicator":"EMA_233"}
    ]
  },
  "sell_spec": {
    "conditions":[
      {"indicator":"EMA_55","operator":"crosses_below","other_indicator":"EMA_233"}
    ]
  },
  "additional_buy_condition": {"indicator":"Close","operator":"<","other_indicator":"EMA_233"},
  "trade_management": {
    "entry_tracking": true,
    "stop_loss_pct": null,
    "take_profit_pct": null,
    "reentry_rule": "After any trade exit, set waiting_for_reset=True; reset only when EMA_55 < EMA_233; do not allow additional buys while waiting_for_reset==True"
  }
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
-If structured_query includes an "additional_buy_condition", always:
        - Evaluate additional_buy_condition even when position > 0 (already holding shares).
        - If True, buy as many shares as possible using available cash.
        - Update trades list, cash, shares, and portfolio_series accordingly.
        - Include explicit code_tasks for additional buy logic after normal buy logic in the daily loop.
        - Ensure portfolio update reflects any additional buys before selling logic.

For any generated strategy:
- Always implement a re-entry safeguard using the following pattern.
- Insert the actual buy signal logic into <buy_signal>.
- Insert the actual sell signal logic into <sell_signal>.
- Always include `and not waiting_for_reset` in buy conditions.
- After a sell, set waiting_for_reset = True.
- Reset waiting_for_reset only when the relevant indicator fully exits the trigger zone.
- Do not allow additional buys if waiting_for_reset==True.
- Reset waiting_for_reset when indicator fully exits oversold/overbought zone.

"""

# --- Validation: indicators <-> conditions consistency ---
def validate_indicators(translator_json):
    errors = []
    indicators = {ind["output_column"] for ind in translator_json.get("indicators", [])}
    indicators_to_plot = set(translator_json.get("indicators_to_plot", []))

    # Collect all condition indicators
    used_indicators = set()
    for cond in translator_json.get("buy_spec", {}).get("conditions", []):
        if "indicator" in cond:
            used_indicators.add(cond["indicator"])
        if "other_indicator" in cond:
            used_indicators.add(cond["other_indicator"])
    for cond in translator_json.get("sell_spec", {}).get("conditions", []):
        if "indicator" in cond:
            used_indicators.add(cond["indicator"])
        if "other_indicator" in cond:
            used_indicators.add(cond["other_indicator"])

    for cond in translator_json.get("sell_spec", {}).get("conditions", []):
        if cond.get("value_type") == "percent" and "relative_to" not in cond:
            cond["relative_to"] = "entry_price"


    # Rule 1: every used indicator must be declared
    for ind in used_indicators:
        if ind not in indicators:
            errors.append(f"Condition uses {ind} but indicators[] missing it")

    # Rule 2: every declared indicator must appear in indicators_to_plot
    for ind in indicators:
        if ind not in indicators_to_plot:
            errors.append(f"Indicator {ind} declared but missing in indicators_to_plot")

    return errors


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

    response = llm.invoke(messages)
    
    content = response.content.strip()

    print("DEBUG — Raw LLM output from translator:", content)

    try:
        translator_json = json.loads(content)

        # --- Skip chained ordering for explicit crossovers ---
        condition_code_snippets = {"buy": [], "sell": []}

        for spec_key in ["buy_spec", "sell_spec"]:
            for cond in translator_json.get(spec_key, {}).get("conditions", []):
                if cond.get("operator") in ["crosses_above", "crosses_below"] and "other_indicator" in cond:
                    if cond["operator"] == "crosses_above":
                        snippet = f"(df['{cond['indicator']}'] > df['{cond['other_indicator']}']) & " \
                                f"(df['{cond['indicator']}'].shift(1) <= df['{cond['other_indicator']}'].shift(1))"
                    else:
                        snippet = f"(df['{cond['indicator']}'] < df['{cond['other_indicator']}']) & " \
                                f"(df['{cond['indicator']}'].shift(1) >= df['{cond['other_indicator']}'].shift(1))"
                    condition_code_snippets[spec_key.replace("_spec","")].append(snippet)

                translator_json["condition_code_snippets"] = condition_code_snippets


        # --- Consolidated indicator validation ---
        errors = validate_indicators(translator_json)

        # Check buy_spec indicators
        indicators_list = [ind["output_column"] for ind in translator_json.get("indicators", [])]
        for cond in translator_json.get("buy_spec", {}).get("conditions", []):
            if cond.get("indicator") not in indicators_list:
                errors.append(f"buy_spec uses {cond.get('indicator')} but indicators[] missing it")
        # Check sell_spec indicators
        for cond in translator_json.get("sell_spec", {}).get("conditions", []):
            if cond.get("indicator") not in indicators_list:
                errors.append(f"sell_spec uses {cond.get('indicator')} but indicators[] missing it")

        # If additional_buy_condition exists, include its indicator(s) in indicators_to_plot
        additional_buy = translator_json.get("additional_buy_condition", {})
        if additional_buy:
            for cond in additional_buy.get("conditions", []):
                if "indicator" in cond and cond["indicator"] not in indicators_list:
                    indicators_list.append(cond["indicator"])
                if "other_indicator" in cond and cond["other_indicator"] not in indicators_list:
                    indicators_list.append(cond["other_indicator"])

                # Every indicator must be in indicators_to_plot
                indicators_to_plot = translator_json.get("indicators_to_plot", [])
                for ind in indicators_list:
                    if ind not in indicators_to_plot:
                        errors.append(f"indicator {ind} missing in indicators_to_plot")

                if errors:
                    translator_json["errors"] = errors

        # --- Ensure safety_checks ---
        required_safety = [
            "non-empty df",
            "indicators present",
            "buy/sell alternation enforced",
            "portfolio_series length matches df.index",
            "final forced-sell updates portfolio_series"
        ]
        if "safety_checks" not in translator_json or not isinstance(translator_json["safety_checks"], list):
            translator_json["safety_checks"] = required_safety
        else:
            for check in required_safety:
                if check not in translator_json["safety_checks"]:
                    translator_json["safety_checks"].append(check)

        # --- Ensure code_tasks exists ---
        if "code_tasks" not in translator_json or not isinstance(translator_json["code_tasks"], list):
            translator_json["code_tasks"] = []

        # --- Re-entry safeguard tasks ---
        reentry_tasks = [
            "Initialize waiting_for_reset = False at the start of the loop",
            "After a sell, set waiting_for_reset = True",
            "Do not evaluate normal or additional buy while waiting_for_reset == True",
            "Reset waiting_for_reset only when the relevant indicator fully exits the trigger zone",
            "Include waiting_for_reset in all buy condition pandas expressions: 'and not waiting_for_reset'"
        ]
        # Prepend reentry tasks
        translator_json["code_tasks"] = reentry_tasks + translator_json["code_tasks"]

        portfolio_tasks = [
            "After processing all tickers, aggregate portfolio_series across all tickers",
            "Calculate portfolio-level metrics: annualized return, volatility, max drawdown, Sharpe ratio, gain-to-loss ratio",
            "Generate portfolio_summary_{{timestamp}}.html with overall metrics and per-ticker contribution",
            "For each ticker, analyze trades to calculate win rate, avg win/loss, profit factor",
            "Generate trade_analysis_{{timestamp}}.html with trade accuracy metrics per ticker",
            "Create portfolio_equity_curve_{{timestamp}}.html showing aggregated portfolio value over time"
        ]
        translator_json["code_tasks"].extend(portfolio_tasks)

        # --- Handle additional_buy_condition if present ---
        additional_buy = structured_query["structured_query"].get("additional_buy_condition")
        print(structured_query)
        print(additional_buy)
        if additional_buy:
            translator_json["additional_buy_condition"] = additional_buy

            # Update features
            features = set(translator_json.get("features", []))
            features.add("additional buying")
            translator_json["features"] = list(features)

            # Update indicators_to_plot
            indicators_to_plot = set(translator_json.get("indicators_to_plot", []))
            for key in ["indicator", "other_indicator"]:
                if key in additional_buy:
                    indicators_to_plot.add(additional_buy[key])
            translator_json["indicators_to_plot"] = list(indicators_to_plot)

            # Append code_tasks for additional buy
            additional_buy_tasks = [
                "Evaluate additional_buy_condition even when position > 0 (already holding).",
                "If True, buy as many shares as possible using available cash.",
                "Update trades list, cash, shares, and portfolio_series accordingly.",
                "Ensure portfolio update reflects any additional buys before selling logic."
            ]
            translator_json["code_tasks"].extend(additional_buy_tasks)
        
        for cond in translator_json.get("buy_spec", {}).get("conditions", []):
            if cond.get("operator") in ["crosses_above","crosses_below"]:
                indicator_a = cond["indicator"]
                indicator_b = cond.get("other_indicator")
                if indicator_b:
                    reentry_rule_generic = f"Reset waiting_for_reset only when {indicator_a} crosses opposite direction relative to {indicator_b}"
                    reentry_tasks.append(reentry_rule_generic)

            if "threshold" in cond:
                reentry_rule_generic = f"Reset waiting_for_reset only when {cond['indicator']} exits the trigger zone"
                reentry_tasks.append(reentry_rule_generic)


        # --- Additional buy condition tasks ---
        
        if "additional_buy_condition" in structured_query and structured_query["additional_buy_condition"]:
            translator_json["additional_buy_condition"] = structured_query["additional_buy_condition"]

            additional_buy_steps = [
                "Evaluate additional_buy_condition even when position > 0 (already holding).",
                "If True, buy as many shares as possible using available cash.",
                "Update trades list, cash, shares, and portfolio_series accordingly.",
                "Ensure portfolio update reflects any additional buys before selling logic."
            ]
            translator_json["code_tasks"].extend(additional_buy_steps)

            # Append after normal buy logic
            for step in additional_buy_steps:
                if step not in translator_json["code_tasks"]:
                    translator_json["code_tasks"].append(step)

        # --- Stop-loss / take-profit handling ---
        tm = translator_json.get("trade_management", {})
        tm["entry_tracking"] = True
        tm.setdefault("stop_loss_pct", structured_query.get("stop_loss", None))
        tm.setdefault("take_profit_pct", structured_query.get("take_profit", None))
        translator_json["trade_management"] = tm

    except json.JSONDecodeError:
        raise ValueError(f"Translator output not valid JSON: {content}")

    # Final safeguard: ensure code_tasks exists
    if "code_tasks" not in translator_json or not isinstance(translator_json["code_tasks"], list):
        translator_json["code_tasks"] = []

    return translator_json

