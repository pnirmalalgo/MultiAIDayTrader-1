from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature=0.2,
    openai_api_key=api_key,
    model="gpt-4o-mini"
)

def get_code_tasks(translated):
    for key, value in translated.items():
        #print(key, ":", value)
        if key == "code_tasks":
            return value if isinstance(value, list) else []
        
    return []

def codegen_mcp(payload: dict) -> dict:
    curr_time_stamp = datetime.now().isoformat()


    #translated = payload.get("translated_query", {})
    #structured_query = payload.get("structured_query", {})
    # find translator output in common places
    translated = {}
    # priority: explicit translated_query -> instructions -> top-level translated/instructions
    if isinstance(payload.get("translated_query"), dict) and payload.get("translated_query"):
        translated = payload["translated_query"]
    elif isinstance(payload.get("instructions"), dict) and payload.get("instructions"):
        translated = payload["instructions"]
    elif isinstance(payload.get("translated"), dict) and payload.get("translated"):
        translated = payload["translated"]
    else:
        # last resort: use structured_query if it already contains instruction-like keys
        translated = payload.get("translated_query") or payload.get("instructions") or payload.get("structured_query") or {}

    structured_query = payload.get("structured_query", {}) or {}
    trade_management = translated.get("trade_management", {})
    reentry_rule = trade_management.get("reentry_rule", "")


    #print("Input to CodeGen MCP:", payload)
    if not translated and not structured_query:
        return {
            "thought": "Neither translated_query nor structured_query provided.",
            "action": "AskClarification",
            "action_input": "Please provide a valid input with at least structured_query."
        }

    # Prefer translator output
    strategy = translated.get("strategy_plan") or structured_query.get("strategy_description", "")
    #buy_condition = translated.get("buy_condition_code") or structured_query.get("buy_condition", "")
    buy_condition = translated.get("buy_condition", {}).get("conditions", []) or translated.get("buy_spec", {}).get("conditions", [])
    sell_condition = translated.get("sell_condition", {}).get("conditions", []) or translated.get("sell_spec", {}).get("conditions", [])
    #code_tasks = translated.get("code_tasks", [])
    # Try multiple locations in order, stop at the first non-empty list

    code_tasks = get_code_tasks(payload)
    #print("Final code_tasks:", code_tasks)


    #print("Buy conditions:", buy_condition)
    #print("Sell conditions:", sell_condition)
    #print("Code tasks for CodeGen:", code_tasks)

    #sell_condition = translated.get("sell_condition_code") or structured_query.get("sell_condition", "")
    duration_type = structured_query.get("duration_type", "")
    duration_days = int(structured_query.get("duration_days", 0))
    remarks = translated.get("remarks") or structured_query.get("remarks", "")

    translator_notes = translated.get("codegen_instructions", "")
    #code_tasks = input.get("code_tasks", [])



    CODEGEN_PROMPT = """
You are the Code Generator agent in a multi-agent trading system.

INPUT: translator_instructions JSON from the Translator agent.

OUTPUT: Executable Python script that strictly implements all strategy rules. Do not include '''python''' or any markdown formatting.
Before writing the code, explain your reasoning under ---THOUGHTS---. 
Then output the executable code under ---CODE---. 
Do not mix them.

---

RULES FOR CODE GENERATION:

GOLDEN RULES (you must respect these before anything else):
1. Never recompute indicators inside the loop — always precompute. 
2. Always shift rolling windows by 1 day so today is compared to the past only. 
3. Always update portfolio value after executing buy/sell logic, not before. 
4. Always use trades list (not df masks) for entry_price, stop-loss, take-profit, and plotting. 
5. Always force-close last open trade on final date and set portfolio_series.iloc[-1] = cash. 
6. All calculations, trades, metrics, and plots are per ticker independently. 
7. Do not truncate the last buy trade.
        1. If the strategy ends with an open position (i.e., last trade is a buy), then automatically close it on the final available closing price in the dataset.
        2. Record this as a sell trade at the last date.
        3. Use this final sell to ensure that the portfolio value and performance metrics reflect a fully closed position by the end of the backtest.
        4. If a final forced sell is executed at the last closing price, make sure to update the portfolio series so that the last element reflects the new cash-only balance. Explicitly set portfolio_series.iloc[-1] = cash after this forced sell to ensure the portfolio plot and metrics are consistent.

- For any strategy, always interpret translator conditions (buy_spec, sell_spec, additional_buy(only if additional buy is mentioned) thresholds) dynamically rather than hardcoding EMA or RSI levels.
- All formulas or thresholds from translator instructions must be applied as-is using precomputed DataFrame columns.
- Always guard entry_price usage:
      if position == 1 and entry_price is not None:
          # stop-loss / take-profit / additional buy(only if additional buy is mentioned)
        
- All sell, stop-loss, take-profit, reentry, and additional buy(only if additional buy is mentioned) checks must be inside:

      if position == 1 and entry_price is not None:
          ...

  This guard is MANDATORY. Never allow code like `current_price < entry_price * ...` outside this block.

    - Never include `entry_price` in any vectorized pandas Series expressions. 
    - `entry_price` must only be used inside the backtest loop, guarded with `if position == 1 and entry_price is not None:`.
    - All stop_loss and take_profit checks must be loop-based using entry_price (scalar), not vectorized.
- Do not reference entry_price in calculations unless it is not None.
- This prevents NoneType errors in stop-loss/take-profit logic.
MUST: Guard all entry_price arithmetic.
- Anywhere the generated code uses `entry_price` in arithmetic (stop-loss, take-profit, risk calcs), it MUST first check `position == 1` AND `entry_price is not None` (or `shares > 0`) before performing the calculation.
- Example requirement wording to include: "Do not perform `entry_price * ...` unless `position == 1 and entry_price is not None`."
**ENFORCE STRICT GUARD**:
- Do NOT perform any arithmetic using `entry_price` (e.g., *, /, +, -) **unless**:
      if position == 1 and entry_price is not None:
- If a generated line violates this rule, your code MUST raise an Exception or skip it.
- All reset, take-profit, stop-loss, reentry checks that reference entry_price must live inside this guard.
- Include explicit examples in the loop:

    if position == 1 and entry_price is not None:
        if current_price >= entry_price * (1 + TAKE_PROFIT_PERCENT/100):
            # sell logic

- Any line outside this guard that uses entry_price should never appear.

8. MUTUAL EXCLUSIVITY (MUST): Trade decision code MUST use mutually-exclusive branches so a single bar cannot execute both Buy and Sell.
   - The generated backtest loop MUST follow the provided TRADE LOOP TEMPLATE below exactly (or an equivalent that uses `if ... elif ...` semantics and an executed_action guard).
   - Do not produce two independent `if` blocks for buy and sell. If buy logic executes on a bar, sell logic must be skipped for that same bar.

   
9. #### ADDITIONAL BUY RULES #####(only if additional buy is mentioned)
- Additional Buy is only executed if position > 0, cash > 0, additional_buy_condition is True, and waiting_for_reset == False.
- Do NOT merge normal buy and additional buy conditions into one if statement.
- Always update cash, shares, and portfolio_series immediately after executing an additional buy.
- Include shares_bought in the trade tuple:
    Normal Buy tuple: ("Buy", current_date, current_price, shares_bought)
    Additional Buy tuple: ("Additional Buy", current_date, current_price, shares_bought)
- Check additional buy **after normal buy logic** but **before sell logic** in each iteration.

For additional buys below entry price, implement sequential buys: 
- Track number of additional buys per trade. 
- First additional buy triggers only after first threshold is hit; second triggers only after second threshold. 
- Prevent multiple additional buys at the same threshold on the same trade.

####Pseudo-code when additional_buy_condition is mentioned:#####(only if additional buy is mentioned)
additional_buy_done = False

for i in range(len(df)):
    executed_action = None

    # Normal Buy
    if position == 0 and not waiting_for_reset and normal_buy_condition:
        execute_normal_buy()
        position = 1
        additional_buy_done = False
        executed_action = "Buy"

    # Additional Buy (safe guarded against NoneType)
    if position == 1 and entry_price is not None:
        additional_buy_cond = (
            df['additional_buy_signal'].iloc[i]
            if pd.notna(df['additional_buy_signal'].iloc[i]) else False
        )
    else:
        additional_buy_cond = False

    if position > 0 and cash > 0 and not waiting_for_reset and additional_buy_cond and not additional_buy_done:
        execute_additional_buy()
        additional_buy_done = True
        executed_action = "Additional Buy"
        # Immediately update portfolio value
        portfolio_series[i] = cash + shares * current_price

    # Sell
    if position == 1 and entry_price is not None and sell_condition:
        execute_sell()
        position = 0
        waiting_for_reset = True
        additional_buy_done = False
        executed_action = "Sell"
        portfolio_series[i] = cash + shares * current_price

    # If no trade executed this bar, still update portfolio value
    if executed_action is None:
        portfolio_series[i] = cash + shares * current_price

-------#####--------
0. ABSOLUTE NONE-SAFETY RULE:
   - Never perform arithmetic using `entry_price` unless explicitly guarded.
   - Any use of entry_price in a calculation (multiplication, division, addition, subtraction) MUST be wrapped in:

        if position == 1 and entry_price is not None:
            # safe to use entry_price here

   - Outside this guard, entry_price may only be assigned or reset (e.g., entry_price = current_price on Buy, entry_price = None on Sell).
   - If code would otherwise attempt `entry_price * ...` or similar without this guard, you must skip or raise Exception in generated code.

1. **Data Handling**
   - Load data from SQLite (no external APIs). Database: market_data.db Table: stock_data Columns: "Date", "Open", "High", "Low", "Close", "Volume"
   - Convert Date to datetime, sort ascending, set as index.
   - Fill missing values using both bfill + ffill.
   - When calculating a rolling statistic over the past N periods, shift it by 1 period so that today’s value is only compared against the previous N periods, excluding today.

   # For multiple tickers:
        - Load each ticker’s data independently from the SQLite database.
        - Each ticker’s DataFrame should include a 'Ticker' column for identification (optional for single-ticker backtests).
        - Perform all calculations (rolling statistics, indicators, buy/sell signals, portfolio simulation) **per ticker independently**.
        - Do NOT combine data from multiple tickers into a single DataFrame for calculations unless simulating a combined portfolio is intended.
        - If combining multiple tickers, use pd.concat(list_of_dataframes) instead of df.append(), and sort by ['Ticker', 'Date'] to preserve ticker separation.
        - Apply rolling statistics, indicators, and trade logic **per ticker** to avoid cross-ticker contamination.

2. **Indicators**
   - Use ta library for indicators (e.g., ta.momentum.RSIIndicator, ta.trend.SMAIndicator).
   - Precompute all indicators from translator_instructions["indicators"].
   - Ensure column names match translator_instructions["indicators"][].output_column exactly.
   - Do not recompute indicators inside the backtest loop.
   - If translator_instructions contains `indicators_to_plot`, ensure every item in that list is explicitly plotted in the strategy figure.


3. ***Implement Buy/Sell logic exactly as defined in translator_instructions:***
   - Only execute Buy if position == 0.
   - Only execute Sell if position == 1.
   - In the loop, always check:
    if position == 1 and entry_price is not None:
        sell_cond_rsi = current_rsi > 70
        sell_cond_sl = current_price <= entry_price * (1 - STOP_LOSS_PERCENT/100)
        sell_cond_tp = current_price >= entry_price * (1 + TAKE_PROFIT_PERCENT/100)
        if sell_cond_rsi or sell_cond_sl or sell_cond_tp:
            # execute sell
    - The Buy condition must include `and not waiting_for_reset`.
    - The Sell condition must always set `shares = 0` after executing the sell.
    - This ensures the portfolio is fully liquid after any exit.
   - Always **check `position` before executing sell**, so sells do not occur without a prior buy.
   - Respect stop-loss and take-profit only if explicitly provided.
   - Track trades as (action, date, price) in a list.
   - **Always use the trades list for determining entry price and for any sell conditions — do not compute sell signals from the DataFrame alone.**
   - For crossovers (golden/death cross, RSI thresholds, etc.):
     - Use explicit detection: `cond = (A > B) & (A.shift(1) <= B.shift(1))` for cross above.
     - Similarly, `cond = (A < B) & (A.shift(1) >= B.shift(1))` for cross below.
   - Respect stop_loss / take_profit:
     - Use `entry_price` per trade.
     - stop_loss: `current_price <= entry_price * (1 - stop_loss_pct/100)`.
     - take_profit: `current_price >= entry_price * (1 + take_profit_pct/100)`.

- **Calculate all indicator series (RSI, SMA, etc.) before the loop**, do not recalc per iteration.

- If operator is "chained_trend" or "stepwise_trend", use the precomputed formula string in `formula` directly in your loop.
    - Example: formula = "(df['EMA_20'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_250'])"
    - Use this as part of `normal_buy_condition` or `sell_condition` inside the backtest loop.


##### Additional Buy Handling#### (Only if additional buy is mentioned in input)
##### ADDITIONAL BUY RULES ##### (Only if additional buy is mentioned in input)
- Only execute Additional Buy if:
    1. position > 0
    2. cash > 0
    3. additional_buy_condition is True
    4. waiting_for_reset == False
- Additional Buy must be **separate from normal buy**; do NOT combine conditions.
- Only allow **one additional buy per bar** (avoid repeated buys on consecutive bars unless a new trade opens).
- Immediately update:
    cash = cash - shares_bought * current_price
    shares += shares_bought
    portfolio_series[i] = cash + shares * current_price
- Append to trades list as ("Additional Buy", date, price, shares_bought)
- Reset any per-trade flag after sell to allow future additional buys in new trades.
- Use a flag `additional_buy_done = False` per open position:
    - Set to True after executing additional buy
    - Reset to False after the position is closed

- Inside the backtest loop, check if translator_instructions contains "cond_additional_buy".
- Execute the additional buy **only if position > 0 and cash > 0**.
- Buy as many shares as possible using available cash at current_price.
- Append a trade tuple: ("Additional Buy", current_date, current_price, shares_bought) to trades list.
- Immediately update cash = cash - shares_bought * current_price and shares += shares_bought.
- Immediately update portfolio_series[i] = cash + shares * current_price.
- Ensure waiting_for_reset rules are respected: do not execute additional buy if waiting_for_reset == True.
- Additional buys are separate from normal buy; normal buy only occurs if position == 0.

- Implement sequential additional buys based on multiple thresholds (e.g., -10%, -20%) without overlapping.
- Track number of additional buys per open trade (e.g., additional_buy_count).
- Execute first additional buy only if additional_buy_count == 0 and threshold met.
- Execute second additional buy only if additional_buy_count == 1 and next threshold met.
- Prevent multiple buys at same threshold on the same trade.
- Always update cash, shares, portfolio_series, and trades immediately after each additional buy.
- Ensure this is dynamic: thresholds and number of additional buys come from translator instructions.

##### Additional Notes for Buy Logic ####(Only if additional buy is mentioned in input)
- Normal buy and additional buy are handled in separate conditional blocks.
- Normal buy executes only if position == 0 and not waiting_for_reset and buy condition is met.
- Additional buy executes only if position > 0, cash > 0, additional buy condition is met, and waiting_for_reset == False.
- Do not combine normal and additional buy in a single if condition.
- In each iteration of the backtest loop:
    1. Check for normal buy if position == 0
    2. Check for additional buy if position > 0
    3. Check for sell conditions
    4. Update portfolio_series after each trade (buy, additional buy, or sell)
- Include "shares_bought" in the trade tuple for additional buys: ("Additional Buy", current_date, current_price, shares_bought)
- Normal buy tuple: ("Buy", current_date, current_price, shares_bought)
- Always update portfolio_series[i] immediately after executing any trade.

- If a condition in translator_instructions contains "relative_to": "entry_price", generate Python code that evaluates the threshold relative to the current entry_price of the open position.
  - This calculation must occur inside the guard: if position == 1 and entry_price is not None.
  - Do not precompute or vectorize this threshold outside the backtest loop.
  - Use entry_price to calculate stop-loss, take-profit, or additional buy(only if additional buy is mentioned) thresholds as needed.
  - Example for an additional buy at -10%:  
        threshold_price = entry_price * (1 - 0.10)
        if current_price <= threshold_price:
            # execute additional buy
  - Repeat this logic for all percentage-based buy/additional buy(only if additional buy is mentioned)/sell conditions referencing "relative_to": "entry_price".




4. **Position & Trade Tracking**
   - Track `position` (0 = no position, 1 = holding position), `entry_price`, `cash`, `shares`.
        - Only execute Buy if position == 0.
        - Only execute Sell if position == 1.
        - Append trades with (action, date, price).
        - Ensure every Buy has a later matching Sell (align trades before plotting).
   - Append trades with (`action`, `date`, `price`).
   - No Sell without an active Buy.
   - Ensure every Buy has a later matching Sell (align trades before plotting).
   - When calculating daily portfolio value, always update `portfolio_value = cash + shares * current_price` **after executing Buy/Sell logic** for that day, not before.
   - When executing a Sell (due to RSI, stop-loss, or take-profit):
        - Add `shares * current_price` to `cash`.
        - Explicitly set `shares = 0` immediately after the sell.
        - Reset `entry_price = None`.
        - Set `position = 0`.
        - Update `waiting_for_reset = True` (per re-entry rules).

- Implement `waiting_for_reset` to prevent immediate re-entry after a sell.
    - After a sell, set `waiting_for_reset = True`.
    - Reset `waiting_for_reset = False` only when the next valid re-entry condition occurs 
      (e.g., golden cross for EMA strategies, or the translator-specified buy setup turning 
      negative → positive again).
    - Do NOT reset waiting_for_reset inside sell logic.
    - Always include `and not waiting_for_reset` in buy condition.
    - This ensures multiple trade cycles can occur: Sell → wait for reset → Buy again.


##### FORCE-CLOSE LAST OPEN TRADE #####
- After the main loop finishes, check:
    if position == 1 and entry_price is not None and shares > 0:
        # execute sell at last available closing price
        cash += shares * df['Close'].iloc[-1]
        trades.append(("Sell", df.index[-1], df['Close'].iloc[-1], shares))
        shares = 0
        position = 0
        entry_price = None
        portfolio_series.iloc[-1] = cash
- This ensures portfolio_series and performance metrics reflect a fully closed position at the end.
- Include this step **after daily loop** and before metrics calculation.

VERY IMPORTANT:
- Whenever generating Python code with multiple logical conditions using & and |, always use parentheses to make the intended order of operations explicit. Ensure the final condition evaluates exactly as described in the logic, rather than relying on Python’s operator precedence.

5. **Portfolio Simulation**
   - Start with initial_capital = 100000 (or 10,000 if specified in translator_instructions).
   - **Update portfolio_series after executing trades** each day.
   - Store as `portfolio_series` aligned with df index.
   - Ensure initial capital line spans full index length when plotting.
   - After evaluating buy/sell logic **for each day**, immediately calculate and record portfolio value as `cash + shares * close_price`.
   - Store this daily portfolio value in `portfolio_series` aligned with df index.
   - portfolio_series must be a pandas Series indexed by df.index, length exactly == len(df). 
   - If loop starts from i=1, slice df.index[1:]; if from i=0, guard lookbacks with if i > 0. 
   - Align portfolio_series with df index.
   - Only plot portfolio_series in portfolio_fig; do not include other price/indicator traces.
   - Aggregated Buy/Sell markers should appear only in strategy_fig.
   - After evaluating Buy/Sell logic for the day, immediately calculate:
      portfolio_value = cash + shares * current_price
      and store in portfolio_series for that day.
    - After the final forced sell (if any), **forward-fill the portfolio_series** for remaining days so that the portfolio value remains flat when no shares are held:
        portfolio_series.ffill(inplace=True)
    - Ensure that at any point after a Sell, `shares = 0` so portfolio_series correctly reflects a cash-only balance.


6. **Performance Metrics**
    Always compute metrics in this order after trades are complete: 
        1. cumulative_return 
        2. daily_returns 
        3. volatility 
        4. max_drawdown 
        5. annualized_return
   - Cumulative Return = (Vend/Vstart - 1) * 100
   - Annualized Return = ((Vend/Vstart) ** (252 / total_days) - 1) * 100. Use total_days = len(df).
   - Volatility = std(daily_returns) * sqrt(252) * 100
   - Max Drawdown = max(1 - portfolio / cummax(portfolio)) * 100
   
   - Guard against empty DataFrame or zero trades.
   - Save results in form of table with columns Ticker, Cumulative Return, Annualized Return, Volatility and Max Drawdown and one row for each ticker. Save table to trading_results.html.
   Aggregate-metrics and output file rule (MANDATORY):
- DO NOT save `trading_results.html` inside the per-ticker processing function. The per-ticker function (e.g., process_ticker) MUST return: `(metrics_dict, generated_files_for_this_ticker)`.
- The top-level script MUST:
    1. Loop over tickers, call process_ticker, collect all metrics dicts into `all_metrics` list and extend `all_generated_files` with each ticker's files.
    2. After the loop, create `results_df = pd.DataFrame(all_metrics)` and SAVE `results_df.to_html('trading_results.html', index=False)` ONCE — so the results table has one row per ticker.
    3. Append 'trading_results.html' to `all_generated_files`.
    4. Write `all_generated_files` into `generated_files_{timestamp}.json` and print json.dumps(all_generated_files).
        - Example pseudocode to be implemented exactly:
            all_metrics = []
            all_generated_files = []
            for ticker in tickers:
                metrics, files = process_ticker(ticker)
                all_metrics.append(metrics)
                all_generated_files.extend(files)
            results_df = pd.DataFrame(all_metrics)
            results_df.to_html('trading_results.html', index=False)
            all_generated_files.append('trading_results.html')
            with open(f'generated_files_{timestamp}.json','w') as f:
                json.dump(all_generated_files, f)
    print(json.dumps(all_generated_files))

       ****VERY IMPORTANT*****
            
            2. Use pandas Series for portfolio values for easier pct_change(), cummax(), and indexing.
            3. Compute daily returns as `portfolio_series.pct_change().dropna()`.
            4. Volatility = daily_returns.std() * sqrt(252) * 100.
            5. Max drawdown = (1 - portfolio_series / portfolio_series.cummax()).max() * 100.
            6. Annualized return = ((final_portfolio_value / initial_capital) ** (252 / n_days) - 1) * 100, where n_days = len(df).
            7. All calculations (MA, RSI, returns) should be **per ticker**, even for multi-ticker backtests.
            8. If no trades are executed during the backtest, set all performance metrics (cumulative return, annualized return, volatility, max drawdown) to 0 and print "No trades executed in this period.".
                - Only calculate metrics using the portfolio_series when trades exist.
                - Plotting should also skip buy/sell markers gracefully if no trades were executed.
        
######Approach#######:
        Load data per ticker separately

        Query historical OHLCV data from SQLite for each ticker.

        Convert the Date column to datetime, set as index, sort, and fill missing values.

        Calculate indicators per ticker

        Compute moving averages, RSI, or other indicators on a per-ticker basis.

        Use .rolling() and .shift() as needed.

        Backtest logic per ticker

        Initialize variables: position, cash, shares, portfolio_series, trades.

        Loop over the rows of the ticker’s DataFrame:

        Buy condition: Check strategy-specific rules, execute buy if not already holding.

        Sell condition: Check exit rules, execute sell if holding.

        - Store executed price in `trades` at the time of Buy/Sell.
        - For stop-loss or sell checks, use the stored price from the trades list (e.g., trades[-1][2]) instead of re-fetching it from the DataFrame.
        - Do not use df.iloc with a timestamp (like trades[-1][1]) because it causes type errors.

        Portfolio update: After buy/sell logic, update portfolio_value = cash + shares * current_price and store in portfolio_series.
        - **Important:** Do not update portfolio before the buy/sell logic — always update after executing trades for the day.

        Calculate performance metrics per ticker.

        Cumulative Return = (final_portfolio_value / initial_capital - 1) * 100

        Annualized Return = ((final_portfolio_value / initial_capital) ** (252 / total_days) - 1) * 100

        Daily returns = portfolio_series.pct_change().dropna()

        Volatility = daily_returns.std() * sqrt(252) * 100

        Max Drawdown = (1 - portfolio_series / portfolio_series.cummax()).max() * 100

        Store results efficiently

        Instead of using the deprecated DataFrame.append(), either:

        Use pd.concat([existing_df, new_df], ignore_index=True) after creating a small DataFrame with the new row, or

        Append dictionaries to a list and convert the list to a DataFrame after the loop (preferred for speed and clarity).

        ##### Portfolio Update Timing ####
        - **Do NOT update portfolio_series at the start of the loop.**
        - Update portfolio_series **only after executing all trades** (normal buy, additional buy(only if additional buy is mentioned), or sell) for the current iteration.
        - At the **end of each backtest loop iteration**, set:
            portfolio_series.iloc[i] = cash + shares * current_price
        - This ensures that portfolio value on buy or additional buy(only if additional buy is mentioned) days reflects the newly acquired shares.
        - Always update portfolio_series immediately after executing any trade:
            1. Normal Buy (position == 0)
            2. Additional Buy (position > 0)(only if additional buy is mentioned)
            3. Sell (position == 1)
        - Do not compute portfolio value twice per iteration; only compute **once after all trade logic**.

        ##### PORTFOLIO UPDATE SEQUENCE #####
        - For each iteration of the backtest loop:
            1. Check Normal Buy (position == 0)
                - If executed, immediately update cash, shares, portfolio_series[i]
            2. Check Additional Buy (position > 0)(only if additional buy is mentioned)
                - If executed, immediately update cash, shares, portfolio_series[i]
            3. Check Sell (position == 1)
                - If executed, immediately update cash, shares, portfolio_series[i]
        - Only compute portfolio_series **once per trade**, never at the start of the loop.

        
        Plotting per ticker

        Create two plots per ticker:

        Strategy plot with price, indicators, buy/sell markers.

        Portfolio value plot with initial capital line.

        Save results

        Save per-ticker HTML plots and the final results table (results.to_html()).

*****Important implementation details:******
- IMPORTANT: Never call .shift() on scalar variables like current_rsi. 
  .shift() must only be applied to pandas Series, e.g. df['RSI'].shift(1).
- Precompute all crossover signals at the DataFrame/Series level before entering the backtest loop. 
  Example: df['buy_signal'] = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)
- Inside the loop, refer to df['buy_signal'].iloc[i] (or .iat[i]) instead of applying shift again.

- Always initialize variables before the backtest loop.
  For example:
    position = 0
    entry_price = None
    shares = 0
    cash = INITIAL_CAPITAL
- When entering a trade (buy), set entry_price = current_price.
- When exiting a trade (sell), reset entry_price = None.
- Always initialize `entry_price = None` before the backtest loop.
- Update `entry_price` to the trade’s price whenever a buy order is executed.
- Reset `entry_price` back to None whenever a position is closed.
- Do NOT use entry_price inside vectorized DataFrame conditions.
- All stop_loss and take_profit checks that depend on entry_price must be handled INSIDE the backtest loop, only when position == 1 and entry_price is not None.
- At the DataFrame level, only precompute indicator-based signals (like RSI crossovers). Do not mix entry_price-dependent logic with Series operations.
- Example:
    df['buy_signal'] = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)

    # In loop:
    if position == 1 and entry_price is not None:
        sell_cond_rsi = current_rsi > 70
        sell_cond_sl = current_price <= entry_price * 0.95
        sell_cond_tp = current_price >= entry_price * 1.15
        if sell_cond_rsi or sell_cond_sl or sell_cond_tp:
        
- When checking stop-loss or take-profit, always guard the calculation:
    only evaluate conditions if `entry_price is not None` and `position == 1`.
    Example:
        if position == 1 and entry_price is not None:
            sell_cond_stop_loss = current_price <= entry_price * (1 - STOP_LOSS_PERCENT / 100)
            sell_cond_take_profit = current_price >= entry_price * (1 + TAKE_PROFIT_PERCENT / 100)

- In sell conditions, always check entry_price is not None before using it.
- When writing the backtesting loop:
    - Always structure trade conditions using `if ... elif ...` instead of two separate `if` blocks, so that buy and sell actions cannot both trigger on the same day.
    - Ensure that once a Buy or Sell executes, the other condition is skipped for that bar.


7. **Plots** (Use plotly, save as HTML)
    "IMPORTANT: Your trades list already contains executed buys/sells (tuples (action,date,price)). Use that list to plot markers and to derive entry_price for stop-loss/take-profit logic — do NOT recompute or infer trades from indicator boolean masks."

   - Strategy Plot: - Include price, all indicators in `indicators_to_plot`, and buy/sell markers.
   - Buy/Sell markers on strategy plot must reflect the **trades list**, NOT the raw indicator conditions.
   - Do not use DataFrame boolean masks (like `RSI<30`) for plotting; only use dates/prices from executed Buy/Sell tuples (i.e. from the trades list).
   
   - Plot a single Scatter trace for all Buys and another for all Sells to avoid multiple legend entries.
   - Portfolio Plot: daily portfolio value (initial investment baseline if specified).
   *PLOT RULES — REQUIRED IMPLEMENTATION DETAILS (copy/paste safe)*
    - Use plotly.subplots.make_subplots with secondary_y=True for strategy plots:
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    - ALWAYS add price/moving-average traces using: fig.add_trace(<trace>, secondary_y=False)
    - ALWAYS add oscillator traces (RSI, MACD, etc.) using: fig.add_trace(<trace>, secondary_y=True)
    - Do NOT plot oscillators on the primary axis (do not use boolean df masks like df[df['RSI']<30] for marker plotting).
    - **Use the trades list for Buy/Sell markers only**. Do not compute markers from indicator conditions. Example for markers:
        buy_dates = [t[1] for t in trades if t[0] == "Buy"]
        buy_prices = [t[2] for t in trades if t[0] == "Buy"]
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy', marker=dict(color='green', size=8)), secondary_y=False)
    - After adding traces call:
        fig.update_layout(yaxis=dict(title="Price"), yaxis2=dict(title="Oscillators", overlaying="y", side="right"))
    - Create TWO separate figures: strategy_fig (make_subplots secondary_y=True) and portfolio_fig (single axis). Save each to its own HTML file.
    - **If translator_instructions includes `"axis":"secondary"` for an indicator, respect that mapping.**
   - Save as HTML files with timestamp in filename.

   - **Create two distinct Figure objects**:
    1. Strategy Figure (`strategy_fig`) for Price, indicators, and Buy/Sell markers.
    2. Portfolio Figure (`portfolio_fig`) for daily portfolio value (and optional initial capital line).
    - When plotting initial capital in portfolio_fig, use:
            x = portfolio_series.index
            y = [initial_capital] * len(portfolio_series)
            mode = 'lines'
            line = dict(dash='dash')
            name = 'Initial Capital'

    - Do **not** combine strategy and portfolio lines into a single figure.
    - Save each figure to its own HTML file with timestamped filename.

8. **Output Files**
   - {{ticker}}_strategy_plot_{{timestamp}}.html
   - {{ticker}}_portfolio_value_{{timestamp}}.html
   - trading_results.html
   - **MUST**: The generated Python code must collect the exact filenames it creates (plots and results) into a list variable named `generated_files`. After saving files, the code must:
       1. Write `generated_files` to a JSON file named `generated_files_{curr_time_stamp}.json` in the working directory.
       2. Print the JSON-serialized `generated_files` list to stdout (so the executor/orchestrator can capture it).
       3. Ensure the filenames match the naming conventions above and any `required_files` provided by the translator instructions; if translator provides `required_files`, use those exact filenames (substituting {ticker} and {curr_time_stamp} accordingly) and add them to `generated_files`.
       4. If the code creates additional supporting files (logs, CSVs), add them to `generated_files` as well.

9. **Safety Checks**
    - Always include all necessary library imports at the top, such as import pandas as pd, import numpy as np, import ta, and import sqlite3.
   - Ensure Buy/Sell alignment.
   - Verify Buy/Sell alignment: no consecutive Buys or consecutive Sells.
   - After trades are generated, ensure no two trades occur on the same bar/timestamp. If both Buy and Sell conditions would hold simultaneously, only the Buy should take precedence (or vice versa if more natural for the strategy).
   - Ensure portfolio_series updates correctly according to position state.
   - Verify portfolio_series length matches df.index and reflects trade updates.
   - Verify buy/sell marker traces appear only once in legend each.
   - Ensure portfolio_series length matches df length.
   - When generating backtest loops, ensure that portfolio tracking produces a series aligned exactly with the dataframe index (no off-by-one). If you start iteration from i=1, align portfolio_series with df.index[1:]. If you start from i=0, make sure to guard lookbacks with i > 0.
   - Ensure metrics don’t crash on empty datasets.
   - Avoid deprecated methods like DataFrame.append(); use pd.concat() or build a list of dicts.
   - Perform all calculations per ticker; treat a single ticker as a special case.
   - Indicators cross-check (CODEGEN must implement):
        - Before running a backtest, assert that all `indicators` named in translator_instructions exist as DataFrame columns after computation. If any are missing, raise a clear error and stop.
        - If translator_instructions contains buy_spec/sell_spec conditions referencing indicator columns that do not exist, raise/return an error rather than generate code silently.
   - Safety check — generated_files:
        - The generated_files list must contain exactly the file names produced by the script (per-ticker plots + aggregated trading_results.html). Do not overwrite generated_files between tickers.

    - RUNTIME SAFETY (MUST include simple runtime guards):
        1. Before any sell logic that uses entry_price, require:
        - if not (position == 1 and entry_price is not None and shares > 0):
            # skip sell checks that depend on entry_price (or set these sell flags to False)
        - Before any multiplication/division with entry_price, check:
            if position == 1 and entry_price is not None
        - If entry_price is None or position == 0, skip any calculation that references it
        - Never assume entry_price exists
        - Initialize entry_price = None before the loop

        2. When appending a Sell trade, ensure `shares > 0` before cash update and appending. If shares == 0, do not append a Sell; instead log or raise error.
        3. Final forced close must check `if position == 1 and entry_price is not None and shares > 0:` before executing the final sell.
    ****Pandas deprecation: Use df.ffill() and df.bfill() instead of df.fillna(method='ffill') / df.fillna(method='bfill').


10. **Code Style**
   - Must be executable as-is (imports included).
   - No explanations, just code.
   - Clear, modular structure with comments.

---SPECIAL REQUIREMENTS FOR CROSSOVERS & RE-ENTRY---
- For every translator `buy_spec` or `sell_spec` condition with `operator` == "crosses_above" or "crosses_below", implement the exact pandas shift-based condition using .shift(1). Example code hint you MUST include in code_tasks:
    - `buy_cond = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)`
    - `sma_cross = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))`
- For re-entry rules included under `trade_management.reentry_rule`:
    - Implement a boolean `can_reenter` (or `waiting_for_reset`) state.
    - After any Sell, set `can_reenter = False`.
    - Only reset `can_reenter` according to the reentry_rule (examples: wait until RSI < 30 and then new crossover above 30; or wait until indicator leaves overbought zone).
    - Prevent new Buys while `can_reenter == False` even if buy condition is True.
    - Document the state reset condition in the code as comments and implement it exactly (e.g., `if current_rsi < 30: can_reenter = True` or a more specific rule if translator provided one).
- For every translator buy_spec or sell_spec with operator "crosses_above" or "crosses_below":
    - Implement a boolean state variable: waiting_for_reset or can_reenter.
    - After a trade is executed (buy or sell), set can_reenter = False.
    - Only reset can_reenter according to the reentry_rule:
        * Wait until the relevant indicator leaves the overbought/oversold zone.
        * Only allow new trades when can_reenter == True AND crossover condition occurs.
    - Include exact pandas hint with .shift(1), e.g.,
        buy_cond = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30) & (can_reenter == True)
    - Document the reset condition in code comments.
- Buy condition safeguard:
    - When coding the buy rule, always enforce waiting_for_reset == False as part of the condition.
    - After a sell, set waiting_for_reset = True.
    - Do not allow a new buy while waiting_for_reset == True, even if the RSI or crossover condition is true.
    - Reset waiting_for_reset = False only when the relevant indicator (e.g., RSI) fully exits the oversold/overbought region, so that the next trade requires a fresh crossover.
    - Explicitly include and not waiting_for_reset in the buy condition.
        Re-entry safeguard (must be applied to ANY buy/sell condition):

        # At initialization
        waiting_for_reset = False

        # Buy condition
        if position == 0 and not waiting_for_reset and <buy_signal>:
            BUY
            position = 1
            entry_price = current_price

        # Sell condition
        if position == 1 and entry_price is not None and <sell_signal>:
            SELL
            position = 0
            entry_price = None
            waiting_for_reset = True

        # Reset condition
        waiting_for_reset should remain True until the relevant indicator
        (first used in <buy_signal> or <sell_signal>) fully exits its trigger zone.  
        Only then set waiting_for_reset = False, so the next trade requires a fresh signal.


---OUTPUT FORMAT REQUIRED FROM LLM---
- Produce **only** two sections:
    1. `---THOUGHTS---` (plain text reasoning).
    2. `---CODE---` (pure Python code implementing everything).
- Do not return any other text, framing, or markdown.
- The code under `---CODE---` **must** create and save files named above, collect them into `generated_files`, write `generated_files_{curr_time_stamp}.json`, and print the JSON representation of `generated_files` to stdout before exiting.

---ADDITIONAL NOTES---
- If translator_instructions contains `required_files`, the generated code must honor those exact filenames (substituting {ticker}, {curr_time_stamp}) and include them in `generated_files`.
- If translator_instructions includes `indicators_to_plot`, ensure they are plotted and included in strategy_fig.
- If no trades executed, code should still create `trading_results.html` containing a one-row table where metrics are zero and include a note "No trades executed in this period." and include that filename in `generated_files`.
- Use initial_capital value from translator_instructions if provided; otherwise default to 10000.
- Implement `waiting_for_reset` logic exactly as described in the provided REENTRY_RULE.
  * After a Sell, set waiting_for_reset = True.
  * Do not allow any new Buy while waiting_for_reset is True.
  * Only set waiting_for_reset = False once the reset condition from REENTRY_RULE is satisfied.
  * Explicitly include `and not waiting_for_reset` in the Buy condition.

---END PROMPT---
"""

    try:
        messages = [
    SystemMessage(content="You are the Code Generator agent in a multi-agent trading system. You must strictly follow the provided translator_instructions."),
    HumanMessage(content=f"""
        TRANSLATOR_INSTRUCTIONS (JSON):
        {translated}

        REENTRY_RULE (from translator):
        {reentry_rule}

        CODE_TASKS (ordered steps):
        {code_tasks}

        ---

        FOLLOW THESE INSTRUCTIONS AND RULES:
        {CODEGEN_PROMPT}
        IMPORTANT: First explain your reasoning under ---THOUGHTS---. 
        Then write the executable Python code under ---CODE---.
        """)
    ]
        
        print("input to codegen: ", messages)

        response = llm.invoke(messages)
        raw = response.content

        if "---CODE---" in raw:
            thoughts_part, code_part = raw.split("---CODE---", 1)
            thoughts_part = thoughts_part.replace("---THOUGHTS---", "").strip()
        else:
            thoughts_part = ""
            code_part = raw.strip()

        thought_summary = "\n".join(thoughts_part.splitlines()[:2]).strip() or "Code generation complete."

        return {
            "thought": thought_summary,
            "full_thought": thoughts_part.strip(),
            "action": "CodeReady",
            "action_input": code_part.strip()
        }

    except Exception as e:
        return {
            "thought": f"Code generation failed due to error: {str(e)}",
            "action": "AskClarification",
            "action_input": "An error occurred during code generation. Please check your input and try again."
        }
