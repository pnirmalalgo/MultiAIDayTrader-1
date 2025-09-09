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

def codegen_mcp(input: dict) -> dict:
    curr_time_stamp = datetime.now().isoformat()
    """
    MCP-compliant code generator with foolproof alternating buy/sell signals.
    Expects input dict with key 'structured_query' containing parsed intent dict.
    Returns dict with keys: thought (str), action (str), action_input (str).
    """
    structured_query = input.get("structured_query")
    if not structured_query:
        return {
            "thought": "No structured query provided. Cannot generate code.",
            "action": "AskClarification",
            "action_input": "Please provide a valid structured_query JSON object."
        }

    # Extract strategy variables
    strategy = structured_query.get("strategy_description", "")
    buy_condition = structured_query.get("buy_condition", "")
    sell_condition = structured_query.get("sell_condition", "")
    duration_type = structured_query.get("duration_type", "")
    duration_days = int(structured_query.get("duration_days", 0))

    # Build the dynamic prompt with CoT + pseudo-code
    prompt = f"""
Given a trading strategy intent, first explain your reasoning step by step under ---THOUGHTS---,
then write executable Python code under ---CODE---.

Constraints:
- Use only pandas and ta libraries (and numpy if needed).
- Do not fetch data from the internet.
- Use sqlite3 and pandas.read_sql() to read from market_data.db, table: stock_data.
- Expected columns: Date, Open, High, Low, Close, Volume, Ticker.
- Always parse Date to datetime, sort by Date ascending, and set Date as the DataFrame index.


Strategy: {strategy}
Buy Condition: {buy_condition}
Sell Condition: {sell_condition}
Duration Type: {duration_type}
Duration Days: {duration_days}

Important: Start your code with
print("GPT is using bla bla bla prompt")

### Data & indexing (MANDATORY) ###
1. Read the full table into a DataFrame `df`. **Fetch tickers dynamically from the database**: tickers = df['Ticker'].unique()
2. Immediately run:
   df['Date'] = pd.to_datetime(df['Date'])
   df = df.sort_values('Date').set_index('Date')
   # Ensure Date is a DatetimeIndex for all tickers and downstream calculations.
3. When filtering per ticker always use .copy():
   ticker_df = df[df['Ticker'] == ticker].copy()

### Event-Based Alternating Signals (Generic, FOOLPROOF) ###
4. Initialize signal columns with np.nan:
   ticker_df['Buy'] = np.nan
   ticker_df['Sell'] = np.nan

5. Compute all indicators required by the buy/sell conditions (do not hardcode any indicator).
 - **If Buy/Sell conditions mention SMA, EMA, or MACD, compute them and plot them on the strategy chart**
 - When implementing MACD, do not call ta.trend.macd(). Instead, use the MACD class from ta.trend. Instantiate it with close, window_slow, window_fast, and window_sign, then call .macd(), .macd_signal(), and .macd_diff() to get the values.
 
 - When using the ta library, always use the correct indicator class names:

        SMAIndicator for SMA

        EMAIndicator for EMA

        RSIIndicator for RSI

        MACD for MACD

        BollingerBands for Bollinger Bands

        StochasticOscillator for Stochastic

    - Never import RSI or call functions like macd() — they do not exist. Always instantiate the class and then call the appropriate method (.sma_indicator(), .ema_indicator(), .rsi(), .macd(), etc.).
 
    - ###Stop-loss / Take-profit instructions for the agent(If operator contains stop_loss or take_profit)###

        Reference price

        When a BUY signal executes, set entry_price = current_price.

        For SELL conditions with "indicator": "Price" and "value_type": "percent", calculate thresholds relative to entry_price.

        Do NOT use previous day’s close or pct_change() for stop-loss / take-profit calculations.

        Operator mapping

        "operator": "stop_loss" → trigger if current_price <= entry_price * (1 - value/100)

        "operator": "take_profit" → trigger if current_price >= entry_price * (1 + value/100)

        Evaluate only when currently holding a position (in_position = True).

        Combining sell conditions

        Combine percent-based stop-loss / take-profit with other sell conditions (e.g., RSI > 70, MACD < Signal) using logical OR.

        Sell if any of these conditions is True.

        Updating entry price

        After executing a SELL (stop-loss, take-profit, or strategy-based condition), reset entry_price = None.

        The next BUY sets a new entry_price.

        Per-trade thresholds

        Percent thresholds are applied per trade, not globally to the whole series.

        Each trade’s stop-loss / take-profit is independent and based only on its entry_price.

        Example pseudo-code for clarity: (This is only example logic, do NOT copy-paste)

        if not in_position and BUY_condition:
            buy at current_price
            entry_price = current_price
            in_position = True

        if in_position:
            SELL_condition_stop_loss = current_price <= entry_price * (1 - stop_loss_percent/100)
            SELL_condition_take_profit = current_price >= entry_price * (1 + take_profit_percent/100)
            SELL_condition_combined = SELL_condition_RSI | SELL_condition_stop_loss | SELL_condition_take_profit
            if SELL_condition_combined:
                sell at current_price
                entry_price = None
                in_position = False

        - Evaluate SELL_condition_combined only when in_position = True.
            - Reset entry_price = None after executing a SELL.
            - Percent thresholds are per trade, relative to entry_price, not the entire series.

    -   - If the strategy has default sell rules (e.g., RSI > 70, MACD < Signal), and the user provides stop-loss or take-profit, **merge them using logical OR**.
            - Do NOT overwrite the default strategy sell condition when user adds percent-based thresholds.
            - For example:
                default_sell_condition = (RSI > 70)
                user_stop_loss_condition = current_price <= entry_price * (stop_loss_percent/100)
                user_take_profit_condition = current_price >= entry_price * (take_profit_percent/100)
                SELL_condition_combined = default_sell_condition | user_stop_loss_condition | user_take_profit_condition        
        
        Important notes for code generation

        Never calculate stop-loss / take-profit using pct_change() or daily differences.

        Always base percentage thresholds on the entry price of the current trade.

        Logical OR combination ensures position exits when any sell condition triggers.

        Reset entry_price after position closure.

        - When combining default strategy rules with user-provided stop-loss/take-profit, **always use logical OR**, even if the original sell_condition logic is "and".
        - For multiple indicators in sell_condition (e.g., RSI > 70 AND MACD < Signal), keep their original AND/OR logic inside, then OR the percent-based thresholds.

 6. Detect crossings (False → True) only. Example helper logic:
   BUY_curr = <evaluate buy_condition boolean on current row>
   BUY_prev = BUY_curr.shift(1)  # or compute prev boolean with .iloc[idx-1]
   BUY_crossed(idx) = (not BUY_prev) and BUY_curr
   Same for SELL_crossed.

  
7. Apply strict alternation using `in_position` and **use .loc for assignments**:
   in_position = False
   for idx in range(1, len(ticker_df)):
       if not in_position and BUY_crossed(idx):
           ticker_df.loc[ticker_df.index[idx], 'Buy'] = ticker_df.loc[ticker_df.index[idx], 'Close']
           in_position = True
       if in_position and SELL_crossed(idx):
           ticker_df.loc[ticker_df.index[idx], 'Sell'] = ticker_df.loc[ticker_df.index[idx], 'Close']
           in_position = False
   - Do NOT use chained assignments like `ticker_df['Buy'].iloc[...] = ...`.
   - Only mark the exact crossing bar; do NOT repeat buy/sell markers on consecutive bars.
   - Always cast BUY_condition and SELL_condition to boolean before crossing detection:
        BUY_condition = (<expression>).astype(bool)
        SELL_condition = (<expression>).astype(bool)
    - Use .shift(1).fillna(False) for previous condition checks.

### Match trades & remove unmatched last buy ###
8.Always ensure Buy and Sell signals are paired correctly:
   - After detecting raw Buy/Sell signals, drop extra unmatched signals.
   - Keep only the first N buys and first N sells, where N = min(len(Buy), len(Sell)).
   - Reset ticker_df['Buy'] and ticker_df['Sell'] to NaN, then reassign only the matched Buy/Sell indexes.
   - This ensures no stray unmatched signals remain in the DataFrame.

After marking raw crossing events:
   trades = ticker_df[['Buy','Sell']].dropna(how='all')
   buy_prices = trades['Buy'].dropna()
   sell_prices = trades['Sell'].dropna()
   # Ensure only matched pairs are used:
   min_len = min(len(buy_prices), len(sell_prices))
   buy_prices = buy_prices.iloc[:min_len]
   sell_prices = sell_prices.iloc[:min_len]
   # Replace unmatched entries in ticker_df so only completed pairs remain:
   matched_buy_idx = buy_prices.index
   matched_sell_idx = sell_prices.index
   # Clear non-matched buy/sell markers using .loc (no chained assignment):
   ticker_df.loc[~ticker_df.index.isin(matched_buy_idx), 'Buy'] = np.nan
   ticker_df.loc[~ticker_df.index.isin(matched_sell_idx), 'Sell'] = np.nan

### Build Position column from cleaned signals ###
9. Build position strictly from matched Buy/Sell:
   ticker_df['Position'] = 0
   ticker_df.loc[ticker_df['Buy'].notna(), 'Position'] = 1
   ticker_df.loc[ticker_df['Sell'].notna(), 'Position'] = 0
   ticker_df['Position'] = ticker_df['Position'].ffill().astype(int)

### Portfolio calculation — ALL-IN / ALL-OUT (cash + integer shares) ###
10. Use all-in / all-out so SELL updates cash and portfolio value:
    initial_capital = 100000.0
    cash = initial_capital
    shares = 0
    portfolio_values = []
    # iterate over the Date index (DatetimeIndex)
    for d in ticker_df.index:
        price = float(ticker_df.loc[d, 'Close'])
        if d in buy_prices.index:
            shares = int(cash // price)
            cash -= shares * price
        if d in sell_prices.index:
            cash += shares * price
            shares = 0
        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)
    portfolio_series = pd.Series(portfolio_values, index=ticker_df.index).astype(float)
    # If portfolio_series is empty, set metrics to zeros and print "No trades executed for {{ticker}}"

### Metrics computed from portfolio_series (as percentages) ###
11. Compute metrics safely (DatetimeIndex required). Use fallbacks:
    if len(portfolio_series) == 0:
        cumulative_return_pct = annualized_return_pct = volatility_pct = max_drawdown_pct = 0.0
    else:
        cumulative_curve = portfolio_series / float(portfolio_series.iloc[0])
        # total_days must be based on datetime index
        total_days = (portfolio_series.index[-1] - portfolio_series.index[0]).days
        total_days = total_days if total_days > 0 else len(portfolio_series)
        cumulative_return_pct = (cumulative_curve.iloc[-1] - 1.0) * 100
        annualized_return_pct = ((cumulative_curve.iloc[-1]) ** (365.0 / total_days) - 1.0) * 100 if total_days > 0 else 0.0
        volatility_pct = cumulative_curve.pct_change().dropna().std() * (252 ** 0.5) * 100
        max_drawdown_pct = (1.0 - (cumulative_curve / cumulative_curve.cummax())).max() * 100
    # Round to 2 decimals before saving: e.g. round(value, 2)

### Plotting (explicit axes) (Use Plotly) ###

12. Plot 1 — Strategy chart:
    - X-axis = Date (use ticker_df.index)
    - Y-axis = Close Price (blue line)
    - Plot completed Buy markers (green) and Sell markers (red) at the Close price of matched trades only.
    - Also plot SMA/EMA/MACD/RSI if used in buy/sell conditions.
    - Save to HTML file using fig.write_html(file_strategy).

13. Plot 2 — Portfolio chart:
    - X-axis = Date
    - Y-axis = portfolio_series values (purple line)
    - Save to HTML file using fig.write_html(file_portfolio).

### Output files & formatting ###
14. Save per-ticker files with a timestamp: {{ticker}}_strategy_plot_{curr_time_stamp}.html and {{ticker}}_portfolio_value_{curr_time_stamp}.html. Ensure files are written with fig.write_html().
15. Append filenames to an `output_files` list and print them at the end (so orchestrator/celery can capture them).
    **Include summary file trading_results.html in output with its filename so UI can display it**

16. Build per-ticker metrics dict with keys:
    'Ticker', 'Cumulative Return (%)', 'Annualized Return (%)', 'Volatility (%)', 'Max Drawdown (%)'
    - All numeric metric values should be percentages rounded to two decimals before constructing the DataFrame.
17. Save the summary DataFrame to `trading_results.html` and print the filename.

18. Before the per-ticker loop:
- Create a single, filesystem-safe timestamp once and reuse it for all filenames:
    curr_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
- Initialize an empty list to collect per-ticker metrics:
    metrics_list = []
- Initialize output_files = []

Inside the per-ticker loop, after computing and rounding the metrics:
- Build metrics_dict for the current ticker and append it to metrics_list:
    metrics_dict = {{ 'Ticker': ticker, 'Cumulative Return (%)': cumulative_return_pct, ... }}
    metrics_list.append(metrics_dict)

- Create filename variables (use the same curr_time_stamp) and write files with those variables:
    file_strategy = f"{{ticker}}_strategy_plot_{{curr_time_stamp}}.html"
    strategy_fig.write_html(file_strategy)
    output_files.append(file_strategy)

    file_portfolio = f"{{ticker}}_portfolio_value_{{curr_time_stamp}}.html"
    portfolio_fig.write_html(file_portfolio)
    output_files.append(file_portfolio)

- Do NOT call pd.Timestamp.now() (or datetime.now()) again when building filenames — always reuse curr_time_stamp so names are consistent and not accidentally different per append.

19. After the per-ticker loop:
- Build the summary DataFrame from metrics_list (not a single metrics_dict):
    summary_df = pd.DataFrame(metrics_list)
- Save the summary to the fixed filename the UI expects and append that filename:
    summary_df.to_html('trading_results.html', index=False)
    output_files.append('trading_results.html')
- Print the trading_results.html filename (so Gradio/orchestrator can pick it up) and print the output_files list.
- When saving tables or summaries (e.g., trading results), do not use only DataFrame.to_html().
    Always wrap the table output into a complete standalone HTML page with <html>, <head>, <body>, and basic CSS styling.
    This ensures it displays correctly inside an iframe in Gradio.

Important Notes for Agent:
- Never use chained assignment; always use df.loc[...] = ... for column updates.
- Ensure Date is converted to datetime and set as the index before any time-based math.
- Use .copy() when slicing per ticker to avoid SettingWithCopyWarning.
- Remove any unmatched buy (no sell) before computing Position and portfolio.
- Compute metrics from `portfolio_series` (pandas Series) and return values in percentages (two decimals).
- When generating pandas code, strictly follow these rules to avoid FutureWarnings:

    1. Do NOT use `.fillna(False)` or `.fillna(True)` directly on boolean/object Series.  
    Instead, first ensure the dtype is correct:  
        BUY_prev = BUY_condition.shift(1).fillna(False).astype(bool)  
    Or, equivalently:  
        BUY_prev = BUY_condition.shift(1).fillna(False).infer_objects(copy=False)

    2. Do NOT index Series directly with `series[idx]` where idx is an integer.  
    Use `.iloc[idx]` for positional access and `.loc[label]` for label-based access.  
    Example:  
        if not in_position and BUY_crossed.iloc[idx]:  
            ...

    3. If you must fill NA values on object/bool dtype, always cast afterwards:  
        SELL_prev = SELL_condition.shift(1).fillna(False).astype(bool)

    4. If you want to future-proof silently, set once at the top of the script:  
        import pandas as pd  
        pd.set_option("future.no_silent_downcasting", True)

    5. Suppress all other warnings.
    6.  Whenever creating shifted BUY/SELL signals:
    - Use .shift(1).infer_objects(copy=False).fillna(False).astype(bool)
    - Also set at the top of the script:
        import pandas as pd
        pd.set_option("future.no_silent_downcasting", True)
    - This avoids FutureWarnings about downcasting.

- Your output MUST have exactly two sections:

---THOUGHTS--- (required, step-by-step reasoning)
1) Restate the parsed intent from the structured_query in one clear sentence.
2) Identify every indicator needed to evaluate the provided Buy/Sell conditions.
   - List each indicator exactly (variable name + parameters). Example: SMA_20, EMA_50, RSI_14, MACD_fast12_slow26_signal9.
   - If the user did not specify indicator parameters (e.g., SMA length), choose sensible defaults (SMA:20,50; EMA:12,26,50; RSI:14; MACD: fast=12, slow=26, signal=9) and state them here.
3) Produce a short numbered plan mapping to code sections (data load, per-ticker loop, indicator computation, signal crossing detection, alternation enforcement, matching trades, portfolio sim, metrics, plotting, file output).
4) State assumptions and edge-cases you will handle (e.g., insufficient bars to compute an indicator, identical buy & sell at same bar, no trades executed).
5) List what will be plotted for the strategy chart and portfolio chart (including RSI/SMA/EMA/MACD if used).
6) End with a single-line confirmation: "Ready to produce code."

---CODE---
(executable Python code only, no comments, no markdown, no code fences)
"""


    try:
        messages = [
            SystemMessage(content="You're a trading algorithm assistant. Reason step by step, then write working code. No markdown or code fences."),
            HumanMessage(content=prompt)
        ]

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
