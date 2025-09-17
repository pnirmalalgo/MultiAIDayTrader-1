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

    translated = input.get("translated_query", {})
    structured_query = input.get("structured_query", {})

    print("Input to CodeGen MCP:", input)
    if not translated and not structured_query:
        return {
            "thought": "Neither translated_query nor structured_query provided.",
            "action": "AskClarification",
            "action_input": "Please provide a valid input with at least structured_query."
        }

    # Prefer translator output
    strategy = translated.get("strategy_plan") or structured_query.get("strategy_description", "")
    buy_condition = translated.get("buy_condition_code") or structured_query.get("buy_condition", "")
    sell_condition = translated.get("sell_condition_code") or structured_query.get("sell_condition", "")
    duration_type = structured_query.get("duration_type", "")
    duration_days = int(structured_query.get("duration_days", 0))
    remarks = translated.get("remarks") or structured_query.get("remarks", "")

    translator_notes = translated.get("codegen_instructions", "")
    code_tasks = input.get("code_tasks", [])
    
    print("Code tasks for CodeGen:", code_tasks)

    CODEGEN_PROMPT = """
You are the Code Generator agent in a multi-agent trading system.

INPUT: translator_instructions JSON from the Translator agent.

OUTPUT: Executable Python script that strictly implements all strategy rules. Do not include '''python''' or any markdown formatting.
Before writing the code, explain your reasoning under ---THOUGHTS---. 
Then output the executable code under ---CODE---. 
Do not mix them.

---

RULES FOR CODE GENERATION:
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
   - Always **check `position` before executing sell**, so sells do not occur without a prior buy.
   - Respect stop-loss and take-profit only if explicitly provided.
   - Track trades as (action, date, price) in a list.
   - **Always use the trades list for determining entry price and for any sell conditions — do not compute sell signals from the DataFrame alone.**
   - Do not truncate the last buy trade.
        1. If the strategy ends with an open position (i.e., last trade is a buy), then automatically close it on the final available closing price in the dataset.
        2. Record this as a sell trade at the last date.
        3. Use this final sell to ensure that the portfolio value and performance metrics reflect a fully closed position by the end of the backtest.
        4. If a final forced sell is executed at the last closing price, make sure to update the portfolio series so that the last element reflects the new cash-only balance. Explicitly set portfolio_series.iloc[-1] = cash after this forced sell to ensure the portfolio plot and metrics are consistent.
   - For crossovers (golden/death cross, RSI thresholds, etc.):
     - Use explicit detection: `cond = (A > B) & (A.shift(1) <= B.shift(1))` for cross above.
     - Similarly, `cond = (A < B) & (A.shift(1) >= B.shift(1))` for cross below.
   - Respect stop_loss / take_profit:
     - Use `entry_price` per trade.
     - stop_loss: `current_price <= entry_price * (1 - stop_loss_pct/100)`.
     - take_profit: `current_price >= entry_price * (1 + take_profit_pct/100)`.
   - **Calculate all indicator series (RSI, SMA, etc.) before the loop**, do not recalc per iteration.

4. **Position & Trade Tracking**
   - Track `position` (0 = no position, 1 = holding position), `entry_price`, `cash`, `shares`.
        - Only execute Buy if position == 0.
        - Only execute Sell if position == 1.
        - Append trades with (action, date, price).
        - Ensure every Buy has a later matching Sell (align trades before plotting).
   - Append trades with (`action`, `date`, `price`).
   - No Sell without an active Buy.
   - Ensure every Buy has a later matching Sell (align trades before plotting).
   - Do not truncate the last buy trade.
        1. If the strategy ends with an open position (i.e., last trade is a buy), then automatically close it on the final available closing price in the dataset.
        2. Record this as a sell trade at the last date.
        3. Use this final sell to ensure that the portfolio value and performance metrics reflect a fully closed position by the end of the backtest.
        4. If a final forced sell is executed at the last closing price, make sure to update the portfolio series so that the last element reflects the new cash-only balance. Explicitly set portfolio_series.iloc[-1] = cash after this forced sell to ensure the portfolio plot and metrics are consistent.
   - When calculating daily portfolio value, always update `portfolio_value = cash + shares * current_price` **after executing Buy/Sell logic** for that day, not before.

5. **Portfolio Simulation**
   - Start with initial_capital = 100000 (or 10,000 if specified in translator_instructions).
   - **Update portfolio_series after executing trades** each day.
   - Store as `portfolio_series` aligned with df index.
   - Ensure initial capital line spans full index length when plotting.
   - After evaluating buy/sell logic **for each day**, immediately calculate and record portfolio value as `cash + shares * close_price`.
   - Store this daily portfolio value in `portfolio_series` aligned with df index.
   - Align portfolio_series with df index.
   - Only plot portfolio_series in portfolio_fig; do not include other price/indicator traces.
   - Aggregated Buy/Sell markers should appear only in strategy_fig.

6. **Performance Metrics**
   - Cumulative Return = (Vend/Vstart - 1) * 100
   - Annualized Return = ((Vend/Vstart) ** (252 / total_days) - 1) * 100. Use total_days = len(df).
   - Volatility = std(daily_returns) * sqrt(252) * 100
   - Max Drawdown = max(1 - portfolio / cummax(portfolio)) * 100
   - Guard against empty DataFrame or zero trades.
   - Save results in form of table with columns Ticker, Cumulative Return, Annualized Return, Volatility and Max Drawdown and one row for each ticker. Save table to trading_results.html.

       ****VERY IMPORTANT*****
            1. Always update portfolio value **after executing buy/sell** for that day.
            2. Use pandas Series for portfolio values for easier pct_change(), cummax(), and indexing.
            3. Compute daily returns as `portfolio_series.pct_change().dropna()`.
            4. Volatility = daily_returns.std() * sqrt(252) * 100.
            5. Max drawdown = (1 - portfolio_series / portfolio_series.cummax()).max() * 100.
            6. Annualized return = ((final_portfolio_value / initial_capital) ** (252 / n_days) - 1) * 100, where n_days = len(df).
            7. Do not truncate the last buy trade.
                - If the strategy ends with an open position (i.e., last trade is a buy), then automatically close it on the final available closing price in the dataset.
                - Record this as a sell trade at the last date.
                - Use this final sell to ensure that the portfolio value and performance metrics reflect a fully closed position by the end of the backtest.
                - If a final forced sell is executed at the last closing price, make sure to update the portfolio series so that the last element reflects the new cash-only balance. Explicitly set portfolio_series.iloc[-1] = cash after this forced sell to ensure the portfolio plot and metrics are consistent.
            8. All calculations (MA, RSI, returns) should be **per ticker**, even for multi-ticker backtests.
            9. If no trades are executed during the backtest, set all performance metrics (cumulative return, annualized return, volatility, max drawdown) to 0 and print "No trades executed in this period.".
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

        Handle unmatched trades: - Do not truncate the last buy trade.
        1. If the strategy ends with an open position (i.e., last trade is a buy), then automatically close it on the final available closing price in the dataset.
        2. Record this as a sell trade at the last date.
        3. Use this final sell to ensure that the portfolio value and performance metrics reflect a fully closed position by the end of the backtest.
        4. If a final forced sell is executed at the last closing price, make sure to update the portfolio series so that the last element reflects the new cash-only balance. Explicitly set portfolio_series.iloc[-1] = cash after this forced sell to ensure the portfolio plot and metrics are consistent.

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

        Plotting per ticker

        Create two plots per ticker:

        Strategy plot with price, indicators, buy/sell markers.

        Portfolio value plot with initial capital line.

        Save results

        Save per-ticker HTML plots and the final results table (results.to_html()).

7. **Plots** (Use plotly, save as HTML)
    "IMPORTANT: Your trades list already contains executed buys/sells (tuples (action,date,price)). Use that list to plot markers and to derive entry_price for stop-loss/take-profit logic — do NOT recompute or infer trades from indicator boolean masks."

   - Strategy Plot: - Include price, all indicators in `indicators_to_plot`, and buy/sell markers.
   - Buy/Sell markers on strategy plot must reflect the **trades list**, NOT the raw indicator conditions.
   - Do not use DataFrame boolean masks (like `RSI<30`) for plotting; only use dates/prices from executed Buy/Sell tuples (i.e. from the trades list).
   - Do not truncate the last buy trade.
        1. If the strategy ends with an open position (i.e., last trade is a buy), then automatically close it on the final available closing price in the dataset.
        2. Record this as a sell trade at the last date.
        3. Use this final sell to ensure that the portfolio value and performance metrics reflect a fully closed position by the end of the backtest.
        4. If a final forced sell is executed at the last closing price, make sure to update the portfolio series so that the last element reflects the new cash-only balance. Explicitly set portfolio_series.iloc[-1] = cash after this forced sell to ensure the portfolio plot and metrics are consistent.

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
   - Ensure Buy/Sell alignment.
   - Verify Buy/Sell alignment: no consecutive Buys or consecutive Sells.
   - Ensure portfolio_series updates correctly according to position state.
   - Verify portfolio_series length matches df.index and reflects trade updates.
   - Verify buy/sell marker traces appear only once in legend each.
   - Ensure portfolio_series length matches df length.
   - When generating backtest loops, ensure that portfolio tracking produces a series aligned exactly with the dataframe index (no off-by-one). If you start iteration from i=1, align portfolio_series with df.index[1:]. If you start from i=0, make sure to guard lookbacks with i > 0.
   - Ensure metrics don’t crash on empty datasets.
   - Avoid deprecated methods like DataFrame.append(); use pd.concat() or build a list of dicts.
   - Perform all calculations per ticker; treat a single ticker as a special case.
   - Update portfolio after buy/sell execution.

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

---END PROMPT---
"""

    try:
        messages = [
    SystemMessage(content="You are the Code Generator agent in a multi-agent trading system. You must strictly follow the provided translator_instructions."),
    HumanMessage(content=f"""
        TRANSLATOR_INSTRUCTIONS (JSON):
        {translated}

        CODE_TASKS (ordered steps):
        {code_tasks}

        ---

        FOLLOW THESE INSTRUCTIONS AND RULES:
        {CODEGEN_PROMPT}

        IMPORTANT: First explain your reasoning under ---THOUGHTS---. 
        Then write the executable Python code under ---CODE---.
        """)
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
