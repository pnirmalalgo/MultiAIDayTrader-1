from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import os
import datetime

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key, model="gpt-4o-mini")

def generate_code(intent_json: str, stock_data: pd.DataFrame) -> dict:
    print(intent_json)
    parsed = json.loads(intent_json)
    ticker = parsed.get("ticker")
    
    strategy = parsed.get("strategy_description", "")
    buy_condition = parsed.get("buy_condition", "")
    sell_condition = parsed.get("sell_condition", "")
    date_range = parsed.get("date_range", "")
    duration_type = parsed.get("duration_type", "")
    duration_days = int(parsed.get("duration_days", 0))

    # Convert the JSON to a string to pass into the prompt
    intent_json_str = json.dumps(intent_json)
    prompt = f"""
Write Python code to implement the following stock trading strategy.

Ensure the output is ONLY executable Python code — no comments, no markdown, no `pip install`, and NO code fences like ```python.

##############################
# GLOBAL ENFORCEMENT (READ FIRST)
##############################
⚠️ VERY IMPORTANT: Implement **only** the logic that is explicitly required by the given {strategy}, {buy_condition}, {sell_condition}, and any explicit {duration_type} or stop-loss clauses in the user's query.
- If the user's query does NOT mention stop-loss, consecutive-day duration, or any other extra constraint, DO NOT add that logic in the produced code.
- Do NOT invent stop-loss, consecutive-day counters, or any other “helpful” features unless they are explicitly present in the provided strategy/buy/sell/duration inputs.
- The code must be minimal and strictly focused on the requested strategy.

GENERAL SETUP:
1. Use only `pandas` and `ta` (assume pre-installed).
2. Do NOT fetch data from yfinance or APIs.
3. Use `sqlite3` and `pandas.read_sql()` to load data from `market_data.db`, table: `stock_data`.
    - Columns: 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date'
    - Sort data by ascending `Date`

4. Multi-Ticker Logic:
    - Filter and process each ticker individually using a for-loop.
    - If there’s only one ticker, treat it as a special case of multi-ticker logic — the code should still loop over unique tickers.
    - Save one plot per ticker (`{ticker}_plot.html`) and append summary metrics for each ticker in a file named `trading_results.html`.

STRATEGY INPUTS:
5. Strategy is passed as:
    Strategy: {strategy}
    Buy Condition: {buy_condition}
    Sell Condition: {sell_condition}
    Duration: {duration_type}        # (if present; otherwise empty/None)
    

INDICATOR COMPUTATION:
6. Compute only indicators mentioned in the strategy. Example: if SMA20 and SMA50 are required, compute them using `.shift(1)` to avoid lookahead bias.
7. For turning points like “macd turns positive”, detect sign change:
    Example: `(macd[i-1] < 0) & (macd[i] > 0)`

SIGNAL LOGIC:
8. Use vectorized boolean expressions (with `&` or `|`) wrapped in parentheses — DO NOT use `and`/`or` (they crash with pandas Series).
    Only compute indicators explicitly required by {strategy}, {buy_condition}, {sell_condition}.

    If {buy_condition} includes "Golden Cross", compute and use golden_cross.

    If {sell_condition} includes "Death Cross", compute and use death_cross.

    If the user's query explicitly includes "stop-loss" (e.g., "stop-loss 5%"), then apply stop-loss check during sell execution per the STOP-LOSS section below. If not included in the user's query, DO NOT implement any stop-loss logic.

    You don’t just paste raw code — you add it as a mandatory rule for computing Golden/Death Cross:

    When strategy mentions "Golden Cross" or "Death Cross", compute as follows:
        short_ma = df['Close'].rolling(window=short_window).mean().shift(1)
        long_ma = df['Close'].rolling(window=long_window).mean().shift(1)
        golden_cross = (short_ma.shift(1) <= long_ma.shift(1)) & (short_ma > long_ma)
        death_cross = (short_ma.shift(1) >= long_ma.shift(1)) & (short_ma < long_ma)

    Use `golden_cross` and `death_cross` as boolean Series for buy/sell signals.

9. CLARIFY INDICATOR DEFINITIONS:
- "50-day high" → compute using: `df['High'].rolling(window=50).max().shift(1)`
    ✅ Represents breakout above recent 50-day price high
- "50-day SMA" or "SMA50" → compute using: `df['Close'].rolling(window=50).mean().shift(1)`
    ❌ DO NOT confuse "50-day high" with "50-day SMA"

STOP-LOSS (CONDITIONAL)
- Implement stop-loss **ONLY IF the user's query explicitly mentions** stop-loss and provides the threshold (e.g., "stop-loss 5%").
- If stop-loss is present:
    - Follow these exact rules:
        - When a buy is executed, store the executed buy price in `last_buy_price`.
        - To trigger stop-loss: `if df['Close'].iloc[i] < last_buy_price * (1 - stop_loss_pct):` then trigger sell.
        - Reset `last_buy_price = None` after position is closed.
- If stop-loss is NOT present in the user's query, DO NOT create `last_buy_price`, `stop_loss_pct`, or any stop-loss checks.

DURATION / CONSECUTIVE CONDITIONS (CONDITIONAL)
- Implement consecutive-day counters (e.g., `below_sma_counter`) **ONLY IF the user's query explicitly asks** for a consecutive-day condition such as "for 3 consecutive days".
- If the query requires consecutive-day logic, use the exact pattern described below; otherwise, DO NOT include counters or consecutive-day checks.

10. For duration checks - if {duration_type} is consecutive: (e.g. RSI < 30 for 3 consecutive days), Use a counter to track how many consecutive days the condition is true.
     - Use a counter (e.g., `below_sma_counter = 0`) inside the loop to track how many **consecutive** days the price is below the SMA.
        - Reset the counter to 0 if the condition breaks
        - Only trigger the buy signal if counter >= {duration_days}

    Example:
        if close[i] < sma[i]:
            below_sma_counter += 1
        else:
            below_sma_counter = 0

        if rsi[i] < 25 and below_sma_counter >= 3:
            # trigger buy

BUY/SELL EXECUTION RULES:
11. Use `position_open` boolean to track trade state.
12. Buy only if position is not open. Sell only if position is open.
13. Signals must strictly alternate: buy → sell → buy. No consecutive buys/sells allowed.
        Generate Buy/Sell signals strictly in a stateful manner:

        Initialize in_position = False.

        Loop over each date (or vectorized equivalent):

        BUY Condition: If buy criteria are met and in_position == False, mark Buy, set in_position = True, store last_buy_price (only if stop-loss is requested).

        SELL Condition: If sell criteria are met and in_position == True, mark Sell, set in_position = False.

        Never generate a Sell without an active Buy (ignore Sell signals when in_position == False).

        Ensure buy_signals and sell_signals arrays are aligned to this logic.

14. Buy/sell only on the bar where the condition is triggered (i.e., at `iloc[i]`).
15. Do NOT prefill Buy/Sell columns. Keep signals sparse (i.e., only trigger points marked).
16. DO NOT forward fill signals.

    - If position is open (Do not check stop-loss or death cross if strategy does not include them):
    - Check stop-loss first (ONLY if stop-loss is part of the strategy and explicitly requested in the query):
        if df['Close'].iloc[i] < last_buy_price * (1 - stop_loss_pct):
            trigger sell
    - Otherwise, check sell conditions:
        - Only If "Death Cross" is part of the strategy:
            if death_cross.iloc[i]:
                trigger sell
        - If other sell conditions are provided, evaluate them here.

    - Ensure that a sell is triggered if ANY one of these conditions is met.

⚠️ INDEXING + ALIGNMENT RULES (STRICT):
17. NEVER use the following (causes IndexError):
        ❌ buy_prices.index = ticker_data.index[buy_prices.index]
        ❌ sell_prices.index = ticker_data.index[sell_prices.index]
        ❌ buy_prices.index = data.index[buy_prices.index]
    
    CRITICAL DATE ALIGNMENT RULES:

    - After filtering the data for each ticker, set the 'Date' column as the index:
        ticker_data.set_index('Date', inplace=True)

    - This ensures that any extracted Series (e.g., buy_prices, sell_prices) inherit a proper DatetimeIndex.

    - ONLY if 'Date' is the index:
        ✅ days_held = (sell_prices.index[-1] - buy_prices.index[0]).days

    - ❌ If 'Date' is just a column and not set as the index, buy_prices.index will be integers (0, 1, ...) — and date math will crash with:
            AttributeError: 'numpy.int64' object has no attribute 'days'

    - Therefore, always call:
        ticker_data.set_index('Date', inplace=True)
        immediately after parsing 'Date' with pd.to_datetime

18. ✅ Buy/Sell prices extracted via `.dropna()` already have correct datetime index — NEVER modify or reassign their index manually.

19. ❌ DO NOT try to reindex anything using `.index[...]` unless it's a boolean mask or integer position. Doing so will crash the code with "arrays used as indices must be of integer or boolean type".

20. DO NOT reset index on any filtered price series (e.g., trades, buy_prices, sell_prices). Keep the inherited datetime index.

21. Buy/sell signals must be stored as new columns in `ticker_data`:
    ✅ Example:
        ticker_data['Buy'] = buy_signals
        ticker_data['Sell'] = sell_signals

TRADES HANDLING:
22. Create `trades = ticker_data[['Buy', 'Sell']]`, then:
    - Use `.dropna(how='all')` to filter rows with any signal.
    - Extract `buy_prices = trades['Buy'].dropna()`
    - Extract `sell_prices = trades['Sell'].dropna()`
    - Truncate longer list:
        ```python
        min_len = min(len(buy_prices), len(sell_prices))
        buy_prices = buy_prices.iloc[:min_len]
        sell_prices = sell_prices.iloc[:min_len]
        ```

RETURNS & METRICS:
23. Only use completed trades (buy followed by sell) to compute metrics:
    - - ALWAYS compute returns as a pandas Series — NOT a NumPy array.
    - For example:
        ✅ returns = pd.Series((sell_prices.values - buy_prices.values) / buy_prices.values)
    - Then compute:
        ✅ cumulative_returns = (1 + returns).cumprod()
        ✅ max_drawdown = (1 - cumulative_returns / cumulative_returns.cummax()).max()
    
    ❌ DO NOT do:
        returns = (sell_prices.values - buy_prices.values) / buy_prices.values
        (this makes `returns` a NumPy array and breaks .cumprod() / .cummax())

    - Cumulative Return = cumulative_returns.iloc[-1] - 1
    - Annualized Return = cumulative_returns.iloc[-1] ** (365 / days_held) - 1
        - Where `days_held = (sell_prices.index[-1] - buy_prices.index[0]).days`
        - Only compute if `days_held > 0`
    
    - Volatility = returns.std() * sqrt(252)
    - Max Drawdown: from cumulative return series
    
    ⚠️ Handle edge case:
        if returns is empty → set all metrics to default (0, None, etc.)

    ⚠️ NEVER call .cummax() or .iloc on NumPy arrays
    

    ⚠️ If no valid trades, set all metrics to:
        Cumulative Return = 0  
        Annualized Return = None  
        Volatility = None  
        Max Drawdown = None

    Also: ✅ print a message: `No trades executed for {ticker}`

    Initialize all metric variables before conditionals to avoid NameError.

24. PLOTTING (Plotly) 

    Create a 2-row subplot using plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True).

    Row 1 = Price chart with all indicators & Buy/Sell markers.

    Row 2 = Equity curve (portfolio_value) as a continuous line.

    Use specs=[[{{"secondary_y": True}}], [{{}}]] so Row 1 supports a secondary y-axis.

    Price + Indicators + Buy/Sell Markers (Row 1):

    Always plot Close price as a blue line on the primary y-axis (secondary_y=False).

    Only plot Buy markers where buy_signals is True.

    Only plot Sell markers where sell_signals is True and preceded by a Buy.

    Ensure no consecutive Sell markers appear without an intervening Buy.

    Plot all indicators referenced in the strategy:

    SMA/MAs:

        Compute using .shift(1) to prevent lookahead bias.

        Plot each SMA as a dotted line with distinct colors (e.g., SMA10 = orange, SMA30 = purple).

        Ensure SMAs always appear even if there are no trades.

    MACD, Signal Line, RSI thresholds, etc. (if applicable): plot on secondary y-axis (secondary_y=True).

    Plot Buy markers as green triangle-up markers at the Close price.

    Plot Sell markers as red triangle-down markers at the Close price.

    Equity Curve (Row 2):

    Assume starting capital = 100,000 (100K).

    All-in/all-out trading logic:

        On a Buy signal: invest full cash balance to buy as many shares as possible.

        On a Sell signal: liquidate all shares into cash.

    Daily portfolio value calculation (continuous):

        Initialize portfolio_value as a pandas Series indexed by all dates in ticker_data.

        Loop through each date in order, updating:

            - If holding shares: portfolio_value[date] = shares * Close[date] + cash_balance

            - If no position: portfolio_value[date] = cash_balance

        Do NOT update only on Buy/Sell dates—update for every date so value tracks daily market movements.

        After processing all dates, forward-fill NaN values (if any gaps remain).

        Plot this portfolio_value as a continuous purple line in Row 2.

    Layout & Axis Configuration:

        Ensure ticker_data.index is a DatetimeIndex.

        Use x=ticker_data.index for all traces (Close, indicators, buy/sell markers, portfolio_value).

        Set fig.update_layout(xaxis=dict(type='date')) for proper date alignment.

    In Row 1:

        Set Close Price axis: fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False).

        Set Indicators axis: fig.update_yaxes(title_text="Indicator", row=1, col=1, secondary_y=True).

        Keep y-axis ranges independent between Price and Indicators.

        Trade Presence Check (Before Plotting):

            If no valid Buy-Sell pairs exist (i.e., sum(buy_signals) == 0 or sum(sell_signals) == 0), skip plotting trade markers.
    
    IMPORTANT: Ensure the backtest plot is clear and aligned:

        - Use make_subplots with secondary_y=True for RSI and other indicators. Do not manually set yaxis='y2'.

        - Plot buy/sell markers only at valid non-NaN signal points using their exact index positions.

        - Initialize portfolio value as NaN except for the starting capital, update dynamically during trades, and forward-fill only at the end.

        - Align Close Price, SMA, RSI, buy/sell markers, and portfolio value on the same date index throughout the backtest date range.

        - Set xaxis=dict(type='date') in layout to ensure proper time-series plotting.

    Save Final Combined Plot:

        Save the complete figure with:

        fig.write_html(f"{ticker}_plot.html")

        Ensure both subplots appear in one file, vertically stacked.

26. When using the ta library for MACD:

    Use MACD(data['Close']) to create the MACD object.

    Use macd.macd() for the MACD line.

    Use macd.macd_signal() for the signal line (not macd.signal()).

    Use macd.macd_diff() for the histogram.

    Always .shift(1) MACD and signal values to avoid look-ahead bias.

    Ensure buy condition is MACD crosses above MACD Signal when no open position.

    Ensure sell condition is MACD crosses below MACD Signal only if a position is open.

FINAL OUTPUT:
27. Append all per-ticker metrics to a summary list and save as:
    `trading_results.html` using `DataFrame.to_html(index=False)`

DEBUGGING + STABILITY:
28. Print SQL query result for debugging
29. Avoid ambiguous conditions — wrap all boolean expressions in parentheses
30. DO NOT use `.between()` inside loops
31. For cumulative metrics like returns/drawdown, keep as pandas Series — do not convert to NumPy array unless doing element-wise math
32. NEVER call `.iloc[]` on NumPy arrays — only on Series or DataFrames
33. Use `warnings.filterwarnings("ignore")` to suppress ta-lib warnings

ADDITIONAL:
34. Print trades rows showing actual Buy/Sell dates, prices, and indicators
35. Ensure reproducible structure — keep all per-ticker logic inside the loop
36. All extracted price series must have inherited datetime index from `ticker_data` — do NOT reindex manually

"""           
    print(prompt)
    messages = [
        SystemMessage(content="Write simple python code. Do not use yfinance. VERY IMP: DO NOT include'''python. '''python causes code to break. Required data is stored in SQLite3 table provided in the query. Use the ta library, importing MACD from ta.trend and RSI from ta.momentum if needed. Initialize ta class."\
                      "Include required libraries eg. numpy. Initialize ta class properly. Use: from ta.momentum import RSIIndicator. Please note: DO NOT use ta.add_all_ta_features(This is not required and gives error). "\
                      "DO NOT include any comments. Only executable python code. Print output as per instructions." \
                      "The generated Python code must be compatible with pandas 2.0+."\
                      "Replace any usage of the deprecated Series.append() or DataFrame.append() with pd.concat([obj1, obj2])." \
                      "When generating code that creates signal lists (e.g., Buy/Sell, Long/Short), ensure the lists are exactly the same length as the DataFrame index."\
                      "Always start the lists pre-filled with np.nan for all rows (e.g., [np.nan] * len(data)) or append values for every iteration so the final list length equals len(data)."\
                      "Do not start loops at range(1, len(data)) unless you also pre-fill the first element(s) to keep lengths equal."\
                      "Before assigning to data['column'], validate that len(list) == len(data)."\
                      "The fix must be generic so it works for MACD, RSI, SMA, or any other indicator as applicable."\
                      "Add checks whereever necessary to see if there is no data before accessing data."
                      ),
        HumanMessage(content=prompt)
    ]
    

    response = llm.invoke(messages)
    print("Generated code: ",response.content)
    return {"code": response.content}