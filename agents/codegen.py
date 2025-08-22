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

INDICATOR COMPUTATION:
6. Compute only indicators mentioned in the strategy. Example: if SMA20 and SMA50 are required, compute them using `.shift(1)` to avoid lookahead bias.
7. For turning points like “macd turns positive”, detect sign change:
    Example: `(macd[i-1] < 0) & (macd[i] > 0)`

SIGNAL LOGIC:
8. Use vectorized boolean expressions (with `&` or `|`) wrapped in parentheses — DO NOT use `and`/`or` (they crash with pandas Series).

9. CLARIFY INDICATOR DEFINITIONS:
- "50-day high" → compute using: `df['High'].rolling(window=50).max().shift(1)`
    ✅ Represents breakout above recent 50-day price high
- "50-day SMA" or "SMA50" → compute using: `df['Close'].rolling(window=50).mean().shift(1)`
    ❌ DO NOT confuse "50-day high" with "50-day SMA"

    STOP LOSS LOGIC:
        Avoid comparing prices against signal arrays like buy_signals[i-1], as they may contain NaN or misaligned values. Instead, store the actual executed buy price in a separate variable (e.g., last_buy_price) when a position is opened. Use this variable to evaluate stop loss conditions (e.g., trigger sell if Close < stop_threshold * last_buy_price). Ensure this variable is reset (e.g., set to None) once the position is closed.

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
14. Buy/sell only on the bar where the condition is triggered (i.e., at `iloc[i]`).
15. Do NOT prefill Buy/Sell columns. Keep signals sparse (i.e., only trigger points marked).
16. DO NOT forward fill signals.

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

PLOTTING (Plotly):
24. Generate one plot per ticker using `plotly.graph_objects`.
    - Always plot Close price (blue line)
    - If present, plot Buy markers: green triangle-up at Close price
    - If present, plot Sell markers: red triangle-down at Close price
    - If moving averages are part of strategy:
        - Compute with `.shift(1)`
        - Plot them as dotted lines
        - Use distinct colors (e.g., SMA20 = orange, SMA50 = purple)

25. Before plotting:
    - Ensure buy_prices and sell_prices are not empty
    - Do NOT plot if no trades

26. Save plot using:
    `fig.write_html(f"{ticker}_plot.html")`

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