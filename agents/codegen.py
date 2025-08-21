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

    # Convert the JSON to a string to pass into the prompt
    intent_json_str = json.dumps(intent_json)
    prompt = f"""
    Write Python code to implement the following stock trading strategy.
    Ensure the output is only executable Python code — no comments, markdown, pip install statements, or ```python wrappers.

    General Instructions:
    1. Use only `pandas` and the `ta` library (assume pre-installed).
    2. DO NOT fetch data from yfinance or any other API.
    3. Use `sqlite3` and `pandas.read_sql` to load data from `market_data.db`, table: `stock_data`.
    - Columns: 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date'
    - Order data in ascending order of `Date`.

    Strategy Requirements:
    4. Strategy type, buy/sell logic, and indicators (e.g., RSI, MACD, MA) are given below:
    Strategy: {strategy}
    Buy Condition: {buy_condition}
    Sell Condition: {sell_condition}    
    5. Compute only the indicators mentioned in the strategy.
    6. For `turning` with `value: positive`: detect sign change from negative to positive.
    Example: (macd[i-1] < 0) & (macd[i] > 0)

    Signal Logic:
    7. Combine multiple conditions using the logic key (`and`/`or`) in a vectorized way using `&` / `|` with parentheses.
    - DO NOT use Python's plain `and` / `or` on pandas Series.
    8. For duration_type = "consecutive", use rolling windows to check if condition is met for duration_days.
    Example: RSI < 30 for 2 consecutive days → use `.rolling(window=2).apply(...)`

    Buy/Sell Execution Rules:
    9. Use a flag `position_open` to track whether a position is currently open.
    - Start with position_open = False
    - Set to True on a buy; set to False on a sell
    10. Sell signals should only be generated if a position is open.
    11. Ignore sells before the first buy.
    12. Do not allow consecutive buys or sells — signals must strictly alternate: buy → sell → buy.
    
    Buy/Sell Signal Alignment:

    - Do not forward-fill Buy/Sell columns.
    - DO NOT reassign index using data.index[buy_prices.index] — THIS CAUSES INDEXERROR.
    - Avoid manually resetting or reassigning indices on filtered signal series; trust their inherited indices from the source data to maintain correct time alignment.
    - After generating buy_signals and sell_signals, create the trades DataFrame, then drop rows where both Buy and Sell are NaN:
        trades.dropna(how='all', inplace=True)
    - Extract buy_prices and sell_prices using .dropna():
        buy_prices = trades['Buy'].dropna()
        sell_prices = trades['Sell'].dropna()
    - If the number of buy/sell signals is unequal, truncate the longer one:
        min_len = min(len(buy_prices), len(sell_prices))
        buy_prices = buy_prices.iloc[:min_len]
        sell_prices = sell_prices.iloc[:min_len]
    - This ensures signals alternate and are aligned for return calculations and plotting.
    - Always define buy_prices and sell_prices as empty pd.Series(dtype=float) before the if trades.empty: block, to avoid NameError when trades are empty.
    - This will prevent the IndexError, keep buy/sell signals aligned, and make your code more robust and clear.

    ROLLING INDICATOR:
        For any rolling indicator (e.g., 50-day high, average volume, moving average, etc.), always apply a .shift(1) to the rolling series before comparison. This ensures the current value is compared against the most recent past indicator value — avoiding self-referential logic which prevents valid breakouts or threshold crossings from being detected.
    
    IMPORTANT: Ensure that moving average crossover strategies (like golden cross or death cross) are not mistakenly implemented using MACD crossovers. A death cross should specifically refer to shorter MA < longer MA (e.g., 50MA < 200MA), and not MACD < signal line.
    Always maintain and update a variable for the current position’s entry price when trades are opened, especially if future logic depends on it (like stop-loss or target exits).
    
    Returns & Metrics:
    14. Use completed trades only to compute: - Store results in a comparison table (trading_results.html). Print the name of the file.
        - Cumulative returns
        - Annualized return
        - Volatility
        - Maximum drawdown
    IMPORTANT: Before using variables like annualized_return, volatility, etc., ensure they are always defined — even if trades are not executed. Initialize them to None before the conditional block.
    15. Annualized return:
            Use number of calendar days between first buy and last sell.
            IMPORTANT: When calculating annualized_return:

            First, reassign datetime indexes to buy_prices and sell_prices:

            buy_prices.index = data.index[buy_prices.index]
            sell_prices.index = data.index[sell_prices.index]


            Then extract the first and last trade dates directly from the reassigned indexes:

            buy_date = pd.to_datetime(buy_prices.index[0])
            sell_date = pd.to_datetime(sell_prices.index[-1])


            ✅ Do NOT index into data.index[...] again after reassignment. The buy_prices.index[...] is already datetime-like, and trying to do data.index[...] on it will fail.
            Use this value to calculate annualized return:
            annualized_return = cumulative_returns.iloc[-1] ** (365 / days_held) - 1 if days_held > 0 else 0

            Other rules:
            If days_held == 0, return 0 for annualized return.
            When calculating number of days between two dates ((sell_date - buy_date).days), always make sure both are datetime objects.
            If dates are stored in columns or come from indexes, always cast them using pd.to_datetime(...) before subtraction.
            Do not assume .index is datetime unless you've explicitly set it with df.index = pd.to_datetime(df.index).
    16. Volatility = standard deviation of trade returns × sqrt(252)
    17. Handle empty trades gracefully: if no trades, return 0 or None for all metrics and print a log message.
    IMPORTANT: - When implementing crossover-based trading signals (e.g., SMA crossovers), detect the *actual crossover event* by comparing the current and previous values of the moving averages:
    - A **buy signal** should be generated only when the short-term MA crosses *from below to above* the long-term MA (i.e., previous short-term MA ≤ previous long-term MA AND current short-term MA > current long-term MA).
    - A **sell signal** should be generated only when the short-term MA crosses *from above to below* the long-term MA (i.e., previous short-term MA ≥ previous long-term MA AND current short-term MA < current long-term MA).
    - Avoid generating buy/sell signals repeatedly if the condition is just sustained (e.g., short-term MA > long-term MA) without an actual crossover event.
    
    Plotting: IMPORTANT: Do not run plotting logic if there is no data.
    18. Use `plotly.graph_objects`:
        - Plot 'Close' price as blue line
        - Plot Buy markers (green triangle-up), Sell markers (red triangle-down)
        - No y-offset for markers — use actual Close values
        - Save as {ticker}_plot.html. Print the name of file.
    Ensure plotting code that uses buy_prices or sell_prices does not crash when these are empty. Use conditional checks like: if not buy_prices.empty: and if not sell_prices.empty:
    Ensure that buy_prices and sell_prices have a proper DatetimeIndex before plotting.
        When you extract them using .dropna(), the index may be a default integer index.
        You must reassign the correct datetime index using the original data DataFrame like this:

        buy_prices = trades['Buy'].dropna()
        buy_prices.index = data.index[buy_prices.index]

        sell_prices = trades['Sell'].dropna()
        sell_prices.index = data.index[sell_prices.index]

        This guarantees that Plotly plots markers at the correct positions on the time axis.
        If you skip this, the buy/sell markers may incorrectly appear around 1970 or outside the actual data range.
        IMPORTANT: Once you've reassigned buy_prices.index and sell_prices.index using data.index[...], access the first/last dates directly via .index[0] and .index[-1]. Do not attempt to index back into data.index[...] using those values — it will cause an error.
    
        IMPORTANT: After generating the buy and sell signals as arrays (e.g., buy_signals and sell_signals), attach them as columns to the main ticker_data DataFrame before attempting to access or visualize them.

            ✅ Example:

        ticker_data['Buy'] = buy_signals  
        ticker_data['Sell'] = sell_signals  


        This ensures the 'Buy' and 'Sell' columns are available in ticker_data when printing or plotting.

        🔁 Also, if you're using a separate trades DataFrame to store signals, ensure you use it consistently or integrate the signals back into the main DataFrame before referencing them.

        Avoid using variables like buy_signals, sell_signals, or trades across different tickers without recalculating them.
            Either:
            Move plotting inside the main loop per ticker, or
            Store buy/sell data separately for each ticker to avoid index mismatches.    
    
    19. Print the rows where buy/sell signals occur — include Date, Close, and indicators.
        - Do not print trades.dropna() or trades[['Buy', 'Sell']].dropna() at the end of the script, as it may misleadingly appear empty. Instead, print the full trades DataFrame or use a filtered version only if it has meaningful rows (e.g., only drop rows where both Buy and Sell are NaN using dropna(how='all')).
        
                
    Moving Averages:
    20. Only compute and plot MA if explicitly mentioned in the strategy.
        - Plot as dotted line
        - Use different colors for different MA periods

    Multi-Ticker Logic:
    21. If multiple tickers are present:
        - Generate plots and metrics for each ticker separately.
        - Filter the data by ticker to generate plot and metrics for each ticker.
        - There should be one plot html file for each ticker.
        - And trading_results.html should contain the table containing metrics from each ticker. i.e. one row per ticker.
        

    Edge Cases & Debugging:
    22. Print SQL query result for debugging.
    23. Avoid ambiguous pandas boolean errors — always use parentheses in conditions
    24. Do not use `.between()` in loops; use it only with full pandas Series
    25. VERY IMPORTANT: For cumulative calculations like cumulative returns, max drawdown, and cumulative sums:
        Always keep the data as pandas Series, never convert to NumPy arrays before these operations.
        Use pandas Series methods .cumprod(), .cummax(), .cumsum().
        Access elements with .iloc[] or .loc[] only on pandas Series or DataFrames, never on NumPy arrays.
        When calculating returns for vectorized operations, you may convert to NumPy arrays temporarily only for element-wise arithmetic, but do not convert the resulting Series for cumulative or indexed operations.
        To avoid errors, do vectorized math like (sell_prices - buy_prices) / buy_prices on NumPy arrays but then wrap the result back into a pandas Series before .cumprod() or .iloc usage. For example:
            returns = pd.Series((sell_prices - buy_prices) / buy_prices)  
            cumulative_returns = (1 + returns).cumprod()  
            annualized_return = cumulative_returns.iloc[-1] ** (365 / days_held) - 1  
        Never call .iloc on a NumPy array.
    26. If no trades are executed:
        - Print message: "No trades executed for {ticker}"
        - Return 0 or None for all calculated metrics
    27. When setting a column (e.g., Date) as index in a DataFrame, ensure subsequent code accessing that column uses .index instead of referring to it as a column.
        - For example, replace df['Date'] with df.index if Date is the index.
        - Maintain consistency between column and index usage to avoid KeyError.
        - Alternatively, avoid setting important columns as index if they need to be accessed as columns later.

    Final Output:
    28. For each ticker:
        - Save interactive plot as HTML file
        - Print DataFrame rows for buy/sell points
        - Include a summary HTML comparing metrics across tickers
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