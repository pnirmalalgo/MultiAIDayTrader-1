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

    
    # Convert DataFrame to JSON string
    df_json = stock_data.to_json(orient='records')  # Each record as a JSON object
    print(df_json)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt = (f"Write code for following: All conditions included(DO NOT include any comments. Only executable python code. VERY IMP: DO NOT include'''python. '''python causes code to break. Do not include pip install statements.): 1. Do not use yfinance. Required data is passed with the query. and `ta` libraries. But do not install. No pip install statements." \
        f"You are given a JSON object describing a stock trading strategy."\
        f"Your task is to generate working Python code using pandas and ta that:"\
        f"Computes all indicators mentioned in the strategy (RSI, MACD, etc.). Code should calulate only the strategy asked for. "\
        f"Implements buy and sell logic as described in the JSON."\
        f"For operator: turning, value: positive: Detect when an indicator changes sign from negative to positive between consecutive days. Example for MACD: (macd[i-1] < 0) & (macd[i] > 0)."\
        f"Combine multiple conditions using the logic key (and/or) in a vectorized way (& / | with parentheses). Do not use plain Python and/or on pandas Series."\
        f"For sell conditions involving Profit: Calculate percentage change from the last buy price and check if it meets the threshold."\
        f"Avoid ambiguous boolean errors by always using vectorized pandas comparisons and combining with & / |."\
        f"3. Write code for {strategy} and following buy_condition={buy_condition} and sell condition={sell_condition}" \
        f"ONLY If duration_type = 'consecutive' then for duration_days: Check that the condition has been met for the required number of consecutive days (e.g., RSI < 30 for 2 consecutive days). Use rolling windows to do this."\
        f"4. IMPORTANT WHEN MORE THAN ONE TICKER: If there are more than one tickers, Generate the below results and graph for each ticker. Also compare the results of all tickers in a table."\
        f"5. Calculate and PRINT: Annualized returns, maximum drawdown, volatility of the backtested strategy for each of the tickers."\
        f"6. Plot the 'Close' price of the stock along with the buy and sell signals on the same chart using `Plotly` for each ticker. Also plot Moving Average ONLY IF MA IS MENTIONED in strategy as dotted lines of different colors."\
        f"When plotting buy and sell signals, use different marker shapes (for example, triangle-up for buy and triangle-down for sell) so that if buy and sell occur on the same date and price, both are visible and don’t overlap as a single dot. Do not offset the y-axis values. Plot it as a blue line."\
        f"VERY IMPORTANT: After generating buy and sell signals, print the exact rows of the DataFrame that will be plotted as buy and sell markers. Show them separately: first print the buy signals DataFrame (only rows where buy is not NaN), then print the sell signals DataFrame (only rows where sell is not NaN). Ensure the printed data includes the date index, price, and any relevant indicator values used for the decision. This will help verify that plotted points match the actual signals generated."\
        f"7. Save the plot as an html file and output the name and path of the file. Use filename={ticker}_plot"\
        f"8. Data can be found from SQLLite3 database: market_data.db, table:  stock_data. Query this table using pandas.read_sql. Also make sure the data is ordered in ascending order of Date."\
        f"IMPORTANT: Add code to print the data from above query for debugging purposes."\
        f"9. Name of columns to be used: 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date' accordingly."\
        f"10. VERY IMPORTANT — Position Management Rules Clarification:"\
        f"A sell signal must ONLY be generated inside the loop if there is an open position (position_open=True)."\
        f"At the start, position_open=False. When a buy is triggered, set position_open=True. When a sell is triggered, set position_open=False."\
        f"This ensures no sell signal ever occurs before the first buy signal. Do not simply filter sells after generation — implement this logic in the loop."\
        f"Buy and sell signals must strictly alternate: buy → sell → buy → sell. If a condition for sell occurs but no buy is active, ignore it."\
        f"Do not allow consecutive buys without a sell in between, and do not allow consecutive sells without a buy in between."\
        f"Make sure that when you create buy_df and sell_df, they still contain the 'buy' and 'sell' columns along with any other indicators. Do not drop these columns when subsetting. Alternatively, you can just use data['buy'].dropna() and data['sell'].dropna() directly instead of relying on the subset DataFrames. Avoid writing code that later tries to access columns that were not included in the DataFrame slice."\
        f"VERY IMPORTANT: When calculating returns, align buy and sell prices so their ARRAYS HAVE SAME LENGTH. Use only completed trades (buy followed by sell). If the last trade is open, ignore it. i.e. drop the last buy row if no. of rows in buy is greater than no. of rows in sell."\
        f"Also calculate cumulative returns, annualized return, volatility, and maximum drawdown using only completed trades."\
        #f"VERY IMPORTANT: After generating buy and sell signals, do not assign trade returns directly to the full stock data DataFrame."\
       # f"Instead, align buy and sell signals, and create a separate DataFrame called trades. When creating the trades DataFrame, always align buy and sell trades by using the minimum length of both. Do not let extra buys or sells cause index mismatch errors."\
        f"If asked for cumulative returns over time, map them back only to the sell dates in the main DataFrame."
        f"14. Ensure these DataFrames exactly match the markers plotted on the chart — no mismatches."\
        f"VERY IMPORTANT: After generating buy and sell signals, REMOVE any sell signal whose date occurs before the first buy signal. This ensures that the plotted signals and backtest only show sells that happen after a corresponding buy. This is very important. Seems like there are still sell signals whose date is before any other buy signal. No such sell signal should remain whose date is before ANY buy signal."
        f"VERY IMPORTANT: Add code to check empty datasets. ValueError: zero-size array to reduction operation minimum which has no identity"\
        f"This happens when there are no buy or sell signals, so buy_prices or sell_prices is empty."\
        f"VERY IMPORTANT (Missing this instruction gives error): Ensure any cumulative operations (cummax(), cumprod(), cumsum(), pct_change()) are called only on pandas Series or DataFrame objects, never directly on NumPy arrays."\
        f"If the variable is a NumPy array but needs a cumulative method eg. cumulative_returns, convert it first using pd.Series() or pd.DataFrame()."\
        f"Do not convert Pandas Series to NumPy arrays (by using .values) when calculating returns, cumulative returns, drawdowns, or volatility. Keep them as Pandas Series so that methods like .cumprod() and .cummax() are available."\
        f"variable_name must be a pandas Series/DataFrame, not a NumPy array. Convert with pd.Series() first."
        f"Replace variable_name with the actual variable."\
        f"When checking a Pandas Series or DataFrame for truth values, always use .any() or .all() instead of relying on if with the object directly, because the truth value of a Series is ambiguous."\
        f"When checking conditions inside a loop (working with single float values), do not use Pandas Series methods like .between(). Instead, use direct comparisons like 40 <= value <= 60. Only use .between() when working with the whole Pandas Series."\
        f"Please modify the code so that it handles empty arrays gracefully. For example, if no trades occur, return None (or 0) for annualized_return, max_drawdown, and volatility instead of crashing. Also add a log/print message so we know no trades were executed."\
    )
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