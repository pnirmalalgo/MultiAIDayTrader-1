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
        f"2. Download stock data for '{ticker}' from {date_range}."\
        f"3. Write code for {strategy} and following buy_condition={buy_condition} and sell condition={sell_condition}" \
        f"4. If there are more than one tickers, Generate the below results and graph for each ticker. Also compare the results of all tickers."
        f"5. Calculate and PRINT Annualized returns, maximum drawdown, volatility of the backtested strategy."\
        f"6. Plot the 'Close' price of the stock along with the buy and sell signals on the same chart using `matplotlib`."\
        f"7. Also plot moving average if applicable. And save the plot as an image file and output the name and path of the file. Use filename={ticker}_{timestamp}"
        f"8. Data can be found from SQLLite3 database: market_data.db, table:  stock_data. Query this table using pandas.read_sql"\
        f"9. Name of columns to be used: 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date' accordingly." 
        f"IMPORTANT: Handle position management logic."
        f"When generating code:"
        f"A buy signal should trigger only when no active position is held."
        f"After a buy, the position should be considered open, and further buy signals should be ignored until a sell signal appears."
        f"A sell signal should trigger only when a position is currently held."
        f"After a sell, the position is considered closed, and further sell signals should be ignored until a buy signal appears."
        f"This ensures that multiple consecutive buy or sell signals are not acted on — the strategy must alternate between buy and sell."
        f"You may use a variable like position to track the current state."
        )
    print(prompt)
    messages = [
        SystemMessage(content="Write simple python code. Do not use yfinance. VERY IMP: DO NOT include'''python. '''python causes code to break. Required data is stored in SQLite3 table provided in the query. Use ta library. Initialize ta class."\
                      "Include required libraries eg. numpy. Initialize ta class properly. Use: from ta.momentum import RSIIndicator. Please note: DO NOT use ta.add_all_ta_features(This is not required and gives error). We only need Close data as of now. "\
                      "DO NOT include any comments. Only executable python code. Print output as per instructions." \
                      "The generated Python code must be compatible with pandas 2.0+."\
                      "Replace any usage of the deprecated Series.append() or DataFrame.append() with pd.concat([obj1, obj2])." \
                      "When generating code that creates signal lists (e.g., Buy/Sell, Long/Short), ensure the lists are exactly the same length as the DataFrame index."\
                      "Always start the lists pre-filled with np.nan for all rows (e.g., [np.nan] * len(data)) or append values for every iteration so the final list length equals len(data)."\
                      "Do not start loops at range(1, len(data)) unless you also pre-fill the first element(s) to keep lengths equal."\
                      "Before assigning to data['column'], validate that len(list) == len(data)."\
                      "The fix must be generic so it works for MACD, RSI, SMA, or any other indicator."\
                      "Add checks whereever necessary to see if there is no data before accessing data."),
        HumanMessage(content=prompt)
    ]
    

    response = llm.invoke(messages)
    print("Generated code: ",response.content)
    return {"code": response.content}