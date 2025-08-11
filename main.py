from agents.interpreter import interpret_query
from agents.ticker_lookup import resolve_ticker
from agents.codegen import generate_code
from agents.code_cleaner import clean_code
from tasks.executor import app
from tasks.executor import run_python_code  # Celery task
from celery.result import AsyncResult
from typing import TypedDict
from langgraph.graph import StateGraph, END
import pandas as pd
import yfinance as yf
import json
import sqlite3
import sys
import requests
import os
from dotenv import load_dotenv

class GraphState(TypedDict):
    input: str
    intent: str
    code: str
    clean_code: str
    execution_result: str

def node_interpreter(state):
    user_input = state["input"]
    result = interpret_query(user_input)  # This should return {"intent": "..."}
    return result

def save_dataframe_to_sqlite(df, db_name='market_data.db', table_name='stock_data'):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    # Close the connection
    conn.close()


def fetch_fmp_single_ticker(tkr, ticker_try, start_date, end_date):
    load_dotenv()
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    """Fetch stock data from FMP for a single ticker in yfinance-like format."""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_try}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
    resp = requests.get(url)
    
    if resp.status_code != 200:
        print(f"HTTP Error {resp.status_code} for {ticker_try}")
        return pd.DataFrame()
    
    data = resp.json()
    if "historical" not in data or not data["historical"]:
        return pd.DataFrame()
    
    df = pd.DataFrame(data["historical"])
    df.rename(columns={
        "date": "Date",
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
        "adjClose": "Adj Close" if "adjClose" in df.columns else "Close"
    }, inplace=True)
    
    # Keep only required columns (some might be missing)
    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep_cols]
    
    # Add Ticker column
    df["Ticker"] = tkr
    return df

def get_fmp_stock_data(tickers, start_date, end_date):
    try:
        # Normalize tickers input
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(",")]
        elif not isinstance(tickers, list):
            raise ValueError("Tickers must be a string or list")
        
        dfs = []
        for tkr in tickers:
            for suffix in [".NS", ".BS", ""]:
                ticker_try = tkr + suffix if suffix else tkr
                print(f"Trying ticker: {ticker_try}")
                df = fetch_fmp_single_ticker(tkr, ticker_try, start_date, end_date)
                if not df.empty:
                    dfs.append(df)
                    break  # Success, no need to try next suffix
            else:
                print(f"No data found for {tkr} with any suffix")
        
        if not dfs:
            raise Exception("No data fetched for any ticker.")
        
        final_df = pd.concat(dfs, ignore_index=True)
        save_dataframe_to_sqlite(final_df)
        return final_df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)


def fetch_stock_data(tickers, start_date, end_date):
    print(tickers)
    #stock_data = pd.DataFrame(columns=["Date", "Close"])
    print(start_date)
    print(end_date)
    try:
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(',')]
        elif isinstance(tickers, list):
            tickers = tickers
        else:
            raise ValueError("Ticker must be a string or list")
        # Instead, fetch separately for each ticker and concat with ticker column
        dfs = []

        for tkr in tickers:
            print("tkr:", tkr)
            df = yf.download(tkr, start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                print(f"No data returned for {tkr}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten columns to only first level (Open, High, Low, etc.)
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()  # Make 'Date' a column instead of index
            df['Ticker'] = tkr     # Add ticker column
            df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Reorder columns
            dfs.append(df)
            print("df:", df)
            print("dffs:", dfs)

        # Only concat if dfs is not empty
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            print(final_df.head())
        else:
            print("No data fetched for any ticker.")
        
        if final_df.empty:
            raise Exception("yfinance did not return data. Please try another query or try again later.")
    except Exception as e:
        print(f"Error fetching data for: {e}")
        sys.exit(1)

    # Save to SQLite
    save_dataframe_to_sqlite(final_df)
    return final_df
    

def node_codegen(state):
    cleaned_content = state["intent"].replace("```json\n", "").replace("\n```", "")
    #parsed_query = state["intent"]Test RELIANCE: buy RSI under 35, sell RSI over 65, last 6 months
    print("Cleaned content:", cleaned_content)
    parsed_query = json.loads(cleaned_content)
    print(parsed_query)
    ticker = parsed_query["ticker"]
    start_date = parsed_query["start_date"]
    end_date = parsed_query["end_date"]
    buy_condition = parsed_query["buy_condition"]
    sell_condition = parsed_query["sell_condition"]

    print(ticker)
    
    # Fetch stock data for the relevant dates
    #stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = get_fmp_stock_data(ticker, start_date, end_date)
    cleaned_content = state["intent"].replace("```json\n", "").replace("\n```", "")
    print("intent before code generation:", cleaned_content)
    return generate_code(cleaned_content, stock_data)

def node_cleaner(state):
    return clean_code(state["code"])

def node_executor(state):
    result = run_python_code.delay(state["clean_code"])
    return {"execution_result": f"Task submitted: {result.id}"}

builder = StateGraph(GraphState)
builder.add_node("interpreter", node_interpreter)
#builder.add_node("ticker_lookup", node_ticker_lookup)
builder.add_node("codegen", node_codegen)
builder.add_node("code_cleaner", node_cleaner)
builder.add_node("executor", node_executor)

builder.set_entry_point("interpreter")
#builder.add_edge("interpreter", "ticker_lookup")
builder.add_edge("interpreter", "codegen")
builder.add_edge("codegen", "code_cleaner")
builder.add_edge("code_cleaner", "executor")
builder.add_edge("executor", END)

langgraph_app = builder.compile()

if __name__ == "__main__":
    query = input("📈 Your query:\n> ")
    final = langgraph_app.invoke({"input": query})
    print("\n📊 Task ID / Result:\n", final["execution_result"])

    execution_result = final['execution_result']

    # Split by colon and strip whitespace
    task_id = execution_result.split(":")[-1].strip()

    print("Task ID:", task_id)
    # Create an AsyncResult instance

    async_result = AsyncResult(task_id, app=app)

    # Wait and fetch result
    # Waits up to 10 seconds, polls every 0.5 seconds
    try:
        result = async_result.get(timeout=10, interval=0.5)
        print("Task result:", result)
    except Exception as e:
        print("Error during result.get():", e)
        print("Task state:", async_result.state)
        print("Task traceback:", async_result.traceback)    