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
    query = "SELECT * FROM stock_data"
    data = pd.read_sql(query, conn)
    print("data from database:")
    print(data)
    conn.close()

def fetch_stock_data(ticker, start_date, end_date):
    #stock_data = pd.DataFrame(columns=["Date", "Close"])
    print(start_date)
    print(end_date)
    try:
        
        stock_data_point = yf.download(ticker, start=start_date, end=end_date,  auto_adjust=True)
        stock_data_point.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        stock_data_point['Ticker'] = ticker
        #print(stock_data_point)
        #if not stock_data_point.empty:
            #stock_data = stock_data.append({"Date": date, "Close": stock_data_point['Close'][0]}, ignore_index=True)
    except Exception as e:
        print(f"Error fetching data for: {e}")

    # Save to SQLite
    save_dataframe_to_sqlite(stock_data_point)
    return stock_data_point
'''
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = date_range.strftime('%Y-%m-%d').tolist()

    print(dates)
    
    for date in dates:
        try:
            stock_data_point = yf.download(ticker, start=date, end=date)
            if not stock_data_point.empty:
                stock_data = stock_data.append({"Date": date, "Close": stock_data_point['Close'][0]}, ignore_index=True)
        except Exception as e:
            print(f"Error fetching data for {date}: {e}")
   ''' 
    

def node_codegen(state):
    cleaned_content = state["intent"].replace("```json\n", "").replace("\n```", "")
    #parsed_query = state["intent"]
    print("Cleaned content:", cleaned_content)
    parsed_query = json.loads(cleaned_content)
    print(parsed_query)
    ticker = parsed_query["ticker"]
    start_date = parsed_query["start_date"]
    end_date = parsed_query["end_date"]
    buy_condition = parsed_query["buy_condition"]
    sell_condition = parsed_query["sell_condition"]

    # Fetch stock data for the relevant dates
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print("last dataframe:", stock_data)
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