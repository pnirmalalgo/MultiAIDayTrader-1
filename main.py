# main.py

import os
import sys
import json
import sqlite3
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv
from typing import TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from celery.result import AsyncResult

# Import your multi-agent/langgraph modules
from agents.interpreter import interpret_query
from agents.codegen import generate_code
from agents.code_cleaner import clean_code
from langgraph.graph import StateGraph, END

# Import Celery app and task
from tasks.executor import app as celery_app
from tasks.executor import run_python_code  # Celery task

import os
from fastapi.responses import FileResponse
# -----------------------------
# Type Definitions
# -----------------------------

HTML_DIR = "."


class GraphState(TypedDict):
    input: str
    intent: str
    code: str
    clean_code: str
    execution_result: str

class QueryRequest(BaseModel):
    query: str

# -----------------------------
# Utility Functions
# -----------------------------
def save_dataframe_to_sqlite(df, db_name='market_data.db', table_name='stock_data'):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def fetch_fmp_single_ticker(tkr, ticker_try, start_date, end_date):
    load_dotenv()
    FMP_API_KEY = os.getenv("FMP_API_KEY")
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
    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep_cols]
    df["Ticker"] = tkr
    return df

def get_fmp_stock_data(tickers, start_date, end_date):
    try:
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(",")]
        elif not isinstance(tickers, list):
            raise ValueError("Tickers must be a string or list")
        
        dfs = []
        for tkr in tickers:
            for suffix in [".NS", ".BS", ""]:
                ticker_try = tkr + suffix if suffix else tkr
                df = fetch_fmp_single_ticker(tkr, ticker_try, start_date, end_date)
                if not df.empty:
                    dfs.append(df)
                    break
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

# -----------------------------
# LangGraph Nodes
# -----------------------------
def node_interpreter(state):
    user_input = state["input"]
    return interpret_query(user_input)

def node_codegen(state):
    cleaned_content = state["intent"].replace("```json\n", "").replace("\n```", "")
    parsed_query = json.loads(cleaned_content)
    ticker = parsed_query["ticker"]
    start_date = parsed_query["start_date"]
    end_date = parsed_query["end_date"]
    buy_condition = parsed_query["buy_condition"]
    sell_condition = parsed_query["sell_condition"]

    # Fetch stock data
    stock_data = get_fmp_stock_data(ticker, start_date, end_date)
    return generate_code(cleaned_content, stock_data)

def node_cleaner(state):
    return clean_code(state["code"])

def node_executor(state):
    result = run_python_code.delay(state["clean_code"])  # Submit task to Celery
    return {"execution_result": f"Task submitted: {result.id}"}

# -----------------------------
# Build LangGraph
# -----------------------------
builder = StateGraph(GraphState)
builder.add_node("interpreter", node_interpreter)
builder.add_node("codegen", node_codegen)
builder.add_node("code_cleaner", node_cleaner)
builder.add_node("executor", node_executor)

builder.set_entry_point("interpreter")
builder.add_edge("interpreter", "codegen")
builder.add_edge("codegen", "code_cleaner")
builder.add_edge("code_cleaner", "executor")
builder.add_edge("executor", END)

langgraph_app = builder.compile()

# -----------------------------
# FastAPI App
# -----------------------------
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.post("/api/submit-query")
async def submit_query(req: QueryRequest):
    final = langgraph_app.invoke({"input": req.query})
    execution_result = final["execution_result"]
    task_id = execution_result.split(":")[-1].strip()
    return {"task_id": task_id}

@fastapi_app.get("/api/task-status/{task_id}")
async def task_status(task_id: str):
    async_result = AsyncResult(task_id, app=celery_app)  # <-- use Celery app here
    if async_result.ready():
        return {"status": async_result.state, "result": async_result.result}
    else:
        return {"status": async_result.state}
    
@fastapi_app.get("/api/list-html")
def list_html_files():
    try:
        files = [
            f for f in os.listdir(HTML_DIR)
            if f.endswith(".html")
        ]
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}

@fastapi_app.get("/api/html/{file_name}")
def get_html(file_name: str):
    file_path = os.path.join(HTML_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    return {"error": "File not found"}

# -----------------------------
# Optional: Run standalone
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, reload=True)

app = fastapi_app  # alias for Uvicorn
