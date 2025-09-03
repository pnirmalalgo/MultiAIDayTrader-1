# main.py

import os
import json
import sqlite3
import pandas as pd
import requests
import urllib.parse
from dotenv import load_dotenv
from typing import TypedDict
import re      # ✅ missing
import ast     # ✅ missing
import logging  # ✅ missing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from celery.result import AsyncResult

# your multi-agent/langgraph imports
from agents.interpreter import interpret_query
from agents.codegen import generate_code
from agents.code_cleaner import clean_code
from agents.ticker_lookup import resolve_ticker
from langgraph.graph import StateGraph, END

# Celery app + task
from tasks.executor import app as celery_app
from tasks.executor import run_python_code  # Celery task

# -----------------------------
# Config
# -----------------------------
PLOTS_DIR = os.path.abspath(".")  # directory where .html plots are written
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Types & Models
# -----------------------------
class GraphState(TypedDict):
    input: str
    intent: str
    code: str
    clean_code: str
    execution_result: str

class QueryRequest(BaseModel):
    query: str

# -----------------------------
# Utilities
# -----------------------------
def save_dataframe_to_sqlite(df: pd.DataFrame, db_name: str = "market_data.db", table_name: str = "stock_data"):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

def fetch_fmp_single_ticker(tkr: str, ticker_try: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for a single ticker_try from FMP.
    Returns empty DataFrame on no-data or HTTP error.
    """
    load_dotenv()
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
    url = f"{FMP_BASE_URL}/historical-price-full/{urllib.parse.quote(ticker_try)}"
    params = {"from": start_date, "to": end_date, "apikey": FMP_API_KEY}

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        # return empty DataFrame (caller will try other suffixes)
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

def get_fmp_stock_data(tickers, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch data for a list or comma-separated string of tickers.
    Raises RuntimeError if nothing fetched.
    """
    # normalize tickers
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",")]
    elif not isinstance(tickers, list):
        raise ValueError("Tickers must be string or list")

    dfs = []
    for tkr in tickers:
        found = False
        for suffix in [".NS", ".BS", ""]:
            ticker_try = tkr + suffix if suffix else tkr
            df = fetch_fmp_single_ticker(tkr, ticker_try, start_date, end_date)
            if not df.empty:
                dfs.append(df)
                found = True
                break
        if not found:
            print(f"No data found for {tkr} with any suffix")

    if not dfs:
        raise RuntimeError("No data fetched for any ticker.")

    final_df = pd.concat(dfs, ignore_index=True)
    save_dataframe_to_sqlite(final_df)
    return final_df

# -----------------------------
# LangGraph nodes
# -----------------------------
def node_interpreter(state):
    user_input = state["input"]
    return interpret_query(user_input)

def node_ticker_lookup(state):
    try:
        intent_json = state["intent"].replace("```json\n", "").replace("\n```", "")
        parsed = json.loads(intent_json)
        ticker_or_company = parsed["ticker"]
        resolved = resolve_ticker(ticker_or_company)
        parsed["ticker"] = resolved
        return {"intent": json.dumps(parsed)}
    except Exception as e:
        return {"intent": state.get("intent", ""), "error": str(e)}

def node_codegen(state):
    cleaned_content = state["intent"].replace("```json\n", "").replace("\n```", "")
    parsed_query = json.loads(cleaned_content)
    tickers = parsed_query["ticker"]
    start_date = parsed_query["start_date"]
    end_date = parsed_query["end_date"]
    _buy_condition = parsed_query.get("buy_condition")
    _sell_condition = parsed_query.get("sell_condition")

    # Fetch data and then call generate_code
    stock_data = get_fmp_stock_data(tickers, start_date, end_date)
    return generate_code(cleaned_content, stock_data)

def node_cleaner(state):
    return clean_code(state["code"])

def node_executor(state):
    # queue Celery task with the cleaned code
    result = run_python_code.delay(state["clean_code"])
    return {"execution_result": f"Task submitted: {result.id}"}

# -----------------------------
# Build LangGraph
# -----------------------------
builder = StateGraph(GraphState)
builder.add_node("interpreter", node_interpreter)
builder.add_node("ticker_lookup", node_ticker_lookup)
builder.add_node("codegen", node_codegen)
builder.add_node("code_cleaner", node_cleaner)
builder.add_node("executor", node_executor)

builder.set_entry_point("interpreter")
builder.add_edge("interpreter", "ticker_lookup")
builder.add_edge("ticker_lookup", "codegen")
builder.add_edge("codegen", "code_cleaner")
builder.add_edge("code_cleaner", "executor")
builder.add_edge("executor", END)

langgraph_app = builder.compile()

# -----------------------------
# FastAPI app
# -----------------------------
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated HTML files under /plots/* for iframe usage
fastapi_app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# ---- submit-query (queue task and return task_id) ----
@fastapi_app.post("/api/submit-query")
async def submit_query(req: QueryRequest):
    """
    Run the LangGraph pipeline which will eventually enqueue a Celery task.
    Returns the Celery task id so the frontend can poll /api/task-status/<id>.
    """
    try:
        # run the graph synchronously (it will call node_executor which enqueues Celery)
        final = langgraph_app.invoke({"input": req.query})

        # LangGraph node_executor returns {"execution_result": "Task submitted: <id>"}
        execution_result = final.get("execution_result", "") if isinstance(final, dict) else ""
        if not execution_result:
            # As fallback, try to extract from different structure
            execution_result = final

        # parse the result like "Task submitted: <id>"
        task_id = None
        if isinstance(execution_result, str):
            if ":" in execution_result:
                task_id = execution_result.split(":", 1)[1].strip()
            else:
                task_id = execution_result.strip()

        if not task_id:
            return {"status": "ERROR", "error": "Could not parse Celery task id from pipeline output", "pipeline_result": execution_result}

        return {"status": "PENDING", "task_id": task_id}

    except Exception as e:
        # don't crash the server; return error to frontend
        return {"status": "ERROR", "error": str(e)}

# ---- task-status endpoint (single canonical) ----
@fastapi_app.get("/api/task-status/{task_id}")
async def task_status(task_id: str):
    async_result = AsyncResult(task_id, app=celery_app)
    state = async_result.state

    if state == "SUCCESS":
        res = async_result.result
        logging.info("DEBUG result from Celery: %r", res)

        files = []
        output = ""

        if isinstance(res, dict):
            files = res.get("files") or res.get("html_files") or []
            output = res.get("output", "")
        else:
            output = str(res)

        # --- NEW: fallback parser for "Generated files: ['...']"
        if not files and output:
            m = re.search(r"Generated files:\s*(\[.*\])", output)
            if m:
                try:
                    candidate = m.group(1)
                    parsed = ast.literal_eval(candidate)
                    if isinstance(parsed, (list, tuple)):
                        files = list(parsed)
                except Exception:
                    logging.exception("Failed to parse Generated files from output")

        return {"status": "SUCCESS", "files": files}

    elif state == "FAILURE":
        err = async_result.result
        return {"status": "FAILURE", "error": str(err)}

    meta = async_result.info or {}
    logs = meta.get("logs", [])
    return {"status": state, "logs": logs}

    # For ongoing states return state (PENDING/PROGRESS/STARTED)
    # If the worker updates meta logs, include them for frontend visibility
    meta = async_result.info or {}
    logs = meta.get("logs", [])
    return {"status": state, "logs": logs}

# ---- optional: list all html files under PLOTS_DIR ----
@fastapi_app.get("/api/list-html")
def list_html_files():
    try:
        files = [f for f in os.listdir(PLOTS_DIR) if f.endswith(".html")]
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}

# ---- serve a single html file (if needed) ----
@fastapi_app.get("/api/html/{file_name}")
def get_html(file_name: str):
    file_path = os.path.join(PLOTS_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    return {"error": "File not found"}

# -----------------------------
# Run standalone (dev)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, reload=True)

# expose app for ASGI
app = fastapi_app
