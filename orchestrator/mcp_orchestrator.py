import urllib.parse
from dotenv import load_dotenv
import os
import sys
import json
import sqlite3
import pandas as pd
import requests
from typing import Generator

# -----------------------------
# Utility Functions
# -----------------------------
def save_dataframe_to_sqlite(df, db_name='market_data.db', table_name='stock_data'):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def fetch_fmp_single_ticker(tkr, ticker_try, start_date, end_date):
    print('here')
    load_dotenv()
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
    url = f"{FMP_BASE_URL}/historical-price-full/{urllib.parse.quote(ticker_try)}"
    params = {
        "from": start_date,
        "to": end_date,
        "apikey": FMP_API_KEY
    }

    print(url, params)
    resp = requests.get(url, params=params)
    print(resp)

    if resp.status_code != 200:
        print(f"HTTP Error {resp.status_code} for {ticker_try}")
        return pd.DataFrame()  # <-- just return, no yield

    data = resp.json()
    if "historical" not in data or not data["historical"]:
        print(f"No historical data for {ticker_try}")
        return pd.DataFrame()  # <-- just return, no yield

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

    print(f"Fetched {len(df)} rows for {ticker_try}")
    return df  # <-- return a DataFrame

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
            raise RuntimeError("No data fetched for any ticker.")
        final_df = pd.concat(dfs, ignore_index=True)
        save_dataframe_to_sqlite(final_df)
        return final_df
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise  # propagate exception instead of sys.exit

# -----------------------------
# MCP Orchestrator
# -----------------------------
class MCPOrchestrator:
    def __init__(self, interpreter, ticker_lookup, code_generator, code_cleaner, executor, lg_client=None):
        self.interpreter = interpreter
        self.ticker_lookup = ticker_lookup
        self.code_generator = code_generator
        self.code_cleaner = code_cleaner
        self.executor = executor
        self.lg = lg_client

    def run(self, query: str):
        context = {"input": query, "logs": []}

        # ----------------------------
        # Run Interpreter
        # ----------------------------
        interpreter_output = self.interpreter(context["input"])
        context["interpreter"] = interpreter_output  # stores both intent and logs

        json_str = interpreter_output["intent"]
        intent_parsed = json.loads(json_str.replace("```json\n", "").replace("\n```", ""))
        
        tickers = intent_parsed.get("ticker")
        if not tickers:
            raise ValueError("Interpreter output does not contain 'ticker' key")

        # Optional: resolve tickers
        resolved_tickers = self.ticker_lookup(tickers)
        intent_parsed["ticker"] = resolved_tickers
        context["intent"] = intent_parsed

        # ----------------------------
        # Fetch stock data
        # ----------------------------
        start_date = intent_parsed["start_date"]
        end_date = intent_parsed["end_date"]
        stock_data = get_fmp_stock_data(resolved_tickers, start_date, end_date)
        context["stock_data"] = stock_data

        # ----------------------------
        # Run Code Generator
        # ----------------------------
        codegen_output = self.code_generator(intent_parsed)
        context["code"] = codegen_output["code"]
        context["code_logs"] = codegen_output["logs"]

        print("Generated code:", context["code"])

        # ----------------------------
        # Run Code Cleaner
        # ----------------------------
        code_cleaner_output = self.code_cleaner(context["code"])
        if isinstance(code_cleaner_output, dict):
            context.setdefault("logs", []).extend(code_cleaner_output.get("context", []))
            context["clean_code"] = code_cleaner_output["clean_code"]
        else:
            context["clean_code"] = code_cleaner_output

        # ----------------------------
        # Execute code
        # ----------------------------
        context["execution"] = self.executor(context["clean_code"])

        # ----------------------------
        # Return full context including logs
        # ----------------------------
        return context
    def decide_action(self, context):
        """
        ReAct-style reasoning with logging
        """
        if 'interpreter' not in context:
            action = "INTERPRETER"
            input_text = context.get("input", "")
        elif 'code' not in context:
            action = "CODE_GENERATOR"
            input_text = context.get("intent", "")
        elif 'execution' not in context:
            action = "EXECUTE"
            input_text = context.get("clean_code", "")
        elif context.get('execution', {}).get('success') is False:
            if 'clean_code' not in context:
                action = "CODE_CLEANER"
                input_text = context.get("code", "")
            else:
                action = "CODE_GENERATOR"
                input_text = context.get("clean_code", "")
        else:
            action = "APPROVE"
            input_text = ""

        context.setdefault("logs", [])
        context["logs"].append({
            "step": "orchestrator",
            "react": f"Decided to call {action} with input: {input_text}"
        })
        return action
