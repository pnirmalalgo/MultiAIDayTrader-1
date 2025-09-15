import os
import requests
import pandas as pd
import sqlite3
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

def save_dataframe_to_sqlite(df: pd.DataFrame, db_name: str = "market_data.db", table_name: str = "stock_data"):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(df)
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
    #print(f"Fetching {ticker_try} from {url} with params {params}")
    
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
