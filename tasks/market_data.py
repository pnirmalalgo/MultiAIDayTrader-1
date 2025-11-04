import os
import asyncio
import aiohttp
import pandas as pd
import sqlite3
import urllib.parse
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

def save_dataframe_to_sqlite(df: pd.DataFrame, db_name: str = "market_data.db", table_name: str = "stock_data"):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

async def fetch_fmp_single_ticker(session, tkr: str, ticker_try: str, start_date: str, end_date: str):
    url = f"{FMP_BASE_URL}/historical-price-full/{urllib.parse.quote(ticker_try)}"
    params = {"from": start_date, "to": end_date, "apikey": FMP_API_KEY}
    
    try:
        async with session.get(url, params=params, timeout=15) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None

    if "historical" not in data or not data["historical"]:
        return None

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

async def fetch_ticker_with_suffixes(session, tkr, start_date, end_date):
    for suffix in [".NS", ".BS", ""]:
        df = await fetch_fmp_single_ticker(session, tkr, tkr + suffix if suffix else tkr, start_date, end_date)
        if df is not None:
            return df
    return None

async def get_fmp_stock_data_async(tickers, start_date, end_date):
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",")]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ticker_with_suffixes(session, tkr, start_date, end_date) for tkr in tickers]
        results = await asyncio.gather(*tasks)

    dfs = [df for df in results if df is not None]
    
    if not dfs:
        raise RuntimeError("No data fetched for any ticker.")

    final_df = pd.concat(dfs, ignore_index=True)
    save_dataframe_to_sqlite(final_df)
    return final_df

def get_fmp_stock_data(tickers, start_date, end_date):
    return asyncio.run(get_fmp_stock_data_async(tickers, start_date, end_date))
