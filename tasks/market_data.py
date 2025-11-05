import os
import asyncio
import aiohttp
import pandas as pd
import sqlite3
import urllib.parse
from dotenv import load_dotenv
from aiohttp import ClientTimeout, ClientError
import random
import time

load_dotenv()


def save_dataframe_to_sqlite(df: pd.DataFrame, db_name="market_data.db", table_name="stock_data"):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print("✅ Saved to DB:", table_name)


async def fetch_fmp_single_ticker(session, tkr, ticker_try, start, end, retries=3):
    """
    Fetch one ticker + suffix with retry.
    """
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    base = "https://financialmodelingprep.com/api/v3"
    url = f"{base}/historical-price-full/{urllib.parse.quote(ticker_try)}"
    params = {"from": start, "to": end, "apikey": FMP_API_KEY}

    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "historical" in data and data["historical"]:
                        df = pd.DataFrame(data["historical"])
                        df.rename(columns={
                            "date": "Date", "close": "Close", "open": "Open",
                            "high": "High", "low": "Low", "volume": "Volume",
                            "adjClose": "Adj Close" if "adjClose" in df.columns else "Close"
                        }, inplace=True)
                        df = df[["Date","Open","High","Low","Close","Adj Close","Volume"] if "Adj Close" in df.columns else ["Date","Open","High","Low","Close","Volume"]]
                        df["Ticker"] = tkr
                        return df
                else:
                    print(f"❌ HTTP {resp.status} for {ticker_try}, attempt {attempt}")
        except (ClientError, asyncio.TimeoutError) as e:
            print(f"⚠️ Error {ticker_try} attempt {attempt}: {e}")

        await asyncio.sleep(0.5 + random.random())  # Respect rate limits

    return pd.DataFrame()  # after retries failed


async def fetch_with_suffix(session, tkr, start, end):
    for suffix in [".NS", ".BS", ""]:
        df = await fetch_fmp_single_ticker(session, tkr, tkr + suffix if suffix else tkr, start, end)
        if not df.empty:
            return df
        await asyncio.sleep(0.3)  # prevent API spam
    print(f"🚫 No data returned for {tkr}")
    return pd.DataFrame()


async def get_fmp_stock_data_async(tickers, start, end, max_concurrency=4):
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",")]

    timeout = ClientTimeout(total=None)  # allow retries properly
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch_with_suffix(session, t, start, end) for t in tickers]
        results = await asyncio.gather(*tasks)

    dfs = [df for df in results if not df.empty]
    if not dfs:
        raise RuntimeError("No data fetched at all")

    final_df = pd.concat(dfs, ignore_index=True)
    save_dataframe_to_sqlite(final_df)
    return final_df


def get_fmp_stock_data(tickers, start, end):
    return asyncio.run(get_fmp_stock_data_async(tickers, start, end))
