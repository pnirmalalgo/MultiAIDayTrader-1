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
import logging

load_dotenv()

# ---------------- Logging Setup ----------------
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("market_data")
logger.setLevel(logging.INFO)

# Prevent duplicate logs if this file is imported multiple times
if not logger.handlers:
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler (visible in `docker logs`)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # File Handler (saved to /app/logs/market_data.log)
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "market_data.log"))
    file_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Optional: also propagate to FastAPI root logger
    logger.propagate = True


def save_dataframe_to_sqlite(df: pd.DataFrame, db_name="market_data.db", table_name="stock_data"):
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        logger.info(f"✅ Saved to DB: {table_name}, Rows: {len(df)}")
    except Exception as e:
        logger.error(f"❌ Error saving to DB {table_name}: {e}")


async def fetch_fmp_single_ticker(session, tkr, ticker_try, start, end, retries=3):
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    base = "https://financialmodelingprep.com/api/v3"
    url = f"{base}/historical-price-full/{urllib.parse.quote(ticker_try)}"
    params = {"from": start, "to": end, "apikey": FMP_API_KEY}

    logger.info(f"🔄 Fetching {ticker_try} | Original: {tkr}")

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
                        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
                                if "Adj Close" in df.columns else
                                ["Date", "Open", "High", "Low", "Close", "Volume"]]
                        df["Ticker"] = tkr
                        logger.info(f"✅ Success: {tkr} | Rows: {len(df)} | Attempt {attempt}")
                        return df
                else:
                    logger.warning(f"❌ HTTP {resp.status} for {ticker_try}, attempt {attempt}")
        except (ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"⚠️ Error {ticker_try} attempt {attempt}: {e}")

        await asyncio.sleep(0.5 + random.random())

    logger.error(f"🚫 Failed to fetch data for {tkr} after {retries} attempts")
    return pd.DataFrame()


async def fetch_with_suffix(session, tkr, start, end):
    for suffix in [""]:
        ticker_try = tkr + suffix if suffix else tkr
        logger.info(f"🔍 Trying ticker variation: {ticker_try}")
        df = await fetch_fmp_single_ticker(session, tkr, ticker_try, start, end)
        if not df.empty:
            logger.info(f"✅ Data found with suffix {suffix} for {tkr}")
            return df
        await asyncio.sleep(0.3)

    logger.error(f"🚫 No data for any suffix of {tkr}")
    return pd.DataFrame()


async def get_fmp_stock_data_async(tickers, start, end, max_concurrency=4):
    logger.info(f"🚀 Starting fetch for tickers: {tickers} from {start} to {end}")

    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",")]

    timeout = ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch_with_suffix(session, t, start, end) for t in tickers]
        results = await asyncio.gather(*tasks)

    dfs = [df for df in results if not df.empty]
    if not dfs:
        logger.error("❌ No data fetched for any ticker")
        raise RuntimeError("No data fetched at all")

    final_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"📊 Combined DataFrame shape: {final_df.shape}")
    save_dataframe_to_sqlite(final_df)
    logger.info("✅ All tickers fetched and saved")
    return final_df


def get_fmp_stock_data(tickers, start, end):
    logger.info(f"📞 get_fmp_stock_data called with: {tickers}, {start}, {end}")
    return asyncio.run(get_fmp_stock_data_async(tickers, start, end))
