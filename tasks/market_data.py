import os
import asyncio
import aiohttp
import pandas as pd
import sqlite3
import urllib.parse
import random
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Optimized parameters
MAX_CONCURRENT_REQUESTS = 10   # Reduced to respect API limits
RETRY_LIMIT = 4                # Increased retry attempts
BASE_TIMEOUT = 20              # Increased timeout
REQUEST_DELAY = 0.1            # Minimum delay between requests
AUTO_BATCH_THRESHOLD = 50      # Automatically batch if more than 50 tickers
BATCH_SIZE = 50                # Size of each batch
BATCH_COOLDOWN = 2             # Seconds to wait between batches

sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def save_dataframe_to_sqlite(df: pd.DataFrame, db_name: str = "market_data.db", table_name: str = "stock_data"):
    """Save dataframe to SQLite with connection management"""
    conn = None
    try:
        conn = sqlite3.connect(db_name, timeout=30.0)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"✓ Saved {len(df)} rows to {db_name}")
    except Exception as e:
        print(f"✗ Database save failed: {e}")
        raise
    finally:
        if conn:
            conn.close()

async def fetch_fmp_single_ticker(
    session: aiohttp.ClientSession, 
    tkr: str, 
    ticker_try: str, 
    start_date: str, 
    end_date: str
) -> Optional[pd.DataFrame]:
    """Fetch data for a single ticker with exponential backoff retry"""
    url = f"{FMP_BASE_URL}/historical-price-full/{urllib.parse.quote(ticker_try)}"
    
    for attempt in range(1, RETRY_LIMIT + 1):
        async with sem:
            try:
                # Add small delay to prevent burst requests
                await asyncio.sleep(REQUEST_DELAY)
                
                params = {"from": start_date, "to": end_date, "apikey": FMP_API_KEY}
                
                # Dynamic timeout: increase with retry attempts
                timeout = aiohttp.ClientTimeout(total=BASE_TIMEOUT * attempt)
                
                async with session.get(url, params=params, timeout=timeout) as resp:
                    # Handle rate limiting explicitly
                    if resp.status == 429:
                        wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff
                        print(f"⚠ Rate limit hit for {ticker_try}. Waiting {wait_time:.1f}s (attempt {attempt}/{RETRY_LIMIT})")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Handle other non-200 responses
                    if resp.status != 200:
                        print(f"✗ HTTP {resp.status} for {ticker_try} (attempt {attempt}/{RETRY_LIMIT})")
                        if attempt < RETRY_LIMIT:
                            wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                            await asyncio.sleep(wait_time)
                        continue

                    data = await resp.json()

                    # Check if data exists
                    if "historical" not in data or not data["historical"]:
                        if attempt == 1:
                            print(f"ℹ No data for {ticker_try}")
                        return None  # Don't retry if no data exists

                    # Process successful response
                    df = pd.DataFrame(data["historical"])
                    df.rename(columns={
                        "date": "Date",
                        "close": "Close",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "volume": "Volume",
                        "adjClose": "Adj Close"
                    }, inplace=True, errors='ignore')

                    df["Ticker"] = tkr
                    print(f"✓ Fetched {len(df)} records for {ticker_try}")
                    return df

            except asyncio.TimeoutError:
                wait_time = 2 ** attempt
                print(f"⏱ Timeout for {ticker_try} (attempt {attempt}/{RETRY_LIMIT}). Waiting {wait_time}s")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(wait_time)
                    
            except aiohttp.ClientError as e:
                print(f"✗ Connection error for {ticker_try}: {e} (attempt {attempt}/{RETRY_LIMIT})")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                print(f"✗ Unexpected error for {ticker_try}: {e} (attempt {attempt}/{RETRY_LIMIT})")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(1)

    print(f"✗ Failed to fetch {ticker_try} after {RETRY_LIMIT} attempts")
    return None

async def fetch_ticker_with_suffixes(
    session: aiohttp.ClientSession, 
    tkr: str, 
    start_date: str, 
    end_date: str
) -> Optional[pd.DataFrame]:
    """Try multiple suffixes for Indian stocks"""
    # Try most common suffix first for Indian stocks
    suffixes = [".NS", ".BO", ""]  # .BO is Bombay Stock Exchange
    
    for suffix in suffixes:
        ticker_try = tkr + suffix if suffix else tkr
        df = await fetch_fmp_single_ticker(session, tkr, ticker_try, start_date, end_date)
        if df is not None and len(df) > 0:
            return df
    
    print(f"✗ No data found for {tkr} with any suffix")
    return None

async def _process_batch_internal(
    tickers: List[str], 
    start_date: str, 
    end_date: str,
    batch_num: int = 0,
    total_batches: int = 1
) -> pd.DataFrame:
    """Internal function to process a single batch"""
    if batch_num > 0:
        print(f"\n{'='*60}")
        print(f"📦 Batch {batch_num}/{total_batches}")
        print(f"{'='*60}")
    
    # Configure session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS * 2,
        limit_per_host=MAX_CONCURRENT_REQUESTS,
        ttl_dns_cache=300
    )
    
    timeout = aiohttp.ClientTimeout(total=BASE_TIMEOUT * RETRY_LIMIT * 2)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_ticker_with_suffixes(session, tkr, start_date, end_date) for tkr in tickers]
        
        # Use gather to collect results with progress tracking
        results = []
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            if completed % 10 == 0 or completed == len(tickers):
                progress = completed / len(tickers) * 100
                print(f"Progress: {completed}/{len(tickers)} tickers processed ({progress:.1f}%)")
            results.append(result)

    # Filter successful results
    dfs = [df for df in results if isinstance(df, pd.DataFrame) and len(df) > 0]
    
    failed_count = len(tickers) - len(dfs)
    if failed_count > 0:
        print(f"⚠️ Batch result: {len(dfs)}/{len(tickers)} successful, {failed_count} failed")
    else:
        print(f"✓ Batch completed: {len(dfs)}/{len(tickers)} successful")

    if not dfs:
        return pd.DataFrame()  # Return empty dataframe instead of raising

    # Combine batch results
    batch_df = pd.concat(dfs, ignore_index=True)
    return batch_df

async def get_fmp_stock_data_async(
    tickers: List[str], 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Fetch data for multiple tickers asynchronously with automatic batching.
    
    Automatically batches requests if ticker count exceeds threshold for better reliability.
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",")]

    num_tickers = len(tickers)
    print(f"\n{'='*60}")
    print(f"📊 FMP Data Fetch")
    print(f"{'='*60}")
    print(f"Tickers: {num_tickers}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Settings: {MAX_CONCURRENT_REQUESTS} concurrent, {RETRY_LIMIT} retries")
    
    # Automatic batching decision
    if num_tickers > AUTO_BATCH_THRESHOLD:
        num_batches = (num_tickers + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"🔄 Auto-batching: {num_batches} batches of ~{BATCH_SIZE} tickers")
        print(f"{'='*60}\n")
        
        all_dfs = []
        
        for i in range(0, num_tickers, BATCH_SIZE):
            batch = tickers[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            batch_df = await _process_batch_internal(
                batch, 
                start_date, 
                end_date,
                batch_num=batch_num,
                total_batches=num_batches
            )
            
            if not batch_df.empty:
                all_dfs.append(batch_df)
            
            # Cooldown between batches (except after last batch)
            if i + BATCH_SIZE < num_tickers:
                print(f"⏸ Cooling down {BATCH_COOLDOWN}s before next batch...\n")
                await asyncio.sleep(BATCH_COOLDOWN)
        
        if not all_dfs:
            raise RuntimeError("❌ No data fetched for any ticker in any batch. Check API key, rate limits, and ticker symbols.")
        
        final_df = pd.concat(all_dfs, ignore_index=True)
        
    else:
        # Process all at once for smaller lists
        print(f"⚡ Single batch processing")
        print(f"{'='*60}\n")
        
        final_df = await _process_batch_internal(tickers, start_date, end_date)
        
        if final_df.empty:
            raise RuntimeError("❌ No data fetched for any ticker. Check API key, rate limits, and ticker symbols.")
    
    # Sort and save
    print(f"\n{'='*60}")
    print(f"📊 Final Results")
    print(f"{'='*60}")
    
    final_df = final_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    unique_tickers = final_df['Ticker'].nunique()
    failed_tickers = num_tickers - unique_tickers
    
    print(f"✓ Successfully fetched: {unique_tickers}/{num_tickers} tickers")
    if failed_tickers > 0:
        print(f"✗ Failed to fetch: {failed_tickers} tickers")
    print(f"✓ Total records: {len(final_df):,}")
    
    save_dataframe_to_sqlite(final_df)
    print(f"{'='*60}\n")
    
    return final_df

def get_fmp_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Synchronous wrapper for fetching FMP stock data.
    
    Automatically handles batching for large ticker lists.
    No code changes needed - just call with any number of tickers.
    
    Args:
        tickers: List of ticker symbols or comma-separated string
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with columns: Ticker, Date, Open, High, Low, Close, Volume, Adj Close
    
    Example:
        # Works for small lists
        df = get_fmp_stock_data(['RELIANCE', 'TCS'], '2024-01-01', '2024-12-31')
        
        # Automatically batches for large lists
        df = get_fmp_stock_data(large_ticker_list, '2024-01-01', '2024-12-31')
    """
    try:
        return asyncio.run(get_fmp_stock_data_async(tickers, start_date, end_date))
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        raise
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise