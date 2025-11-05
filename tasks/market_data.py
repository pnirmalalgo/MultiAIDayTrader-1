import os
import asyncio
import aiohttp
import pandas as pd
import sqlite3
import urllib.parse
import random
import time
from dotenv import load_dotenv
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Optimized parameters for server environment
MAX_CONCURRENT_REQUESTS = 5    # More conservative for server
RETRY_LIMIT = 5                # More retry attempts
BASE_TIMEOUT = 30              # Longer timeout for slower networks
REQUEST_DELAY = 0.2            # Longer delay between requests
AUTO_BATCH_THRESHOLD = 30      # Smaller batches for server
BATCH_SIZE = 30                # Smaller batch size
BATCH_COOLDOWN = 5             # Longer cooldown between batches
TOTAL_TIMEOUT = 300            # 5 minutes max per batch

sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def save_dataframe_to_sqlite(df: pd.DataFrame, db_name: str = "market_data.db", table_name: str = "stock_data"):
    """Save dataframe to SQLite with connection management and retry logic"""
    conn = None
    max_db_retries = 3
    
    for attempt in range(1, max_db_retries + 1):
        try:
            conn = sqlite3.connect(db_name, timeout=60.0)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info(f"✓ Saved {len(df)} rows to {db_name}")
            return
        except sqlite3.OperationalError as e:
            logger.warning(f"Database locked (attempt {attempt}/{max_db_retries}): {e}")
            if attempt < max_db_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        except Exception as e:
            logger.error(f"✗ Database save failed: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

async def fetch_fmp_single_ticker(
    session: aiohttp.ClientSession, 
    tkr: str, 
    ticker_try: str, 
    start_date: str, 
    end_date: str
) -> Optional[pd.DataFrame]:
    """Fetch data for a single ticker with exponential backoff retry and robust error handling"""
    url = f"{FMP_BASE_URL}/historical-price-full/{urllib.parse.quote(ticker_try)}"
    
    for attempt in range(1, RETRY_LIMIT + 1):
        async with sem:
            try:
                # Add jittered delay to prevent thundering herd
                await asyncio.sleep(REQUEST_DELAY + random.uniform(0, 0.1))
                
                params = {"from": start_date, "to": end_date, "apikey": FMP_API_KEY}
                
                # Progressive timeout increase
                timeout = aiohttp.ClientTimeout(
                    total=BASE_TIMEOUT * attempt,
                    connect=10,  # Connection timeout
                    sock_read=BASE_TIMEOUT  # Socket read timeout
                )
                
                async with session.get(url, params=params, timeout=timeout, ssl=False) as resp:
                    # Handle rate limiting
                    if resp.status == 429:
                        retry_after = resp.headers.get('Retry-After', 2 ** attempt)
                        try:
                            wait_time = float(retry_after)
                        except:
                            wait_time = 2 ** attempt + random.uniform(0, 2)
                        
                        logger.warning(f"⚠ Rate limit for {ticker_try}. Waiting {wait_time:.1f}s (attempt {attempt}/{RETRY_LIMIT})")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Handle server errors (500, 502, 503, 504)
                    if resp.status >= 500:
                        wait_time = 2 ** attempt
                        logger.warning(f"⚠ Server error {resp.status} for {ticker_try}. Retrying in {wait_time}s")
                        if attempt < RETRY_LIMIT:
                            await asyncio.sleep(wait_time)
                        continue
                    
                    # Handle client errors (except 404)
                    if resp.status == 404:
                        logger.info(f"ℹ Ticker {ticker_try} not found (404)")
                        return None
                    
                    if resp.status != 200:
                        logger.warning(f"✗ HTTP {resp.status} for {ticker_try} (attempt {attempt}/{RETRY_LIMIT})")
                        if attempt < RETRY_LIMIT:
                            wait_time = min(2 ** attempt, 10)
                            await asyncio.sleep(wait_time)
                        continue

                    # Read response with timeout
                    try:
                        text = await asyncio.wait_for(resp.text(), timeout=30)
                        data = await asyncio.wait_for(resp.json(), timeout=30)
                    except asyncio.TimeoutError:
                        logger.warning(f"⏱ Response read timeout for {ticker_try}")
                        if attempt < RETRY_LIMIT:
                            await asyncio.sleep(2 ** attempt)
                        continue

                    # Validate response structure
                    if not isinstance(data, dict):
                        logger.warning(f"⚠ Invalid response format for {ticker_try}: expected dict, got {type(data)}")
                        if attempt < RETRY_LIMIT:
                            await asyncio.sleep(1)
                        continue
                    
                    # Check if data exists
                    if "historical" not in data or not data["historical"]:
                        if attempt == 1:
                            logger.info(f"ℹ No data for {ticker_try}")
                        return None

                    # Process successful response
                    try:
                        df = pd.DataFrame(data["historical"])
                        
                        # Validate DataFrame
                        if df.empty:
                            logger.info(f"ℹ Empty data for {ticker_try}")
                            return None
                        
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
                        logger.info(f"✓ Fetched {len(df)} records for {ticker_try}")
                        return df
                    
                    except Exception as e:
                        logger.error(f"✗ Data processing error for {ticker_try}: {e}")
                        if attempt < RETRY_LIMIT:
                            await asyncio.sleep(1)
                        continue

            except asyncio.TimeoutError:
                wait_time = 2 ** attempt
                logger.warning(f"⏱ Timeout for {ticker_try} (attempt {attempt}/{RETRY_LIMIT}). Waiting {wait_time}s")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(wait_time)
            
            except aiohttp.ClientConnectorError as e:
                logger.error(f"✗ Connection failed for {ticker_try}: {e}")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(2 ** attempt)
                    
            except aiohttp.ClientError as e:
                logger.error(f"✗ Client error for {ticker_try}: {e} (attempt {attempt}/{RETRY_LIMIT})")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(2 ** attempt)
            
            except aiohttp.ServerDisconnectedError as e:
                logger.error(f"✗ Server disconnected for {ticker_try}: {e}")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"✗ Unexpected error for {ticker_try}: {type(e).__name__}: {e}")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(2)

    logger.error(f"✗ Failed to fetch {ticker_try} after {RETRY_LIMIT} attempts")
    return None

async def fetch_ticker_with_suffixes(
    session: aiohttp.ClientSession, 
    tkr: str, 
    start_date: str, 
    end_date: str
) -> Optional[pd.DataFrame]:
    """Try multiple suffixes for Indian stocks with fallback"""
    # Try most common suffix first for Indian stocks
    suffixes = [".NS", ".BO", ""]
    
    for suffix in suffixes:
        ticker_try = tkr + suffix if suffix else tkr
        try:
            df = await fetch_fmp_single_ticker(session, tkr, ticker_try, start_date, end_date)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            logger.error(f"✗ Error trying {ticker_try}: {e}")
            continue
    
    logger.warning(f"✗ No data found for {tkr} with any suffix")
    return None

async def _process_batch_internal(
    tickers: List[str], 
    start_date: str, 
    end_date: str,
    batch_num: int = 0,
    total_batches: int = 1
) -> pd.DataFrame:
    """Internal function to process a single batch with comprehensive error handling"""
    if batch_num > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 Batch {batch_num}/{total_batches}")
        logger.info(f"{'='*60}")
    
    # More conservative connection settings for server environment
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS,
        limit_per_host=MAX_CONCURRENT_REQUESTS,
        ttl_dns_cache=300,
        force_close=True,  # Force close connections
        enable_cleanup_closed=True  # Clean up closed connections
    )
    
    # Global timeout for entire batch
    timeout = aiohttp.ClientTimeout(
        total=TOTAL_TIMEOUT,
        connect=10,
        sock_read=BASE_TIMEOUT
    )
    
    try:
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            trust_env=True  # Respect proxy settings if any
        ) as session:
            tasks = [fetch_ticker_with_suffixes(session, tkr, start_date, end_date) for tkr in tickers]
            
            # Use gather with return_exceptions to prevent one failure from stopping all
            results = []
            completed = 0
            
            try:
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await asyncio.wait_for(coro, timeout=TOTAL_TIMEOUT/len(tickers))
                        completed += 1
                        if completed % 5 == 0 or completed == len(tickers):
                            progress = completed / len(tickers) * 100
                            logger.info(f"Progress: {completed}/{len(tickers)} tickers processed ({progress:.1f}%)")
                        results.append(result)
                    except asyncio.TimeoutError:
                        logger.error(f"⏱ Ticker timeout in batch processing")
                        results.append(None)
                    except Exception as e:
                        logger.error(f"✗ Error in batch processing: {e}")
                        results.append(None)
                        
            except asyncio.TimeoutError:
                logger.error(f"⏱ Batch timeout reached after {TOTAL_TIMEOUT}s")

        # Filter successful results
        dfs = [df for df in results if isinstance(df, pd.DataFrame) and len(df) > 0]
        
        failed_count = len(tickers) - len(dfs)
        if failed_count > 0:
            logger.warning(f"⚠️ Batch result: {len(dfs)}/{len(tickers)} successful, {failed_count} failed")
        else:
            logger.info(f"✓ Batch completed: {len(dfs)}/{len(tickers)} successful")

        if not dfs:
            logger.warning("⚠️ No successful fetches in this batch")
            return pd.DataFrame()

        # Combine batch results
        batch_df = pd.concat(dfs, ignore_index=True)
        return batch_df
    
    except Exception as e:
        logger.error(f"✗ Batch processing error: {e}")
        return pd.DataFrame()
    
    finally:
        # Ensure connector is closed
        if connector:
            await connector.close()

async def get_fmp_stock_data_async(
    tickers: List[str], 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Fetch data for multiple tickers asynchronously with automatic batching and robust error handling.
    
    Automatically batches requests if ticker count exceeds threshold for better reliability.
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",")]

    num_tickers = len(tickers)
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 FMP Data Fetch")
    logger.info(f"{'='*60}")
    logger.info(f"Tickers: {num_tickers}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info(f"Settings: {MAX_CONCURRENT_REQUESTS} concurrent, {RETRY_LIMIT} retries")
    
    # Automatic batching decision
    if num_tickers > AUTO_BATCH_THRESHOLD:
        num_batches = (num_tickers + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"🔄 Auto-batching: {num_batches} batches of ~{BATCH_SIZE} tickers")
        logger.info(f"{'='*60}\n")
        
        all_dfs = []
        failed_batches = []
        
        for i in range(0, num_tickers, BATCH_SIZE):
            batch = tickers[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            try:
                batch_df = await _process_batch_internal(
                    batch, 
                    start_date, 
                    end_date,
                    batch_num=batch_num,
                    total_batches=num_batches
                )
                
                if not batch_df.empty:
                    all_dfs.append(batch_df)
                else:
                    failed_batches.append(batch_num)
                    logger.warning(f"⚠️ Batch {batch_num} returned no data")
                
            except Exception as e:
                logger.error(f"✗ Batch {batch_num} failed: {e}")
                failed_batches.append(batch_num)
            
            # Cooldown between batches (except after last batch)
            if i + BATCH_SIZE < num_tickers:
                logger.info(f"⏸ Cooling down {BATCH_COOLDOWN}s before next batch...\n")
                await asyncio.sleep(BATCH_COOLDOWN)
        
        if not all_dfs:
            error_msg = "❌ No data fetched for any ticker in any batch. Check API key, rate limits, network, and ticker symbols."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if failed_batches:
            logger.warning(f"⚠️ {len(failed_batches)} batches failed: {failed_batches}")
        
        final_df = pd.concat(all_dfs, ignore_index=True)
        
    else:
        # Process all at once for smaller lists
        logger.info(f"⚡ Single batch processing")
        logger.info(f"{'='*60}\n")
        
        try:
            final_df = await _process_batch_internal(tickers, start_date, end_date)
            
            if final_df.empty:
                error_msg = "❌ No data fetched for any ticker. Check API key, rate limits, network, and ticker symbols."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            logger.error(f"✗ Single batch processing failed: {e}")
            raise
    
    # Sort and save
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Final Results")
    logger.info(f"{'='*60}")
    
    final_df = final_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    unique_tickers = final_df['Ticker'].nunique()
    failed_tickers = num_tickers - unique_tickers
    
    logger.info(f"✓ Successfully fetched: {unique_tickers}/{num_tickers} tickers")
    if failed_tickers > 0:
        logger.warning(f"✗ Failed to fetch: {failed_tickers} tickers")
    logger.info(f"✓ Total records: {len(final_df):,}")
    
    try:
        save_dataframe_to_sqlite(final_df)
    except Exception as e:
        logger.error(f"✗ Failed to save to database: {e}")
        # Still return the dataframe even if save fails
    
    logger.info(f"{'='*60}\n")
    
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
    max_retries = 2  # Retry entire operation if it fails
    
    for retry in range(max_retries):
        try:
            logger.info(f"\n{'='*60}")
            if retry > 0:
                logger.info(f"🔄 Retry attempt {retry + 1}/{max_retries}")
            logger.info(f"{'='*60}")
            
            return asyncio.run(get_fmp_stock_data_async(tickers, start_date, end_date))
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Process interrupted by user")
            raise
            
        except RuntimeError as e:
            if "No data fetched" in str(e) and retry < max_retries - 1:
                wait_time = 10 * (retry + 1)
                logger.warning(f"⚠️ Retrying entire operation in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"\n❌ Fatal error: {e}")
                raise
                
        except Exception as e:
            logger.error(f"\n❌ Fatal error: {type(e).__name__}: {e}")
            if retry < max_retries - 1:
                wait_time = 10 * (retry + 1)
                logger.warning(f"⚠️ Retrying entire operation in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    raise RuntimeError("Failed to fetch data after all retries")