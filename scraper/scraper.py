import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import sqlite3
from typing import List, Dict
import re
from pathlib import Path

csv_path = "tickers/NIFTY_TOTAL_LIST.csv"
db_path = "../market_data.db"
table_name = "company_metrics"

def clean_number(text: str) -> float:
    """
    Clean and convert text to number.
    Handles formats like: "1,234.56", "12.5%", "1,234 Cr.", etc.
    """
    if not text or text.strip() in ['-', 'N/A', '']:
        return None
    
    # Remove whitespace
    text = text.strip()
    
    # Remove percentage sign
    text = text.replace('%', '')
    
    # Remove 'Cr.' (Crores)
    text = text.replace('Cr.', '').replace('Cr', '')
    
    # Remove commas
    text = text.replace(',', '')
    
    try:
        return float(text)
    except ValueError:
        return None


def scrape_screener_data(ticker: str) -> Dict:
    """
    Scrape data from screener.in for a given ticker.
    
    Args:
        ticker: Stock ticker (e.g., "TCS", "RELIANCE")
    
    Returns:
        Dictionary with scraped data
    """
    # Screener.in uses ticker without exchange suffix
    base_ticker = ticker.replace('.NS', '').replace('.BO', '')
    url = f"https://www.screener.in/company/{base_ticker}/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"Scraping {base_ticker}...", end=' ')
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Error {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        data = {
            'Ticker': ticker,
            'Company_Name': None,
            'Market_Cap': None,
            'PE_Ratio': None,
            'PB_Ratio': None,
            'PS_Ratio': None,
            'ROCE': None,
            'ROE': None,
            'Net_Profit': None,
            'Promoter_Holding': None
        }
        
        # Get company name
        try:
            company_name = soup.find('h1')
            if company_name:
                data['Company_Name'] = company_name.text.strip()
        except:
            pass
        
        # Find all ratio/metric sections
        try:
            # Find all list items with ratios
            ratio_list = soup.find_all('li', class_='flex flex-space-between')
            
            for item in ratio_list:
                name_elem = item.find('span', class_='name')
                value_elem = item.find('span', class_='number')
                
                if name_elem and value_elem:
                    name = name_elem.text.strip()
                    value = value_elem.text.strip()
                    
                    # Map to our fields
                    if 'Market Cap' in name:
                        data['Market_Cap'] = clean_number(value)
                    elif 'Stock P/E' in name or 'PE Ratio' in name:
                        data['PE_Ratio'] = clean_number(value)
                    elif 'Book Value' in name or 'Price to Book' in name:
                        data['PB_Ratio'] = clean_number(value)
                    elif 'ROCE' in name:
                        data['ROCE'] = clean_number(value)
                    elif 'ROE' in name:
                        data['ROE'] = clean_number(value)
                    elif 'Promoter holding' in name:
                        data['Promoter_Holding'] = clean_number(value)
        except Exception as e:
            print(f"⚠️ Error parsing ratios: {e}")
        
        # Try alternate structure for metrics in top box
        try:
            top_ratios = soup.find('div', id='top-ratios')
            if top_ratios:
                ratio_items = top_ratios.find_all('li')
                for item in ratio_items:
                    text = item.text
                    if 'Market Cap' in text:
                        nums = re.findall(r'[\d,\.]+', text)
                        if nums and not data['Market_Cap']:
                            data['Market_Cap'] = clean_number(nums[0])
                    elif 'P/E' in text:
                        nums = re.findall(r'[\d,\.]+', text)
                        if nums and not data['PE_Ratio']:
                            data['PE_Ratio'] = clean_number(nums[0])
                    elif 'ROCE' in text:
                        nums = re.findall(r'[\d,\.]+', text)
                        if nums and not data['ROCE']:
                            data['ROCE'] = clean_number(nums[0])
                    elif 'ROE' in text:
                        nums = re.findall(r'[\d,\.]+', text)
                        if nums and not data['ROE']:
                            data['ROE'] = clean_number(nums[0])
        except:
            pass

        # --- Extract Promoter Holding ---
        try:
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if not cells:
                        continue
                    first_cell = cells[0].get_text(strip=True)
                    if re.search(r"Promoters\s*\+", first_cell, re.I):
                        latest_val = cells[-1].get_text(strip=True)
                        data["Promoter_Holding"] = clean_number(latest_val)
                        break
        except Exception as e:
            print(f"⚠️ Promoter Holding not found: {e}")

        # --- Extract Net Profit ---
        try:
            for table in tables:
                if "Profit & Loss" in str(table):
                    for row in table.find_all("tr"):
                        cols = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
                        if len(cols) < 2:
                            continue
                        if re.search(r"(Net Profit|Profit after tax)", cols[0], re.I):
                            latest_val = cols[-1]
                            data["Net_Profit"] = clean_number(latest_val)
                            break
        except Exception as e:
            print(f"⚠️ Net Profit not found: {e}")

        # --- Compute P/S Ratio ---
        try:
            if data.get("Market_Cap") and data.get("Net_Profit"):
                # try to approximate sales if available in tables
                for table in tables:
                    if "Profit & Loss" in str(table):
                        for row in table.find_all("tr"):
                            cols = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
                            if len(cols) < 2:
                                continue
                            if re.search(r"(Net Sales|Total Revenue|Revenue from Operations)", cols[0], re.I):
                                latest_val = clean_number(cols[-1])
                                if latest_val and latest_val > 0:
                                    data["PS_Ratio"] = data["Market_Cap"] / (latest_val * 1e7)  # since Cr → ₹
                                break
                        break
        except Exception as e:
            print(f"⚠️ Error computing P/S Ratio: {e}")

                # --- Extract Sales and Net Profit from Quarterly Results ---
        try:
            quarterly_header = soup.find(lambda tag: tag.name in ["h2", "h3"] and "Quarterly Results" in tag.get_text())
            if quarterly_header:
                table = quarterly_header.find_next("table")
                if table:
                    rows = table.find_all("tr")
                    for row in rows:
                        cols = [c.get_text(strip=True).replace("+", "") for c in row.find_all(["th", "td"])]
                        if not cols or len(cols) < 2:
                            continue

                        # Sales row
                        if re.search(r"^Sales$", cols[0], re.I):
                            # Last column = latest quarter
                            latest_value = clean_number(cols[-1])
                            if latest_value:
                                data["Sales"] = latest_value

                        # Net Profit row
                        elif re.search(r"^Net Profit$", cols[0], re.I):
                            latest_value = clean_number(cols[-1])
                            if latest_value:
                                data["Net_Profit"] = latest_value

            # --- Compute P/S Ratio (Market Cap / Annualized Sales) ---
            if data.get("Market_Cap") and data.get("Sales"):
                # Sales is quarterly (₹ Cr) — multiply by 4 to annualize
                annual_sales = data["Sales"] * 4
                if annual_sales > 0:
                    data["PS_Ratio"] = round(data["Market_Cap"] / annual_sales, 2)

        except Exception as e:
            print(f"⚠️ Error extracting quarterly Sales/Net Profit: {e}")

        print(f"✅ (Market Cap={data['Market_Cap']}, PE={data['PE_Ratio']}, ROCE={data['ROCE']}%)")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None




def load_from_database(db_path: str = "../market_data.db", table_name: str = "company_metrics") -> pd.DataFrame:
    """
    Load data from SQLite database.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of the table
    
    Returns:
        DataFrame with data from database
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        print(f"✅ Loaded {len(df)} records from database")
        return df
    except Exception as e:
        print(f"❌ Error loading from database: {e}")
        return pd.DataFrame()


def scrape_multiple_tickers(tickers: List[str], delay: float = 2.0) -> pd.DataFrame:
    """
    Scrape data for multiple tickers with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        delay: Delay between requests in seconds (to avoid rate limiting)
    
    Returns:
        DataFrame with scraped data
    """
    results = []
    
    print(f"Starting scrape for {len(tickers)} tickers...")
    print("=" * 80)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] ", end='')
        
        data = scrape_screener_data(ticker)
        if data:
            results.append(data)
        
        # Rate limiting - be nice to the server
        if i < len(tickers):
            time.sleep(delay)
    
    print("=" * 80)

    df = pd.DataFrame(results)
    

    print(f"✅ Scraping complete! Successfully scraped {len(results)}/{len(tickers)} tickers")
    
    return df


def load_tickers_from_csv(csv_path: str) -> List[str]:
    """
    Load tickers from CSV file.
    
    Args:
        csv_path: Path to CSV file with tickers
    
    Returns:
        List of ticker symbols
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Try different column names
        ticker_col = None
        for col in ['Ticker', 'ticker', 'Symbol', 'symbol', 'SYMBOL']:
            if col in df.columns:
                ticker_col = col
                break
        
        if ticker_col is None:
            # Use first column if no standard name found
            ticker_col = df.columns[0]
            print(f"⚠️ Using first column '{ticker_col}' as ticker column")
        
        tickers = df[ticker_col].dropna().tolist()
        print(f"✅ Loaded {len(tickers)} tickers from {csv_path}")
        return tickers
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []


def display_database_summary(db_path: str = "../market_data.db", table_name: str = "company_metrics"):
    """
    Display summary of data in database.
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Get sample data
        df_sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print("\n📋 Sample Data (first 5 rows):")
        print(df_sample.to_string())
        
        # Get statistics
        df_stats = pd.read_sql_query(f"""
            SELECT 
                COUNT(*) as Total_Companies,
                AVG(Market_Cap) as Avg_Market_Cap,
                AVG(PE_Ratio) as Avg_PE,
                AVG(ROCE) as Avg_ROCE,
                AVG(ROE) as Avg_ROE,
                AVG(Promoter_Holding) as Avg_Promoter_Holding
            FROM {table_name}
        """, conn)
        
        print("\n📊 Summary Statistics:")
        print(df_stats.to_string())
        
        # Companies with best metrics
        df_top_roce = pd.read_sql_query(f"""
            SELECT Ticker, Company_Name, ROCE, ROE, PE_Ratio
            FROM {table_name}
            WHERE ROCE IS NOT NULL
            ORDER BY ROCE DESC
            LIMIT 5
        """, conn)
        
        print("\n🏆 Top 5 Companies by ROCE:")
        print(df_top_roce.to_string())
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error displaying summary: {e}")


if __name__ == "__main__":
    # Configuration
    
    # Load tickers from CSV
    tickers = load_tickers_from_csv(csv_path)
    
    if not tickers:
        print("❌ No tickers found. Using sample tickers...")
        tickers = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK"]
    
    # Limit to first 1 for testing (remove this for full scrape)
    print(f"\n⚠️ Testing with first 20 tickers. Remove limit for full scrape.")
    tickers = tickers[:1]
    
    # Scrape data
    df = scrape_multiple_tickers(tickers, delay=2.0)
    
    
    # Display summary
    display_database_summary(db_path=db_path, table_name=table_name)
    
    