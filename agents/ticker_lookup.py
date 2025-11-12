import requests
import os
from dotenv import load_dotenv
import pandas as pd
import re

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

PLACEHOLDER_TO_CSV = {
    "NIFTY_50_LIST": "tickers/NIFTY_50_LIST.csv",
    "NIFTY_NEXT_50_LIST": "tickers/NIFTY_NEXT_50_LIST.csv",
    "NIFTY_MIDCAP_100_LIST": "tickers/NIFTY_MIDCAP_100_LIST.csv",
    "NIFTY_SMALLCAP_250_LIST": "tickers/NIFTY_SMALLCAP_250_LIST.csv",
    "NIFTY_500_LIST": "tickers/NIFTY_500_LIST.csv",
    "NIFTY_TOTAL_LIST": "tickers/NIFTY_TOTAL_LIST.csv"
}

def clean_name(raw_name: str) -> str:
    """
    Remove unwanted quotes, brackets, and whitespace.
    Examples:
      "'TCS.NS'" -> TCS.NS
      " 'INFY.NS' " -> INFY.NS
    """
    return re.sub(r"^[\"'\s]+|[\"'\s]+$", "", str(raw_name).strip())

def resolve_ticker(company_names):
    """
    Resolve one or multiple company names into a list of ticker symbols.
    Always returns a list of strings (tickers).
    """
    print(f"Resolving tickers for: {company_names}")

    if isinstance(company_names, str):
        company_names = [company_names]
    elif not isinstance(company_names, list):
        raise ValueError("company_names must be a string or list of strings")

    tickers = []
    for raw_name in company_names:
        name = clean_name(raw_name)

        # Check for placeholder list
        if name in PLACEHOLDER_TO_CSV:
            csv_path = PLACEHOLDER_TO_CSV[name]
            df = pd.read_csv(csv_path, header=None)
            print(f"{name} tickers:", df[0].tolist())
            tickers.extend(df[0].tolist())
            continue

        try:
            resp = requests.get(
                f"{FMP_BASE_URL}/search",
                params={"query": name, "apikey": FMP_API_KEY}
            )
            resp.raise_for_status()
            results = resp.json()

            if results:
                # Prefer NSE, then BSE
                nse_match = next((r for r in results if r.get("exchangeShortName") == "NSE"), None)
                bse_match = next((r for r in results if r.get("exchangeShortName") == "BSE"), None)

                if nse_match:
                    tickers.append(nse_match["symbol"])
                elif bse_match:
                    tickers.append(bse_match["symbol"])
                else:
                    tickers.append(results[0]["symbol"])
            else:
                print(f"⚠️ No ticker found for {name}")
        except Exception as e:
            print(f"❌ Error resolving {name}: {e}")

    print(f"Resolved tickers: {tickers}")
    return tickers
