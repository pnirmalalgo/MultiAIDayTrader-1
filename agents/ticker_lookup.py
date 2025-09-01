import requests
import os
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

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
    for name in company_names:
        try:
            resp = requests.get(
                f"{FMP_BASE_URL}/search",
                params={"query": name, "apikey": FMP_API_KEY}
            )
            resp.raise_for_status()
            results = resp.json()
            if results:
                # Try to find NSE first
                nse_match = next((r for r in results if r.get("exchangeShortName") == "NSE"), None)
                if nse_match:
                    tickers.append(nse_match["symbol"])
                    continue

                # Try to find BSE next
                bse_match = next((r for r in results if r.get("exchangeShortName") == "BSE"), None)
                if bse_match:
                    tickers.append(bse_match["symbol"])
                    continue

                # Fallback: just take the first result
                tickers.append(results[0]["symbol"])
            else:
                print(f"⚠️ No ticker found for {name}")
        except Exception as e:
            print(f"❌ Error resolving {name}: {e}")

    print(f"Resolved tickers: {tickers}")
    return tickers
