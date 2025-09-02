import pandas as pd
import sqlite3
import warnings
from ta.momentum import RSIIndicator
from plotly.subplots import make_subplots
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

conn = sqlite3.connect('market_data.db')
query = "SELECT * FROM stock_data"
data = pd.read_sql(query, conn)

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

tickers = data['Ticker'].unique()
results = []

for ticker in tickers:
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data.set_index('Date', inplace=True)

    rsi = RSIIndicator(ticker_data['Close']).rsi()
    ticker_data['RSI'] = rsi

    ticker_data['Buy'] = None
    ticker_data['Sell'] = None

    trades = ticker_data[['Buy', 'Sell']].dropna(how='all')
    buy_prices = trades['Buy'].dropna()
    sell_prices = trades['Sell'].dropna()

    cumulative_return = 0
    annualized_return = 0
    volatility = 0
    max_drawdown = 0

    if not buy_prices.empty and not sell_prices.empty:
        min_len = min(len(buy_prices), len(sell_prices))
        buy_prices = buy_prices.iloc[:min_len]
        sell_prices = sell_prices.iloc[:min_len]

        returns = (sell_prices.values - buy_prices.values) / buy_prices.values
        cumulative_return = (returns.sum() * 100)
        total_days = (ticker_data.index[-1] - ticker_data.index[0]).days
        if total_days > 0:
            annualized_return = ((1 + (returns.mean())) ** (365 / total_days) - 1) * 100
        volatility = returns.std() * (252 ** 0.5) * 100
        max_drawdown = (1 - (returns.cumsum() / (1 + returns.cumsum()).cummax())).max() * 100

    results.append({
        'Ticker': ticker,
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown
    })

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['RSI'], name='RSI', line=dict(color='orange')), row=1, col=1, secondary_y=True)
    fig.update_layout(title=f'Trading Signals for {ticker}')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

print("No trades executed for the following tickers:", [result['Ticker'] for result in results])
print("Generated HTML files for plots and trading results.")