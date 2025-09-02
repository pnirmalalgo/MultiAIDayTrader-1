import pandas as pd
import sqlite3
from ta.momentum import RSIIndicator
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

conn = sqlite3.connect('market_data.db')
data = pd.read_sql('SELECT * FROM stock_data', conn)
conn.close()

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

tickers = data['Ticker'].unique()
results = []

for ticker in tickers:
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data.set_index('Date', inplace=True)

    rsi = RSIIndicator(ticker_data['Close'], window=14).rsi()
    ticker_data['RSI'] = rsi

    ticker_data['Buy'] = (ticker_data['RSI'] < 35) * ticker_data['Close']
    ticker_data['Sell'] = (ticker_data['RSI'] > 65) * ticker_data['Close']

    trades = ticker_data[['Buy', 'Sell']].dropna(how='all')
    buy_prices = trades['Buy'].dropna()
    sell_prices = trades['Sell'].dropna()

    min_len = min(len(buy_prices), len(sell_prices))
    buy_prices = buy_prices.iloc[:min_len]
    sell_prices = sell_prices.iloc[:min_len]

    returns = (sell_prices.values - buy_prices.values) / buy_prices.values
    cumulative_return = (1 + returns).prod() - 1 if len(returns) > 0 else 0
    annualized_return = ((1 + cumulative_return) ** (365 / (ticker_data.index[-1] - ticker_data.index[0]).days) - 1) * 100 if len(returns) > 0 else 0
    volatility = ticker_data['Close'].pct_change().std() * (252 ** 0.5) * 100 if len(returns) > 0 else 0
    max_drawdown = (1 - (ticker_data['Close'].cummax() / ticker_data['Close'])).max() * 100 if len(returns) > 0 else 0

    results.append({
        'Ticker': ticker,
        'Cumulative Return': cumulative_return * 100,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown
    })

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], mode='lines', name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['RSI'], mode='lines', name='RSI'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Buy'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Sell'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)
    
    fig.update_layout(title=f'Trading Signals for {ticker}', xaxis_title='Date', yaxis_title='Price')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

for ticker in tickers:
    print(f"Generated HTML for {ticker}: {ticker}_plot.html")
print("Generated trading results: trading_results.html")