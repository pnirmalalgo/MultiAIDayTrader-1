import pandas as pd
import sqlite3
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

conn = sqlite3.connect('market_data.db')
query = "SELECT * FROM stock_data"
data = pd.read_sql(query, conn)

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date', inplace=True)

results = []
tickers = data['Ticker'].unique()

for ticker in tickers:
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data.set_index('Date', inplace=True)

    ticker_data['RSI'] = ta.momentum.RSIIndicator(ticker_data['Close']).rsi()
    ticker_data['20_day_SMA'] = ticker_data['Close'].rolling(window=20).mean().shift(1)

    ticker_data['Buy'] = None
    ticker_data['Sell'] = None
    in_position = False

    for i in range(len(ticker_data)):
        if not in_position and (ticker_data['RSI'].iloc[i] < 30) and (ticker_data['Close'].iloc[i] > ticker_data['20_day_SMA'].iloc[i]):
            ticker_data['Buy'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = True
        elif in_position and ((ticker_data['RSI'].iloc[i] > 70) or (ticker_data['Close'].iloc[i] < ticker_data['20_day_SMA'].iloc[i])):
            ticker_data['Sell'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = False

    trades = ticker_data[['Buy', 'Sell']].dropna(how='all')
    buy_prices = trades['Buy'].dropna()
    sell_prices = trades['Sell'].dropna()

    min_len = min(len(buy_prices), len(sell_prices))
    buy_prices = buy_prices.iloc[:min_len]
    sell_prices = sell_prices.iloc[:min_len]

    returns = (sell_prices.values - buy_prices.values) / buy_prices.values
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = ((1 + cumulative_return) ** (365 / (ticker_data.index[-1] - ticker_data.index[0]).days) - 1) * 100 if (ticker_data.index[-1] - ticker_data.index[0]).days > 0 else 0
    volatility = ticker_data['Close'].pct_change().std() * (252 ** 0.5) * 100
    max_drawdown = (1 - (ticker_data['Close'] / ticker_data['Close'].cummax())).max() * 100

    results.append({
        'Ticker': ticker,
        'Cumulative Return': cumulative_return * 100,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown
    })

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], mode='lines', name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['20_day_SMA'], mode='lines', name='20-day SMA', line=dict(dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['RSI'], mode='lines', name='RSI'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Buy'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Sell'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=(1 + returns).cumprod(), mode='lines', name='Equity Curve'), row=2, col=1)

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', yaxis2_title='RSI')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

for ticker in tickers:
    print(f"Generated plot for {ticker}: {ticker}_plot.html")
print("Generated trading results: trading_results.html")