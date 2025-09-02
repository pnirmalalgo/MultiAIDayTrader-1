import sqlite3
import pandas as pd
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
from math import sqrt

warnings.filterwarnings("ignore")

conn = sqlite3.connect('market_data.db')
query = "SELECT Open, High, Low, Close, Volume, Ticker, Date FROM stock_data"
data = pd.read_sql(query, conn)

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by=['Ticker', 'Date'], inplace=True)

results = []

for ticker in data['Ticker'].unique():
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data.set_index('Date', inplace=True)

    ticker_data['RSI'] = ta.momentum.RSIIndicator(ticker_data['Close']).rsi()

    ticker_data['Buy'] = (ticker_data['RSI'] < 35) * ticker_data['Close']
    ticker_data['Sell'] = (ticker_data['RSI'] > 65) * ticker_data['Close']

    trades = ticker_data[['Buy', 'Sell']].dropna(how='all')
    buy_prices = trades['Buy'].dropna()
    sell_prices = trades['Sell'].dropna()

    min_len = min(len(buy_prices), len(sell_prices))
    buy_prices = buy_prices.iloc[:min_len]
    sell_prices = sell_prices.iloc[:min_len]

    returns = (sell_prices.values - buy_prices.values) / buy_prices.values

    initial_capital = 100000
    cash_balance = initial_capital
    shares = 0
    portfolio_value = pd.Series(index=ticker_data.index, data=0.0)

    for date in ticker_data.index:
        if date in buy_prices.index:
            shares = int(cash_balance / ticker_data['Close'].loc[date])
            cash_balance -= shares * ticker_data['Close'].loc[date]
        if date in sell_prices.index:
            cash_balance += shares * ticker_data['Close'].loc[date]
            shares = 0
        portfolio_value.loc[date] = cash_balance + shares * ticker_data['Close'].loc[date]

    cumulative_curve = portfolio_value / float(portfolio_value.iloc[0])
    cumulative_return = (cumulative_curve.iloc[-1] - 1) * 100
    total_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    annualized_return = ((cumulative_curve.iloc[-1]) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0
    daily_rets = portfolio_value.pct_change().dropna()
    volatility = daily_rets.std() * sqrt(252) * 100 if len(daily_rets) > 0 else 0
    max_drawdown = (1 - cumulative_curve / cumulative_curve.cummax()).max() * 100 if len(cumulative_curve) > 0 else 0

    results.append({
        'Ticker': ticker,
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown
    })

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], mode='lines', name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['RSI'], mode='lines', name='RSI'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=buy_prices.index, y=buy_prices.values, mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_prices.index, y=sell_prices.values, mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfolio Value'), row=2, col=1)

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', yaxis2_title='RSI')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

for ticker in data['Ticker'].unique():
    print(f"Generated file: {ticker}_plot.html")
print("Generated file: trading_results.html")