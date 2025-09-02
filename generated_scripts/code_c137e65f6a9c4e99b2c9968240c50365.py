import pandas as pd
import sqlite3
from ta.trend import SMAIndicator
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
from math import sqrt

warnings.filterwarnings("ignore")

conn = sqlite3.connect('market_data.db')
df = pd.read_sql('SELECT * FROM stock_data', conn)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

tickers = df['Ticker'].unique()
results = []

for ticker in tickers:
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data.set_index('Date', inplace=True)

    ticker_data['SMA_20'] = SMAIndicator(ticker_data['Close'], window=20).sma_indicator().shift(1)

    ticker_data['Buy'] = ((ticker_data['Close'] < ticker_data['SMA_20'] * 0.95) & (ticker_data['Close'].shift(1) >= ticker_data['SMA_20'] * 0.95)).astype(float) * ticker_data['Close']
    ticker_data['Sell'] = ((ticker_data['Close'] > ticker_data['SMA_20'] * 1.05) & (ticker_data['Close'].shift(1) <= ticker_data['SMA_20'] * 1.05)).astype(float) * ticker_data['Close']

    in_position = False
    for i in range(len(ticker_data)):
        if not in_position and not pd.isna(ticker_data['Buy'].iloc[i]):
            ticker_data['Buy'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = True
        elif in_position and not pd.isna(ticker_data['Sell'].iloc[i]):
            ticker_data['Sell'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = False

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
    portfolio_value = []

    for date in ticker_data.index:
        if date in buy_prices.index:
            shares = int(cash_balance / ticker_data['Close'].loc[date])
            cash_balance -= shares * ticker_data['Close'].loc[date]
        if date in sell_prices.index:
            cash_balance += shares * ticker_data['Close'].loc[date]
            shares = 0
        portfolio_value.append(cash_balance + shares * ticker_data['Close'].loc[date])

    portfolio_value = pd.Series(portfolio_value, index=ticker_data.index)

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
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['SMA_20'], mode='lines', name='20-day SMA', line=dict(dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=buy_prices.index, y=buy_prices.values, mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_prices.index, y=sell_prices.values, mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfolio Value', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', yaxis2_title='Indicator')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

for ticker in tickers:
    print(f"Generated plot for {ticker}_plot.html")
print("Generated trading_results.html")