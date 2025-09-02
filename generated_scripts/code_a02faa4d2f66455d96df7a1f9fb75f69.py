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
data.sort_values(by='Date', inplace=True)

results = []

for ticker in data['Ticker'].unique():
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data.set_index('Date', inplace=True)

    ticker_data['SMA20'] = ticker_data['Close'].rolling(window=20).mean().shift(1)
    ticker_data['SMA50'] = ticker_data['Close'].rolling(window=50).mean().shift(1)

    ticker_data['Buy'] = ((ticker_data['SMA20'].shift(1) <= ticker_data['SMA50'].shift(1)) & 
                          (ticker_data['SMA20'] > ticker_data['SMA50'])) * ticker_data['Close']
    ticker_data['Sell'] = ((ticker_data['SMA20'].shift(1) >= ticker_data['SMA50'].shift(1)) & 
                           (ticker_data['SMA20'] < ticker_data['SMA50'])) * ticker_data['Close']

    in_position = False
    cash_balance = 100000
    shares = 0
    portfolio_value = []

    for i in range(len(ticker_data)):
        if not in_position and not pd.isna(ticker_data['Buy'].iloc[i]):
            shares = int(cash_balance / ticker_data['Close'].iloc[i])
            cash_balance -= shares * ticker_data['Close'].iloc[i]
            in_position = True
        elif in_position and not pd.isna(ticker_data['Sell'].iloc[i]):
            cash_balance += shares * ticker_data['Close'].iloc[i]
            shares = 0
            in_position = False
        
        portfolio_value.append(cash_balance + shares * ticker_data['Close'].iloc[i])

    portfolio_value = pd.Series(portfolio_value, index=ticker_data.index)

    if shares == 0:
        cumulative_return = 0
        annualized_return = 0
        volatility = 0
        max_drawdown = 0
    else:
        cumulative_curve = portfolio_value / portfolio_value.iloc[0]
        cumulative_return = (cumulative_curve.iloc[-1] - 1) * 100
        total_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
        annualized_return = ((cumulative_curve.iloc[-1]) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0
        daily_rets = portfolio_value.pct_change().dropna()
        volatility = daily_rets.std() * sqrt(252) * 100
        max_drawdown = (1 - cumulative_curve / cumulative_curve.cummax()).max() * 100

    results.append({
        'Ticker': ticker,
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown
    })

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], mode='lines', name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['SMA20'], mode='lines', name='20-day SMA', line=dict(dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['SMA50'], mode='lines', name='50-day SMA', line=dict(dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Buy'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Sell'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfolio Value', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', yaxis2_title='Indicator')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

for ticker in data['Ticker'].unique():
    print(f"Generated plot for {ticker}_plot.html")
print("Generated trading_results.html")