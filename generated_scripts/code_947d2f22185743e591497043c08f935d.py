import pandas as pd
import sqlite3
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
from math import sqrt

warnings.filterwarnings("ignore")

conn = sqlite3.connect('market_data.db')
data = pd.read_sql('SELECT * FROM stock_data', conn)

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

tickers = data['Ticker'].unique()
results = []

for ticker in tickers:
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data.set_index('Date', inplace=True)

    ticker_data['MACD'] = ta.trend.macd(ticker_data['Close'])
    ticker_data['Signal_Line'] = ta.trend.macd_signal(ticker_data['Close'])
    
    ticker_data['Buy'] = None
    ticker_data['Sell'] = None
    in_position = False

    for i in range(1, len(ticker_data)):
        if not in_position and (ticker_data['MACD'].iloc[i-1] < ticker_data['Signal_Line'].iloc[i-1]) and (ticker_data['MACD'].iloc[i] > ticker_data['Signal_Line'].iloc[i]):
            ticker_data['Buy'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = True
        elif in_position and (ticker_data['MACD'].iloc[i-1] > ticker_data['Signal_Line'].iloc[i-1]) and (ticker_data['MACD'].iloc[i] < ticker_data['Signal_Line'].iloc[i]):
            ticker_data['Sell'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = False

    trades = ticker_data[['Buy', 'Sell']].dropna(how='all')
    buy_prices = trades['Buy'].dropna()
    sell_prices = trades['Sell'].dropna()

    min_len = min(len(buy_prices), len(sell_prices))
    buy_prices = buy_prices.iloc[:min_len]
    sell_prices = sell_prices.iloc[:min_len]

    returns = (sell_prices.values - buy_prices.values) / buy_prices.values
    portfolio_value = pd.Series(index=ticker_data.index, data=0.0)
    initial_capital = 100000
    cash_balance = initial_capital
    shares = 0

    for date in ticker_data.index:
        if date in buy_prices.index:
            shares = int(cash_balance / ticker_data['Close'].loc[date])
            cash_balance -= shares * ticker_data['Close'].loc[date]
        if date in sell_prices.index:
            cash_balance += shares * ticker_data['Close'].loc[date]
            shares = 0
        portfolio_value.loc[date] = cash_balance + (shares * ticker_data['Close'].loc[date])

    cumulative_curve = portfolio_value / initial_capital
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
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['MACD'], mode='lines', name='MACD', line=dict(color='orange')), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='red')), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Buy'], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Sell'], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfolio Value', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', yaxis2_title='Indicator')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

for ticker in tickers:
    print(f"Generated plot for {ticker}_plot.html")
print("Saved trading results to trading_results.html")