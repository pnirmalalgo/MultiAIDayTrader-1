import pandas as pd
import sqlite3
from ta.trend import SMAIndicator
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings

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

    ticker_data['10_MA'] = ticker_data['Close'].rolling(window=10).mean().shift(1)
    ticker_data['30_MA'] = ticker_data['Close'].rolling(window=30).mean().shift(1)

    ticker_data['Buy'] = None
    ticker_data['Sell'] = None
    in_position = False

    for i in range(len(ticker_data)):
        if not in_position and ticker_data['10_MA'].iloc[i] > ticker_data['30_MA'].iloc[i]:
            ticker_data['Buy'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = True
        elif in_position and ticker_data['10_MA'].iloc[i] < ticker_data['30_MA'].iloc[i]:
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
    volatility = daily_rets.std() * (252 ** 0.5) * 100 if len(daily_rets) > 0 else 0
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
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['10_MA'], mode='lines', name='10-day MA', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['30_MA'], mode='lines', name='30-day MA', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Buy'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Sell'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfolio Value', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', yaxis2_title='Indicator')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

for ticker in tickers:
    print(f"Generated plot for {ticker}: {ticker}_plot.html")
print("Trading results saved to trading_results.html")