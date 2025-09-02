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

    ticker_data['MA10'] = ticker_data['Close'].rolling(window=10).mean().shift(1)
    ticker_data['MA30'] = ticker_data['Close'].rolling(window=30).mean().shift(1)

    ticker_data['Buy_Signal'] = (ticker_data['MA10'] > ticker_data['MA30']) & (ticker_data['MA10'].shift(1) <= ticker_data['MA30'].shift(1))
    ticker_data['Sell_Signal'] = (ticker_data['MA10'] < ticker_data['MA30']) & (ticker_data['MA10'].shift(1) >= ticker_data['MA30'].shift(1))

    in_position = False
    cash_balance = 100000
    shares = 0
    portfolio_value = []

    for i in range(len(ticker_data)):
        if not in_position and ticker_data['Buy_Signal'].iloc[i]:
            shares = int(cash_balance / ticker_data['Close'].iloc[i])
            cash_balance -= shares * ticker_data['Close'].iloc[i]
            in_position = True
        elif in_position and ticker_data['Sell_Signal'].iloc[i]:
            cash_balance += shares * ticker_data['Close'].iloc[i]
            shares = 0
            in_position = False
        
        portfolio_value.append(cash_balance + shares * ticker_data['Close'].iloc[i])

    ticker_data['Portfolio_Value'] = portfolio_value

    cumulative_return = (ticker_data['Portfolio_Value'].iloc[-1] - 100000) / 100000 * 100
    annualized_return = ((ticker_data['Portfolio_Value'].iloc[-1] / 100000) ** (365 / (ticker_data.index[-1] - ticker_data.index[0]).days) - 1) * 100 if (ticker_data.index[-1] - ticker_data.index[0]).days > 0 else 0
    daily_rets = ticker_data['Portfolio_Value'].pct_change().dropna()
    volatility = daily_rets.std() * (252 ** 0.5) * 100
    max_drawdown = (1 - ticker_data['Portfolio_Value'] / ticker_data['Portfolio_Value'].cummax()).max() * 100

    results.append({
        'Ticker': ticker,
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown
    })

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], mode='lines', name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['MA10'], mode='lines', name='10-day MA', line=dict(dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['MA30'], mode='lines', name='30-day MA', line=dict(dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index[ticker_data['Buy_Signal']], y=ticker_data['Close'][ticker_data['Buy_Signal']], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index[ticker_data['Sell_Signal']], y=ticker_data['Close'][ticker_data['Sell_Signal']], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Portfolio_Value'], mode='lines', name='Portfolio Value', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(title=f'Trading Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', yaxis2_title='Portfolio Value')
    fig.write_html(f"{ticker}_plot.html")

results_df = pd.DataFrame(results)
results_df.to_html('trading_results.html', index=False)

for ticker in tickers:
    print(f"Generated plot for {ticker}: {ticker}_plot.html")
print("Trading results saved to trading_results.html")