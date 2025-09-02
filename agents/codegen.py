from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import os
import datetime
import re

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key, model="gpt-4o-mini")

def extract_code_blocks(response_text: str) -> str:
    """Extract <code>...</code> block, discard <reasoning>."""
    match = re.search(r"<code>(.*?)</code>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

def generate_code(intent_dict: dict) -> dict:
    """
    Generate code based on the parsed intent dictionary.
    """
    print(intent_dict)

    ticker = intent_dict.get("ticker")
    strategy = intent_dict.get("strategy_description", "")
    buy_condition = intent_dict.get("buy_condition", "")
    sell_condition = intent_dict.get("sell_condition", "")
    date_range = intent_dict.get("date_range", "")
    duration_type = intent_dict.get("duration_type", "")
    duration_days = int(intent_dict.get("duration_days", 0))
    # Convert the JSON to a string to pass into the prompt
    
    prompt = f"""
You are a professional trading code generation agent.

First, think step by step about:
- Which indicators and rules are explicitly required by {strategy}, {buy_condition}, {sell_condition}, {duration_type}
- Whether stop-loss, consecutive days, or duration are explicitly mentioned (if not, do NOT add them)
- Which technical indicators must be computed.
- Data when fetched from SQLite, must be parsed and sorted by Date.
- How to ensure correct indexing, no lookahead bias, vectorization, etc.

Wrap your reasoning inside <reasoning> ... </reasoning>.
Then output ONLY executable Python code inside <code> ... </code>.

<reasoning>
Explain (to yourself) how to transform the inputs into working Python backtest code.
This reasoning is hidden from the user and never executed.
</reasoning>

<code>
# Python code goes here
</code>
##############################

GLOBAL ENFORCEMENT (READ FIRST)

##############################
⚠️ VERY IMPORTANT: Implement only the logic that is explicitly required by the given {strategy}, {buy_condition}, {sell_condition}, and any explicit {duration_type} or stop-loss clauses in the user's query.

If the user's query does NOT mention stop-loss, consecutive-day duration, or any other extra constraint, DO NOT add that logic in the produced code.

Do NOT invent stop-loss, consecutive-day counters, or any other “helpful” features unless they are explicitly present in the provided strategy/buy/sell/duration inputs.

The code must be minimal and strictly focused on the requested strategy.

##############################

GENERAL SETUP

##############################

Use only pandas and ta (assume pre-installed).

Do NOT fetch data from yfinance or other APIs.

Use sqlite3 and pandas.read_sql() to load data from market_data.db, table: stock_data.

Expected columns: 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date'

Parse Date to datetime and sort ascending by Date.

Multi-Ticker Logic:

Filter and process each ticker individually using a for-loop over unique tickers.

If there’s only one ticker, still loop over the single unique ticker.

Save one plot per ticker as {ticker}_plot.html.

Append per-ticker summary metrics to a list and save them all to trading_results.html.

##############################

STRATEGY INPUTS

##############################
5. Strategy is passed as:
Strategy: {strategy}
Buy Condition: {buy_condition}
Sell Condition: {sell_condition}
Duration: {duration_type} # (if present; otherwise empty/None)

##############################

INDICATOR COMPUTATION

##############################
6. Compute only indicators explicitly mentioned by the strategy/buy/sell inputs.

Moving averages (e.g., SMA, EMA) must be computed with .shift(1) to avoid lookahead bias.

RSI MUST be computed directly from Close without any shift.

MACD and other indicators should be computed only if requested.

Turning points/sign changes (if requested), detect using prior/cur values, e.g.:
(series.shift(1) < 0) & (series > 0).

Golden/Death Cross (only if explicitly referenced):

short_ma = df['Close'].rolling(window=short_window).mean().shift(1)
long_ma  = df['Close'].rolling(window=long_window).mean().shift(1)
golden_cross = (short_ma.shift(1) <= long_ma.shift(1)) & (short_ma > long_ma)
death_cross  = (short_ma.shift(1) >= long_ma.shift(1)) & (short_ma < long_ma)

Use these boolean Series for buy/sell when asked.

CLARIFY INDICATOR DEFINITIONS:

"50-day high" → df['High'].rolling(50).max().shift(1)

"50-day SMA" → df['Close'].rolling(50).mean().shift(1)

Do NOT confuse highs with moving averages.

##############################

STOP-LOSS (ONLY IF REQUESTED)

##############################

Implement stop-loss ONLY IF the user's query explicitly mentions it (e.g., "stop-loss 5%").

If present:

Store executed buy price in last_buy_price on entry.

Trigger stop-loss if df['Close'].iloc[i] < last_buy_price * (1 - stop_loss_pct).

Reset last_buy_price = None after closing the position.

If not present in the query, do not create stop-loss variables or checks.

##############################

DURATION / CONSECUTIVE CONDITIONS (ONLY IF REQUESTED)

##############################
10. Implement consecutive-day counters only if the query asks for them (e.g., "for 3 consecutive days").
- Use a counter pattern inside the loop:
- Increment when condition holds, reset to 0 when it breaks.
- Fire the signal only when the counter >= required days.

##############################

SIGNAL LOGIC

##############################
11. Use vectorized pandas boolean expressions (&, |) wrapped in parentheses when possible. Do not use Python and/or on pandas Series.

State handling:

SEQUENTIAL TRADING LOGIC (CRITICAL)

- Maintain a boolean `in_position` for each ticker.
- Loop sequentially over ticker_data.index:
    for i in range(len(ticker_data)):
        if not in_position and buy_condition_met(i):
            ticker_data['Buy'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = True
        elif in_position and sell_condition_met(i):
            ticker_data['Sell'].iloc[i] = ticker_data['Close'].iloc[i]
            in_position = False
- Do NOT create Buy/Sell series by multiplying booleans by Close.
- Ensure signals strictly alternate buy → sell → buy.
- This preserves sequential trading and ensures portfolio_value calculations work.

- Store signals in new columns: ticker_data['Buy'], ticker_data['Sell'] containing prices at signal bars and NaN elsewhere.

- When generating buy/sell signals, do NOT output booleans. Instead, output the actual stock price at which the buy/sell occurs. 
    For example, if a buy signal occurs when RSI < 35, set:
    Buy = (RSI < 35) * Close
    Sell = (RSI > 65) * Close

- When constructing daily portfolio_value, check buy and sell signals independently for each date:
    - Do NOT use `elif` between buy and sell.
    - For each date d:
        if d in buy_prices.index: execute buy
        if d in sell_prices.index: execute sell
    - This ensures same-day buy and sell are both processed if strategy allows.

- Only When handling Bollinger Bands:

    - Buy if in_position == False and Close <= lower_band (and any other conditions like RSI < 30)
    - Sell if in_position == True and Close >= upper_band
##############################

INDEXING + ALIGNMENT RULES (STRICT)

##############################
14. Immediately after parsing Date, set it as the index for each filtered ticker:
ticker_data.set_index('Date', inplace=True)
Ensure the index is a DatetimeIndex.

Do not manually reassign indices for buy_prices/sell_prices. The .dropna() extracts preserve the DatetimeIndex.

Do not use index slicing like series.index[...] unless it is a boolean mask or integer positions. Do not reindex trades. Do not reset index on these extracted Series.

Date math (for days differences) must only be done when index is a DatetimeIndex.

##############################

Bollinger Bands:

If the strategy/buy/sell conditions reference Bollinger Bands, compute them using ta.volatility.BollingerBands on Close:

- lower_band = BollingerBands(df['Close']).bollinger_lband()
- upper_band = BollingerBands(df['Close']).bollinger_hband()
- middle_band = BollingerBands(df['Close']).bollinger_mavg()  # if needed

Do NOT shift bands unless explicitly requested.  

Use these series to construct buy/sell signals when the query references "touching lower/upper band".

#############################

TRADES HANDLING

##############################
18. Build trades = ticker_data[['Buy', 'Sell']], then:
- trades = trades.dropna(how='all')
- buy_prices = trades['Buy'].dropna()
- sell_prices = trades['Sell'].dropna()
- Truncate to matched pairs:
min_len = min(len(buy_prices), len(sell_prices)) 
buy_prices = buy_prices.iloc[:min_len] 
sell_prices = sell_prices.iloc[:min_len]

##############################

RETURNS & METRICS (PERCENTAGES)

##############################
19. Trade-level returns (completed pairs only):
returns = pd.Series((sell_prices.values - buy_prices.values) / buy_prices.values)

Construct **daily portfolio_value** using all-in/all-out logic over every date in ticker_data.index:

Start with initial_capital = 100000.

Maintain cash_balance and fractional shares:
shares = cash_balance / Close[d] on Buy signal.

- When buying shares:
    - Restrict to integer shares: shares = int(cash_balance / Close)
    - Adjust cash_balance accordingly
- Make sure portfolio_value is computed consistently with chosen share method


Update cash_balance and portfolio_value for each date.

Forward-fill portfolio_value if needed for continuity.

Define normalized equity curve:
cumulative_curve = portfolio_value / float(portfolio_value.iloc[0])

Metrics (all in percentages):

Cumulative Return = (cumulative_curve.iloc[-1] - 1) * 100

Annualized Return:

total_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
If total_days > 0:
((cumulative_curve.iloc[-1]) ** (365 / total_days) - 1) * 100
Else: 0

Volatility:

daily_rets = portfolio_value.pct_change().dropna()
volatility = daily_rets.std() * sqrt(252) * 100
If no trades executed, set = 0

Max Drawdown:

max_drawdown = (1 - cumulative_curve / cumulative_curve.cummax()).max() * 100
If no trades executed i.e. len(cumulative_curve)=0, then set = 0

Edge cases:

If there are no completed trades (empty returns or no buy/sell pairs), set:

Cumulative Return = 0
Annualized Return = 0
Volatility = 0
Max Drawdown = 0

Print: No trades executed for {ticker}

Initialize all metric variables before conditionals to avoid NameError.

##############################

PLOTTING (PLOTLY)

##############################
25. Create a 2-row subplot via:
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{{"secondary_y": True}}], [{{}}]])

Row 1 (Price & Indicators & Markers):

Plot Close price as a blue line on the primary y-axis (secondary_y=False).

Plot indicators referenced by the strategy:

SMAs/EMAs: computed with .shift(1), plot as dotted lines with distinct colors.

RSI, MACD, and similar oscillator/secondary indicators on the secondary y-axis (secondary_y=True).

Buy markers: green triangle-up markers at ticker_data['Buy'] values.

Sell markers: red triangle-down markers at ticker_data['Sell'] values.

Plot markers only at valid (non-NaN) signal points; Plotly will ignore NaNs automatically.

Row 2 (Equity Curve):

Plot portfolio_value as a continuous purple line.

Layout & Axes:

Ensure ticker_data.index is a DatetimeIndex and is used as x for all traces.

fig.update_layout(xaxis=dict(type='date'))

fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)

fig.update_yaxes(title_text="Indicator", row=1, col=1, secondary_y=True)

Keep Price and Indicator y-axes ranges independent.

Give proper titles, legends. Plot axis labels appropriately.

Save Final Combined Plot:

fig.write_html(f"{ticker}_plot.html")

##############################

FINAL OUTPUT

##############################
30. Append per-ticker metrics into a list of dicts with keys:
Ticker, Cumulative Return, Annualized Return, Volatility, Max Drawdown
(All metric values in percentages.)

Convert to DataFrame and save:
DataFrame.to_html('trading_results.html', index=False)

Also print the names of generated HTML files for each ticker at the end.
##############################

DEBUGGING + STABILITY

##############################
32. Print a brief confirmation of SQL read (e.g., shape or head).
33. Wrap boolean expressions in parentheses to avoid ambiguity.
34. Do NOT use .between() inside loops.
35. For cumulative calculations (returns/drawdown), operate on pandas Series (not NumPy arrays).
36. NEVER call .iloc[] on NumPy arrays — only on Series/DataFrames.
37. Use warnings.filterwarnings("ignore") to suppress ta warnings.
38. Print a small table of executed trades (buy/sell dates and prices) per ticker for inspection.
39. Keep all per-ticker logic self-contained in the loop.
40. All extracted price series must inherit the datetime index from ticker_data — do not reindex manually.

"""

          
    print(prompt)
    messages = [
        SystemMessage(content="Follow all enforcement rules strictly. Output format must include <reasoning>...</reasoning> and <code>...</code>. Do not include ```python fences."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    logs = []

    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response.content, re.DOTALL)
    reasoning_text = reasoning_match.group(1).strip() if reasoning_match else "No reasoning captured."

    logs.append({
        "step": "code_generator",
        "cot": reasoning_text
    })

    final_code = extract_code_blocks(response.content)
    print("Extracted code: ", final_code)

    return {
        "code": final_code,
        "logs": logs
    }
