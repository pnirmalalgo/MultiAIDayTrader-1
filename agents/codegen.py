from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import os
import datetime

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key, model="gpt-4o-mini")

def generate_code(intent_json: str, stock_data: pd.DataFrame) -> dict:
    print(intent_json)
    parsed = json.loads(intent_json)
    ticker = parsed.get("ticker")

    strategy = parsed.get("strategy_description", "")
    buy_condition = parsed.get("buy_condition", "")
    sell_condition = parsed.get("sell_condition", "")
    date_range = parsed.get("date_range", "")
    duration_type = parsed.get("duration_type", "")
    duration_days = int(parsed.get("duration_days", 0))


    format_example = """
    Here’s the required format:

    ---THOUGHTS---
    First, I will load the stock data from SQLite using pandas.read_sql().
    Next, I will compute the 14-day RSI using ta.momentum.RSIIndicator.
    The buy condition is RSI < 30, which suggests the stock is oversold.
    The sell condition is RSI > 70, which signals overbought.

    I'll loop through each ticker individually, generate buy/sell signals on crossovers, and track positions.
    Finally, I’ll plot the results and save them to HTML.

    ---CODE---
    # actual code starts here...
    """

    # This is the only instruction prompt we are sending now — minimalist
    prompt = f"""
Given the following intent, first explain your reasoning step by step under ---THOUGHTS---, 
and then write executable Python code under ---CODE---.

{format_example}

Use only pandas and ta. Do not fetch data from the internet. Use SQLite3 and read data using pandas.read_sql().
Data is stored in a SQLite database named market_data.db and table: stock_data.

---INTENT---
{intent_json}

---THOUGHTS---

---COT Instructions---
Think step by step:

1. Understand the problem: What is the goal of the task?

2. Break down the requirements: What inputs/outputs are expected? Are there any constraints?

3. Design a plan: How should we approach the problem? What functions or data structures will we use?

4. Write pseudocode: Sketch a high-level version of the logic.

5. Translate to actual code: Implement the logic in [desired language].

6. Test the code: Include sample input/output to verify correctness.
------------------------------

Instructions:
Write Python code to implement the following stock trading strategy.

Ensure the output is ONLY executable Python code — no comments, no markdown, no pip install, and NO code fences like ```python.

##############################

GLOBAL ENFORCEMENT (READ FIRST)

##############################
⚠️ VERY IMPORTANT: Implement only the logic that is explicitly required by the given {strategy}, {buy_condition}, {sell_condition}, and any explicit {duration_type} or stop-loss clauses in the user's query.

If the user's query does NOT mention stop-loss, consecutive-day duration, or any other extra constraint, DO NOT add that logic in the produced code.

Do NOT invent stop-loss, consecutive-day counters, or any other “helpful” features unless they are explicitly present in the provided strategy/buy/sell/duration inputs.

The code must be minimal and strictly focused on the requested strategy. Understand the strategy and buy & sell conditions very carefully. 

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

##############################
Important: When generating buy/sell signals based on indicator comparisons (e.g., SMA(10) vs SMA(30), RSI vs threshold, MACD vs Signal), do not generate signals continuously while the condition is true.
Instead, generate a signal only at the crossover/moment of change, i.e.:

A buy signal occurs only when the indicator condition changes from false to true (crossing up).

A sell signal occurs only when the indicator condition changes from true to false (crossing down).

This ensures signals are triggered once at the event, not continuously.
####################

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

Maintain in_position boolean.

Buy only if in_position == False and the buy condition is met.

Sell only if in_position == True and the sell condition is met.

Signals must alternate strictly: buy → sell → buy. Ignore sells with no open position.

Mark signals at the exact bar/time they trigger (use the current index position).

Keep signals sparse:

Do NOT forward-fill buy/sell columns.

Store signals in new columns: ticker_data['Buy'], ticker_data['Sell'] containing prices at signal bars and NaN elsewhere.

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

- After truncating buy_prices and sell_prices to matched pairs, also remove the corresponding entries in ticker_data['Buy'] and ticker_data['Sell'] so that plotted Buy/Sell markers only correspond to completed trades.
- Use vectorized assignment: replace unmatched buy/sell entries with np.nan.
- Example logic:
   ticker_data['Buy'] = np.where(ticker_data.index.isin(buy_prices.index), ticker_data['Buy'], np.nan)
   ticker_data['Sell'] = np.where(ticker_data.index.isin(sell_prices.index), ticker_data['Sell'], np.nan)
- This prevents any buy signal appearing after the last completed sell.
- Ensure there are no unmatched Buy or Sell markers at the edges of the plot.


##############################

RETURNS & METRICS (PERCENTAGES)

##############################
19. Trade-level returns (completed pairs only):
returns = pd.Series((sell_prices.values - buy_prices.values) / buy_prices.values)

Construct **daily portfolio_value** (absolute ₹ or $ values) using all-in/all-out logic.

Start with initial_capital = 100000.

Maintain cash_balance and integer shares:
- On Buy: shares = int(cash_balance / Close[d]); adjust cash_balance
- On Sell: liquidate all shares; update cash_balance
- Each day: portfolio_value[d] = cash_balance + shares * Close[d]

NEW: For all metric calculations (Cumulative Return, Annualized Return, Volatility, Max Drawdown), use portfolio_series / cumulative_curve derived from it. 
- ALWAYS use the pandas Series portfolio_series (or cumulative_curve) for ALL calculations of returns, volatility, and drawdowns.
- Do NOT use any Python list (like portfolio_value_list) for .pct_change(), .std(), .cummax(), or any metric calculation.
- If the Series is empty, set all metrics to 0. Never calculate metrics on lists.
- If portfolio_series is empty (no trades executed), set all metrics = 0. 
- Always validate that total_days > 0 before computing annualized return. 
- Wrap cumulative_curve / cumulative_curve.cummax() and pct_change() calls with checks for non-empty Series.

After loop, build:
portfolio_series = pd.Series(portfolio_value_list, index=ticker_data.index)

⚠️ Always normalize before metrics:
cumulative_curve = portfolio_series / float(portfolio_series.iloc[0])

Metrics must always use this normalized `cumulative_curve`:
- Cumulative Return = (cumulative_curve.iloc[-1] - 1) * 100
- Annualized Return = ((cumulative_curve.iloc[-1]) ** (365 / total_days) - 1) * 100
- Volatility = cumulative_curve.pct_change().dropna().std() * sqrt(252) * 100
- Max Drawdown = (1 - cumulative_curve / cumulative_curve.cummax()).max() * 100

When generating trading strategy scripts, always:

Convert the portfolio value list into a Pandas Series with the date index.

Use this Series (portfolio_series) for computing daily returns, volatility, cumulative return, annualized return, and max drawdown.

Do not use raw Python lists for .pct_change() or other Pandas methods.”

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

- If metrics is a single dictionary, convert scalars to lists.
- If metrics is a list of dictionaries, pass it directly to pd.DataFrame.
- Use dynamic code like:

if isinstance(metrics, dict):
    trading_results = pd.DataFrame({{k: [v] if not isinstance(v, list) else v for k, v in metrics.items()}})
else:
    trading_results = pd.DataFrame(metrics)

##############################
PLOTTING (PLOTLY)

##############################
25. Create a 2-row subplot via:
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    specs=[[{{"secondary_y": True}}], [{{}}]])

Row 1 (Price & Indicators & Markers):

- Plot Close price as a blue line on the primary y-axis (secondary_y=False).
- Plot trend indicators (SMA, EMA, Bollinger Bands, etc.) on the primary y-axis with dotted lines.
- Plot oscillators (RSI, MACD, Stochastic, etc.) **always on the secondary y-axis (secondary_y=True)**.
- 🔹 For RSI, explicitly set the y-axis range to [0, 100]. 
- 🔹 Never plot oscillators on the primary y-axis with price.

- Plot Buy markers: green triangle-up markers at ticker_data['Buy'] values.
- Plot Sell markers: red triangle-down markers at ticker_data['Sell'] values.
- Plot markers only at valid (non-NaN) signal points.

Row 2 (Equity Curve):

- Plot portfolio_value as a continuous purple line.

Layout & Axes:

- Ensure ticker_data.index is a DatetimeIndex and is used as x for all traces.
- fig.update_layout(xaxis=dict(type='date'))
- fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
- 🔹 fig.update_yaxes(title_text="Oscillator / Indicator", row=1, col=1, 
                      secondary_y=True, range=[0, 100] if RSI is plotted)

- Keep Price and Indicator y-axes ranges independent.
- Provide proper titles, legends, and axis labels.

Save Final Combined Plot:

fig.write_html(f"{ticker}_plot.html")

##############################

FINAL OUTPUT

##############################
30. Append per-ticker metrics into a list of dicts with keys:
Ticker, Cumulative Return, Annualized Return, Volatility, Max Drawdown
(All metric values in percentages.)

After finishing the ticker loop:
- Convert all_metrics list into DataFrame
- Save once: trading_results.to_html("trading_results.html", index=False)
- Append to outputs: output_files.append("trading_results.html")
⚠️ Do NOT save or append trading_results.html inside the per-ticker loop.
Save it once after processing all tickers, then append to output_files.


When you generate code that saves any output files (e.g., plots, reports, HTML):

1. Collect all filenames into a list called `output_files`.
   - Example: output_files = []

2. Each time you save a file, append its name:
   - output_files.append("filename.html")

3. At the very end of the script, ALWAYS add:

   try:
       context["execution"]["files"] = output_files
   except NameError:
       pass  # context not defined in standalone runs

   print("Generated files:", output_files)

This ensures:
- Filenames are captured in the LangGraph execution context.
- They can be returned to the orchestrator/React frontend.
- The script still runs safely when executed outside LangGraph.

This must be done **every time** output files are generated.

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

    

    messages = [
        SystemMessage(content="You're a trading algorithm assistant. Reason step by step, then write working code. Do not use markdown or code fences."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    # Split CoT and code
    raw = response.content
    if "---CODE---" in raw:
        thoughts_part, code_part = raw.split("---CODE---", 1)
        thoughts_part = thoughts_part.replace("---THOUGHTS---", "").strip()
    else:
        thoughts_part = ""
        code_part = raw.strip()

    # Debug print (optional)
    print("Generated thoughts:\n", thoughts_part)
    print("Generated code:\n", code_part)

    return {
        "code": code_part,
        "thoughts": [line.strip() for line in thoughts_part.splitlines() if line.strip()]
    }