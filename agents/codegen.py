from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key)

def generate_code(intent_json: str) -> dict:
    parsed = json.loads(intent_json)
    ticker = parsed.get("ticker")
    strategy = parsed.get("strategy_description", "")
    buy_condition = parsed.get("buy_condition", "")
    sell_condition = parsed.get("sell_condition", "")
    date_range = parsed.get("date_range", "")

    
    prompt = (f"Write code for following: All conditions included(DO NOT include any comments. Only executable python code. Do not include'''python. Do not include pip install statements.): 1. Use the `yfinance` and `ta` libraries. But do not install. No pip install statements." \
        f"2. Download stock data for '{ticker}' from {date_range} using `yfinance`.Use: from ta.momentum import RSIIndicator"\
        f"3. Calculate the Relative Strength Index (RSI) using the `ta` library on the 'Close' price. Use data['Close'].iloc[:, 0]" \
        f"4. Generate buy signals when the RSI crosses below {buy_condition} and sell signals when the RSI crosses above {sell_condition}."\
        f"5. Backtest a simple trading strategy based on these signals: buy when a buy signal occurs (if not already in a position) and sell when a sell signal occurs (if in a position)."\
        f"6. Calculate and Print Annualized returns(Use: annualized_returns = ((data['Cumulative_Returns'].iloc[-1])**(252/len(data))) - 1), maximum drawdown, volatility of the backtested strategy."\
        f"7. No need to calculate Daily_Returns. Use .iloc[:, 0] wherever series is expected as input for eg. data['Cumulative_Returns'] = data['Close'].iloc[-1].pct_change() * data['Position'].shift(1)"\
        "7. Plot the 'Close' price of the stock along with the buy and sell signals on the same chart using `matplotlib`."
        )
    print(prompt)
    messages = [
        SystemMessage(content="Write simple python code. User yfinance, ta. Initialize ta class. Please note that you will need to use: data['Close'].iloc[:, 0] and iterate through data."\
                      "Only show data. Initialize ta class properly. Use: from ta.momentum import RSIIndicator. Please note: DO NOT use ta.add_all_ta_features(This is not required and gives error). We only need close data as of now. "\
                      "DO NOT include any comments. Only executable python code."),
        HumanMessage(content=prompt)
    ]
    '''
    prompt = "Write code for Test RELIANCE: buy RSI under 35, sell RSI over 65, last 6 months using closing data." \
             "Do not plot graph. Only show data. Initialize ta class properly. Use data['Close'].iloc[:, 0] and iterate through data."\
             "Please note: DO NOT use ta.add_all_ta_features(This is not required and gives error). We only need close data as of now. "\
             
                        
    messages = [
        SystemMessage(content="Write simple python code. User yfinance, ta. Initialize ta class. Please note that you will need to use: data['Close'].iloc[:, 0] and iterate through data."\
                      "Do not plot graph. Only show data. Initialize ta class properly. Please note: DO NOT use ta.add_all_ta_features(This is not required and gives error). We only need close data as of now. "\
                      "DO NOT include any comments. Only executable python code."),
        HumanMessage(content=prompt)
    ]'''

    response = llm.invoke(messages)
    print("Generated code: ",response.content)
    return {"code": response.content}