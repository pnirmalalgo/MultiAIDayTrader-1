PROMPT = """9. #### ADDITIONAL BUY RULES #####(only if additional buy is mentioned)
- Additional Buy is only executed if position > 0, cash > 0, additional_buy_condition is True, and waiting_for_reset == False.
- Do NOT merge normal buy and additional buy conditions into one if statement.
- Always update cash, shares, and portfolio_series immediately after executing an additional buy.
- Include shares_bought in the trade tuple:
    Normal Buy tuple: ("Buy", current_date, current_price, shares_bought)
    Additional Buy tuple: ("Additional Buy", current_date, current_price, shares_bought)
- Check additional buy **after normal buy logic** but **before sell logic** in each iteration.

For additional buys below entry price, implement sequential buys: 
- Track number of additional buys per trade. 
- First additional buy triggers only after first threshold is hit; second triggers only after second threshold. 
- Prevent multiple additional buys at the same threshold on the same trade.

####Pseudo-code when additional_buy_condition is mentioned:#####(only if additional buy is mentioned)
additional_buy_done = False

for i in range(len(df)):
    executed_action = None

    # Normal Buy
    if position == 0 and not waiting_for_reset and normal_buy_condition:
        execute_normal_buy()
        position = 1
        additional_buy_done = False
        executed_action = "Buy"

    # Additional Buy (safe guarded against NoneType)
    if position == 1 and entry_price is not None:
        additional_buy_cond = (
            df['additional_buy_signal'].iloc[i]
            if pd.notna(df['additional_buy_signal'].iloc[i]) else False
        )
    else:
        additional_buy_cond = False

    if position > 0 and cash > 0 and not waiting_for_reset and additional_buy_cond and not additional_buy_done:
        execute_additional_buy()
        additional_buy_done = True
        executed_action = "Additional Buy"
        # Immediately update portfolio value
        portfolio_series[i] = cash + shares * current_price

    # Sell
    if position == 1 and entry_price is not None and sell_condition:
        execute_sell()
        position = 0
        waiting_for_reset = True
        additional_buy_done = False
        executed_action = "Sell"
        portfolio_series[i] = cash + shares * current_price

    # If no trade executed this bar, still update portfolio value
    if executed_action is None:
        portfolio_series[i] = cash + shares * current_price



##### Additional Buy Handling#### (Only if additional buy is mentioned in input)
##### ADDITIONAL BUY RULES ##### (Only if additional buy is mentioned in input)
- Only execute Additional Buy if:
    1. position > 0
    2. cash > 0
    3. additional_buy_condition is True
    4. waiting_for_reset == False
- Additional Buy must be **separate from normal buy**; do NOT combine conditions.
- Only allow **one additional buy per bar** (avoid repeated buys on consecutive bars unless a new trade opens).
- Immediately update:
    cash = cash - shares_bought * current_price
    shares += shares_bought
    portfolio_series[i] = cash + shares * current_price
- Append to trades list as ("Additional Buy", date, price, shares_bought)
- Reset any per-trade flag after sell to allow future additional buys in new trades.
- Use a flag `additional_buy_done = False` per open position:
    - Set to True after executing additional buy
    - Reset to False after the position is closed

- Inside the backtest loop, check if translator_instructions contains "cond_additional_buy".
- Execute the additional buy **only if position > 0 and cash > 0**.
- Buy as many shares as possible using available cash at current_price.
- Append a trade tuple: ("Additional Buy", current_date, current_price, shares_bought) to trades list.
- Immediately update cash = cash - shares_bought * current_price and shares += shares_bought.
- Immediately update portfolio_series[i] = cash + shares * current_price.
- Ensure waiting_for_reset rules are respected: do not execute additional buy if waiting_for_reset == True.
- Additional buys are separate from normal buy; normal buy only occurs if position == 0.

- Implement sequential additional buys based on multiple thresholds (e.g., -10%, -20%) without overlapping.
- Track number of additional buys per open trade (e.g., additional_buy_count).
- Execute first additional buy only if additional_buy_count == 0 and threshold met.
- Execute second additional buy only if additional_buy_count == 1 and next threshold met.
- Prevent multiple buys at same threshold on the same trade.
- Always update cash, shares, portfolio_series, and trades immediately after each additional buy.
- Ensure this is dynamic: thresholds and number of additional buys come from translator instructions.

##### Additional Notes for Buy Logic ####(Only if additional buy is mentioned in input)
- Normal buy and additional buy are handled in separate conditional blocks.
- Normal buy executes only if position == 0 and not waiting_for_reset and buy condition is met.
- Additional buy executes only if position > 0, cash > 0, additional buy condition is met, and waiting_for_reset == False.
- Do not combine normal and additional buy in a single if condition.
- In each iteration of the backtest loop:
    1. Check for normal buy if position == 0
    2. Check for additional buy if position > 0
    3. Check for sell conditions
    4. Update portfolio_series after each trade (buy, additional buy, or sell)
- Include "shares_bought" in the trade tuple for additional buys: ("Additional Buy", current_date, current_price, shares_bought)
- Normal buy tuple: ("Buy", current_date, current_price, shares_bought)
- Always update portfolio_series[i] immediately after executing any trade.

- If a condition in translator_instructions contains "relative_to": "entry_price", generate Python code that evaluates the threshold relative to the current entry_price of the open position.
  - This calculation must occur inside the guard: if position == 1 and entry_price is not None.
  - Do not precompute or vectorize this threshold outside the backtest loop.
  - Use entry_price to calculate stop-loss, take-profit, or additional buy(only if additional buy is mentioned) thresholds as needed.
  - Example for an additional buy at -10%:  
        threshold_price = entry_price * (1 - 0.10)
        if current_price <= threshold_price:
            # execute additional buy
  - Repeat this logic for all percentage-based buy/additional buy(only if additional buy is mentioned)/sell conditions referencing "relative_to": "entry_price".

  ##### PORTFOLIO UPDATE SEQUENCE #####
        - For each iteration of the backtest loop:
            1. Check Normal Buy (position == 0)
                - If executed, immediately update cash, shares, portfolio_series[i]
            2. Check Additional Buy (position > 0)(only if additional buy is mentioned)
                - If executed, immediately update cash, shares, portfolio_series[i]
            3. Check Sell (position == 1)
                - If executed, immediately update cash, shares, portfolio_series[i]
        - Only compute portfolio_series **once per trade**, never at the start of the loop.

"""