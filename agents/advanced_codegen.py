from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
from datetime import datetime
from .prompts import CODEGEN_PROMPT_BASE
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature=0.2,
    openai_api_key=api_key,
    model="gpt-4o-mini"
)

def get_code_tasks(translated):
    for key, value in translated.items():
        #print(key, ":", value)
        if key == "code_tasks":
            return value if isinstance(value, list) else []
        
    return []

def advanced_codegen_mcp(payload: dict) -> dict:
    curr_time_stamp = datetime.now().isoformat()

    # locate translator output (same logic as before)
    translated = {}
    if isinstance(payload.get("translated_query"), dict) and payload.get("translated_query"):
        translated = payload["translated_query"]
    elif isinstance(payload.get("instructions"), dict) and payload.get("instructions"):
        translated = payload["instructions"]
    elif isinstance(payload.get("translated"), dict) and payload.get("translated"):
        translated = payload["translated"]
    else:
        translated = payload.get("translated_query") or payload.get("instructions") or payload.get("structured_query") or {}

    structured_query = payload.get("structured_query", {}) or {}
    trade_management = translated.get("trade_management", {})
    reentry_rule = trade_management.get("reentry_rule", "")

    # --------------- FIX: pull code_tasks from translated first (fallback to payload)
    code_tasks = translated.get("code_tasks") or payload.get("code_tasks") or []
    if not isinstance(code_tasks, list):
        code_tasks = []

    # Detect additional_buy_condition in either translated or structured_query
    additional_buy = translated.get("additional_buy_condition") or structured_query.get("additional_buy_condition") or translated.get("additional_buy") or structured_query.get("additional_buy")

    # Build advanced prompt lines (base + explicit additional-buy template)
    prompt_lines = [CODEGEN_PROMPT_BASE]
    prompt_lines.append("\n# -------- ADVANCED STRATEGY LOGIC --------")
    prompt_lines.append(
        "In addition to coding instructions above, follow below rules for advanced query handling:\n"
        "- ALWAYS derive buy/sell logic from `buy_spec` and `sell_spec` if available. Prioritize these over freeform strategy text.\n"
        "- If translator JSON includes `additional_buy_condition` (exact key), `averaging`, or `pyramiding` rules, you MUST implement them exactly as described (percentage-based or condition-based). Do NOT invent defaults.\n"
        "- If `trade_management.reentry_rule` exists, implement re-entry/reset logic accordingly (e.g., wait until crossover resets before entering again).\n"
        "- If `stop_loss` or `take_profit` rules are specified, enforce them strictly within trade loop.\n"
        "- Translator `code_tasks` are mandatory ordered steps. Always follow them in order when generating code.\n"
        "- If multiple strategies are present, implement each in its own function.\n"
        "- Ensure code is modular, commented, and executable without manual fixes."
        "- Always plot additional buy signals (e.g., averaging/pyramiding) as separate markers on the strategy chart.\n"
        "- Use a distinct color (e.g., light green) and label them 'Add Buy'."
        " - Ensure they appear alongside initial buy and sell markers, without overwriting any existing markers."

    )

    prompt_lines.append(
"""
- Start with an initial capital of 100,000 units (base capital).
- When executing a **normal buy**, invest using available capital.
- Whenever an **additional buy condition** (averaging or pyramiding rule) triggers:
  - Inject an **additional 100,000 units of capital** into the portfolio (treat this as new cash inflow).
  - Execute an additional buy immediately using that new capital.
  - Record the new capital injection as a negative cash flow (buy event) for XIRR computation.
- Maintain a record of all trade cash flows:
  - Buy (including additional buys) → Negative cash flow (outflow)
  - Sell → Positive cash flow (inflow)
- At the end of the strategy, compute total performance using **XIRR** (Extended Internal Rate of Return).
  - Use `numpy_financial.xirr(cash_flows, dates)` or equivalent.
  - Report both final portfolio value and XIRR (%).
- Do NOT limit to one action per day; allow multiple additional buys if funds exist.
- Always update portfolio value immediately after every trade.
- Ensure all buys and sells update the trade log and portfolio equity series.
- Plot markers distinctly:
  - Buy → Green upward triangle
  - Additional Buy → Light green diamond
  - Sell → Red downward triangle
- Do NOT hardcode conditions; use those from the translator or structured_query.
- Follow all translator `code_tasks` exactly in order.
- Implement reentry/reset logic according to `trade_management.reentry_rule`.

- When plotting:
  - Show price on the primary y-axis.
  - Plot EMA indicators (e.g., EMA 55, EMA 233) on a secondary y-axis.
  - Use `make_subplots(specs=[[{"secondary_y": True}]])`.
  - Mark all buy/additional-buy/sell events aligned with their date indices.
"""
)


    # Add structured_query JSON for reference (so LLM sees the actual conditions)
    prompt_lines.append("\n# Structured query for reference:\n")
    prompt_lines.append(json.dumps(structured_query, indent=2))

    # Include translator code_tasks (if any)
    if code_tasks:
        prompt_lines.append("\n# Translator code tasks / steps:\n")
        for t in code_tasks:
            prompt_lines.append(f"- {t}")

    # --------------- NEW: If additional_buy_condition present, append explicit instructions + example snippet
    if additional_buy:
        # Ensure code_tasks enforces additional-buy placement after normal buy and before sell
        add_steps = [
            "Precompute `additional_buy_signal` (pandas Series) from additional_buy_condition; handle crossovers with .shift(1) when operator is crosses_above/crosses_below.",
            "Evaluate additional_buy_signal inside the loop only when position > 0 and not waiting_for_reset (and not executed_action).",
            "If additional_buy_signal is True, buy as many shares as possible using available cash, update trades list (use label 'AddBuy' or 'Buy'), update cash and shares, and update portfolio_series before sell checks.",
            "Do not allow additional buys if waiting_for_reset == True.",
            "Ensure additional_buy indicators are included in indicators_to_plot and plotted in strategy figure."
        ]
        for s in add_steps:
            if s not in code_tasks:
                code_tasks.append(s)

        # Provide an explicit code example — copy-paste ready template for additional buying
        prompt_lines.append("\n# ADDITIONAL BUY HANDLING - TEMPLATE:\n")
        prompt_lines.append(            """
# Example snippet for reference:
if df['additional_buy_signal'].iloc[i]:
    capital += 100000  # inject new capital
    extra_shares = 100000 // current_price
    if extra_shares > 0:
        shares += extra_shares
        capital -= extra_shares * current_price
        trades.append(('AddBuy', df.index[i], current_price, -100000))
"""

)

    # Build final prompt body
    final_prompt = "\n".join(prompt_lines)

    # Build messages for LLM
    try:
        messages = [
            SystemMessage(content="You are the Code Generator agent in a multi-agent trading system. You must strictly follow the provided translator_instructions."),
            HumanMessage(content=f"""
TRANSLATOR_INSTRUCTIONS (JSON):
{json.dumps(translated, indent=2)}

REENTRY_RULE (from translator):
{reentry_rule}

CODE_TASKS (ordered steps):
{json.dumps(code_tasks, indent=2)}

---

FOLLOW THESE INSTRUCTIONS AND RULES:
{final_prompt}
IMPORTANT: First explain your reasoning under ---THOUGHTS---.
Then write the executable Python code under ---CODE---.
""")
        ]

        print("input to advanced_code_gen (trimmed):", {"has_additional_buy": bool(additional_buy), "code_tasks_len": len(code_tasks)})
        response = llm.invoke(messages)
        raw = response.content

        if "---CODE---" in raw:
            thoughts_part, code_part = raw.split("---CODE---", 1)
            thoughts_part = thoughts_part.replace("---THOUGHTS---", "").strip()
        else:
            thoughts_part = ""
            code_part = raw.strip()

        thought_summary = "\n".join(thoughts_part.splitlines()[:2]).strip() or "Code generation complete."

        return {
            "thought": thought_summary,
            "full_thought": thoughts_part.strip(),
            "action": "CodeReady",
            "action_input": code_part.strip()
        }

    except Exception as e:
        return {
            "thought": f"Code generation failed due to error: {str(e)}",
            "action": "AskClarification",
            "action_input": "An error occurred during code generation. Please check your input and try again."
        }
