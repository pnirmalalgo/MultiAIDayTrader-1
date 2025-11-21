from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
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

def codegen_mcp(payload: dict) -> dict:
    curr_time_stamp = datetime.now().isoformat()


    #translated = payload.get("translated_query", {})
    #structured_query = payload.get("structured_query", {})
    # find translator output in common places
    translated = {}
    if isinstance(payload.get("instructions"), dict) and payload["instructions"]:
        print("Using translator_instructions")
        translated = payload["instructions"]
    elif isinstance(payload.get("translated_query"), dict) and payload["translated_query"]:
        print("Using translated_query")
        translated = payload["translated_query"]
    elif isinstance(payload.get("translated"), dict) and payload["translated"]:
        print("Using translated")
        translated = payload["translated"]
    else:
        # last resort
        translated = payload.get("structured_query", {}) or {}

    structured_query = payload.get("structured_query", {}) or {}
    trade_management = translated.get("trade_management", {})
    reentry_rule = trade_management.get("reentry_rule", "")


    #print("Input to CodeGen MCP:", payload)
    if not translated and not structured_query:
        return {
            "thought": "Neither translated_query nor structured_query provided.",
            "action": "AskClarification",
            "action_input": "Please provide a valid input with at least structured_query."
        }

    # Prefer translator output
    strategy = translated.get("strategy_plan") or structured_query.get("strategy_description", "")
    #buy_condition = translated.get("buy_condition_code") or structured_query.get("buy_condition", "")
    buy_condition = translated.get("buy_condition", {}).get("conditions", []) or translated.get("buy_spec", {}).get("conditions", [])
    sell_condition = translated.get("sell_condition", {}).get("conditions", []) or translated.get("sell_spec", {}).get("conditions", [])
    #code_tasks = translated.get("code_tasks", [])
    # Try multiple locations in order, stop at the first non-empty list

    code_tasks = get_code_tasks(payload)

    duration_type = structured_query.get("duration_type", "")
    duration_days = int(structured_query.get("duration_days", 0))
    remarks = translated.get("remarks") or structured_query.get("remarks", "")

    translator_notes = translated.get("codegen_instructions", "")
    #code_tasks = input.get("code_tasks", [])

    CODEGEN_PROMPT = CODEGEN_PROMPT_BASE.replace("{{ticker}}", ", ".join(structured_query.get("ticker", ""))) \
                                    .replace("{{start_date}}", structured_query.get("start_date", "")) \
                                    .replace("{{end_date}}", structured_query.get("end_date", ""))


    try:
        messages = [
    SystemMessage(content="You are the Code Generator agent in a multi-agent trading system. You must strictly follow the provided translator_instructions."),
    HumanMessage(content=f"""
        TRANSLATOR_INSTRUCTIONS (JSON):
        {translated}

        REENTRY_RULE (from translator):
        {reentry_rule}

        CODE_TASKS (ordered steps):
        {code_tasks}

        ---

        FOLLOW THESE INSTRUCTIONS AND RULES:
        {CODEGEN_PROMPT}
        IMPORTANT: First explain your reasoning under ---THOUGHTS---. 
        Then write the executable Python code under ---CODE---.
        """)
    ]
        
        print("input to codegen: ", messages)

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
