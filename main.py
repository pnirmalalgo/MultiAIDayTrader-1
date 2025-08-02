from agents.interpreter import interpret_query
from agents.ticker_lookup import resolve_ticker
from agents.codegen import generate_code
from agents.code_cleaner import clean_code
from tasks.executor import run_python_code  # Celery task

from typing import TypedDict
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    input: str
    intent: str
    code: str
    clean_code: str
    execution_result: str

def node_interpreter(state):
    user_input = state["input"]
    result = interpret_query(user_input)  # This should return {"intent": "..."}
    return result

def node_ticker_lookup(state):
    return resolve_ticker(state["intent"])

def node_codegen(state):
    return generate_code(state["intent"])

def node_cleaner(state):
    return clean_code(state["code"])

def node_executor(state):
    result = run_python_code.delay(state["clean_code"])
    return {"execution_result": f"Task submitted: {result.id}"}

builder = StateGraph(GraphState)
builder.add_node("interpreter", node_interpreter)
builder.add_node("ticker_lookup", node_ticker_lookup)
builder.add_node("codegen", node_codegen)
builder.add_node("code_cleaner", node_cleaner)
builder.add_node("executor", node_executor)

builder.set_entry_point("interpreter")
builder.add_edge("interpreter", "ticker_lookup")
builder.add_edge("ticker_lookup", "codegen")
builder.add_edge("codegen", "code_cleaner")
builder.add_edge("code_cleaner", "executor")
builder.add_edge("executor", END)

app = builder.compile()

if __name__ == "__main__":
    query = input("📈 Your query:\n> ")
    final = app.invoke({"input": query})
    print("\n📊 Task ID / Result:\n", final["execution_result"])