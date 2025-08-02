from langgraph.graph import StateGraph, END
from langchain.schema import BaseOutputParser
from agents.interpreter import interpret_query
from agents.codegen import generate_code
from typing import TypedDict

class GraphState(TypedDict):
    input: str
    intent: str
    code: str

# State will carry messages across the flow
def node_interpreter(state):
    user_query = state["input"]
    result = interpret_query(user_query)
    return {"intent": result["intent"]}

def node_codegen(state):
    intent = state["intent"]
    result = generate_code(intent)
    return {"code": result["code"]}

# Define the LangGraph
builder = StateGraph(GraphState)
builder.add_node("interpreter", node_interpreter)
builder.add_node("codegen", node_codegen)

# Define flow: interpreter → codegen → END
builder.set_entry_point("interpreter")
builder.add_edge("interpreter", "codegen")
builder.add_edge("codegen", END)

app = builder.compile()

# Run it!
if __name__ == "__main__":
    user_input = input("🔍 Enter your stock strategy query:\n> ")
    final_state = app.invoke({"input": user_input})
    print("\n🧠 Structured Intent:\n", final_state["intent"])
    print("\n🧾 Generated Code:\n", final_state["code"])