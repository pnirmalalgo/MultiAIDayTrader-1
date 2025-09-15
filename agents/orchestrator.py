from agents.interpreter import interpret_query_mcp
from agents.codegen import codegen_mcp
from tasks.executor import run_python_code
from agents.ticker_lookup import resolve_ticker
from tasks.market_data import get_fmp_stock_data  # Ensure this is accessible
from agents.translator import translator_mcp

class OrchestratorAgent:
    def __init__(self):
        self.thoughts = []
        self.logs = []

    def run(self, user_input_or_query):
        self.thoughts = []  # Reset thoughts per run
        self.log_plan("Let's understand the query and decide next steps.")

        if user_input_or_query == "REJECTED":
            return {
                "status": "CLARIFY",
                "question": "Okay, could you rephrase or specify corrections?",
                "structured_query": {},
                "thoughts": self.thoughts
            }

        # Step 1: Interpret user input
        self.log_command("call:interpreter")
        result = interpret_query_mcp({"query": user_input_or_query})

        # Step 2: Collect interpreter thoughts
        thoughts = result.get("thought") or result.get("thoughts", "")
        self.log_message("interpreter", thoughts)

        # Step 3: Action and input
        action = result.get("action", "")
        action_input = result.get("action_input", {})

        # ✅ If clarification is needed
        if action == "AskClarification":
            question = action_input.get("question", "Could you clarify?")
            structured_query = action_input.get("structured_query", {})
            return {
                "status": "CLARIFY",
                "question": question,
                "structured_query": structured_query,
                "thoughts": self.thoughts
            }

        # ✅ If structured query is ready, confirm with user
        elif action == "StructuredQueryReady":
            structured_query = action_input or {}

            if not structured_query or "ticker" not in structured_query:
                return self.ask_user("Could you clarify which stock/ticker you're referring to?")

            return {
                "status": "CLARIFY",
                "question": "Is this interpretation correct?",
                "structured_query": structured_query,
                "thoughts": self.thoughts
            }

        # ❌ Unknown action fallback
        return {
            "status": "ERROR",
            "error": f"Unknown action: {action}",
            "thoughts": self.thoughts
        }

    def execute_confirmed_query(self, structured_query: dict, cot: str = None):
        self.thoughts = []
        self.log_plan("Executing confirmed structured query.")

        # Step 1: Resolve tickers
        try:
            self.log_command("call:ticker_lookup")
            raw_tickers = structured_query.get("ticker", [])
            resolved_tickers = resolve_ticker(raw_tickers)
            structured_query["ticker"] = resolved_tickers
            self.log_message("ticker_lookup", f"Resolved tickers: {resolved_tickers}")
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Ticker resolution failed: {str(e)}",
                "thoughts": self.thoughts
            }

        # Step 2: Fetch stock data
        try:
            self.log_command("call:fetch_stock_data")
            start = structured_query["start_date"]
            end = structured_query["end_date"]
            df = get_fmp_stock_data(resolved_tickers, start, end)
            self.log_message("data_fetcher", f"Fetched {len(df)} rows of data for {resolved_tickers}")
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Data fetch failed: {str(e)}",
                "thoughts": self.thoughts
            }

        # ✅ Step 3: Translator — enrich structured query into actionable instructions
        try:
            self.log_command("call:translator")
            if not cot:
                cot = ""
            structured_query["remarks"] = structured_query.get("remarks", "")
            structured_query["remarks"] = (structured_query["remarks"] + " " + cot).strip()
            trans_result = translator_mcp({"structured_query": structured_query})    

            enriched_query = trans_result.get("structured_query", structured_query)
            translator_instructions = trans_result
            code_tasks = trans_result.get("code_tasks", [])

            self.log_message("translator", f"Translator output keys: {list(trans_result.keys())}")

            # Log translator reasoning
            translator_thoughts = trans_result.get("thought", "")
            self.log_message("translator", translator_thoughts)

        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Translator error: {str(e)}",
                "thoughts": self.thoughts
            }

        # Step 4: CodeGen
        try:
            self.log_command("call:codegen")
            code_result = codegen_mcp({
                "structured_query": structured_query,
                "translated_query": enriched_query,   # ✅ use enriched translator output
                "instructions": translator_instructions,  # ✅ forward instructions if needed
                "code_tasks": code_tasks
            })

            if code_result.get("action") != "CodeReady":
                return {
                    "status": "ERROR",
                    "error": f"Codegen returned unexpected action: {code_result.get('action')}",
                    "thoughts": self.thoughts
                }

            code = code_result["action_input"]

            # Log codegen thoughts
            codegen_thoughts = code_result.get("full_thought", code_result.get("thought", ""))
            for line in codegen_thoughts.strip().splitlines():
                self.log_message("codegen", line.strip())

        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Codegen error: {str(e)}",
                "thoughts": self.thoughts
            }
        
        # Step 5: Submit to Celery
        try:
            self.log_command("call:executor")
            task = run_python_code.delay(code)
            self.log_message("executor", f"Submitted task with ID: {task.id}")
            return {
                "status": "PENDING",
                "task_id": task.id,
                "thoughts": self.thoughts
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Task submission failed: {str(e)}",
                "thoughts": self.thoughts
            }

    # ---- Logging helpers ----
    def log_plan(self, content):
        self.thoughts.append({"role": "orchestrator", "type": "plan", "content": content})

    def log_command(self, content):
        self.thoughts.append({"role": "orchestrator", "type": "command", "content": content})

    def log_message(self, role, content):
        self.thoughts.append({"role": role, "type": "message", "content": content})

    def ask_user(self, question):
        self.thoughts.append({
            "role": "orchestrator",
            "type": "message",
            "content": f"[CLARIFICATION NEEDED] {question}"
        })
        return {
            "status": "CLARIFY",
            "question": question,
            "thoughts": self.thoughts
        }
