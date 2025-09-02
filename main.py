# main.py

import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from agents.interpreter import interpret_query
from agents.codegen import generate_code
from agents.code_cleaner import clean_code
from agents.ticker_lookup import resolve_ticker
from orchestrator.mcp_orchestrator import MCPOrchestrator
from wrappers.executor_wrapper import executor_wrapper

# -----------------------------
# Configuration
# -----------------------------
HTML_DIR = "."
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# MCP Orchestrator Instance
# -----------------------------
orchestrator = MCPOrchestrator(
    interpreter=interpret_query,
    code_generator=generate_code,
    ticker_lookup=resolve_ticker,
    code_cleaner=clean_code,
    executor=executor_wrapper,
)

# -----------------------------
# Request Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str

# -----------------------------
# Streaming Logs Endpoint
# -----------------------------
@fastapi_app.post("/api/submit-query")
async def submit_query(req: Request):
    payload = await req.json()
    query = payload.get("query", "")
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    try:
        # Run the orchestrator synchronously and get final context
        context = orchestrator.run(query)

        # Build response for frontend
        response = {
            "status": "SUCCESS",
            "logs": context.get("logs", []),
            "result": context.get("execution", {})  # final execution result
        }

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"status": "FAILURE", "error": str(e)}, status_code=500)
# -----------------------------
# HTML Files Endpoints
# -----------------------------
@fastapi_app.get("/api/list-html")
def list_html_files():
    try:
        files = [f for f in os.listdir(HTML_DIR) if f.endswith(".html")]
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}

@fastapi_app.get("/api/html/{file_name}")
def get_html(file_name: str):
    file_path = os.path.join(HTML_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    return {"error": "File not found"}


# -----------------------------
# Run standalone
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, reload=True)

app = fastapi_app
