# main.py

import os
import json
import sqlite3
import pandas as pd
import requests
import urllib.parse
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import re
import ast
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from celery.result import AsyncResult

# Import your OrchestratorAgent instead of LangGraph pipeline
from agents.orchestrator import OrchestratorAgent
from agents.interpreter import interpret_query_mcp
# from agents.codegen import generate_code --- IGNORE ---
from agents.codegen import codegen_mcp

# Celery app + task
from tasks.executor import app as celery_app
from tasks.executor import run_python_code  # Celery task

# -----------------------------
# Config
# -----------------------------
PLOTS_DIR = os.path.abspath(".")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Types & Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    structured_query: Optional[Dict[str, Any]] = None
    cot: Optional[str] = None
    original_cot: Optional[str] = None
    original_query: Optional[str] = None

# -----------------------------
# Initialize orchestrator
# -----------------------------
orchestrator_agent = OrchestratorAgent()

# -----------------------------
# FastAPI app
# -----------------------------
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated HTML files under /plots/*
fastapi_app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# Interpreter endpoint (same as before)
@fastapi_app.post("/api/interpret-query")
async def interpret_query_endpoint(req: QueryRequest):
    thoughts, structured_query = interpret_query_mcp({"query": req.query})
    return {
        "thoughts": thoughts,
        "structured_query": structured_query
    }

# New submit-query endpoint using OrchestratorAgent
@fastapi_app.post("/api/submit-query")
async def submit_query(req: QueryRequest):
    """
    Handles:
    1. Normal query interpretation (user_query only)
    2. Confirmed structured query (skip interpreter or re-run if CoT changed)
    3. Rejected interpretation (stop execution and ask for rephrase)
    """
    try:
        # Handle rejection first
        if req.query == "REJECTED":
            result = orchestrator_agent.run("REJECTED")
            return result

        # Confirmed structured query
        if req.structured_query:
            result = orchestrator_agent.execute_confirmed_query(
                structured_query=req.structured_query,
                cot=req.cot,
                original_cot=req.original_cot,
                query=req.query,
                original_query=req.original_query,
            )
            print("DEBUG result from code generation:", result)
            return result

        # Normal query flow
        result = orchestrator_agent.run(req.query)
        return result

    except Exception as e:
        logging.exception("Error in submit-query")
        return {"status": "ERROR", "error": str(e)}
    
# Task status endpoint (unchanged)
@fastapi_app.get("/api/task-status/{task_id}")
async def task_status(task_id: str):
    async_result = AsyncResult(task_id, app=celery_app)
    state = async_result.state
    meta = async_result.info or {}
    if isinstance(meta, dict):
        logs = meta.get("logs", [])
    else:
        logs = []

    if state == "SUCCESS":
        res = async_result.result
        logging.info("DEBUG result from Celery: %r", res)

        files = []
        output = ""

        if isinstance(res, dict):
            files = res.get("files") or res.get("html_files") or []
            output = res.get("output", "")
            logs = res.get("logs", logs)
        else:
            output = str(res)

        # Fallback parsing for generated files in output string
        if not files and output:
            m = re.search(r"Generated files:\s*(\[.*\])", output)
            if m:
                try:
                    candidate = m.group(1)
                    parsed = ast.literal_eval(candidate)
                    if isinstance(parsed, (list, tuple)):
                        files = list(parsed)
                except Exception:
                    logging.exception("Failed to parse Generated files from output")

        return {
            "status": "SUCCESS",
            "files": files,
            "logs": logs,
            "output": output,
        }

    elif state == "FAILURE":
        err = async_result.result
        return {
            "status": "FAILURE",
            "error": str(err),
            "logs": logs,
        }

    # For ongoing states like PENDING or PROGRESS
    return {
        "status": state,
        "logs": logs,
    }

# List all html files for plots
@fastapi_app.get("/api/list-html")
def list_html_files():
    try:
        files = [f for f in os.listdir(PLOTS_DIR) if f.endswith(".html")]
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}

# Serve a single html file
@fastapi_app.get("/api/html/{file_name}")
def get_html(file_name: str):
    file_path = os.path.join(PLOTS_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    return {"error": "File not found"}

# Run standalone (dev)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, reload=True)

# Expose ASGI app
app = fastapi_app
