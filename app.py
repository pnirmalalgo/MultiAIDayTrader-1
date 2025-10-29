# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os
import json

from agents.orchestrator import OrchestratorAgent
from celery.result import AsyncResult
from tasks.executor import app as celery_app

# -------------------- FastAPI Backend --------------------
app = FastAPI(title="DayTrader AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for plots
PLOTS_DIR = Path("plots").absolute()
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")

# Initialize orchestrator
orchestrator = OrchestratorAgent()

# -------------------- Utility Functions --------------------
def get_filtered_tickers():
    try:
        if os.path.exists("filtered_tickers.json"):
            with open("filtered_tickers.json", "r") as f:
                tickers = json.load(f)
            return [t.strip() for t in tickers if t.strip()]
        return []
    except Exception as e:
        print("DEBUG: Error loading filtered tickers:", e)
        return []

# -------------------- FastAPI Endpoints --------------------

@app.post("/api/submit-query")
def submit_query(payload: dict):
    """
    Handle query submission - either initial interpretation or confirmed execution
    """
    query = payload.get("query", "")
    structured_query = payload.get("structured_query")
    cot = payload.get("cot")
    original_cot = payload.get("original_cot")
    original_query = payload.get("original_query")
    
    try:
        # If structured_query exists, user confirmed - execute it
        if structured_query:
            result = orchestrator.execute_confirmed_query(
                structured_query=structured_query,
                cot=cot,
                original_cot=original_cot,
                query=query,
                original_query=original_query
            )
        else:
            # Initial interpretation
            result = orchestrator.run(query)
        
        return result
        
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "thoughts": orchestrator.thoughts
        }

@app.get("/api/task-status/{task_id}")
def task_status(task_id: str):
    """
    Check Celery task status and return results
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if result.ready():
            if result.successful():
                # Get task metadata
                meta = result.info if hasattr(result, 'info') else {}
                
                return {
                    "status": "SUCCESS",
                    "files": meta.get("files", []),
                    "file": meta.get("file"),
                    "output": meta.get("output", ""),
                    "logs": meta.get("logs", [])
                }
            else:
                return {
                    "status": "FAILURE",
                    "error": str(result.result),
                    "output": str(result.info) if hasattr(result, 'info') else ""
                }
        else:
            # Task still running - check for progress updates
            meta = result.info if hasattr(result, 'info') else {}
            return {
                "status": "PENDING",
                "output": "\n".join(meta.get("logs", [])) if isinstance(meta, dict) else ""
            }
            
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e)
        }

@app.get("/api/list-html")
def list_html():
    """
    List all HTML files in the plots directory
    """
    try:
        html_files = [
            f.name for f in PLOTS_DIR.glob("*.html")
            if f.is_file() and not f.name.startswith(".")
        ]
        return {"files": sorted(html_files)}
    except Exception as e:
        return {"files": [], "error": str(e)}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# -------------------- Run Server --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)