from celery import Celery, current_task
import subprocess
import uuid
import os
import logging
import time

logging.basicConfig(level=logging.INFO)

app = Celery(
    "executor",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

SCRIPT_DIR = "generated_scripts"
os.makedirs(SCRIPT_DIR, exist_ok=True)

@app.task(bind=True)
def run_python_code(self, code: str):
    filename = os.path.join(SCRIPT_DIR, f"code_{uuid.uuid4().hex}.py")
    logging.info(f"Saved code to {filename} (length={len(code)})")
    
    with open(filename, "w") as f:
        f.write(code)

    logs = []
    try:
        # Example: push logs incrementally
        logs.append("Starting execution...")
        print("DEBUG: pushing logs ->", logs)
        self.update_state(state="PROGRESS", meta={"logs": logs})
        print("DEBUG: logs pushed")
        
        output = subprocess.check_output(
            ["python", filename],
            stderr=subprocess.STDOUT,
            timeout=30
        )
        
        logs.append("Execution finished successfully.")
        self.update_state(state="SUCCESS", meta={"logs": logs})
        return {"output": output.decode(), "file": filename, "logs": logs}

    except subprocess.CalledProcessError as e:
        logs.append(f"Error during execution: {e.output.decode()}")
        self.update_state(state="FAILURE", meta={"logs": logs})
        return {"output": e.output.decode(), "file": filename, "logs": logs}

    except subprocess.TimeoutExpired:
        logs.append("Code execution timed out.")
        self.update_state(state="FAILURE", meta={"logs": logs})
        return {"output": "Code execution timed out.", "file": filename, "logs": logs}
