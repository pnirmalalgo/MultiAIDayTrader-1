# executor.py
from celery import Celery
import subprocess
import uuid
import os
import logging
import ast
import time

logging.basicConfig(level=logging.INFO)

app = Celery(
    "executor",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

SCRIPT_DIR = "generated_scripts"
os.makedirs(SCRIPT_DIR, exist_ok=True)

# Optionally, set the PLOTS_DIR to the same folder your FastAPI serves via StaticFiles.
# If FastAPI uses PLOTS_DIR = os.path.abspath("."), keep this as '.'
PLOTS_DIR = os.path.abspath(".")  

@app.task(bind=True)
def run_python_code(self, code: str):
    """
    Save the incoming code, run it in a subprocess, and return output + discovered HTML files.
    The function tries multiple strategies to discover generated HTML files:
      1) Look for a printed debug line: Generated files: [...]
      2) Detect new .html files created between before/after snapshots of PLOTS_DIR.
    """
    filename = os.path.join(SCRIPT_DIR, f"code_{uuid.uuid4().hex}.py")
    logging.info(f"Saved code to {filename} (length={len(code)})")
    
    with open(filename, "w") as f:
        f.write(code)

    logs = []

    
    try:
        
        # Snapshot before running
        before_html = set([f for f in os.listdir(PLOTS_DIR) if f.endswith(".html")])

        logs.append("Starting execution...")
        self.update_state(state="PROGRESS", meta={"logs": logs})

        # Execute the script
        output = subprocess.check_output(
            ["python", filename],
            stderr=subprocess.STDOUT,
            timeout=60  # increase if scripts take longer
        )
        decoded_output = output.decode()
        logs.append("Execution finished (subprocess returned).")

        # 1) Try parse "Generated files: [...]" in output
        files = []
        logs.append(f"DEBUG — Subprocess output:\n{decoded_output}")
        for line in decoded_output.splitlines():
            #logs.append(line)
            if "Generated files:" in line:
                try:
                    # safely evaluate the list literal
                    candidate = line.split("Generated files:", 1)[1].strip()
                    parsed = ast.literal_eval(candidate)
                    #logs.append("Parsed generated files:", parsed)  # Debugging line
                    if isinstance(parsed, (list, tuple)):
                        #logs.append("Using parsed file list from output.")
                        files = list(parsed)
                        #logs.append(f"Detected files: {files}")
                        break
                except Exception:
                    # ignore parse errors, fallback to directory scan
                    pass

        # 2) Fallback: detect new HTML files created
        if not files:
            after_html = set([f for f in os.listdir(PLOTS_DIR) if f.endswith(".html")])
            new_files = sorted(list(after_html - before_html))
            files = new_files

        # Finalize logs and return
        logs.append(f"files: {files}")
        self.update_state(state="SUCCESS", meta={
            "logs": logs,
            "files": files,
            "output": decoded_output,
            "file": filename
        })

    except subprocess.CalledProcessError as e:
        err_out = e.output.decode() if hasattr(e, "output") else str(e)
        logs.append(f"Error during execution: {err_out}")
        raise RuntimeError(f"Subprocess failed: {err_out}") from e
    except subprocess.TimeoutExpired as e:
        err_out = e.output.decode() if hasattr(e, "output") else str(e)
        logs.append(f"Error during execution: {err_out}")
        raise RuntimeError(f"Subprocess failed: {err_out}") from e

    except Exception as e:
        err_out = e.output.decode() if hasattr(e, "output") else str(e)
        logs.append(f"Error during execution: {err_out}")
        raise RuntimeError(f"Subprocess failed: {err_out}") from e
        