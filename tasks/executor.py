from celery import Celery
import subprocess
import uuid
import os

import logging

logging.basicConfig(level=logging.INFO)

app = Celery("executor", 
             broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
# Requires Redis running

# folder to keep generated scripts
SCRIPT_DIR = "generated_scripts"
os.makedirs(SCRIPT_DIR, exist_ok=True)

@app.task
def run_python_code(code: str):
    print('here')
    # save to persistent folder
    filename = os.path.join(SCRIPT_DIR, f"code_{uuid.uuid4().hex}.py")

    logging.info(f"Saved code to {filename} (length={len(code)})")

    # write code to file
    with open(filename, "w") as f:
        f.write(code)

    try:
        output = subprocess.check_output(
            ["python", filename],
            stderr=subprocess.STDOUT,
            timeout=30
        )
        return {
            "output": output.decode(),
            "file": filename
        }
    except subprocess.CalledProcessError as e:
        return {
            "output": e.output.decode(),
            "file": filename
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "Code execution timed out.",
            "file": filename
        }
