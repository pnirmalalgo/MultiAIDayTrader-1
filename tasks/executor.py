from celery import Celery
import subprocess
import uuid
import os

app = Celery("executor", 
             broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
# Requires Redis running

@app.task
def run_python_code(code: str):
    filename = f"/tmp/code_{uuid.uuid4().hex}.py"
    print(code, flush=True)
    with open(filename, "w") as f:
        f.write(code)

    try:
        output = subprocess.check_output(["python", filename], stderr=subprocess.STDOUT, timeout=30)
        return output.decode()
    except subprocess.CalledProcessError as e:
        return e.output.decode()
    except subprocess.TimeoutExpired:
        return "Code execution timed out."
    finally:
        try:
            os.remove(filename)
        except Exception:
            pass