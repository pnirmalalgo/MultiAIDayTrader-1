from tasks.executor import run_python_code

def executor_wrapper(code):
    """
    Submits the code to Celery and returns structured MCP-compliant output.
    """
    result = run_python_code.delay(code)
    return {"id": result.id, "status": "submitted"}

