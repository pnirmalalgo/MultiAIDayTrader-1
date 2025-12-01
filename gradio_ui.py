import gradio as gr
import requests
import time
import json
import os
import re
import logging
from datetime import datetime

# -------------------- API URLs --------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_TASK_STATUS_URL = f"{API_BASE_URL}/api/task-status"
API_SUBMIT_URL = f"{API_BASE_URL}/api/submit-query"
API_LIST_HTML_URL = f"{API_BASE_URL}/api/list-html"

SCRIPT_DIR = "/app/generated_scripts"

# ---------- Logging setup ----------
logger = logging.getLogger("poll_task_status_logger")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join("logs", "poll_task_status.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

# -------------------- Utility Functions (unchanged) --------------------

def get_filtered_tickers():
    """Load filtered tickers from JSON if available."""
    try:
        if os.path.exists("filtered_tickers.json"):
            with open("filtered_tickers.json", "r") as f:
                tickers = json.load(f)
            return [t.strip() for t in tickers if t.strip()]
        return []
    except Exception as e:
        print("DEBUG: Error loading filtered tickers:", e)
        return []

# -------------------- Gradio Core Functions (unchanged) --------------------
def submit_query_to_orchestrator(user_input, filtered_tickers_text):
    """
    Submit user query to the FastAPI Orchestrator endpoint.
    """
    try:
        payload = {"query": user_input}

        # Step 1: Send initial query
        resp = requests.post(API_SUBMIT_URL, json=payload)
        result = resp.json()
        thoughts = result.get("thoughts", [])
        status = result.get("status", "")
        structured_query = result.get("structured_query", {})

        # Step 2: Merge filtered tickers
        tickers_text = filtered_tickers_text.strip()
        if tickers_text:
            tickers_list = [t.strip() for t in re.split(r'[,\s]+', tickers_text) if t.strip()]
            print("DEBUG: Using tickers from textbox:", tickers_list)
        else:
            tickers_list = get_filtered_tickers()
            print("DEBUG: Using tickers from filtered_tickers.json:", tickers_list)

        if tickers_list:
            structured_query["ticker"] = tickers_list

        payload["structured_query"] = structured_query
        print("DEBUG: Payload being sent after merging tickers:", payload)

        cot_text = "\n".join(
            f"[{t['role'].upper()} - {t['type']}] {t['content']}" for t in thoughts
        )
        structured_query_str = json.dumps(structured_query, indent=2)
        tickers_str = ", ".join(tickers_list) if tickers_list else "No filtered tickers applied."

        if status == "CLARIFY":
            question = result.get("question", "Is this interpretation correct?")
            status_msg = f"[CLARIFICATION] {question}"
        elif status == "PENDING":
            task_id = result.get("task_id")
            status_msg = f"Task submitted with ID: {task_id}"
        else:
            status_msg = f"Status: {status}"

        # CRITICAL: Return exact number of outputs matching the outputs list
        return (
            cot_text,                                      # cot_output
            structured_query_str,                          # structured_query_output (State)
            status_msg,                                    # status_output
            tickers_str,                                   # filtered_tickers_box
            "",                                            # code_output (cleared on initial submit)
            gr.update(visible=True),                       # edit_btn
            gr.update(visible=(status == "CLARIFY")),      # confirm_btn
            gr.update(visible=(status == "CLARIFY")),      # reject_btn
            gr.update(visible=False),                      # flag_btn
            gr.update(visible=False),                      # refresh_btn
            gr.update(visible=False),                      # submit_btn
            user_input,                                    # original_query_state (State)
            cot_text                                       # original_cot_state (State)
        )

    except Exception as e:
        logger.exception(f"Error in submit_query_to_orchestrator: {e}")
        return (
            "",
            "{}",
            f"Exception: {str(e)}",
            "Error loading filtered tickers.",
            "",                                # code_output
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),           # re-enable submit button
            user_input,
            ""
        )

def refresh_filtered_tickers_box():
    """Return current filtered tickers as a comma-separated string."""
    tickers = get_filtered_tickers()
    return ", ".join(tickers) if tickers else "No filtered tickers found."

# -------------------- Poll Task Status (unchanged) --------------------
def poll_task_status(task_id):
    """
    Poll Celery task status and return iframe HTML + code.
    """
    logger.info(f"poll_task_status called with task_id={task_id}")

    try:
        max_attempts = 60
        delay = 2
        files = []
        py_content = ""
        py_file = None

        for attempt in range(max_attempts):
            try:
                resp = requests.get(f"{API_TASK_STATUS_URL}/{task_id}")
                status_data = resp.json()
            except Exception as e:
                logger.exception(f"HTTP or JSON error on attempt {attempt+1}: {e}")
                time.sleep(delay)
                continue

            status = status_data.get("status", "")
            output_log = status_data.get("output", "")
            logger.debug(f"Attempt {attempt+1}: status={status}")

            if status == "SUCCESS" or status == "COMPLETED":
                files = status_data.get("files", []) or []
                py_file = status_data.get("file")
                if "trading_results.html" not in files:
                    files.append("trading_results.html")
                logger.info(f"Task success: files returned count={len(files)}, py_file={py_file}")
                break
            elif status == "FAILURE" or status == "FAILED":
                err = status_data.get("error", "Unknown error")
                logger.error(f"Task failure: {err}")
                return "", f"Task failed: {err}", ""
            else:
                time.sleep(delay)

        # Read Python file content
        if py_file and os.path.exists(py_file):
            try:
                with open(py_file, "r") as f:
                    py_content = f.read()
                logger.info(f"✅ Successfully read py_file. Length: {len(py_content)}")
            except Exception as e:
                logger.exception(f"❌ Error reading py_file {py_file}: {e}")
                py_content = "# Error reading file"
        else:
            logger.warning(f"⚠️ Python file not found: {py_file}")
            py_content = "# Python file not found"

        if not files and not output_log:
            logger.warning("No files and no output_log after polling")
            return "", "Task completed but no files found.", py_content

        # Reorder files (unchanged)
        summary_patterns = [
            r"portfolio_equity_curve",
            r"portfolio_summary",
            r"trading_results"
        ]
        summary_files = []
        for pat in summary_patterns:
            for f in files:
                if re.search(pat, f):
                    if f not in summary_files:
                        summary_files.append(f)

        if "trading_results.html" not in summary_files:
            if "trading_results.html" in files:
                summary_files.append("trading_results.html")
            else:
                summary_files.append("trading_results.html")

        other_files = [f for f in files if f not in summary_files]
        files_ordered = summary_files + other_files

        logger.info(f"files_ordered: {files_ordered}")

        # Build iframe_html (unchanged)
        iframe_html = ""
        for file in files_ordered:
            iframe_html += f"""
                <div style="margin-bottom: 20px; border: 1px solid #ccc; border-radius: 8px; padding: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <p style="margin: 0;"><strong>{file}</strong></p>
                        <button onclick="document.getElementById('iframe_{file}').src='{API_BASE_URL}/plots/{file}?t=' + Date.now()" 
                                style="background: #f0f0f0; border: none; padding: 4px 8px; border-radius: 6px; cursor: pointer;">
                            🔄 Refresh
                        </button>
                    </div>
                    <iframe id="iframe_{file}" 
                        src="{API_BASE_URL}/plots/{file}?t={int(time.time())}" 
                        width="100%" height="500px"
                        style="border: none; border-radius: 8px; margin-top: 8px;">
                    </iframe>
                </div>
            """

        logger.info(f"✅ Returning py_content length: {len(py_content)}")
        return iframe_html, "Task completed successfully.", py_content

    except Exception as e:
        logger.exception(f"Exception polling task status: {e}")
        return "", f"Exception polling task status: {str(e)}", ""

# -------------------- Confirm / Flag (unchanged) --------------------
def on_user_confirms_interpretation(structured_query_str, cot_text, user_query, original_query, original_cot):
    """Send confirmation to Orchestrator and display results."""
    try:
        structured_query = json.loads(structured_query_str) if structured_query_str else {}
    except Exception as e:
        return f"❌ Invalid JSON in structured query: {e}", "", ""

    payload = {
        "query": user_query.strip(),
        "structured_query": structured_query,
        "cot": cot_text.strip(),
        "original_query": original_query,
        "original_cot": original_cot
    }
    
    try:
        resp = requests.post(API_SUBMIT_URL, json=payload)
        result = resp.json()
        task_id = result.get("task_id")
        thoughts = result.get("thoughts", [])
        cot_update = "\n".join(f"[{t['role'].upper()} - {t['type']}] {t['content']}" for t in thoughts)

        if not task_id:
            return f"Error: No task_id returned.\n{cot_update}", "", ""

        # Poll for results
        iframe_html, final_status, code_content = poll_task_status(task_id)
        
        logger.info(f"📊 Code content length: {len(code_content)}")
        logger.debug(f"📄 First 200 chars: {code_content[:200]}")

        # RETURN PLAIN STRING - Gradio will handle display
        return (
            cot_update + "\n" + final_status,
            iframe_html,
            code_content  # ← PLAIN STRING, no gr.update()
        )
        
    except Exception as e:
        logger.exception(f"Error in on_user_confirms_interpretation: {e}")
        return f"❌ Exception: {str(e)}", "", ""

def on_flag_click(structured_query_str):
    """Flag bad queries for review."""
    try:
        structured_query = json.loads(structured_query_str) if structured_query_str else {}
        with open("flagged_queries.jsonl", "a") as f:
            f.write(json.dumps(structured_query) + "\n")
        return "Query flagged for review!"
    except Exception as e:
        return f"Exception: {str(e)}"

def list_plots():
    """List existing plot HTML files."""
    try:
        resp = requests.get(API_LIST_HTML_URL)
        data = resp.json()
        files = data.get("files", [])
        if not files:
            return "<p>No plots found.</p>"

        iframe_html = ""
        for file in files:
            iframe_html += f"""
                <div style="margin-bottom: 20px;">
                    <p><strong>{file}</strong></p>
                    <iframe src="{API_BASE_URL}/plots/{file}" width="100%" height="500px"
                        style="border: 1px solid #ccc; border-radius: 8px;"></iframe>
                </div>
            """
        return iframe_html
    except Exception as e:
        return f"<p>Error fetching plot list: {str(e)}</p>"

# -------------------- Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Trading Query Interpreter with Chain of Thoughts")

    user_input = gr.Textbox(
        label="Enter Trading Query",
        placeholder="E.g., Show me RSI trades for SBIN.NS from Jan 2025 to Mar 2025",
        interactive=True
    )

    filtered_tickers_box = gr.Textbox(
        label="🎯 Filtered Tickers (from Stock Screener)",
        lines=2,
        interactive=True,
        value=refresh_filtered_tickers_box()
    )

    cot_output = gr.Textbox(label="Chain of Thoughts", lines=10, interactive=False)
    structured_query_output = gr.State()
    original_query_state = gr.State()
    original_cot_state = gr.State()

    with gr.Row():
        status_output = gr.Textbox(label="Status / Task ID", lines=5, interactive=False)
        # ✅ CHANGE: Using gr.Textbox instead of gr.Code for stability
        code_output = gr.Textbox(
            label="Generated Python Code",
            interactive=False, # Make it read-only
            value="",
            lines=20 # Control height with lines
        )

    iframe_display = gr.HTML(label="Generated Plots")

    submit_btn = gr.Button("Submit Query")
    edit_btn = gr.Button("✏️ Edit CoT", visible=False)
    confirm_btn = gr.Button("✅ Confirm Interpretation", visible=False)
    reject_btn = gr.Button("❌ Reject Interpretation", visible=False)
    flag_btn = gr.Button("🚩 Flag for Review", visible=False)
    refresh_btn = gr.Button("🔄 Refresh Plots", visible=False)

    submit_btn.click(
        fn=submit_query_to_orchestrator,
        inputs=[user_input, filtered_tickers_box],
        outputs=[
            cot_output,              # 0
            structured_query_output, # 1 (State)
            status_output,           # 2
            filtered_tickers_box,    # 3
            code_output,             # 4 (Content clear)
            edit_btn,                # 5
            confirm_btn,             # 6
            reject_btn,              # 7
            flag_btn,                # 8
            refresh_btn,             # 9
            submit_btn,              # 10
            original_query_state,    # 11 (State)
            original_cot_state       # 12 (State)
        ],
    )

    edit_btn.click(
        fn=lambda: gr.update(interactive=True),
        inputs=[],
        outputs=[cot_output]
    )

    confirm_btn.click(
        fn=on_user_confirms_interpretation,
        inputs=[
            structured_query_output,
            cot_output,
            user_input,
            original_query_state,
            original_cot_state
        ],
        outputs=[
            status_output,    # 0
            iframe_display,   # 1
            code_output       # 2 (Content update)
        ],
    )

    flag_btn.click(
        fn=on_flag_click,
        inputs=structured_query_output,
        outputs=status_output,
    )

    refresh_btn.click(
        fn=list_plots,
        inputs=[],
        outputs=iframe_display,
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0")