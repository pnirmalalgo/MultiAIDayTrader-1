import gradio as gr
import requests
import time
import json
import os
import re

# -------------------- API URLs --------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_TASK_STATUS_URL = f"{API_BASE_URL}/api/task-status"
API_SUBMIT_URL = f"{API_BASE_URL}/api/submit-query"
API_LIST_HTML_URL = f"{API_BASE_URL}/api/list-html"

# -------------------- Utility Functions --------------------
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

# -------------------- Gradio Core Functions --------------------
def submit_query_to_orchestrator(user_input, filtered_tickers_text):
    """
    Submit user query to the FastAPI Orchestrator endpoint.
    Overrides only tickers with filtered_tickers.json if available.
    """
    try:
        payload = {"query": user_input}

        # Step 1: Send initial query to get structured query from orchestrator
        resp = requests.post(API_SUBMIT_URL, json=payload)
        result = resp.json()
        thoughts = result.get("thoughts", [])
        status = result.get("status", "")
        structured_query = result.get("structured_query", {})

        # Step 2: Merge filtered tickers if present
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

        return (
            cot_text,
            structured_query_str,
            status_msg,
            tickers_str,
            "",  # code_output initially empty
            gr.update(visible=(status=="CLARIFY")),
            gr.update(visible=(status=="CLARIFY")),
            gr.update(visible=(status=="CLARIFY")),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=False),
            user_input,
            cot_text
        )

    except Exception as e:
        return (
            "",
            "{}",
            f"Exception: {str(e)}",
            "Error loading filtered tickers.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=True),
            user_input,
            ""
        )

def refresh_filtered_tickers_box():
    """Return current filtered tickers as a comma-separated string for the textbox."""
    tickers = get_filtered_tickers()
    return ", ".join(tickers) if tickers else "No filtered tickers found."

# -------------------- Poll Task Status --------------------
def poll_task_status(task_id):
    """Poll Celery task status and return iframe HTML + code."""
    try:
        max_attempts = 60
        delay = 2
        files = []
        py_content = ""
        py_file = None

        for attempt in range(max_attempts):
            resp = requests.get(f"{API_TASK_STATUS_URL}/{task_id}")
            status_data = resp.json()
            status = status_data.get("status", "")
            output_log = status_data.get("output", "")
            print(f"DEBUG: attempt {attempt+1}, status={status}")

            if status == "SUCCESS":
                files = status_data.get("files", [])
                py_file = status_data.get("file")
                if "trading_results.html" not in files:
                    files.append("trading_results.html")
                break
            elif status == "FAILURE":
                return "", f"Task failed: {status_data.get('error', 'Unknown error')}", ""
            else:
                time.sleep(delay)

        if not files and not output_log:
            return "", "Task completed but no files found.", ""

        if not py_file:
            py_file = next((f for f in files if f.endswith(".py")), None)
        if not py_file:
            match = re.search(r'(generated_scripts/.*?\.py):', output_log)
            py_file = match.group(1) if match else None

        if py_file and os.path.exists(py_file):
            with open(py_file, "r") as f:
                py_content = f.read()
            print(f"DEBUG: Read py_file = {py_file}")

        iframe_html = ""
        for file in files:
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
        return iframe_html, "Task completed successfully.", py_content
    except Exception as e:
        return "", f"Exception polling task status: {str(e)}", ""

# -------------------- Confirm / Flag --------------------
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
    resp = requests.post(API_SUBMIT_URL, json=payload)
    result = resp.json()
    task_id = result.get("task_id")
    thoughts = result.get("thoughts", [])
    cot_text = "\n".join(f"[{t['role'].upper()} - {t['type']}] {t['content']}" for t in thoughts)

    if not task_id:
        return f"Error: No task_id returned.\n{cot_text}", "", ""

    iframe_html, final_status, code_content = poll_task_status(task_id)
    code_update = gr.update(value=code_content, visible=bool(code_content))

    return cot_text + "\n" + final_status, iframe_html, code_update

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
        code_output = gr.Textbox(label="Generated Python Code", lines=20, interactive=False, visible=False)

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
            cot_output,
            structured_query_output,
            status_output,
            filtered_tickers_box,
            code_output,
            gr.State(),
            confirm_btn,
            reject_btn,
            flag_btn,
            submit_btn,
            user_input,
            cot_output
        ],
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
        outputs=[status_output, iframe_display, code_output],
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

demo.launch(server_port=7860, server_name="0.0.0.0")
