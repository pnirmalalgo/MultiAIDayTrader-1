import gradio as gr
import requests
import time
import json
import os
import re

API_TASK_STATUS_URL = "http://127.0.0.1:8000/api/task-status"
API_SUBMIT_URL = "http://127.0.0.1:8000/api/submit-query"
API_LIST_HTML_URL = "http://127.0.0.1:8000/api/list-html"

# -----------------------------------------
# Submit initial user query to Orchestrator
# -----------------------------------------
def submit_query_to_orchestrator(user_input):
    try:
        payload = {"query": user_input}
        resp = requests.post(API_SUBMIT_URL, json=payload)
        result = resp.json()

        thoughts = result.get("thoughts", [])
        status = result.get("status", "")
        structured_query = result.get("structured_query", {})

        # Format Chain of Thought (CoT)
        cot_text = "\n".join(
            f"[{t['role'].upper()} - {t['type']}] {t['content']}" for t in thoughts
        )
        structured_query_str = json.dumps(structured_query, indent=2)

        if status == "CLARIFY":
            question = result.get("question", "Is this interpretation correct?")
            status_msg = f"[CLARIFICATION] {question}"
        elif status == "PENDING":
            task_id = result.get("task_id")
            status_msg = f"Task submitted with ID: {task_id}"
        else:
            status_msg = f"Unexpected status: {status}"

        return (
            cot_text,
            structured_query_str,
            status_msg,
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
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=True),
            user_input,
            ""
        )

# -----------------------------------------
# When user rejects interpretation
# -----------------------------------------
def on_user_rejects_interpretation(structured_query_str):
    try:
        structured_query = json.loads(structured_query_str) if structured_query_str else {}
    except Exception:
        structured_query = {}

    payload = {
        "query": "REJECTED",
        "structured_query": structured_query
    }
    resp = requests.post(API_SUBMIT_URL, json=payload)
    result = resp.json()

    return (
        f"Interpretation rejected. Please rephrase your query.\n{result.get('question','')}",
        "",  # clear iframe
        "",  # clear code_output
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False)
    )

# -----------------------------------------
# Poll task status and show result, return py file content
# -----------------------------------------
def poll_task_status(task_id):
    try:
        max_attempts = 60
        delay = 2
        files = []
        py_content = ""
        py_file = None  # ✅ keep track of script file

        for attempt in range(max_attempts):
            resp = requests.get(f"{API_TASK_STATUS_URL}/{task_id}")
            status_data = resp.json()
            status = status_data.get("status", "")
            output_log = status_data.get("output", "")
            print(f"DEBUG: attempt {attempt+1}, task status data = {status_data}")

            if status == "SUCCESS":
                files = status_data.get("files", [])
                py_file = status_data.get("file")  # ✅ pull script file from API
                if "trading_results.html" not in files:
                    files.append("trading_results.html")
                break
            elif status == "FAILURE":
                return "", f"Task failed: {status_data.get('error', 'Unknown error')}", ""
            else:
                time.sleep(delay)

        if not files and not output_log:
            return "", "Task completed but no files found.", ""

        # ✅ If API didn’t return it, fallback to old logic
        if not py_file:
            py_file = next((f for f in files if f.endswith(".py")), None)
        if not py_file:
            match = re.search(r'(generated_scripts/.*?\.py):', output_log)
            py_file = match.group(1) if match else None

        if py_file and os.path.exists(py_file):
            try:
                with open(py_file, "r") as f:
                    py_content = f.read()
                print(f"DEBUG: successfully read py_file = {py_file}")
            except Exception as e:
                print(f"DEBUG: failed to read py_file = {py_file}, error: {e}")

        iframe_html = ""
        for file in files:
            iframe_html += f"""
                <div style="margin-bottom: 20px;">
                    <p><strong>{file}</strong></p>
                    <iframe src="http://localhost:8000/plots/{file}?t={int(time.time())}" 
                        width="100%" height="500px"
                        style="border: 1px solid #ccc; border-radius: 8px;"></iframe>
                </div>
            """
        return iframe_html, "Task completed successfully.", py_content
    except Exception as e:
        return "", f"Exception polling task status: {str(e)}", ""
# -----------------------------------------
# When user confirms interpretation
# -----------------------------------------
def on_user_confirms_interpretation(structured_query_str, cot_text, user_query, original_query, original_cot):
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
    cot_text = "\n".join(
        f"[{t['role'].upper()} - {t['type']}] {t['content']}" for t in thoughts
    )

    if not task_id:
        return f"Error: No task_id returned.\n{cot_text}", "", ""

    # Get iframe, final status, and .py code content
    iframe_html, final_status, code_content = poll_task_status(task_id)

    code_update = gr.update(value=code_content, visible=bool(code_content))

    return cot_text + "\n" + final_status, iframe_html, code_update

# -----------------------------------------
# Flag bad queries
# -----------------------------------------
def on_flag_click(structured_query_str):
    try:
        structured_query = json.loads(structured_query_str) if structured_query_str else {}
        with open("flagged_queries.jsonl", "a") as f:
            f.write(json.dumps(structured_query) + "\n")
        return "Query flagged for review!"
    except Exception as e:
        return f"Exception: {str(e)}"

# -----------------------------------------
# List existing plots
# -----------------------------------------
def list_plots():
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
                    <iframe src="http://localhost:8000/plots/{file}" width="100%" height="500px"
                        style="border: 1px solid #ccc; border-radius: 8px;"></iframe>
                </div>
            """
        return iframe_html
    except Exception as e:
        return f"<p>Error fetching plot list: {str(e)}</p>"

# -----------------------------------------
# Gradio UI
# -----------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Trading Query Interpreter with Chain of Thoughts")

    user_input = gr.Textbox(
        label="Enter Trading Query",
        placeholder="E.g., Show me RSI trades for SBIN.NS from Jan 2025 to Mar 2025",
        interactive=True
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

    # ----------------- BUTTON CLICKS -----------------
    reject_btn.click(
        fn=on_user_rejects_interpretation,
        inputs=structured_query_output,
        outputs=[
            status_output,
            iframe_display,
            code_output,
            confirm_btn,
            structured_query_output,
            cot_output,
            reject_btn
        ],
    )

    submit_btn.click(
        fn=submit_query_to_orchestrator,
        inputs=user_input,
        outputs=[
            cot_output,
            structured_query_output,
            status_output,
            code_output,
            edit_btn,
            confirm_btn,
            reject_btn,
            flag_btn,
            submit_btn,
            user_input,
            original_query_state,
            original_cot_state
        ],
    )

    def on_edit_cot():
        return gr.update(interactive=True), gr.update(visible=False)

    edit_btn.click(
        fn=on_edit_cot,
        inputs=[],
        outputs=[cot_output, edit_btn],
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

demo.launch()
