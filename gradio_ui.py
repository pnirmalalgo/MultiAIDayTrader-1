import gradio as gr
import requests
import time
import json

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
            return (
                cot_text,
                structured_query_str,
                status_msg,
                gr.update(visible=True),    # confirm_btn
                gr.update(visible=True),    # reject_btn
                gr.update(visible=False),   # flag_btn
                gr.update(visible=False),   # submit_btn
                gr.update(visible=True),    # cot_output
                gr.update(visible=True),    # structured_query_output
                user_input,                 # NEW → original_query
                cot_text                    # NEW → original_cot
            )

        elif status == "PENDING":
            task_id = result.get("task_id")
            status_msg = f"Task submitted with ID: {task_id}"
            return (
                cot_text,
                structured_query_str,
                status_msg,
                gr.update(visible=False),   # confirm_btn
                gr.update(visible=False),   # reject_btn
                gr.update(visible=False),   # flag_btn
                gr.update(visible=False),   # submit_btn
                gr.update(visible=True),    # cot_output
                gr.update(visible=True),    # structured_query_output
                user_input,                 # NEW → original_query
                cot_text                    # NEW → original_cot
            )

        else:
            return (
                cot_text,
                structured_query_str,
                f"Unexpected status: {status}",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                user_input,                 # NEW
                cot_text                    # NEW
            )

    except Exception as e:
        return (
            "",
            "{}",
            f"Exception: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            user_input,                     # NEW (still pass query for debugging)
            ""                              # NEW (empty original_cot if fail)
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
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False)
    )

# -----------------------------------------
# When user confirms interpretation
# -----------------------------------------
def on_user_confirms_interpretation(structured_query_str, cot_text, original_query, original_cot):
    try:
        structured_query = json.loads(structured_query_str) if structured_query_str else {}
    except Exception as e:
        return f"❌ Invalid JSON in structured query: {e}", ""

    payload = {
        "query": "CONFIRMED",
        "structured_query": structured_query,
        "cot": cot_text,
        "original_query": original_query,   # NEW
        "original_cot": original_cot        # NEW
    }
    resp = requests.post(API_SUBMIT_URL, json=payload)
    result = resp.json()

    task_id = result.get("task_id")
    thoughts = result.get("thoughts", [])
    cot_text = "\n".join(
        f"[{t['role'].upper()} - {t['type']}] {t['content']}" for t in thoughts
    )

    if not task_id:
        return f"Error: No task_id returned.\n{cot_text}", ""

    iframe_html, final_status = poll_task_status(task_id)
    return cot_text + "\n" + final_status, iframe_html

# -----------------------------------------
# Poll task status and show result
# -----------------------------------------
def poll_task_status(task_id):
    try:
        max_attempts = 20
        delay = 1.5
        files = []

        for _ in range(max_attempts):
            resp = requests.get(f"{API_TASK_STATUS_URL}/{task_id}")
            status_data = resp.json()
            status = status_data.get("status", "")
            if status == "SUCCESS":
                files = status_data.get("files", [])
                if "trading_results.html" not in files:
                    files.append("trading_results.html")
                break
            elif status == "FAILURE":
                return "", f"Task failed: {status_data.get('error', 'Unknown error')}"
            else:
                time.sleep(delay)

        if not files:
            return "", "Task completed but no files found."

        iframe_html = ""
        for file in files:
            iframe_html += f"""
                <div style="margin-bottom: 20px;">
                    <p><strong>{file}</strong></p>
                    <iframe src="http://localhost:8000/plots/{file}" width="100%" height="500px"
                        style="border: 1px solid #ccc; border-radius: 8px;"></iframe>
                </div>
            """
        return iframe_html, "Task completed successfully."

    except Exception as e:
        return "", f"Exception polling task status: {str(e)}"

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
        placeholder="E.g., Show me RSI trades for SBIN.NS from Jan 2025 to Mar 2025"
    )

    cot_output = gr.Textbox(label="Chain of Thoughts (editable)", lines=10, interactive=True)

    structured_query_output = gr.State()

    status_output = gr.Textbox(label="Status / Task ID")
    iframe_display = gr.HTML(label="Generated Plots")

    # NEW hidden states
    original_query_state = gr.State()
    original_cot_state = gr.State()

    submit_btn = gr.Button("Submit Query")
    confirm_btn = gr.Button("✅ Confirm Interpretation", visible=False)
    reject_btn = gr.Button("❌ Reject Interpretation", visible=False)
    flag_btn = gr.Button("🚩 Flag for Review", visible=False)
    refresh_btn = gr.Button("🔄 Refresh Plots", visible=False)

    reject_btn.click(
        fn=on_user_rejects_interpretation,
        inputs=structured_query_output,
        outputs=[
            status_output,
            iframe_display,
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
            confirm_btn,
            reject_btn,
            flag_btn,
            submit_btn,
            cot_output,
            structured_query_output,
            original_query_state,   # NEW
            original_cot_state      # NEW
        ],
    )

    confirm_btn.click(
        fn=on_user_confirms_interpretation,
        inputs=[structured_query_output, cot_output, original_query_state, original_cot_state],  # NEW
        outputs=[status_output, iframe_display],
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
