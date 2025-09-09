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

        if status == "CLARIFY":
            question = result.get("question", "Is this interpretation correct?")
            status_msg = f"[CLARIFICATION] {question}"
            return cot_text, structured_query, status_msg, gr.update(visible=True), gr.update(visible=True)


        elif status == "PENDING":
            task_id = result.get("task_id")
            status_msg = f"Task submitted with ID: {task_id}"
            return cot_text, {}, status_msg, gr.update(visible=False), gr.update(visible=False)


        else:
            return cot_text, {}, f"Unexpected status: {status}", gr.update(visible=False), gr.update(visible=False)


    except Exception as e:
        return "", {}, f"Exception: {str(e)}", gr.update(visible=False)


# -----------------------------------------
# When user confirms interpretation
# -----------------------------------------
def on_user_confirms_interpretation(structured_query_dict):
    try:
        payload = {
            "query": "CONFIRMED",
            "structured_query": structured_query_dict
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

        # Start polling
        iframe_html, final_status = poll_task_status(task_id)
        return cot_text + "\n" + final_status, iframe_html

    except Exception as e:
        return f"Exception during confirmation: {str(e)}", ""


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

        # HTML for plots
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
# Flag bad queries for review
# -----------------------------------------
def on_flag_click(structured_query_dict):
    try:
        with open("flagged_queries.jsonl", "a") as f:
            f.write(json.dumps(structured_query_dict) + "\n")
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

    cot_output = gr.Textbox(label="Chain of Thoughts", lines=10)
    structured_query_output = gr.JSON(label="Structured Query (editable)", visible=True)

    status_output = gr.Textbox(label="Status / Task ID")
    iframe_display = gr.HTML(label="Generated Plots")

    submit_btn = gr.Button("Submit Query")
    confirm_btn = gr.Button("✅ Confirm Interpretation", visible=False)
    flag_btn = gr.Button("🚩 Flag for Review")
    refresh_btn = gr.Button("🔄 Refresh Plots")

    # Submit user query
    submit_btn.click(
        fn=submit_query_to_orchestrator,
        inputs=user_input,
        outputs=[
            cot_output,
            structured_query_output,
            status_output,
            confirm_btn,
            structured_query_output,  # <-- added to allow hiding on submit
        ],
    )

    # Confirm structured query
    confirm_btn.click(
        fn=on_user_confirms_interpretation,
        inputs=structured_query_output,
        outputs=[
            status_output,
            iframe_display,
        ],
    )

    # Flag bad interpretation
    flag_btn.click(
        fn=on_flag_click,
        inputs=structured_query_output,
        outputs=status_output,
    )

    # Refresh plot list
    refresh_btn.click(
        fn=list_plots,
        inputs=[],
        outputs=iframe_display,
    )

demo.launch()
