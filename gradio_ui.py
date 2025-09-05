# gradio_ui.py
import gradio as gr
import json
import requests

from agents.interpreter import interpreter_with_cot  # your CoT-enabled interpreter
import time

API_TASK_STATUS_URL = "http://127.0.0.1:8000/api/task-status"
API_SUBMIT_URL = "http://127.0.0.1:8000/api/submit-query"
API_LIST_HTML_URL = "http://127.0.0.1:8000/api/list-html"

def on_confirm_click(structured_query_dict):
    try:
        payload = {"query": json.dumps(structured_query_dict)}
        resp = requests.post(API_SUBMIT_URL, json=payload)
        result = resp.json()

        if not result.get("task_id"):
            return f"Error submitting query: {result.get('error')}", "", [], gr.update(visible=True)

        task_id = result["task_id"]
        thoughts = result.get("thoughts", [])  # ✅ Get actual CoT thoughts

        # --- Poll for completion ---
        status = "PENDING"
        files = []
        max_attempts = 20
        delay = 1.5  # seconds

        for _ in range(max_attempts):
            status_resp = requests.get(f"{API_TASK_STATUS_URL}/{task_id}")
            status_data = status_resp.json()
            print("DEBUG — Task status data:", status_data)
            status = status_data.get("status", "")
            if status == "SUCCESS":
                print("DEBUG — Task completed successfully.")
                files = status_data.get("files", [])
                print("DEBUG — Files returned:", files)
                break
            elif status == "FAILURE":
                return f"Task failed: {status_data.get('error', 'Unknown error')}", "", thoughts, gr.update(visible=False)
            else:
                time.sleep(delay)

        if not files:
            return f"Task completed but no files found.", "", thoughts, gr.update(visible=False)

        # --- Build iframe HTML for returned files only ---
        iframe_html = ""
        for file in files:
            iframe_html += f"""
                <div style="margin-bottom: 20px;">
                    <p><strong>{file}</strong></p>
                    <iframe src="http://localhost:8000/plots/{file}" width="100%" height="500px"
                        style="border: 1px solid #ccc; border-radius: 8px;"></iframe>
                </div>
            """

        return f"Query submitted! Task ID: {task_id}", iframe_html, thoughts, gr.update(visible=False)

    except Exception as e:
        return f"Exception: {str(e)}", "", [], gr.update(visible=True)
   
def on_flag_click(structured_query_dict):
    try:
        with open("flagged_queries.jsonl", "a") as f:
            f.write(json.dumps(structured_query_dict) + "\n")
        return "Query flagged for review!"
    except Exception as e:
        return f"Exception: {str(e)}"

def list_plots():
    try:
        resp = requests.get(API_LIST_HTML_URL)
        data = resp.json()
        files = data.get("files", [])
        if not files:
            return "<p>No plots found.</p>"

        # Create iframes for each HTML file
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

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:

    gr.Markdown("## 🧠 Trading Query Interpreter with Chain of Thoughts")

    # Input
    user_input = gr.Textbox(
        label="Enter Trading Query",
        placeholder="E.g., Show me RSI trades for SBIN.NS from Jan 2025 to Mar 2025"
    )

    # Outputs
    cot_output = gr.Textbox(label="Chain of Thoughts", lines=10)
    structured_query_output = gr.JSON(label="Structured Query (editable)")
    status_output = gr.Textbox(label="Status / Task ID")
    iframe_display = gr.HTML(label="Generated Plots")

    # Buttons
    generate_btn = gr.Button("Generate CoT & Structured Query")
    confirm_btn = gr.Button("Confirm & Submit")
    flag_btn = gr.Button("Flag for Review")
    refresh_btn = gr.Button("🔄 Refresh Plots")

    # Callbacks
    generate_btn.click(
        fn=interpreter_with_cot,
        inputs=user_input,
        outputs=[cot_output, structured_query_output]
    )

    confirm_btn.click(
        fn=on_confirm_click,
        inputs=structured_query_output,
        outputs=[
            status_output,     # Status / Task ID
            iframe_display,    # HTML plots
            cot_output,        # Chain of Thought
            structured_query_output  # 👈 Update visibility
        ]
    )

    flag_btn.click(
        fn=on_flag_click,
        inputs=structured_query_output,
        outputs=status_output
    )

    refresh_btn.click(
        fn=list_plots,
        inputs=[],
        outputs=iframe_display
    )

# Launch Gradio UI
demo.launch()
