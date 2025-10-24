import gradio as gr
import os
import pandas as pd
import sqlite3
import json
from dotenv import load_dotenv
import openai
from scraper.scraper import scrape_multiple_tickers, load_tickers_from_csv

# ------------------ Setup ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

DB_PATH = "./market_data.db"

# ------------------ Utility Functions ------------------

def check_metrics_table():
    if not os.path.exists(DB_PATH):
        return False
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='company_metrics';"
    )
    exists = bool(cursor.fetchone())
    conn.close()
    return exists

def save_to_database(df: pd.DataFrame, db_path: str = DB_PATH, table_name: str = "company_metrics"):
    """Save DataFrame to SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            Ticker TEXT PRIMARY KEY,
            Company_Name TEXT,
            Market_Cap REAL,
            PE_Ratio REAL,
            PB_Ratio REAL,
            PS_Ratio REAL,
            ROCE REAL,
            ROE REAL,
            Net_Profit REAL,
            Promoter_Holding REAL,
            Last_Updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        conn.execute(create_table_sql)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"💾 ✅ Saved {count} records to {db_path} (table: {table_name})")
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        raise

def load_existing_metrics():
    """Load existing data from company_metrics table if available."""
    try:
        if not check_metrics_table():
            return "<p style='color:gray'>No metrics found. Please click 'Refresh Metrics' first.</p>"
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM company_metrics", conn)
        conn.close()
        if df.empty:
            return "<p style='color:gray'>No records found in company_metrics table.</p>"
        return df.to_html(index=False, justify="center")
    except Exception as e:
        return f"<p style='color:red'>Error loading data: {e}</p>"

def refresh_metrics():
    csv_path = "tickers/NIFTY_50_LIST.csv"
    tickers = load_tickers_from_csv(csv_path)
    df = scrape_multiple_tickers(tickers)
    save_to_database(df, db_path=DB_PATH, table_name='company_metrics')
    return "success"

def sanitize_ai_query(query, df_columns):
    import re
    for col in df_columns:
        pattern = re.compile(rf".*{re.escape(col)}.*", re.IGNORECASE)
        query = pattern.sub(col, query)
    for col in df_columns:
        if not col.isidentifier():
            query = re.sub(rf"\b{re.escape(col)}\b", f"`{col}`", query)
    query = re.sub(r"BACKTICK_QUOTED_STRING__.*?__", "", query)
    return query

def get_filtered_tickers():
    """Return the latest filtered tickers from JSON."""
    try:
        with open("filtered_tickers.json", "r") as f:
            tickers = json.load(f)
        return tickers
    except Exception:
        return []

def screen_stocks(criteria_text, mode="Strict"):
    if not check_metrics_table():
        return "company_metrics table not found. Please refresh metrics first.", []

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM company_metrics", conn)
        conn.close()

        numeric_cols = ["ROCE", "ROE", "PE_Ratio", "PB_Ratio", "PS_Ratio",
                        "Net_Profit", "Market_Cap", "Promoter_Holding"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace("₹", "", regex=False)
                    .str.extract(r"(\d+\.?\d*)")[0]
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if mode == "Strict":
            filtered_df = df.query(criteria_text)
        else:
            query_str, error = criteria_to_query(criteria_text, df.columns.tolist())
            if error:
                return error, []
            query_str = sanitize_ai_query(query_str, df.columns.tolist())
            filtered_df = df.query(query_str)

        tickers = filtered_df.get("Ticker", pd.Series()).dropna().tolist()

        # Save filtered tickers
        with open("filtered_tickers.json", "w") as f:
            json.dump(tickers, f, indent=2)

        if not tickers:
            return "No tickers found matching criteria.", []

        display_table = filtered_df.to_html(index=False, justify="center")
        return display_table, tickers

    except Exception as e:
        return f"Error fetching data: {e}", []

def criteria_to_query(criteria_text, df_columns):
    if not OPENAI_API_KEY:
        return None, "Missing OpenAI API key."
    system_prompt = (
        "You are an expert in translating natural language stock screening criteria "
        "into Python pandas DataFrame query strings. "
        "Only use these columns if they exist: " + ", ".join(df_columns) + ". "
        "Return only a valid pandas query string without explanation."
    )
    user_prompt = f"Convert this criteria into a pandas query string:\n{criteria_text}"
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        query_str = response.choices[0].message.content.strip()
        return query_str, None
    except Exception as e:
        return None, f"Error calling OpenAI API: {e}"

# ------------------ Gradio UI ------------------

with gr.Blocks() as prescreen_demo:
    gr.Markdown("## 📊 Pre-Screen Stocks — Criteria or Natural Language")

    with gr.Row():
        criteria_input = gr.Textbox(
            label="Enter Screening Criteria or Ask in Natural Language",
            placeholder='Examples:\n1️⃣ ROCE > 30 and PE < 25\n2️⃣ Show companies with high ROE and low PE',
            lines=3
        )
        refresh_btn = gr.Button("🔄 Refresh Metrics")

    mode_selector = gr.Radio(
        choices=["Strict", "AI"],
        label="Mode",
        value="Strict",
        info="Use 'Strict' for direct pandas query, 'AI' for GPT-4o-mini natural language queries"
    )

    submit_btn = gr.Button("🔍 Submit")
    results_table = gr.HTML(label="Matching Stocks / Existing Metrics", value=load_existing_metrics())
    tickers_state = gr.State()

    # ---------------- Button Callbacks ----------------
    refresh_btn.click(
        fn=refresh_metrics,
        inputs=[],
        outputs=results_table
    )

    submit_btn.click(
        fn=screen_stocks,
        inputs=[criteria_input, mode_selector],
        outputs=[results_table, tickers_state]
    )

    go_to_ui_btn = gr.Button("➡️ Go to Trading UI")

    def open_trading_ui(tickers_state):
        """
        Use the latest tickers from Gradio state, not old JSON.
        """
        latest_tickers = tickers_state if tickers_state else get_filtered_tickers()
        return (
            f'<p>✅ Loaded {len(latest_tickers)} tickers: {", ".join(latest_tickers)}</p>'
            '<a href="http://127.0.0.1:7861" target="_blank" style="font-size:16px; color:blue;">Click here to open Trading UI</a>',
            latest_tickers
        )

    go_to_ui_btn.click(
        fn=open_trading_ui,
        inputs=[tickers_state],
        outputs=[results_table, tickers_state]
    )

prescreen_demo.launch()
