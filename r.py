# --------------------------------------------------------------
# deploy.py — FINAL VERSION WITH Button Fixes for Upload and Run Analysis
# --------------------------------------------------------------
import sys
import subprocess
from pathlib import Path
import time
import streamlit as st

# Paths
REPORT_HTML = Path("interactive_analysis_report.html")
SUPERSTORE_CSV = Path("Superstore.csv")

# Session State Defaults
st.session_state.setdefault("report_ready", REPORT_HTML.is_file())
st.session_state.setdefault("report_visible", False)

# --------------------------------------------------------------
# Run analysis (main.py)
# --------------------------------------------------------------
def run_analysis():
    if REPORT_HTML.is_file():
        REPORT_HTML.unlink()
        st.session_state.report_ready = False

    with st.spinner("Running analysis… please wait"):
        try:
            subprocess.run([sys.executable, "-u", "main.py"], check=True)
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            return

    if REPORT_HTML.is_file():
        st.session_state.report_ready = True
        st.success("✅ Report generated successfully!")
    else:
        st.error("❌ Report generation failed — report file missing.")

# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Intelligent Data Analyst Agent")
st.title("🧭 Intelligent Data Analyst Agent")

tab1, tab2, tab3 = st.tabs(["📊 Report", "💬 Chatbot", "⚙️ Controls"])

# --------------------------------------------------------------
# TAB 3 — Controls (CSV upload & Run Analysis)
# --------------------------------------------------------------
with tab3:
    st.subheader("⚙️ Controls")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        with open(SUPERSTORE_CSV, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("CSV uploaded successfully.")

    if st.button("Run Analysis (main.py)"):
        if not SUPERSTORE_CSV.is_file():
            st.error("Upload CSV first.")
        else:
            run_analysis()

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

# --------------------------------------------------------------
# TAB 1 — Report Viewer
# --------------------------------------------------------------
with tab1:
    st.subheader("📄 Interactive Analysis Report")

    if not st.session_state.report_ready:
        st.info("No report generated yet. Go to Controls → Run Analysis.")
    else:
        with open(REPORT_HTML, "r", encoding="utf-8") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=900, scrolling=True)

        st.download_button(
            "⬇️ Download Report",
            data=REPORT_HTML.read_bytes(),
            file_name="interactive_analysis_report.html",
            mime="text/html",
        )

# --------------------------------------------------------------
# TAB 2 — Chatbot (Stub for future implementation)
# --------------------------------------------------------------
with tab2:
    st.subheader("🤖 Chatbot Assistant")
    st.write("Chatbot functionality is under development.")
