# --------------------------------------------------------------
# Home.py — Report Viewer and Data Management Page
# --------------------------------------------------------------
import sys
import subprocess
from pathlib import Path
import time
from bs4 import BeautifulSoup
import os
import streamlit as st
import streamlit.components.v1 as components

# Import utilities and shared resources from the new core file
from data_agent_core import (
    load_data, init_db, data, conn, SQLITE_DB_PATH, 
    SUPERSTORE_CSV, REPORT_HTML, CHARTS_HTML_DIR
)

# --------------------------------------------------------------
# Configuration and Setup
# --------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Data Analyst - Report Viewer")
st.title("🧭 Intelligent Data Analyst Agent")

# --------------------------------------------------------------
# Session State Defaults
# --------------------------------------------------------------
st.session_state.setdefault("report_ready", REPORT_HTML.is_file())
st.session_state.setdefault("report_visible", False)
st.session_state.setdefault("last_uploaded_file_id", None)

# --------------------------------------------------------------
# --- Report Handlers (Logic copied from original deploy.py) ---
# --------------------------------------------------------------

def run_analysis_script():
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

def load_full_report_html():
    if not REPORT_HTML.is_file(): return "<h3>No report generated yet.</h3>"

    html_content = REPORT_HTML.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "") or iframe.get("data-src", "")
        if not src: continue
        clean_src = src.replace("file://", "").replace("\\", "/").lstrip("/")
        chart_name = Path(clean_src).name
        chart_path = CHARTS_HTML_DIR / chart_name
        
        if chart_path.is_file():
            chart_html = chart_path.read_text(encoding="utf-8")
            iframe["srcdoc"] = chart_html
            iframe["src"] = ""
            iframe["width"] = "100%"
            iframe["height"] = "600px" 
        else:
            error_div = soup.new_tag("div")
            error_div.string = f"⚠️ Chart file not found: {chart_name}"
            iframe.replace_with(error_div)
    return str(soup)

# --------------------------------------------------------------
# --- UI Layout ---
# --------------------------------------------------------------

with st.sidebar:
    st.subheader("⚙️ Data & Analysis Controls")

    uploaded = st.file_uploader("Upload CSV", type=['csv'], key="csv_uploader")
    
    if uploaded and uploaded != st.session_state.get("last_uploaded_file_id"):
        st.session_state["last_uploaded_file_id"] = uploaded
        
        if 'conn' in globals() and conn: conn.close()
        st.cache_resource.clear()
        st.cache_data.clear()

        with open(SUPERSTORE_CSV, "wb") as f: f.write(uploaded.getbuffer())
        
        if os.path.exists(SQLITE_DB_PATH):
            try:
                time.sleep(0.5)
                os.remove(SQLITE_DB_PATH)
            except Exception: pass

        st.success("CSV uploaded successfully. Reloading...")
        time.sleep(1)
        st.rerun()

    if st.button("Run Analysis (main.py)"):
        if not SUPERSTORE_CSV.is_file():
            st.error("Upload a CSV first.")
        else:
            run_analysis_script()

    st.markdown("---")
    
    st.info("Navigate to the **🤖 Chatbot** page in the sidebar for interactive queries.")


# Main Body - Report Viewer
st.header("📄 Interactive Analysis Report Viewer")
button_label = '✅ Show Report' if not st.session_state.report_visible else '❌ Hide Report'
if st.button(button_label):
    st.session_state.report_visible = not st.session_state.report_visible
    st.rerun()

if st.session_state.report_visible:
    if st.session_state.report_ready:
        final_html = load_full_report_html()
        components.html(final_html, height=900, scrolling=True)
        st.download_button("⬇️ Download Report", data=REPORT_HTML.read_bytes(), file_name="interactive_analysis_report.html", mime="text/html")
    else:
        st.info("No report generated yet. Use the Controls in the sidebar to run analysis.")