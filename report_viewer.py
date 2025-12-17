# --------------------------------------------------------------
# report_viewer.py - Utility functions and UI for the Report Viewer
# --------------------------------------------------------------
import sys
import subprocess
from pathlib import Path
from bs4 import BeautifulSoup
import streamlit as st
import streamlit.components.v1 as components

# --- Paths (Must match paths in main_app.py) ---
REPORT_HTML = Path("interactive_analysis_report.html")
CHARTS_HTML_DIR = Path("charts_html")
SUPERSTORE_CSV = Path("Superstore.csv")

def run_analysis_script(report_ready_state_key):
    """Runs the external main.py analysis script to generate the report."""
    if REPORT_HTML.is_file():
        REPORT_HTML.unlink()
        st.session_state[report_ready_state_key] = False

    with st.spinner("Running analysis… please wait"):
        try:
            # Assumes 'main.py' is the external analysis script that generates the report
            subprocess.run([sys.executable, "-u", "main.py"], check=True)
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}. Check if 'main.py' exists and its dependencies are installed.")
            return

    if REPORT_HTML.is_file():
        st.session_state[report_ready_state_key] = True
        st.success("✅ Report generated successfully!")
    else:
        st.error("❌ Report generation failed — report file missing.")

def load_full_report_html():
    """Loads the main report HTML and embeds iframe charts using srcdoc."""
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

def render_report_tab(report_ready_key, report_visible_key):
    """Renders the Report Viewer tab UI."""
    st.header("📄 Interactive Analysis Report Viewer")
    button_label = '✅ Show Report' if not st.session_state[report_visible_key] else '❌ Hide Report'
    if st.button(button_label):
        st.session_state[report_visible_key] = not st.session_state[report_visible_key]

    if st.session_state[report_visible_key]:
        if st.session_state[report_ready_key]:
            final_html = load_full_report_html()
            components.html(final_html, height=900, scrolling=True)
            if REPORT_HTML.is_file():
                st.download_button("⬇️ Download Report", data=REPORT_HTML.read_bytes(), file_name="interactive_analysis_report.html", mime="text/html")
        else:
            st.info("No report generated yet. Use the Controls in the sidebar to run analysis.")