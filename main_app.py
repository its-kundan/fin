# --------------------------------------------------------------
# main_app.py - Main application file for UI routing and setup
# (Includes fix for Streamlit callback error)
# --------------------------------------------------------------
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# --- Import separated modules ---
from chatbot_agent import create_agent, RESULTS_CACHE, GRAPH_CACHE, State
from report_viewer import run_analysis_script, render_report_tab

# ---------------------------
# Configuration and Paths
# ---------------------------
load_dotenv()
nvidia_api = os.getenv("NVIDIA_API_KEY")

SUPERSTORE_CSV = Path("Superstore.csv")
SQLITE_DB_PATH = "sales_temp.db" 
REPORT_HTML_PATH = Path("interactive_analysis_report.html")

# ---------------------------
# Streamlit UI Configuration & Styling
# ---------------------------
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.markdown(
    """
<style>
.chat-card {background-color: #111827; padding: 18px; border-radius: 16px; margin-bottom: 14px; border: 1px solid #1f2937;}
.sql-card {background-color: #0f172a; padding: 16px; border-radius: 14px; margin-top: 10px; border: 1px solid #1e293b;}
.user-msg {color:#38bdf8; font-weight:600;}
.assistant-msg {color:#f9fafb; font-weight:500;}
.section-title {color:#22c55e; font-size:20px; font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)
st.title("🧭 Intelligent Data Analyst Agent")

# ---------------------------
# Session State Defaults
# ---------------------------
st.session_state.setdefault("report_ready", SUPERSTORE_CSV.is_file() and REPORT_HTML_PATH.is_file())
st.session_state.setdefault("report_visible", False)
st.session_state.setdefault("last_uploaded_file_id", None)
st.session_state.setdefault("conversation", [])
st.session_state.setdefault("user_input_box", "")


# ---------------------------
# Data & DB Initialization (Cached)
# ---------------------------
@st.cache_data(show_spinner="Loading and preparing data...")
def load_data():
    if not SUPERSTORE_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(SUPERSTORE_CSV, encoding="latin1")

@st.cache_resource(show_spinner="Initializing SQLite database...")
def init_db(data: pd.DataFrame):
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    if not data.empty:
        data.to_sql("sales", conn, if_exists="replace", index=False)
    return conn

data = load_data()
conn = init_db(data)

# ---------------------------
# Initialize LLM and Agent
# ---------------------------
llm = None
app = None
if nvidia_api:
    try:
        llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            temperature=0.05,
            api_key=nvidia_api,
        )
        app = create_agent(llm, conn, data)
    except Exception as e:
        llm = None
        st.error(f"LLM initialization failed: {e}")
else:
    st.error("⚠ ERROR: NVIDIA_API_KEY environment variable is missing. Chatbot features will be disabled.")


# --------------------------------------------------------------
# --- Callback Function for Chat Submission ---
# --------------------------------------------------------------

def handle_chat_submit(agent_app, agent_data, agent_conn):
    """Handles the chat submission logic and clears the input box."""
    user_input = st.session_state.user_input_box
    
    if not user_input or agent_app is None:
        return

    # 1. Store the user message
    st.session_state.conversation.append({"role": "user", "content": user_input})

    # 2. Clear input state immediately (critical for fixing the original API error)
    st.session_state.user_input_box = "" 

    # 3. Prepare initial state for LangGraph
    init_state: State = {
        "messages": st.session_state.conversation.copy(),
        "intent": None, "sql_query": None, "sql_results_preview": None, 
        "results_id": None, "chart_config": None, "graph_id": None, 
        "insight": None,
    }
    
    # 4. Run the agent synchronously (this will block the UI, but ensures stable execution)
    with st.spinner("🤖 Thinking and analyzing data..."):
        try:
            result = agent_app.invoke(init_state)

            assistant_msg_content = result["messages"][-1]["content"]

            st.session_state.conversation.append({
                "role": "assistant", 
                "content": assistant_msg_content, 
                "results_id": result.get("results_id"), 
                "graph_id": result.get("graph_id")
            })
        except Exception as e:
            st.session_state.conversation.append({
                "role": "assistant", 
                "content": f"### ❌ Error\nAgent execution failed: {e}",
                "results_id": None, "graph_id": None
            })
    
    # 5. The natural Streamlit rerun (triggered by the session state update) displays the results.


# --------------------------------------------------------------
# --- Sidebar UI (Data Management) ---
# --------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Data & Analysis Controls")

    uploaded = st.file_uploader("Upload CSV", type=['csv'], key="csv_uploader")
    
    if uploaded and uploaded != st.session_state.get("last_uploaded_file_id"):
        # File upload logic (remains the same)
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
            run_analysis_script("report_ready")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.conversation = []
        st.success("Chat history cleared.")


# --------------------------------------------------------------
# --- Chatbot Render Function ---
# --------------------------------------------------------------

def render_chatbot_tab():
    st.header("🤖 AI Data Analyst Chatbot")
    
    if data.empty:
        st.warning("Please upload a CSV file in the sidebar to load the data and start chatting.")
    elif app is None:
        st.error("Chatbot agent failed to initialize. Check the LLM error message above.")
    else:
        # Chat UI Input Section
        col1, col2 = st.columns([8, 2])
        with col1:
            st.text_input(
                "Type your question...", 
                key="user_input_box", 
                label_visibility="collapsed",
                on_change=handle_chat_submit,
                args=(app, data, conn) 
            )
        with col2:
            st.button(
                "Send", 
                use_container_width=True, 
                on_click=handle_chat_submit,
                args=(app, data, conn)
            )

        # Display Chat Messages and interactive controls
        for msg in reversed(st.session_state.conversation):
            
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-card user-msg'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-card assistant-msg'>{msg['content']}</div>", unsafe_allow_html=True)

                graph_id = msg.get("graph_id")
                if graph_id:
                    fig = GRAPH_CACHE.get(graph_id)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                results_id = msg.get("results_id")
                if results_id:
                    with st.expander("📄 View SQL Results & Details"):
                        st.markdown("<div class='section-title'>SQL Query Output</div>", unsafe_allow_html=True)
                        st.markdown("<div class='sql-card'>", unsafe_allow_html=True)
                        
                        df = RESULTS_CACHE.get(results_id, pd.DataFrame())
                        
                        if not df.empty and "error" not in df.columns:
                            st.dataframe(df, use_container_width=True)
                        elif "error" in df.columns:
                            st.error(f"SQL Error: {df.iloc[0]['error']}")
                        else:
                            st.warning("No data available for display.")
                        
                        st.markdown("</div>", unsafe_allow_html=True)

            st.divider()

# --------------------------------------------------------------
# --- Main UI Routing ---
# --------------------------------------------------------------

# This radio button creates the tabs and routes the view
tab_selection = st.radio("Select View:", ["Report", "Chatbot"], index=0, horizontal=True)
st.write("---") 

if tab_selection == "Report":
    render_report_tab("report_ready", "report_visible")
elif tab_selection == "Chatbot":
    render_chatbot_tab()