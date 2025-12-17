import os
import sqlite3
import json
import uuid
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, ValidationError

# ---------------------------
# Load NVIDIA API Key
# ---------------------------
load_dotenv()
nvidia_api = os.getenv("NVIDIA_API_KEY")
assert nvidia_api, "⚠ ERROR: NVIDIA_API_KEY missing"

# ---------------------------
# Streamlit UI Configuration
# ---------------------------
st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.markdown(
    """
<style>
.chat-card {
  background-color: #111827;
  padding: 18px;
  border-radius: 16px;
  margin-bottom: 14px;
  border: 1px solid #1f2937;
}
.sql-card {
  background-color: #0f172a;
  padding: 16px;
  border-radius: 14px;
  margin-top: 10px;
  border: 1px solid #1e293b;
}
.user-msg {color:#38bdf8; font-weight:600;}
.assistant-msg {color:#f9fafb; font-weight:500;}
.section-title {color:#22c55e; font-size:20px; font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("AI Data Analyst Chatbot (Fixed)")

# ---------------------------
# Caches for non-serializables
# ---------------------------
RESULTS_CACHE: Dict[str, pd.DataFrame] = {}
GRAPH_CACHE: Dict[str, Any] = {}

# ---------------------------
# Load CSV into SQLite (cached)
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Superstore.csv", encoding="latin1")

@st.cache_resource
def init_db(data: pd.DataFrame):
    conn = sqlite3.connect("sales_temp.db", check_same_thread=False)
    data.to_sql("sales", conn, if_exists="replace", index=False)
    return conn

data = load_data()
conn = init_db(data)

# ---------------------------
# Initialize NVIDIA LLM
# ---------------------------
llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    temperature=0.05,
    api_key=nvidia_api,
)

# ---------------------------
# Pydantic Schema for ChartConfig
# ---------------------------
class ChartConfig(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter"]
    x_axis_column: str
    y_axis_column: Optional[str] = None
    title_suffix: Optional[str] = Field(default="Data Visualization")

# ---------------------------
# LangGraph State (JSON-serializable only)
# ---------------------------
class State(TypedDict):
    messages: List[Dict[str, str]]
    intent: Optional[str]
    sql_query: Optional[str]
    sql_results_preview: Optional[List[Dict[str, Any]]]
    results_id: Optional[str]  # key into RESULTS_CACHE
    chart_config: Optional[Dict[str, Any]]
    graph_id: Optional[str]  # key into GRAPH_CACHE
    insight: Optional[str]

# ---------------------------
# Node 1 — Intent Detection
# ---------------------------
def intent_node(state: State):
    user_msg = state["messages"][-1]["content"]

    prompt = f"""
Identify user intent as ONLY one of:
SQL
INSIGHT

SQL → If asking for numbers, totals, trends, grouped values, time-based analysis.
INSIGHT → If asking for explanation, reasoning, patterns, or interpretation.

Return ONLY: SQL or INSIGHT.

User query: "{user_msg}"
"""
    reply = llm.invoke([{"role": "user", "content": prompt}])
    intent = getattr(reply, "content", "").strip().upper() or "INSIGHT"
    if intent not in ("SQL", "INSIGHT"):
        intent = "INSIGHT"
    return {**state, "intent": intent}

# ---------------------------
# Node 2 — SQL Generator
# ---------------------------
def sql_node(state: State):
    if state.get("intent") != "SQL":
        return state

    user_msg = state["messages"][-1]["content"]
    columns = ", ".join([f'"{c}"' for c in data.columns])

    prompt = f"""
You are an expert SQLite query generator.
Return ONLY a single valid SQLite SELECT query (no explanations).
TABLE: sales
COLUMNS: {columns}

DATE FORMAT ("Order Date" dd/mm/yyyy → YYYY-MM-DD):
Use: SUBSTR("Order Date",7,4)||'-'||SUBSTR("Order Date",4,2)||'-'||SUBSTR("Order Date",1,2)

User query:
{user_msg}
"""
    reply = llm.invoke([{"role": "user", "content": prompt}])
    sql = getattr(reply, "content", "").strip()

    low = sql.lower()
    if "select" in low:
        sql = sql[low.index("select") :].strip()
    return {**state, "sql_query": sql}

# ---------------------------
# Node 3 — SQL Executor
# ---------------------------
def sql_exec_node(state: State):
    sql = state.get("sql_query")
    if not sql:
        return state

    try:
        df = pd.read_sql_query(sql, conn)
        results_id = str(uuid.uuid4())
        RESULTS_CACHE[results_id] = df
        preview = df.head(100).to_dict(orient="records")
        return {**state, "sql_results_preview": preview, "results_id": results_id}
    except Exception as e:
        err_preview = [{"error": f"SQL Error: {str(e)}"}]
        results_id = str(uuid.uuid4())
        RESULTS_CACHE[results_id] = pd.DataFrame(err_preview)
        return {**state, "sql_results_preview": err_preview, "results_id": results_id}

# ---------------------------
# Node 4 — Visualization Planner
# ---------------------------
def visualization_planner_node(state: State):
    preview = state.get("sql_results_preview")
    results_id = state.get("results_id")
    user_msg = state["messages"][-1]["content"]

    if not preview or len(preview) == 0:
        return {**state, "chart_config": None}

    df = RESULTS_CACHE.get(results_id, pd.DataFrame(preview))
    numeric_cols = list(df.select_dtypes(include="number").columns)
    all_cols = list(df.columns)

    prompt = f"""
Analyze the user's question and the SQL result data preview.
Return a JSON object ONLY with keys:
{{"chart_type": one of ["bar","line","pie","scatter"],
 "x_axis_column": string,
 "y_axis_column": string or null,
 "title_suffix": string}}

Rules:
- Choose x_axis_column as a categorical/grouping column (or "index").
- Choose y_axis_column as a numeric column if available, otherwise null (for pie you must provide values).
- Keep names exactly as column headers in data.

User Question: "{user_msg}"
Columns: {all_cols}
Numeric columns: {numeric_cols}
Data preview (first rows): {json.dumps(preview[:5])}
"""
    reply = llm.invoke([{"role": "user", "content": prompt}])
    raw = getattr(reply, "content", "").strip()

    try:
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
        parsed = json.loads(raw)
        cfg = ChartConfig.parse_obj(parsed)
        cfg_dict = cfg.dict()
        if cfg_dict["x_axis_column"] != "index" and cfg_dict["x_axis_column"] not in all_cols:
            cfg_dict["x_axis_column"] = all_cols[0] if all_cols else "index"
        if cfg_dict.get("y_axis_column") and cfg_dict["y_axis_column"] not in all_cols:
            cfg_dict["y_axis_column"] = numeric_cols[0] if numeric_cols else None
    except (json.JSONDecodeError, ValidationError, Exception):
        x_col = next((c for c in all_cols if c not in numeric_cols), "index")
        y_col = numeric_cols[0] if numeric_cols else None
        chart_type = "bar" if y_col else "pie"
        cfg_dict = {
            "chart_type": chart_type,
            "x_axis_column": x_col,
            "y_axis_column": y_col,
            "title_suffix": "Auto chart"
        }

    return {**state, "chart_config": cfg_dict}

# ---------------------------
# Node 5 — Generate Graph
# ---------------------------
def generate_graph_node(state: State):
    cfg = state.get("chart_config")
    results_id = state.get("results_id")
    if not cfg or not results_id:
        return {**state, "graph_id": None}

    df = RESULTS_CACHE.get(results_id)
    if df is None or df.empty or "error" in df.columns:
        return {**state, "graph_id": None}

    chart_type = cfg.get("chart_type", "bar")
    x_col = cfg.get("x_axis_column")
    y_col = cfg.get("y_axis_column")
    title_suffix = cfg.get("title_suffix") or "Data Visualization"

    fig = None
    try:
        if chart_type == "pie":
            if y_col is None:
                series = df[x_col].value_counts().reset_index()
                series.columns = [x_col, "value"]
                fig = px.pie(series, names=x_col, values="value", title=f"Data Visualization: {title_suffix}")
            else:
                fig = px.pie(df, names=x_col if x_col != "index" else df.columns[0], values=y_col, title=f"Data Visualization: {title_suffix}")
        else:
            x_arg = df.index if x_col == "index" else x_col
            fig = {
                "bar": px.bar,
                "line": px.line,
                "scatter": px.scatter,
            }[chart_type](df, x=x_arg, y=y_col, title=f"Data Visualization: {title_suffix}")
            if chart_type == "bar" and x_col != "index":
                fig.update_layout(xaxis={'categoryorder': 'total descending'})
    except Exception as e:
        print(f"Plotly Generation Error: {e}")
        fig = None

    graph_id = None
    if fig is not None:
        graph_id = str(uuid.uuid4())
        GRAPH_CACHE[graph_id] = fig

    return {**state, "graph_id": graph_id}

# ---------------------------
# Node 6 — Insights
# ---------------------------
def insight_node(state: State):
    user_msg = state["messages"][-1]["content"]
    results_id = state.get("results_id")
    df = RESULTS_CACHE.get(results_id, pd.DataFrame())

    if df.empty or ("error" in df.columns and len(df) > 0):
        err_msg = df.iloc[0]["error"] if "error" in df.columns else "No results to analyze."
        return {**state, "insight": f"Could not analyze data. {err_msg}"}

    # Use to_markdown for clearer tabular data presentation to the LLM
    preview_text = df.head(10).to_markdown(index=False) 
    
    prompt = f"""
You are an expert data analyst. Based on the **USER QUERY** and the **SQL RESULT DATA**, provide a clear and comprehensive business insight summary.

**USER QUERY:**
{user_msg}

**SQL RESULTS (Top 10 rows):**
{preview_text}

You MUST structure your response using the following Markdown headings:
1.  **### Key Findings & Trends** (What are the most important conclusions?)
2.  **### Peaks & Lows / Top Performers** (Identify the highest and lowest values or key categorical leaders.)
3.  **### Business Interpretation** (Provide possible reasons and business implications for the findings.)

Return ONLY the structured text using these headings. Do not include any introductory or concluding remarks outside of the structured sections.
"""
    try:
        reply = llm.invoke([{"role": "user", "content": prompt}])
        insight = getattr(reply, "content", "").strip()
        if not insight:
            insight = "No detailed insight generated by the model."
    except Exception as e:
        print(f"Insight Generation Error: {e}")
        insight = f"Error generating insight: {e}"
        
    return {**state, "insight": insight}
# ---------------------------
# Node 7 — Final Output
# ---------------------------
def output_node(state: State):
    insight_text = state.get('insight', 'No specific insights generated.')
    # The output now uses the structured text generated by insight_node
    out = insight_text 
    
    chart_cfg = state.get("chart_config")
    results_id = state.get("results_id")
    graph_id = state.get("graph_id")

    content_lines = [out]
    if chart_cfg:
        content_lines.append(f"\n\n**Suggested chart**: {chart_cfg.get('chart_type')}, X: {chart_cfg.get('x_axis_column')}, Y: {chart_cfg.get('y_axis_column')}")
    if results_id:
        content_lines.append(f"(results_id: {results_id})")
    if graph_id:
        content_lines.append(f"(graph_id: {graph_id})")

    final_content = "\n\n".join(content_lines)
    new_messages = state["messages"] + [{"role": "assistant", "content": final_content}]
    return {**state, "messages": new_messages}
# ---------------------------
# Build LangGraph Flow
# ---------------------------
graph = StateGraph(State)
graph.add_node("intent", intent_node)
graph.add_node("sql_gen", sql_node)
graph.add_node("sql_exec", sql_exec_node)
graph.add_node("planner", visualization_planner_node)
graph.add_node("graph_gen", generate_graph_node)
graph.add_node("insight", insight_node)
graph.add_node("output", output_node)

graph.set_entry_point("intent")
graph.add_edge("intent", "sql_gen")
graph.add_edge("sql_gen", "sql_exec")
graph.add_edge("sql_exec", "planner")
graph.add_edge("planner", "graph_gen")
graph.add_edge("graph_gen", "insight")
graph.add_edge("insight", "output")
graph.add_edge("output", END)

app = graph.compile()

# ---------------------------
# Session Memory
# ---------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# ---------------------------
# Chat UI
# ---------------------------
st.divider()
st.subheader("Ask Something About Your Data")

col1, col2 = st.columns([8, 2])
with col1:
    user_input = st.text_input("Type your question...", key="user_input_box")
with col2:
    send = st.button("Send")

if send and user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})

    # Invoke the LangGraph flow with a serializable state
    init_state: State = {
        "messages": st.session_state.conversation.copy(),
        "intent": None,
        "sql_query": None,
        "sql_results_preview": None,
        "results_id": None,
        "chart_config": None,
        "graph_id": None,
        "insight": None,
    }
    result = app.invoke(init_state)

    # The result contains state with messages; pull latest assistant message
    assistant_msg = result["messages"][-1]["content"]

    # Append assistant message to session conversation
    st.session_state.conversation.append({"role": "assistant", "content": assistant_msg, "results_id": result.get("results_id"), "graph_id": result.get("graph_id")})

# ---------------------------
# Display Chat Messages and interactive controls
# ---------------------------
for msg in reversed(st.session_state.conversation):
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-card user-msg'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-card assistant-msg'>{msg['content']}</div>", unsafe_allow_html=True)

        # If there is a graph id, retrieve and render the cached figure
        graph_id = msg.get("graph_id")
        if graph_id:
            fig = GRAPH_CACHE.get(graph_id)
            if fig:
                st.plotly_chart(fig, width="stretch")

        # If there are results_id, show a button to display DataFrame from cache
        results_id = msg.get("results_id")
        if results_id:
            if st.button(f"📄 Show SQL Results Table {results_id}", key=f"results_{results_id}"):
                st.markdown("<div class='section-title'>SQL Query Output</div>", unsafe_allow_html=True)
                st.markdown("<div class='sql-card'>", unsafe_allow_html=True)
                
                # Retrieve DataFrame from cache
                df = RESULTS_CACHE.get(results_id, pd.DataFrame())
                
                # Debugging: Check the content of df
                st.write(f"Data for {results_id}:")
                st.write(df)

                if not df.empty:
                    st.dataframe(df, width="stretch")
                else:
                    st.warning("No data available for display.")
                
                st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
