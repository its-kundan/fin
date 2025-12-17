# --------------------------------------------------------------
# chatbot_agent.py - Defines the LangGraph agent structure and nodes
# (Includes FIX for structured insight generation)
# --------------------------------------------------------------
import uuid
import json
from typing_extensions import TypedDict
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, ValidationError

import pandas as pd
import plotly.express as px
from langgraph.graph import StateGraph, END

# --- Shared Caches (will be accessed by main_app.py) ---
RESULTS_CACHE: Dict[str, pd.DataFrame] = {}
GRAPH_CACHE: Dict[str, Any] = {}

# ---------------------------
# Pydantic Schema for ChartConfig
# ---------------------------
class ChartConfig(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter"]
    x_axis_column: str
    y_axis_column: Optional[str] = None
    title_suffix: Optional[str] = Field(default="Data Visualization")

# ---------------------------
# LangGraph State
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

# --------------------------------------------------------------
# LangGraph Creation Function
# --------------------------------------------------------------

def create_agent(llm, conn, data):
    """
    Creates and compiles the LangGraph agent, binding it to the necessary resources.
    """
    if llm is None:
        raise ValueError("LLM is not initialized.")

    # --- Node 1: Intent Detection ---
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

    # --- Node 2: SQL Generator ---
    def sql_node(state: State):
        if state.get("intent") != "SQL": return state
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

    # --- Node 3: SQL Executor ---
    def sql_exec_node(state: State):
        sql = state.get("sql_query")
        if not sql: return state

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

    # --- Node 4: Visualization Planner ---
    def visualization_planner_node(state: State):
        preview = state.get("sql_results_preview")
        results_id = state.get("results_id")
        user_msg = state["messages"][-1]["content"]

        if not preview or len(preview) == 0: return {**state, "chart_config": None}

        df = RESULTS_CACHE.get(results_id, pd.DataFrame(preview))
        if "error" in df.columns: return {**state, "chart_config": None}
        
        numeric_cols = list(df.select_dtypes(include="number").columns)
        all_cols = list(df.columns)

        prompt = f"""
Analyze the user's question and the SQL result data preview.
Return a JSON object ONLY with keys:
{{"chart_type": one of ["bar","line","pie","scatter"],
 "x_axis_column": string,
 "y_axis_column": string or null,
 "title_suffix": string}}
... (rest of prompt)
"""
        reply = llm.invoke([{"role": "user", "content": prompt}])
        raw = getattr(reply, "content", "").strip()

        try:
            if raw.startswith("```"): raw = raw.strip("`").strip()
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
                "chart_type": chart_type, "x_axis_column": x_col, "y_axis_column": y_col, "title_suffix": "Auto chart"
            }

        return {**state, "chart_config": cfg_dict}

    # --- Node 5: Generate Graph ---
    def generate_graph_node(state: State):
        cfg = state.get("chart_config")
        results_id = state.get("results_id")
        if not cfg or not results_id: return {**state, "graph_id": None}

        df = RESULTS_CACHE.get(results_id)
        if df is None or df.empty or "error" in df.columns: return {**state, "graph_id": None}

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
                    "bar": px.bar, "line": px.line, "scatter": px.scatter,
                }[chart_type](df, x=x_arg, y=y_col, title=f"Data Visualization: {title_suffix}")
                if chart_type == "bar" and x_col != "index":
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
        except Exception:
            fig = None

        graph_id = None
        if fig is not None:
            graph_id = str(uuid.uuid4())
            GRAPH_CACHE[graph_id] = fig

        return {**state, "graph_id": graph_id}

    # --- Node 6: Insights (FIXED FOR STRUCTURED OUTPUT) ---
    def insight_node(state: State):
        user_msg = state["messages"][-1]["content"]
        results_id = state.get("results_id")
        df = RESULTS_CACHE.get(results_id, pd.DataFrame())

        if df.empty or ("error" in df.columns and len(df) > 0):
            err_msg = df.iloc[0]["error"] if "error" in df.columns else "No results to analyze. Run an SQL query first."
            return {**state, "insight": f"Could not analyze data. {err_msg}"}
        
        # Use to_markdown and limit rows to ensure clear tabular data format
        preview_text = df.head(10).to_markdown(index=False)
        
        prompt = f"""
You are an expert data analyst. Based on the **USER QUERY** and the **SQL RESULT DATA**, provide a clear and comprehensive business insight summary.

**USER QUERY:**
{user_msg}

**SQL RESULTS (Top {len(df.head(10))} rows of {len(df)} total):**
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
            if not insight: insight = "No detailed insight generated by the model."
        except Exception as e:
            insight = f"Error generating insight: {e}"
            
        return {**state, "insight": insight}

    # --- Node 7: Final Output (Adjusted to remove generic heading) ---
    def output_node(state: State):
        insight_text = state.get("insight") or ""
        chart_cfg = state.get("chart_config")
        results_id = state.get("results_id")
        graph_id = state.get("graph_id")

        # The insight_text already contains the desired headings (### Key Findings, etc.)
        content_lines = [insight_text] 
        
        if chart_cfg: content_lines.append(f"\n\n**Suggested chart**: {chart_cfg.get('chart_type')}, X: {chart_cfg.get('x_axis_column')}, Y: {chart_cfg.get('y_axis_column')}")
        if results_id: content_lines.append(f"(results_id: {results_id})")
        if graph_id: content_lines.append(f"(graph_id: {graph_id})")

        out = "\n\n".join(content_lines)
        new_messages = state["messages"] + [{"role": "assistant", "content": out}]
        return {**state, "messages": new_messages}


    # ---------------------------
    # Build and Compile LangGraph Flow
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

    return graph.compile()