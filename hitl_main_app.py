"""
Main Human-in-the-Loop (HITL) Streamlit Application
Strict HITL workflow with mandatory checkpoints - NO auto-advance
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

from hitl_state import HITLStateManager, Checkpoint, ApprovalStatus
from hitl_ui import render_hitl_main_ui
from orchestrator import IntelligentAnalysisOrchestrator
from state import IntelligentAnalysisState

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "Superstore.csv"
CHARTS_DIR = "charts"
CHARTS_DIR_HTML = "charts_html"

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'hitl_manager' not in st.session_state:
        st.session_state.hitl_manager = HITLStateManager()
    
    if 'orchestrator' not in st.session_state:
        load_dotenv(dotenv_path=".env", override=True)
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key:
            st.session_state.orchestrator = IntelligentAnalysisOrchestrator(
                groq_api_key, 
                st.session_state.hitl_manager
            )
        else:
            st.session_state.orchestrator = None
    
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    
    if 'analysis_state' not in st.session_state:
        st.session_state.analysis_state = None
    
    if 'orchestrator_result' not in st.session_state:
        st.session_state.orchestrator_result = None
    
    if 'workflow_started' not in st.session_state:
        st.session_state.workflow_started = False

# =============================================================================
# WORKFLOW CONTROL FUNCTIONS
# =============================================================================

def start_analysis_workflow(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """Start the HITL analysis workflow"""
    if st.session_state.orchestrator is None:
        st.error("❌ Orchestrator not initialized. Please check GROQ_API_KEY in .env file.")
        return
    
    try:
        # Reset HITL state for new analysis
        st.session_state.hitl_manager = HITLStateManager()
        st.session_state.orchestrator.hitl = st.session_state.hitl_manager
        
        # Initialize analysis state
        st.session_state.analysis_state = IntelligentAnalysisState(
            dataset=df,
            dataset_info={'name': dataset_name, 'description': f"Analysis of {dataset_name}"},
            analysis_plan={},
            specialized_analyses={},
            chart_paths=[],
            chart_paths_html=[],
            reports={},
            insights=[],
            current_step='initialized',
            error_log=[],
            current_chart_index=0
        )
        
        # Start workflow - this will pause at first checkpoint
        st.session_state.workflow_started = True
        result = st.session_state.orchestrator.analyze_dataset(
            df, 
            dataset_name, 
            wait_for_approval=True  # CRITICAL: Always wait for approval
        )
        
        st.session_state.orchestrator_result = result
        
        # Update analysis state if returned
        if 'state' in result:
            st.session_state.analysis_state = result['state']
        
        return result
        
    except Exception as e:
        st.error(f"❌ Error starting analysis workflow: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def continue_from_checkpoint():
    """Continue workflow from current checkpoint after human approval"""
    if st.session_state.orchestrator is None or st.session_state.analysis_state is None:
        st.error("❌ Cannot continue: Orchestrator or analysis state not initialized.")
        return
    
    try:
        checkpoint = Checkpoint(st.session_state.hitl_manager.state['current_checkpoint'])
        
        # Continue based on checkpoint
        result = st.session_state.orchestrator.continue_from_checkpoint(
            st.session_state.dataset,
            st.session_state.analysis_state
        )
        
        st.session_state.orchestrator_result = result
        
        # Update analysis state if returned
        if 'state' in result:
            st.session_state.analysis_state = result['state']
        
        return result
        
    except Exception as e:
        st.error(f"❌ Error continuing from checkpoint: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_start_screen():
    """Render the initial screen before workflow starts"""
    st.header("🚀 Human-in-the-Loop Data Analytics Agent")
    st.markdown("---")
    
    st.info("""
    **Welcome to the HITL Data Analytics Agent**
    
    This system implements a strict Human-in-the-Loop workflow where:
    - ✅ AI generates analysis plans, charts, and insights
    - ⏸️ **You must approve each step before proceeding**
    - ✏️ You can edit insights and reject content
    - 🛑 **No step will auto-advance without your approval**
    """)
    
    # Dataset selection/upload
    st.subheader("📊 Dataset Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if Path(DATA_PATH).exists():
            st.success(f"✅ Dataset found: `{DATA_PATH}`")
            if st.button("📂 Load Dataset", type="primary", use_container_width=True):
                try:
                    df = pd.read_csv(DATA_PATH, encoding="latin1")
                    # Convert date columns
                    if 'Order Date' in df.columns:
                        df['Order_Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, format='mixed', errors='coerce')
                    if 'Ship Date' in df.columns:
                        df['Ship_Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, format='mixed', errors='coerce')
                    
                    st.session_state.dataset = df
                    st.success(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
        else:
            st.warning(f"⚠️ Dataset not found at `{DATA_PATH}`")
            st.info("Please ensure the dataset file exists or upload a CSV file.")
    
    with col2:
        uploaded_file = st.file_uploader("Or upload CSV", type=['csv'], key="csv_uploader")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, encoding="latin1")
                # Convert date columns
                if 'Order Date' in df.columns:
                    df['Order_Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, format='mixed', errors='coerce')
                if 'Ship Date' in df.columns:
                    df['Ship_Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, format='mixed', errors='coerce')
                
                st.session_state.dataset = df
                st.success(f"✅ Dataset uploaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading uploaded file: {str(e)}")
    
    # Start analysis button
    if st.session_state.dataset is not None:
        st.markdown("---")
        st.subheader("▶️ Start Analysis")
        
        dataset_name = st.text_input(
            "Dataset Name:",
            value="Superstore Sales Dataset",
            key="dataset_name_input"
        )
        
        if st.button("🚀 Start HITL Analysis Workflow", type="primary", use_container_width=True):
            with st.spinner("Initializing analysis workflow..."):
                result = start_analysis_workflow(st.session_state.dataset, dataset_name)
                if result:
                    st.rerun()

def render_workflow_ui():
    """Render the main workflow UI with checkpoint management"""
    hitl = st.session_state.hitl_manager
    current_checkpoint = hitl.state.get('current_checkpoint', Checkpoint.INITIALIZED.value)
    
    # Initialize continue flags if not present
    if 'continue_from_charts' not in st.session_state:
        st.session_state.continue_from_charts = False
    if 'continue_from_insights' not in st.session_state:
        st.session_state.continue_from_insights = False
    
    # Check if we need to continue from a checkpoint
    if current_checkpoint != Checkpoint.INITIALIZED.value and current_checkpoint != Checkpoint.COMPLETED.value:
        # Check if checkpoint is complete and we need to continue
        checkpoint_complete = False
        
        if current_checkpoint == Checkpoint.ANALYSIS_PLAN_REVIEW.value:
            # Auto-continue when analysis plan is approved
            checkpoint_complete = hitl.state.get('analysis_plan_approval') == ApprovalStatus.APPROVED.value
        elif current_checkpoint == Checkpoint.CHART_REVIEW.value:
            # Check if user clicked continue
            checkpoint_complete = st.session_state.continue_from_charts
            if checkpoint_complete:
                st.session_state.continue_from_charts = False
        elif current_checkpoint == Checkpoint.INSIGHT_REVIEW.value:
            # Check if user clicked continue
            checkpoint_complete = st.session_state.continue_from_insights
            if checkpoint_complete:
                st.session_state.continue_from_insights = False
        elif current_checkpoint == Checkpoint.FINAL_REPORT_APPROVAL.value:
            # Auto-continue when report is approved
            checkpoint_complete = hitl.state.get('report_approval') == ApprovalStatus.APPROVED.value
        
        # If checkpoint is complete, continue workflow
        if checkpoint_complete and st.session_state.workflow_started:
            with st.spinner("Continuing workflow from checkpoint..."):
                result = continue_from_checkpoint()
                if result:
                    st.session_state.orchestrator_result = result
                    if 'state' in result:
                        st.session_state.analysis_state = result['state']
                    # Always rerun to update UI
                    st.rerun()
    
    # Render the appropriate checkpoint UI
    render_hitl_main_ui(hitl, st.session_state.orchestrator_result)
    
    # If completed, show download options
    if current_checkpoint == Checkpoint.COMPLETED.value:
        final_report_html = st.session_state.orchestrator_result.get('final_report_html', '') if st.session_state.orchestrator_result else None
        if final_report_html:
            st.markdown("---")
            st.success("✅ Report generation complete! Use the download buttons above to save the report.")
    
    # Add continue buttons for chart and insight checkpoints
    if current_checkpoint == Checkpoint.CHART_REVIEW.value:
        approved_charts = hitl.state.get('approved_charts', [])
        if approved_charts:
            st.markdown("---")
            if st.button("➡️ Continue to Insights", type="primary", use_container_width=True):
                st.session_state.continue_from_charts = True
                st.rerun()
    
    elif current_checkpoint == Checkpoint.INSIGHT_REVIEW.value:
        approved_insights = hitl.state.get('approved_insights', [])
        if approved_insights:
            st.markdown("---")
            if st.button("➡️ Continue to Final Report", type="primary", use_container_width=True):
                st.session_state.continue_from_insights = True
                st.rerun()

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="HITL Data Analytics Agent",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main UI routing
    if not st.session_state.workflow_started or st.session_state.hitl_manager.state['current_checkpoint'] == Checkpoint.INITIALIZED.value:
        render_start_screen()
    else:
        render_workflow_ui()
    
    # Sidebar with controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.session_state.dataset is not None:
            st.success(f"✅ Dataset Loaded")
            st.info(f"Shape: {st.session_state.dataset.shape[0]:,} × {st.session_state.dataset.shape[1]}")
        else:
            st.warning("⚠️ No dataset loaded")
        
        st.markdown("---")
        
        if st.session_state.workflow_started:
            st.info("🔄 Workflow Active")
            
            if st.button("🔄 Reset Workflow", type="secondary"):
                st.session_state.workflow_started = False
                st.session_state.hitl_manager = HITLStateManager()
                st.session_state.analysis_state = None
                st.session_state.orchestrator_result = None
                st.rerun()
        else:
            st.info("⏸️ Workflow Not Started")

if __name__ == "__main__":
    main()

