"""
Human-in-the-Loop (HITL) Streamlit UI Components
Provides review and approval interfaces for each checkpoint
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from hitl_state import HITLStateManager, Checkpoint, ApprovalStatus
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from PIL import Image
import os


def render_checkpoint_indicator(current_checkpoint: str):
    """Render visual checkpoint progress indicator"""
    checkpoints = [
        Checkpoint.ANALYSIS_PLAN_REVIEW.value,
        Checkpoint.CHART_REVIEW.value,
        Checkpoint.INSIGHT_REVIEW.value,
        Checkpoint.FINAL_REPORT_APPROVAL.value,
        Checkpoint.COMPLETED.value
    ]
    
    current_idx = checkpoints.index(current_checkpoint) if current_checkpoint in checkpoints else 0
    
    cols = st.columns(len(checkpoints))
    for i, checkpoint in enumerate(checkpoints):
        with cols[i]:
            if i < current_idx:
                st.success(f"✓ {checkpoint.replace('_', ' ').title()}")
            elif i == current_idx:
                st.info(f"⏸ {checkpoint.replace('_', ' ').title()}")
            else:
                st.empty()


def render_analysis_plan_review(hitl: HITLStateManager) -> bool:
    """
    Render analysis plan review UI.
    Returns True if user has approved and wants to continue.
    """
    st.header("📋 Checkpoint 1: Analysis Plan Review")
    st.markdown("---")
    
    plan = hitl.state['analysis_plan']
    
    if not plan:
        st.warning("⚠️ Analysis plan not generated yet. Please start the workflow.")
        return False
    
    # Display analysis plan
    st.subheader("Generated Analysis Plan")
    
    # Specialized Analyses
    analyses = plan.get('specialized_analyses', [])
    selected_analyses = []
    
    if analyses:
        st.markdown("### Specialized Analyses")
        st.info(f"**{len(analyses)} analyses proposed.** Select which analyses to execute:")
        
        for i, analysis in enumerate(analyses):
            col1, col2 = st.columns([1, 4])
            with col1:
                analysis_id = f"analysis_{i}"
                # Get current selection state
                key = f"select_{analysis_id}"
                if key not in st.session_state:
                    st.session_state[key] = True  # Default to selected
                
                selected = st.checkbox(
                    f"Analysis {i+1}",
                    value=st.session_state[key],
                    key=key
                )
                st.session_state[key] = selected
                if selected:
                    selected_analyses.append(analysis)
            
            with col2:
                st.markdown(f"**Function:** `{analysis.get('function', 'N/A')}`")
                st.markdown(f"**Columns:** {', '.join(analysis.get('columns', []))}")
                st.markdown(f"**Justification:** {analysis.get('justification', 'N/A')}")
                if analysis.get('parameters'):
                    st.markdown(f"**Parameters:** {analysis.get('parameters')}")
                st.markdown("---")
    else:
        st.warning("No specialized analyses in the plan.")
    
    # Visualizations
    visualizations = plan.get('visualizations', [])
    if visualizations:
        st.markdown("### Visualizations")
        st.info(f"**{len(visualizations)} visualizations planned.**")
        for i, viz in enumerate(visualizations):
            st.markdown(f"**{i+1}. {viz.get('title', 'Chart')}**")
            st.markdown(f"- Type: `{viz.get('chart_type', 'N/A')}`")
            st.markdown(f"- Columns: {', '.join(viz.get('columns', []))}")
            st.markdown("---")
    
    # Feedback
    st.subheader("Feedback (Optional)")
    feedback = st.text_area(
        "Add comments or feedback about the analysis plan:",
        value=hitl.state.get('analysis_plan_feedback', ''),
        key="plan_feedback"
    )
    
    # Approval buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    approval_status = hitl.state.get('analysis_plan_approval', ApprovalStatus.PENDING.value)
    
    with col1:
        if approval_status != ApprovalStatus.APPROVED.value:
            if st.button("✅ Approve & Continue", type="primary", use_container_width=True):
                if not selected_analyses:
                    st.error("⚠️ Please select at least one analysis to approve.")
                else:
                    hitl.approve_analysis_plan(selected_analyses, feedback)
                    st.success("✅ Analysis plan approved! Continuing workflow...")
                    st.rerun()
        else:
            st.success("✅ Approved")
    
    with col2:
        if approval_status != ApprovalStatus.REJECTED.value:
            if st.button("❌ Reject", use_container_width=True):
                hitl.reject_analysis_plan(feedback)
                st.error("❌ Analysis plan rejected.")
                st.rerun()
        else:
            st.error("❌ Rejected")
    
    # Show current status
    if approval_status == ApprovalStatus.APPROVED.value:
        st.success("✅ Analysis plan has been approved. Workflow will continue automatically.")
        return True
    elif approval_status == ApprovalStatus.REJECTED.value:
        st.error("❌ Analysis plan has been rejected. Please regenerate or restart workflow.")
        return False
    
    return False


def render_chart_review(hitl: HITLStateManager) -> bool:
    """
    Render chart review UI with individual approval.
    Returns True if at least one chart is approved and user wants to continue.
    """
    st.header("📊 Checkpoint 2: Chart Review & Selection")
    st.markdown("---")
    
    charts = hitl.state.get('generated_charts', [])
    
    if not charts:
        st.warning("No charts generated yet.")
        return False
    
    st.info(f"**{len(charts)} charts generated.** Please review and approve the charts you want to include in the final report.")
    
    # Display charts in grid
    approved_count = len(hitl.state.get('approved_charts', []))
    st.metric("Approved Charts", approved_count, len(charts))
    
    # Chart review interface
    for i, chart in enumerate(charts):
        chart_id = chart.get('id', f"chart_{i}")
        approval_status = hitl.state.get('chart_approvals', {}).get(chart_id, ApprovalStatus.PENDING.value)
        
        with st.expander(f"📈 {chart.get('title', 'Chart')} - {approval_status.upper()}", expanded=(approval_status == ApprovalStatus.PENDING.value)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display chart
                chart_path = chart.get('path', '')
                if chart_path and Path(chart_path).exists():
                    try:
                        if chart_path.endswith('.html'):
                            st.components.v1.html(open(chart_path, 'r').read(), height=500, scrolling=True)
                        else:
                            img = Image.open(chart_path)
                            st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load chart: {e}")
                else:
                    st.warning(f"Chart file not found: {chart_path}")
                
                st.markdown(f"**Type:** {chart.get('type', 'unknown')}")
                st.markdown(f"**Path:** `{chart_path}`")
            
            with col2:
                # Approval controls
                st.markdown("### Approval")
                
                feedback_key = f"chart_feedback_{chart_id}"
                feedback = st.text_area(
                    "Feedback:",
                    value=hitl.state.get('chart_feedback', {}).get(chart_id, ''),
                    key=feedback_key,
                    height=100
                )
                
                col_approve, col_reject = st.columns(2)
                
                with col_approve:
                    if st.button("✅ Approve", key=f"approve_{chart_id}", use_container_width=True):
                        hitl.approve_chart(chart_id, feedback)
                        st.rerun()
                
                with col_reject:
                    if st.button("❌ Reject", key=f"reject_{chart_id}", use_container_width=True):
                        hitl.reject_chart(chart_id, feedback)
                        st.rerun()
                
                # Status indicator
                if approval_status == ApprovalStatus.APPROVED.value:
                    st.success("✅ Approved")
                elif approval_status == ApprovalStatus.REJECTED.value:
                    st.error("❌ Rejected")
                else:
                    st.info("⏳ Pending")
        
        st.markdown("---")
    
    # Continue button - handled in main app
    approved_charts = hitl.state.get('approved_charts', [])
    if approved_charts:
        st.success(f"✅ {len(approved_charts)} chart(s) approved.")
        st.info("💡 Click 'Continue to Insights' button below to proceed.")
    else:
        st.warning("⚠️ Please approve at least one chart to continue.")
    
    return len(approved_charts) > 0


def render_insight_review(hitl: HITLStateManager) -> bool:
    """
    Render insight review UI with sentence-level editing.
    Returns True if at least one insight is approved and user wants to continue.
    """
    st.header("💡 Checkpoint 3: Insight Review & Editing")
    st.markdown("---")
    
    insights = hitl.state.get('generated_insights', [])
    
    if not insights:
        st.warning("No insights generated yet.")
        return False
    
    st.info(f"**{len(insights)} insights generated.** Please review, edit if needed, and approve the insights for the final report.")
    
    approved_count = len(hitl.state.get('approved_insights', []))
    st.metric("Approved Insights", approved_count, len(insights))
    
    # Insight review interface
    for i, insight in enumerate(insights):
        insight_id = insight.get('id', f"insight_{i}")
        approval_status = hitl.state.get('insight_approvals', {}).get(insight_id, ApprovalStatus.PENDING.value)
        
        # Get edited content if available, otherwise original
        current_content = hitl.state.get('insight_edits', {}).get(insight_id, insight.get('content', ''))
        
        with st.expander(f"💭 Insight {i+1} - {approval_status.upper()}", expanded=(approval_status == ApprovalStatus.PENDING.value)):
            # Source indicator
            source = insight.get('source', 'UNKNOWN')
            if source == 'LLAMA':
                st.markdown("**Source:** 🦙 Llama 3.3 Analysis")
            elif source == 'GEMINI':
                st.markdown("**Source:** 🔮 Gemini VLM Analysis")
            
            # Editable content area
            st.markdown("### Content (Editable)")
            edited_content = st.text_area(
                "Edit the insight content:",
                value=current_content,
                height=200,
                key=f"insight_edit_{insight_id}",
                help="You can edit this insight. Your edits will override the AI-generated content."
            )
            
            # Sentence-level editing (optional, for advanced users)
            with st.expander("📝 Sentence-Level Editing (Advanced)"):
                sentences = insight.get('sentences', [])
                if sentences:
                    for j, sentence in enumerate(sentences):
                        edited_sentence = st.text_input(
                            f"Sentence {j+1}:",
                            value=sentence,
                            key=f"sentence_{insight_id}_{j}"
                        )
                        # Note: In a full implementation, you'd reconstruct the content from edited sentences
                else:
                    st.info("Sentence breakdown not available for this insight.")
            
            # Approval controls
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("✅ Approve", key=f"approve_insight_{insight_id}", use_container_width=True):
                    # Save edits if content changed
                    if edited_content != insight.get('content', ''):
                        hitl.edit_insight(insight_id, edited_content)
                    else:
                        hitl.approve_insight(insight_id)
                    st.rerun()
            
            with col2:
                if st.button("❌ Reject", key=f"reject_insight_{insight_id}", use_container_width=True):
                    hitl.reject_insight(insight_id)
                    st.rerun()
            
            with col3:
                # Save edits button
                if edited_content != current_content:
                    if st.button("💾 Save Edits", key=f"save_insight_{insight_id}", use_container_width=True):
                        hitl.edit_insight(insight_id, edited_content)
                        st.success("Edits saved!")
                        st.rerun()
            
            # Status indicator
            if approval_status == ApprovalStatus.APPROVED.value:
                st.success("✅ Approved")
            elif approval_status == ApprovalStatus.EDITED.value:
                st.info("✏️ Edited & Approved")
            elif approval_status == ApprovalStatus.REJECTED.value:
                st.error("❌ Rejected")
            else:
                st.info("⏳ Pending")
        
        st.markdown("---")
    
    # Continue button - handled in main app
    approved_insights = hitl.state.get('approved_insights', [])
    if approved_insights:
        st.success(f"✅ {len(approved_insights)} insight(s) approved.")
        st.info("💡 Click 'Continue to Final Report' button below to proceed.")
    else:
        st.warning("⚠️ Please approve at least one insight to continue.")
    
    return len(approved_insights) > 0


def render_final_report_approval(hitl: HITLStateManager) -> bool:
    """
    Render final report preview and approval UI with section-by-section review.
    Returns True if report is approved.
    """
    st.header("📄 Checkpoint 4: Final Report Approval")
    st.markdown("---")
    
    report_structure = hitl.state.get('report_structure', {})
    report_preview = hitl.state.get('report_preview', '')
    
    if not report_structure and not report_preview:
        st.warning("Report preview not generated yet.")
        return False
    
    st.info("**Final report structure generated.** Please review and edit each section before approving.")
    
    # Show summary of approved content
    st.subheader("Approved Content Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Analyses", len(hitl.state.get('approved_analyses', [])))
    
    with col2:
        st.metric("Charts", len(hitl.state.get('approved_charts', [])))
    
    with col3:
        st.metric("Insights", len(hitl.state.get('approved_insights', [])))
    
    st.markdown("---")
    
    # Section-by-section review if structure exists
    if report_structure:
        sections = report_structure.get('sections', [])
        
        if sections:
            st.subheader("📝 Report Sections Review & Editing")
            st.markdown("Review and edit each section. Your edits will override AI-generated content.")
            
            for section in sections:
                section_id = section.get('id', '')
                section_title = section.get('title', 'Section')
                section_type = section.get('type', 'text')
                approval_status = hitl.state.get('report_section_approvals', {}).get(section_id, ApprovalStatus.PENDING.value)
                
                # Get current content (edited if available)
                current_content = hitl.state.get('report_section_edits', {}).get(section_id, section.get('content', ''))
                
                with st.expander(f"📄 {section_title} - {approval_status.upper()}", expanded=(approval_status == ApprovalStatus.PENDING.value)):
                    # Display and edit section based on type
                    if section_type == 'text':
                        edited_content = st.text_area(
                            f"Edit {section_title}:",
                            value=current_content if isinstance(current_content, str) else str(current_content),
                            height=200,
                            key=f"report_section_edit_{section_id}",
                            help="Edit this section. Your edits will override AI-generated content."
                        )
                        
                        if edited_content != current_content:
                            if st.button(f"💾 Save Edits", key=f"save_section_{section_id}"):
                                hitl.edit_report_section(section_id, edited_content)
                                st.success("Edits saved!")
                                st.rerun()
                    
                    elif section_type == 'list':
                        # For lists, show as editable text area (user can edit as text)
                        list_text = '\n'.join(current_content) if isinstance(current_content, list) else str(current_content)
                        edited_text = st.text_area(
                            f"Edit {section_title} (one item per line):",
                            value=list_text,
                            height=200,
                            key=f"report_section_edit_{section_id}",
                            help="Edit list items, one per line."
                        )
                        
                        if edited_text != list_text:
                            if st.button(f"💾 Save Edits", key=f"save_section_{section_id}"):
                                # Convert back to list
                                edited_list = [line.strip() for line in edited_text.split('\n') if line.strip()]
                                hitl.edit_report_section(section_id, edited_list)
                                st.success("Edits saved!")
                                st.rerun()
                        else:
                            # Display as list
                            st.markdown("**Current Content:**")
                            for item in (current_content if isinstance(current_content, list) else [current_content]):
                                st.markdown(f"- {item}")
                    
                    elif section_type == 'charts':
                        # For charts, show chart catalog with editable explanations
                        st.markdown("**Chart Catalog:**")
                        chart_catalog = current_content if isinstance(current_content, list) else []
                        
                        for chart in chart_catalog:
                            chart_id = chart.get('id', '')
                            chart_title = chart.get('title', 'Chart')
                            chart_explanation = chart.get('explanation', '')
                            
                            with st.container():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.markdown(f"**{chart_title}**")
                                    # Try to show chart preview
                                    chart_path = chart.get('path', '')
                                    if chart_path and Path(chart_path).exists():
                                        try:
                                            from PIL import Image
                                            img = Image.open(chart_path)
                                            st.image(img, use_container_width=True)
                                        except:
                                            st.info("Chart preview unavailable")
                                
                                with col2:
                                    edited_explanation = st.text_area(
                                        f"Explanation for {chart_title}:",
                                        value=chart_explanation,
                                        height=100,
                                        key=f"chart_explanation_{chart_id}",
                                        help="Edit the chart explanation."
                                    )
                                    
                                    if edited_explanation != chart_explanation:
                                        chart['explanation'] = edited_explanation
                                        hitl.edit_report_section(section_id, chart_catalog)
                                        st.success("Chart explanation updated!")
                    
                    # Section approval controls
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Approve Section", key=f"approve_section_{section_id}", use_container_width=True):
                            hitl.approve_report_section(section_id)
                            st.rerun()
                    
                    with col2:
                        if st.button(f"❌ Reject Section", key=f"reject_section_{section_id}", use_container_width=True):
                            hitl.reject_report_section(section_id)
                            st.rerun()
                    
                    # Status indicator
                    if approval_status == ApprovalStatus.APPROVED.value:
                        st.success("✅ Section Approved")
                    elif approval_status == ApprovalStatus.EDITED.value:
                        st.info("✏️ Section Edited & Approved")
                    elif approval_status == ApprovalStatus.REJECTED.value:
                        st.error("❌ Section Rejected")
                    else:
                        st.info("⏳ Section Pending")
                    
                    st.markdown("---")
            
            # Check if all sections are approved
            section_approvals = hitl.state.get('report_section_approvals', {})
            approved_sections = [sid for sid, status in section_approvals.items() 
                               if status in [ApprovalStatus.APPROVED.value, ApprovalStatus.EDITED.value]]
            
            st.metric("Approved Sections", len(approved_sections), len(sections))
    
    # Show simple preview if no structure
    if not report_structure and report_preview:
        st.subheader("Report Preview")
        st.markdown(report_preview)
    
    # Feedback
    st.subheader("Feedback (Optional)")
    feedback = st.text_area(
        "Add comments or feedback about the final report:",
        value=hitl.state.get('report_feedback', ''),
        key="report_feedback"
    )
    
    # Final approval buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    approval_status = hitl.state.get('report_approval', ApprovalStatus.PENDING.value)
    
    with col1:
        if approval_status != ApprovalStatus.APPROVED.value:
            if st.button("✅ Approve & Generate Report", type="primary", use_container_width=True):
                # Check if all sections approved (if structure exists)
                if report_structure:
                    sections = report_structure.get('sections', [])
                    section_approvals = hitl.state.get('report_section_approvals', {})
                    all_approved = all(
                        section_approvals.get(s.get('id'), '') in [ApprovalStatus.APPROVED.value, ApprovalStatus.EDITED.value]
                        for s in sections
                    )
                    if not all_approved:
                        st.error("⚠️ Please approve all sections before generating the final report.")
                    else:
                        hitl.approve_report(feedback)
                        st.success("Report approved! Generating final report...")
                        st.rerun()
                else:
                    hitl.approve_report(feedback)
                    st.success("Report approved! Generating final report...")
                    st.rerun()
        else:
            st.success("✅ Approved")
    
    with col2:
        if approval_status != ApprovalStatus.REJECTED.value:
            if st.button("❌ Reject", use_container_width=True):
                hitl.reject_report(feedback)
                st.error("Report rejected. Please review and regenerate.")
                st.rerun()
        else:
            st.error("❌ Rejected")
    
    # Show current status
    if approval_status == ApprovalStatus.APPROVED.value:
        st.success("✅ Final report has been approved. Report will be generated.")
        return True
    elif approval_status == ApprovalStatus.REJECTED.value:
        st.error("❌ Final report has been rejected. Please review and regenerate.")
        return False
    
    return False


def render_completed_report(hitl: HITLStateManager, final_report: str = None, final_report_html: str = None):
    """Render completed report view"""
    st.header("✅ Analysis Complete!")
    st.markdown("---")
    
    # Show HTML report if available
    report_html = hitl.state.get('final_report_html', '') or final_report_html
    
    if report_html:
        st.success("🎉 Final report has been generated successfully!")
        
        # Display HTML report
        st.subheader("📄 Interactive HTML Report")
        st.components.v1.html(report_html, height=800, scrolling=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download HTML Report",
                data=report_html,
                file_name="interactive_analysis_report.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col2:
            if final_report:
                st.download_button(
                    label="📥 Download Text Report",
                    data=final_report,
                    file_name="analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        # Show report file path
        from pathlib import Path
        report_file = Path(__file__).parent / "interactive_analysis_report.html"
        if report_file.exists():
            st.info(f"📁 Report saved to: `{report_file}`")
    
    elif final_report:
        st.markdown("### Text Report")
        st.markdown(final_report)
        
        st.download_button(
            label="📥 Download Report",
            data=final_report,
            file_name="analysis_report.txt",
            mime="text/plain"
        )
    else:
        st.info("Final report will be displayed here.")
    
    # Show summary
    st.markdown("---")
    st.subheader("📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analyses", len(hitl.state.get('approved_analyses', [])))
    
    with col2:
        st.metric("Charts", len(hitl.state.get('approved_charts', [])))
    
    with col3:
        st.metric("Insights", len(hitl.state.get('approved_insights', [])))
    
    with col4:
        sections = hitl.state.get('report_structure', {}).get('sections', [])
        st.metric("Report Sections", len(sections))


def render_hitl_main_ui(hitl: HITLStateManager, orchestrator_result: Dict[str, Any] = None):
    """
    Main HITL UI router that displays the appropriate checkpoint interface.
    """
    current_checkpoint = hitl.state.get('current_checkpoint', Checkpoint.INITIALIZED.value)
    
    # Render checkpoint indicator
    render_checkpoint_indicator(current_checkpoint)
    
    st.markdown("---")
    
    # Route to appropriate checkpoint UI
    if current_checkpoint == Checkpoint.ANALYSIS_PLAN_REVIEW.value:
        render_analysis_plan_review(hitl)
    
    elif current_checkpoint == Checkpoint.CHART_REVIEW.value:
        render_chart_review(hitl)
    
    elif current_checkpoint == Checkpoint.INSIGHT_REVIEW.value:
        render_insight_review(hitl)
    
    elif current_checkpoint == Checkpoint.FINAL_REPORT_APPROVAL.value:
        render_final_report_approval(hitl)
    
    elif current_checkpoint == Checkpoint.COMPLETED.value:
        final_report = orchestrator_result.get('final_report', '') if orchestrator_result else None
        final_report_html = orchestrator_result.get('final_report_html', '') if orchestrator_result else None
        render_completed_report(hitl, final_report, final_report_html)
    
    else:
        st.info("Workflow initialized. Waiting to start analysis...")
    
    # Sidebar with state info
    with st.sidebar:
        st.header("HITL Status")
        st.markdown(f"**Checkpoint:** {current_checkpoint}")
        st.markdown(f"**Session ID:** {hitl.session_id}")
        
        st.markdown("### Approval Status")
        st.markdown(f"Analysis Plan: {hitl.state.get('analysis_plan_approval', 'pending')}")
        st.markdown(f"Charts: {len(hitl.state.get('approved_charts', []))} approved")
        st.markdown(f"Insights: {len(hitl.state.get('approved_insights', []))} approved")
        st.markdown(f"Report: {hitl.state.get('report_approval', 'pending')}")
        
        # Reset button (for development)
        if st.button("🔄 Reset HITL State", type="secondary"):
            hitl.__init__(hitl.session_id)
            st.rerun()

