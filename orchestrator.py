from planner import IntelligentAnalysisPlanner
from insights import EnhancedInsightGenerator
from state import IntelligentAnalysisState
from executor import IntelligentAnalysisExecutor
from visualizer import ComprehensiveVisualizationGenerator
from autoviz_charts import AutovizChartGenerator
from hitl_state import HITLStateManager, Checkpoint, ApprovalStatus

from typing import TypedDict, List, Dict, Any, Optional, Tuple
import pandas as pd
import shutil

# =============================================================================
# INTELLIGENT WORKFLOW ORCHESTRATOR WITH HITL
# =============================================================================

class IntelligentAnalysisOrchestrator:
    """Main orchestrator for intelligent multi-agent data analysis with HITL"""
    
    def __init__(self, groq_api_key: str, hitl_state_manager: HITLStateManager = None):
        self.groq_api_key = groq_api_key
        self.planner = IntelligentAnalysisPlanner(groq_api_key)
        self.insight_generator = EnhancedInsightGenerator(groq_api_key)
        self.hitl = hitl_state_manager or HITLStateManager()
    
    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str = "Dataset", 
                       wait_for_approval: bool = True) -> Dict[str, Any]:
        """
        Intelligent analysis workflow with HITL checkpoints.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            wait_for_approval: If True, pauses at checkpoints for human approval
        
        Returns:
            Dict with results and checkpoint status
        """
        
        print(f"Starting intelligent analysis of {dataset_name}")
        print(f" Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Initialize state
        state = IntelligentAnalysisState(
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
        
        try:
            # =================================================================
            # CHECKPOINT 1: Analysis Plan Review
            # =================================================================
            print("\n" + "="*80)
            print(" CHECKPOINT 1: Creating Analysis Plan")
            print("="*80)
            
            state['analysis_plan'] = self.planner.create_analysis_plan(df, state['dataset_info'])
            self.hitl.set_analysis_plan(state['analysis_plan'])
            self.hitl.update_checkpoint(Checkpoint.ANALYSIS_PLAN_REVIEW)
            
            if wait_for_approval:
                print("\n⏸️  PAUSED: Waiting for human approval of analysis plan...")
                print("   Use the UI to review and approve/reject the plan.")
                return {
                    'status': 'paused',
                    'checkpoint': Checkpoint.ANALYSIS_PLAN_REVIEW.value,
                    'state': state,
                    'hitl_state': self.hitl.to_dict(),
                    'message': 'Analysis plan generated. Please review and approve in the UI.'
                }
            
            # If not waiting, auto-approve all analyses (for testing only)
            # CRITICAL: This should only be used for testing, not in production HITL mode
            if not wait_for_approval:
                approved_analyses = state['analysis_plan'].get('specialized_analyses', [])
                self.hitl.approve_analysis_plan(approved_analyses, "Auto-approved (no HITL - testing only)")
            
            # =================================================================
            # Execute only approved analyses
            # =================================================================
            approved_analyses = self.hitl.state['approved_analyses']
            if not approved_analyses:
                return {
                    'status': 'error',
                    'message': 'No analyses approved. Cannot proceed.',
                    'hitl_state': self.hitl.to_dict()
                }
            
            # Create filtered plan with only approved analyses
            filtered_plan = {
                'specialized_analyses': approved_analyses,
                'visualizations': state['analysis_plan'].get('visualizations', [])
            }
            
            print("\n Executing approved specialized analyses...")
            state['specialized_analyses'] = IntelligentAnalysisExecutor.execute_analysis_plan(
                df, filtered_plan)
            state['current_step'] = 'specialized_analyses_complete'
            
            # =================================================================
            # CHECKPOINT 2: Chart Review
            # =================================================================
            print("\n" + "="*80)
            print(" CHECKPOINT 2: Generating Charts")
            print("="*80)
            
            print(" Creating intelligent visualizations...")
            import os
            if os.path.exists("charts"):
                shutil.rmtree("charts")
            os.makedirs("charts", exist_ok=True)
            
            state['chart_paths'], state['chart_paths_html'] = ComprehensiveVisualizationGenerator.create_intelligent_charts(
                df, filtered_plan, state['specialized_analyses'])
            
            # Generate AutoViz charts
            print(" Generating additional AutoViz charts...")
            autoviz_generator = AutovizChartGenerator()
            autoviz_charts = autoviz_generator.generate_autoviz_charts(df, dataset_name)
            state['chart_paths'].extend(autoviz_charts)
            
            # Format charts for HITL review
            charts_for_review = []
            for i, chart_path in enumerate(state['chart_paths']):
                charts_for_review.append({
                    'id': f"chart_{i}",
                    'path': chart_path,
                    'title': f"Chart {i+1}",
                    'type': self._infer_chart_type(chart_path),
                    'metadata': {}
                })
            
            self.hitl.add_charts(charts_for_review)
            self.hitl.update_checkpoint(Checkpoint.CHART_REVIEW)
            
            if wait_for_approval:
                print("\n⏸️  PAUSED: Waiting for human approval of charts...")
                print(f"   {len(charts_for_review)} charts generated. Please review and approve in the UI.")
                return {
                    'status': 'paused',
                    'checkpoint': Checkpoint.CHART_REVIEW.value,
                    'state': state,
                    'hitl_state': self.hitl.to_dict(),
                    'message': f'{len(charts_for_review)} charts generated. Please review and approve.'
                }
            
            # Auto-approve all charts if not waiting (testing only)
            if not wait_for_approval:
                for chart in charts_for_review:
                    self.hitl.approve_chart(chart['id'], "Auto-approved (testing only)")
            
            # =================================================================
            # CHECKPOINT 3: Insight Review
            # =================================================================
            print("\n" + "="*80)
            print(" CHECKPOINT 3: Generating Insights")
            print("="*80)
            
            print(" Generating comprehensive insights...")
            raw_insights = self.insight_generator.generate_comprehensive_insights(state)
            
            # Format insights for HITL review (sentence-level editable)
            insights_for_review = []
            for i, insight_content in enumerate(raw_insights):
                # Split into sentences for editing
                sentences = self._split_into_sentences(insight_content)
                insights_for_review.append({
                    'id': f"insight_{i}",
                    'content': insight_content,
                    'source': 'LLAMA' if '[LLAMA' in insight_content else 'GEMINI',
                    'sentences': sentences,
                    'original': insight_content
                })
            
            self.hitl.add_insights(insights_for_review)
            self.hitl.update_checkpoint(Checkpoint.INSIGHT_REVIEW)
            
            if wait_for_approval:
                print("\n⏸️  PAUSED: Waiting for human approval/editing of insights...")
                print(f"   {len(insights_for_review)} insights generated. Please review, edit, and approve in the UI.")
                return {
                    'status': 'paused',
                    'checkpoint': Checkpoint.INSIGHT_REVIEW.value,
                    'state': state,
                    'hitl_state': self.hitl.to_dict(),
                    'message': f'{len(insights_for_review)} insights generated. Please review and approve.'
                }
            
            # Auto-approve all insights if not waiting (testing only)
            if not wait_for_approval:
                for insight in insights_for_review:
                    self.hitl.approve_insight(insight['id'], "Auto-approved (testing only)")
            
            # =================================================================
            # CHECKPOINT 4: Final Report Approval
            # =================================================================
            print("\n" + "="*80)
            print(" CHECKPOINT 4: Generating Final Report Preview")
            print("="*80)
            
            # Generate report preview from approved content
            approved_content = self.hitl.get_approved_content()
            report_preview = self._generate_report_preview(approved_content, state)
            
            self.hitl.set_report_preview(report_preview)
            self.hitl.update_checkpoint(Checkpoint.FINAL_REPORT_APPROVAL)
            
            if wait_for_approval:
                print("\n⏸️  PAUSED: Waiting for human approval of final report...")
                print("   Report preview generated. Please review and approve in the UI.")
                return {
                    'status': 'paused',
                    'checkpoint': Checkpoint.FINAL_REPORT_APPROVAL.value,
                    'state': state,
                    'hitl_state': self.hitl.to_dict(),
                    'report_preview': report_preview,
                    'message': 'Final report preview generated. Please review and approve.'
                }
            
            # Auto-approve report if not waiting (testing only)
            if not wait_for_approval:
                self.hitl.approve_report("Auto-approved (testing only)")
            
            # =================================================================
            # Final Report Generation
            # =================================================================
            if self.hitl.state['report_approval'] == ApprovalStatus.APPROVED.value:
                print("\n Generating final report from approved content...")
                final_report = self._generate_final_report(approved_content, state)
                self.hitl.update_checkpoint(Checkpoint.COMPLETED)
                
                results = {
                    'status': 'completed',
                    'dataset_info': {
                        'name': dataset_name,
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
                    },
                    'analysis_plan': state['analysis_plan'],
                    'approved_analyses': approved_content['analyses'],
                    'approved_charts': approved_content['charts'],
                    'approved_insights': approved_content['insights'],
                    'final_report': final_report,
                    'errors': state['error_log'],
                    'hitl_state': self.hitl.to_dict()
                }
                
                print(f"\n✅ Intelligent analysis complete!")
                print(f"   Approved analyses: {len(approved_content['analyses'])}")
                print(f"   Approved charts: {len(approved_content['charts'])}")
                print(f"   Approved insights: {len(approved_content['insights'])}")
                
                return results
            else:
                return {
                    'status': 'error',
                    'message': 'Final report not approved. Cannot generate report.',
                    'hitl_state': self.hitl.to_dict()
                }
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e),
                'dataset_info': {'name': dataset_name, 'shape': df.shape},
                'hitl_state': self.hitl.to_dict()
            }
    
    def continue_from_checkpoint(self, df: pd.DataFrame, state: IntelligentAnalysisState) -> Dict[str, Any]:
        """Continue workflow from current checkpoint after human approval"""
        checkpoint = Checkpoint(self.hitl.state['current_checkpoint'])
        
        if checkpoint == Checkpoint.ANALYSIS_PLAN_REVIEW:
            # Continue from analysis execution
            return self._continue_after_analysis_plan(df, state)
        elif checkpoint == Checkpoint.CHART_REVIEW:
            # Continue from insight generation
            return self._continue_after_charts(df, state)
        elif checkpoint == Checkpoint.INSIGHT_REVIEW:
            # Continue from report generation
            return self._continue_after_insights(df, state)
        elif checkpoint == Checkpoint.FINAL_REPORT_APPROVAL:
            # Generate final report
            return self._continue_after_report_preview(df, state)
        else:
            return {'status': 'error', 'message': f'Unknown checkpoint: {checkpoint.value}'}
    
    def _continue_after_analysis_plan(self, df: pd.DataFrame, state: IntelligentAnalysisState) -> Dict[str, Any]:
        """Continue workflow after analysis plan approval"""
        approved_analyses = self.hitl.state['approved_analyses']
        if not approved_analyses:
            return {'status': 'error', 'message': 'No analyses approved'}
        
        filtered_plan = {
            'specialized_analyses': approved_analyses,
            'visualizations': state['analysis_plan'].get('visualizations', [])
        }
        
        print("\n Executing approved specialized analyses...")
        state['specialized_analyses'] = IntelligentAnalysisExecutor.execute_analysis_plan(df, filtered_plan)
        state['current_step'] = 'specialized_analyses_complete'
        
        # Move to chart generation checkpoint
        return self._continue_to_chart_generation(df, state)
    
    def _continue_to_chart_generation(self, df: pd.DataFrame, state: IntelligentAnalysisState) -> Dict[str, Any]:
        """Generate charts after analyses are complete"""
        print("\n" + "="*80)
        print(" CHECKPOINT 2: Generating Charts")
        print("="*80)
        
        # Get approved analyses for chart generation
        approved_analyses = self.hitl.state['approved_analyses']
        filtered_plan = {
            'specialized_analyses': approved_analyses,
            'visualizations': state['analysis_plan'].get('visualizations', [])
        }
        
        print(" Creating intelligent visualizations...")
        import os
        import shutil
        if os.path.exists("charts"):
            shutil.rmtree("charts")
        os.makedirs("charts", exist_ok=True)
        
        state['chart_paths'], state['chart_paths_html'] = ComprehensiveVisualizationGenerator.create_intelligent_charts(
            df, filtered_plan, state['specialized_analyses'])
        
        # Generate AutoViz charts
        print(" Generating additional AutoViz charts...")
        autoviz_generator = AutovizChartGenerator()
        autoviz_charts = autoviz_generator.generate_autoviz_charts(df, state['dataset_info']['name'])
        state['chart_paths'].extend(autoviz_charts)
        
        # Format charts for HITL review
        charts_for_review = []
        for i, chart_path in enumerate(state['chart_paths']):
            charts_for_review.append({
                'id': f"chart_{i}",
                'path': chart_path,
                'title': f"Chart {i+1}",
                'type': self._infer_chart_type(chart_path),
                'metadata': {}
            })
        
        self.hitl.add_charts(charts_for_review)
        self.hitl.update_checkpoint(Checkpoint.CHART_REVIEW)
        
        print("\n⏸️  PAUSED: Waiting for human approval of charts...")
        print(f"   {len(charts_for_review)} charts generated. Please review and approve in the UI.")
        
        return {
            'status': 'paused',
            'checkpoint': Checkpoint.CHART_REVIEW.value,
            'state': state,
            'hitl_state': self.hitl.to_dict(),
            'message': f'{len(charts_for_review)} charts generated. Please review and approve.'
        }
    
    def _continue_after_charts(self, df: pd.DataFrame, state: IntelligentAnalysisState) -> Dict[str, Any]:
        """Continue workflow after chart approval - generate insights"""
        print("\n" + "="*80)
        print(" CHECKPOINT 3: Generating Insights")
        print("="*80)
        
        print(" Generating comprehensive insights...")
        raw_insights = self.insight_generator.generate_comprehensive_insights(state)
        
        # Format insights for HITL review (sentence-level editable)
        insights_for_review = []
        for i, insight_content in enumerate(raw_insights):
            # Split into sentences for editing
            sentences = self._split_into_sentences(insight_content)
            insights_for_review.append({
                'id': f"insight_{i}",
                'content': insight_content,
                'source': 'LLAMA' if '[LLAMA' in insight_content else 'GEMINI',
                'sentences': sentences,
                'original': insight_content
            })
        
        self.hitl.add_insights(insights_for_review)
        self.hitl.update_checkpoint(Checkpoint.INSIGHT_REVIEW)
        
        print("\n⏸️  PAUSED: Waiting for human approval/editing of insights...")
        print(f"   {len(insights_for_review)} insights generated. Please review, edit, and approve in the UI.")
        
        return {
            'status': 'paused',
            'checkpoint': Checkpoint.INSIGHT_REVIEW.value,
            'state': state,
            'hitl_state': self.hitl.to_dict(),
            'message': f'{len(insights_for_review)} insights generated. Please review and approve.'
        }
    
    def _continue_after_insights(self, df: pd.DataFrame, state: IntelligentAnalysisState) -> Dict[str, Any]:
        """Continue workflow after insight approval - generate report preview"""
        print("\n" + "="*80)
        print(" CHECKPOINT 4: Generating Final Report Preview")
        print("="*80)
        
        # Generate report preview from approved content
        approved_content = self.hitl.get_approved_content()
        report_preview = self._generate_report_preview(approved_content, state)
        
        self.hitl.set_report_preview(report_preview)
        self.hitl.update_checkpoint(Checkpoint.FINAL_REPORT_APPROVAL)
        
        print("\n⏸️  PAUSED: Waiting for human approval of final report...")
        print("   Report preview generated. Please review and approve in the UI.")
        
        return {
            'status': 'paused',
            'checkpoint': Checkpoint.FINAL_REPORT_APPROVAL.value,
            'state': state,
            'hitl_state': self.hitl.to_dict(),
            'report_preview': report_preview,
            'message': 'Final report preview generated. Please review and approve.'
        }
    
    def _continue_after_report_preview(self, df: pd.DataFrame, state: IntelligentAnalysisState) -> Dict[str, Any]:
        """Generate final report after approval"""
        print("\n Generating final report from approved content...")
        approved_content = self.hitl.get_approved_content()
        final_report = self._generate_final_report(approved_content, state)
        self.hitl.update_checkpoint(Checkpoint.COMPLETED)
        
        print(f"\n✅ Intelligent analysis complete!")
        print(f"   Approved analyses: {len(approved_content['analyses'])}")
        print(f"   Approved charts: {len(approved_content['charts'])}")
        print(f"   Approved insights: {len(approved_content['insights'])}")
        
        return {
            'status': 'completed',
            'final_report': final_report,
            'state': state,
            'hitl_state': self.hitl.to_dict(),
            'dataset_info': {
                'name': state['dataset_info']['name'],
                'shape': state['dataset'].shape,
                'columns': state['dataset'].columns.tolist(),
            },
            'approved_analyses': approved_content['analyses'],
            'approved_charts': approved_content['charts'],
            'approved_insights': approved_content['insights'],
        }
    
    def _infer_chart_type(self, chart_path: str) -> str:
        """Infer chart type from filename"""
        path_lower = chart_path.lower()
        if 'correlation' in path_lower:
            return 'correlation'
        elif 'time_series' in path_lower or 'line' in path_lower:
            return 'time_series'
        elif 'bar' in path_lower:
            return 'bar'
        elif 'scatter' in path_lower:
            return 'scatter'
        elif 'histogram' in path_lower:
            return 'histogram'
        else:
            return 'unknown'
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for editing"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_report_preview(self, approved_content: Dict[str, Any], state: IntelligentAnalysisState) -> str:
        """Generate report preview from approved content"""
        preview = f"""
# Analysis Report Preview

## Dataset Information
- Name: {state['dataset_info']['name']}
- Shape: {state['dataset'].shape[0]:,} rows × {state['dataset'].shape[1]} columns

## Approved Analyses
{len(approved_content['analyses'])} analyses were approved and executed.

## Approved Charts
{len(approved_content['charts'])} charts were approved for inclusion.

## Approved Insights
{len(approved_content['insights'])} insights were approved.

### Insights Content:
"""
        for insight in approved_content['insights']:
            content = insight.get('content', '')
            if insight.get('edited'):
                content += " [EDITED BY HUMAN]"
            preview += f"\n{content}\n---\n"
        
        return preview
    
    def _generate_final_report(self, approved_content: Dict[str, Any], state: IntelligentAnalysisState) -> str:
        """Generate final report from approved content"""
        # This would call the report generation module
        # For now, return the preview as final report
        return self._generate_report_preview(approved_content, state)
    
    def print_analysis_plan(self, results: Dict[str, Any]):
        """Pretty print the LLM-generated analysis plan"""
        plan = results.get('analysis_plan', {})
        
        print("\n" + "="*80)
        print(" LLM-GENERATED ANALYSIS PLAN")
        print("="*80)
        
        
        # Specialized Analyses
        analyses = plan.get('specialized_analyses', [])
        if analyses:
            print("\n SPECIALIZED ANALYSES PLANNED:")
            for i, analysis in enumerate(analyses, 1):
                print(f"  {i}. {analysis['function']} on {analysis['columns']}")
                print(f"     → {analysis['justification']}")
        
    
    def print_insights(self, results: Dict[str, Any]):
        """Pretty print the generated insights"""
        print("\n" + "="*80)
        print(" INTELLIGENT INSIGHTS")
        print("="*80)
        
        ins = []
        for i, insight in enumerate(results.get('insights', []), 1):
            print(f"\n{i}. {insight}")
            ins.append(insight)
        return ins