"""
Human-in-the-Loop (HITL) State Management
Tracks checkpoints, approvals, edits, and user feedback
"""

from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum
import json
from datetime import datetime


class Checkpoint(Enum):
    """HITL Checkpoint enumeration"""
    INITIALIZED = "initialized"
    ANALYSIS_PLAN_REVIEW = "analysis_plan_review"
    CHART_REVIEW = "chart_review"
    INSIGHT_REVIEW = "insight_review"
    FINAL_REPORT_APPROVAL = "final_report_approval"
    COMPLETED = "completed"


class ApprovalStatus(Enum):
    """Approval status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


class HITLState(TypedDict):
    """Human-in-the-Loop state management"""
    
    # Current checkpoint
    current_checkpoint: str
    
    # Analysis Plan Review
    analysis_plan: Dict[str, Any]
    analysis_plan_approval: str  # ApprovalStatus
    analysis_plan_feedback: str
    approved_analyses: List[Dict[str, Any]]  # Only approved analyses proceed
    
    # Chart Review
    generated_charts: List[Dict[str, Any]]  # {path, title, type, metadata}
    chart_approvals: Dict[str, str]  # chart_id -> ApprovalStatus
    chart_feedback: Dict[str, str]  # chart_id -> feedback
    approved_charts: List[str]  # chart_ids that are approved
    
    # Insight Review
    generated_insights: List[Dict[str, Any]]  # {id, content, source, sentences}
    insight_edits: Dict[str, str]  # insight_id -> edited_content
    insight_approvals: Dict[str, str]  # insight_id -> ApprovalStatus
    insight_feedback: Dict[str, str]  # insight_id -> feedback
    approved_insights: List[str]  # insight_ids that are approved
    
    # Final Report
    report_preview: str
    report_approval: str  # ApprovalStatus
    report_feedback: str
    final_report_generated: bool
    
    # User feedback and comments
    user_feedback: List[Dict[str, Any]]  # {checkpoint, timestamp, comment}
    
    # Approval flags
    approval_flags: Dict[str, bool]  # Generic approval flags
    
    # Metadata
    session_id: str
    created_at: str
    last_updated: str


class HITLStateManager:
    """Manages HITL state and checkpoint transitions"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"hitl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state = self._create_initial_state()
    
    def _create_initial_state(self) -> HITLState:
        """Create initial HITL state"""
        return HITLState(
            current_checkpoint=Checkpoint.INITIALIZED.value,
            analysis_plan={},
            analysis_plan_approval=ApprovalStatus.PENDING.value,
            analysis_plan_feedback="",
            approved_analyses=[],
            generated_charts=[],
            chart_approvals={},
            chart_feedback={},
            approved_charts=[],
            generated_insights=[],
            insight_edits={},
            insight_approvals={},
            insight_feedback={},
            approved_insights=[],
            report_preview="",
            report_approval=ApprovalStatus.PENDING.value,
            report_feedback="",
            final_report_generated=False,
            user_feedback=[],
            approval_flags={},
            session_id=self.session_id,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def update_checkpoint(self, checkpoint: Checkpoint):
        """Update current checkpoint"""
        self.state['current_checkpoint'] = checkpoint.value
        self.state['last_updated'] = datetime.now().isoformat()
    
    def set_analysis_plan(self, plan: Dict[str, Any]):
        """Set analysis plan and reset approval status"""
        self.state['analysis_plan'] = plan
        self.state['analysis_plan_approval'] = ApprovalStatus.PENDING.value
        self.state['analysis_plan_feedback'] = ""
        self.state['last_updated'] = datetime.now().isoformat()
    
    def approve_analysis_plan(self, approved_analyses: List[Dict[str, Any]], feedback: str = ""):
        """Approve analysis plan with selected analyses"""
        self.state['analysis_plan_approval'] = ApprovalStatus.APPROVED.value
        self.state['approved_analyses'] = approved_analyses
        self.state['analysis_plan_feedback'] = feedback
        self._add_feedback("analysis_plan_review", feedback)
        self.state['last_updated'] = datetime.now().isoformat()
    
    def reject_analysis_plan(self, feedback: str):
        """Reject analysis plan"""
        self.state['analysis_plan_approval'] = ApprovalStatus.REJECTED.value
        self.state['analysis_plan_feedback'] = feedback
        self._add_feedback("analysis_plan_review", feedback)
        self.state['last_updated'] = datetime.now().isoformat()
    
    def add_charts(self, charts: List[Dict[str, Any]]):
        """Add generated charts"""
        self.state['generated_charts'] = charts
        # Initialize all charts as pending
        for chart in charts:
            chart_id = chart.get('id', f"chart_{len(self.state['generated_charts'])}")
            self.state['chart_approvals'][chart_id] = ApprovalStatus.PENDING.value
        self.state['last_updated'] = datetime.now().isoformat()
    
    def approve_chart(self, chart_id: str, feedback: str = ""):
        """Approve a specific chart"""
        self.state['chart_approvals'][chart_id] = ApprovalStatus.APPROVED.value
        if chart_id not in self.state['approved_charts']:
            self.state['approved_charts'].append(chart_id)
        if feedback:
            self.state['chart_feedback'][chart_id] = feedback
            self._add_feedback("chart_review", f"Chart {chart_id}: {feedback}")
        self.state['last_updated'] = datetime.now().isoformat()
    
    def reject_chart(self, chart_id: str, feedback: str = ""):
        """Reject a specific chart"""
        self.state['chart_approvals'][chart_id] = ApprovalStatus.REJECTED.value
        if chart_id in self.state['approved_charts']:
            self.state['approved_charts'].remove(chart_id)
        if feedback:
            self.state['chart_feedback'][chart_id] = feedback
            self._add_feedback("chart_review", f"Chart {chart_id}: {feedback}")
        self.state['last_updated'] = datetime.now().isoformat()
    
    def add_insights(self, insights: List[Dict[str, Any]]):
        """Add generated insights"""
        self.state['generated_insights'] = insights
        # Initialize all insights as pending
        for insight in insights:
            insight_id = insight.get('id', f"insight_{len(self.state['generated_insights'])}")
            self.state['insight_approvals'][insight_id] = ApprovalStatus.PENDING.value
        self.state['last_updated'] = datetime.now().isoformat()
    
    def edit_insight(self, insight_id: str, edited_content: str):
        """Edit an insight (human override)"""
        self.state['insight_edits'][insight_id] = edited_content
        self.state['insight_approvals'][insight_id] = ApprovalStatus.EDITED.value
        if insight_id not in self.state['approved_insights']:
            self.state['approved_insights'].append(insight_id)
        self.state['last_updated'] = datetime.now().isoformat()
    
    def approve_insight(self, insight_id: str, feedback: str = ""):
        """Approve an insight"""
        self.state['insight_approvals'][insight_id] = ApprovalStatus.APPROVED.value
        if insight_id not in self.state['approved_insights']:
            self.state['approved_insights'].append(insight_id)
        if feedback:
            self.state['insight_feedback'][insight_id] = feedback
            self._add_feedback("insight_review", f"Insight {insight_id}: {feedback}")
        self.state['last_updated'] = datetime.now().isoformat()
    
    def reject_insight(self, insight_id: str, feedback: str = ""):
        """Reject an insight"""
        self.state['insight_approvals'][insight_id] = ApprovalStatus.REJECTED.value
        if insight_id in self.state['approved_insights']:
            self.state['approved_insights'].remove(insight_id)
        if feedback:
            self.state['insight_feedback'][insight_id] = feedback
            self._add_feedback("insight_review", f"Insight {insight_id}: {feedback}")
        self.state['last_updated'] = datetime.now().isoformat()
    
    def set_report_preview(self, preview: str):
        """Set final report preview"""
        self.state['report_preview'] = preview
        self.state['report_approval'] = ApprovalStatus.PENDING.value
        self.state['report_feedback'] = ""
        self.state['last_updated'] = datetime.now().isoformat()
    
    def approve_report(self, feedback: str = ""):
        """Approve final report"""
        self.state['report_approval'] = ApprovalStatus.APPROVED.value
        self.state['report_feedback'] = feedback
        self.state['final_report_generated'] = True
        self._add_feedback("final_report_approval", feedback)
        self.state['last_updated'] = datetime.now().isoformat()
    
    def reject_report(self, feedback: str):
        """Reject final report"""
        self.state['report_approval'] = ApprovalStatus.REJECTED.value
        self.state['report_feedback'] = feedback
        self._add_feedback("final_report_approval", feedback)
        self.state['last_updated'] = datetime.now().isoformat()
    
    def _add_feedback(self, checkpoint: str, comment: str):
        """Add user feedback entry"""
        self.state['user_feedback'].append({
            'checkpoint': checkpoint,
            'timestamp': datetime.now().isoformat(),
            'comment': comment
        })
    
    def get_approved_content(self) -> Dict[str, Any]:
        """Get all approved content for final report generation"""
        return {
            'analyses': self.state['approved_analyses'],
            'charts': [c for c in self.state['generated_charts'] 
                      if c.get('id') in self.state['approved_charts']],
            'insights': self._get_approved_insights_with_edits()
        }
    
    def _get_approved_insights_with_edits(self) -> List[Dict[str, Any]]:
        """Get approved insights, using edited versions if available"""
        approved = []
        for insight in self.state['generated_insights']:
            insight_id = insight.get('id')
            if insight_id in self.state['approved_insights']:
                # Use edited version if available, otherwise original
                if insight_id in self.state['insight_edits']:
                    insight_copy = insight.copy()
                    insight_copy['content'] = self.state['insight_edits'][insight_id]
                    insight_copy['edited'] = True
                    approved.append(insight_copy)
                else:
                    approved.append(insight)
        return approved
    
    def is_checkpoint_complete(self, checkpoint: Checkpoint) -> bool:
        """Check if a checkpoint has been completed"""
        if checkpoint == Checkpoint.ANALYSIS_PLAN_REVIEW:
            return self.state['analysis_plan_approval'] == ApprovalStatus.APPROVED.value
        elif checkpoint == Checkpoint.CHART_REVIEW:
            return len(self.state['approved_charts']) > 0
        elif checkpoint == Checkpoint.INSIGHT_REVIEW:
            return len(self.state['approved_insights']) > 0
        elif checkpoint == Checkpoint.FINAL_REPORT_APPROVAL:
            return self.state['report_approval'] == ApprovalStatus.APPROVED.value
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return dict(self.state)
    
    def from_dict(self, data: Dict[str, Any]):
        """Load state from dictionary"""
        self.state.update(data)
    
    def save_to_file(self, filepath: str):
        """Save state to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def load_from_file(self, filepath: str):
        """Load state from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.from_dict(data)

