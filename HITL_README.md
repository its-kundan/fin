# Human-in-the-Loop (HITL) Data Analytics Agent

## Overview

This is a **strict Human-in-the-Loop (HITL) workflow** for a Data Analytics Agent. The system ensures that **NO analysis, charts, insights, or reports are finalized without explicit human approval**. Human decisions and edits always override AI output.

## Core Principles

- ✅ **AI generates** analysis plans, charts, insights, and reports
- ⏸️ **Human must approve** each step before proceeding
- ✏️ **Human can edit** insights and reject content
- 🛑 **NO auto-advance** - every step requires explicit approval

## HITL Checkpoints (Mandatory)

The workflow has 4 mandatory checkpoints where execution pauses:

1. **Analysis Plan Review** - Review and select which analyses to execute
2. **Chart Review & Selection** - Review and approve individual charts
3. **Insight Review & Editing** - Review, edit, and approve insights
4. **Final Report Approval** - Review and approve the final report

## System Architecture

### Key Files

- `hitl_main_app.py` - Main Streamlit application entry point
- `hitl_state.py` - HITL state management and checkpoint tracking
- `hitl_ui.py` - UI components for each checkpoint
- `orchestrator.py` - Workflow orchestrator with HITL integration

### State Management

The `HITLStateManager` tracks:
- Current checkpoint
- Approved analyses, charts, and insights
- Human edits to insights
- User feedback at each checkpoint
- Approval flags

## Usage

### 1. Start the Application

```bash
streamlit run hitl_main_app.py
```

### 2. Load Dataset

- The app will look for `Superstore.csv` by default
- Or upload a CSV file through the UI

### 3. Start Analysis Workflow

- Enter a dataset name
- Click "Start HITL Analysis Workflow"
- The workflow will pause at the first checkpoint

### 4. Review and Approve at Each Checkpoint

#### Checkpoint 1: Analysis Plan Review
- Review the proposed analyses
- Select which analyses to execute (checkboxes)
- Add optional feedback
- Click "Approve & Continue"

#### Checkpoint 2: Chart Review
- Review each generated chart
- Approve or reject individual charts
- Add feedback for each chart
- Click "Continue to Insights" when ready

#### Checkpoint 3: Insight Review
- Review each generated insight
- **Edit insights directly** in the text area
- Approve or reject insights
- Click "Continue to Final Report" when ready

#### Checkpoint 4: Final Report Approval
- Review the report preview
- Add optional feedback
- Click "Approve & Generate Report"

## Important Rules

1. **No Auto-Advance**: The workflow will NEVER proceed without human approval
2. **Human Edits Preserved**: Edited insights are never regenerated unless explicitly requested
3. **Approval Required**: At least one item must be approved at each checkpoint to continue
4. **State Persistence**: All approvals and edits are stored in the HITL state

## Workflow Flow

```
1. Generate Analysis Plan
   ↓
   ⏸️ WAIT for approval
   ↓
2. Execute only approved analyses
   ↓
3. Generate Charts
   ↓
   ⏸️ WAIT for chart approval
   ↓
4. Generate Insights
   ↓
   ⏸️ WAIT for insight approval/editing
   ↓
5. Preview Final Report
   ↓
   ⏸️ WAIT for final approval
   ↓
6. Generate Final Report
```

## Configuration

### Environment Variables

Create a `.env` file with:

```
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # Optional, for VLM insights
```

### Data Requirements

- CSV file with data to analyze
- Date columns should be in a parseable format
- Numeric columns for analysis

## Features

### Analysis Plan Review
- View all proposed specialized analyses
- Select which analyses to execute
- View planned visualizations
- Add feedback

### Chart Review
- Individual chart approval/rejection
- Visual chart display (PNG/HTML)
- Per-chart feedback
- Chart type identification

### Insight Review
- **Editable text areas** for each insight
- Sentence-level editing (advanced)
- Source identification (LLAMA/GEMINI)
- Edit preservation

### Final Report
- Preview before generation
- Summary of approved content
- Feedback collection
- Final report download

## Development Notes

- The orchestrator always uses `wait_for_approval=True` in production
- Auto-approve is only available for testing (not in HITL mode)
- State is managed through Streamlit session state
- Checkpoint transitions are controlled by approval status

## Troubleshooting

### Workflow Not Starting
- Check that dataset is loaded
- Verify GROQ_API_KEY is set in `.env`
- Check console for error messages

### Checkpoint Not Advancing
- Ensure you've approved at least one item
- Check approval status in sidebar
- Verify continue button is clicked (for chart/insight checkpoints)

### Edits Not Saving
- Click "Save Edits" button after editing insights
- Or approve the insight to save edits automatically

## License

This is a production-ready HITL system for data analytics. Ensure all API keys are kept secure and not committed to version control.

