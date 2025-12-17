# Intelligent Data Analytics Agent with Human-in-the-Loop (HITL)

A comprehensive, production-ready data analytics system that combines AI-powered analysis with strict human oversight. This system provides intelligent data analysis, automated visualization, and business insights while ensuring human authority over all AI-generated outputs.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Components](#components)
- [HITL Workflow](#hitl-workflow)
- [Chatbot Features](#chatbot-features)
- [API Keys Setup](#api-keys-setup)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project is an **Intelligent Data Analytics Agent** that provides:

- **AI-Powered Analysis**: Automated data analysis using LLMs (Groq/Llama) and Vision Models (Google Gemini)
- **Strict Human-in-the-Loop**: Mandatory checkpoints requiring human approval at every step
- **Comprehensive Visualizations**: Automated chart generation with multiple visualization types
- **Business Insights**: AI-generated insights from both textual analysis and visual chart analysis
- **Interactive Chatbot**: Stateful chatbot for querying charts and data
- **Interactive Reports**: HTML-based interactive reports with chart catalogs

### Core Principle

**Human authority always overrides AI output.** The system ensures that no analysis, chart, insight, or report is finalized without explicit human approval.

## ✨ Key Features

### 1. Human-in-the-Loop (HITL) Workflow
- ✅ **4 Mandatory Checkpoints** with pause points
- ✅ **No Auto-Advance** - every step requires explicit approval
- ✅ **Editable Insights** - sentence-level editing capability
- ✅ **Individual Chart Approval** - approve/reject charts one by one
- ✅ **State Persistence** - all approvals and edits are tracked

### 2. Intelligent Analysis
- **LLM-Powered Planning**: AI generates comprehensive analysis plans
- **Specialized Analytics**: 
  - Time series decomposition
  - Cohort analysis
  - Customer segmentation (RFM)
  - Correlation network analysis
  - Anomaly detection
  - Distribution comparison
- **Automated Visualization**: 10+ chart types automatically generated
- **Multi-Model Insights**: Combines Llama (text) and Gemini (vision) insights

### 3. Visualization Capabilities
- **Chart Types**: Correlation heatmaps, time series, bar charts, histograms, scatter plots, box plots, violin plots, bubble charts, treemaps, and more
- **Dual Format**: Both PNG (static) and HTML (interactive) versions
- **AutoViz Integration**: Automated chart generation for quick insights
- **Intelligent Selection**: Charts selected based on data characteristics

### 4. Interactive Components
- **Streamlit UI**: Modern, responsive web interface
- **Stateful Chatbot**: Persistent chat session for chart analysis
- **Interactive Reports**: HTML reports with tabbed navigation
- **Chart Catalog**: Sequential chart explanations with VLM insights

### 5. Data Processing
- **CSV Support**: Load and analyze CSV files
- **SQLite Integration**: Automatic database creation for querying
- **Date Handling**: Automatic date column parsing and normalization
- **Data Profiling**: Automatic dataset profiling and schema detection

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  HITL App    │  │  Main App    │  │  Chatbot     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     IntelligentAnalysisOrchestrator                    │  │
│  │  - Workflow Management                                │  │
│  │  - Checkpoint Control                                 │  │
│  │  - State Management                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Analysis Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Planner    │  │  Executor    │  │  Visualizer  │     │
│  │  (LLM)       │  │  (Analytics) │  │  (Charts)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │  Insights    │  │  AutoViz     │                       │
│  │  (LLM+VLM)   │  │  Generator   │                       │
│  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CSV Files  │  │   SQLite DB  │  │   Charts     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- API keys for:
  - Groq API (required)
  - Google Gemini API (optional, for VLM insights)

### Step 1: Clone the Repository

```bash
git clone https://github.com/its-kundan/fin.git
cd fin
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important**: The `.env` file is already in `.gitignore` and will not be committed to version control.

### Step 4: Verify Installation

```bash
python -c "import streamlit; import pandas; import plotly; print('Installation successful!')"
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key for Groq (Llama models) |
| `GEMINI_API_KEY` | Optional | API key for Google Gemini (VLM insights) |

### File Paths

Default file paths (can be modified in respective files):

- **Data File**: `Superstore.csv` (or upload via UI)
- **Charts Directory**: `charts/` (auto-created)
- **HTML Charts**: `charts_html/` (auto-created)
- **Database**: `sales_data.db` (auto-created)
- **Reports**: `analysis_report.txt`, `interactive_analysis_report.html`

## 🚀 Usage

### 1. HITL Workflow (Recommended)

The Human-in-the-Loop workflow provides full control over the analysis process.

```bash
streamlit run hitl_main_app.py
```

**Workflow Steps:**

1. **Load Dataset**: Upload CSV or use default `Superstore.csv`
2. **Start Workflow**: Click "Start HITL Analysis Workflow"
3. **Checkpoint 1 - Analysis Plan Review**:
   - Review proposed analyses
   - Select which analyses to execute
   - Approve to continue
4. **Checkpoint 2 - Chart Review**:
   - Review each generated chart
   - Approve/reject individual charts
   - Continue to insights
5. **Checkpoint 3 - Insight Review**:
   - Review generated insights
   - **Edit insights directly** (human override)
   - Approve to continue
6. **Checkpoint 4 - Final Report Approval**:
   - Review report preview
   - Approve to generate final report
7. **Download Report**: Get the final interactive HTML report

### 2. Automated Analysis (Non-HITL)

For automated analysis without checkpoints:

```bash
python main.py
```

This will:
- Generate charts automatically
- Run all analyses
- Generate insights
- Create reports

**Note**: This mode auto-approves everything (for testing/quick analysis only).

### 3. Interactive Chatbot

Stateful chatbot for querying charts:

```bash
python chatbot_session.py
```

Features:
- Loads all charts from the charts directory
- Maintains context across questions
- Answers questions based on visual chart analysis
- Uses Gemini Vision for chart understanding

### 4. Other Applications

#### Main App (Multi-tab Interface)
```bash
streamlit run main_app.py
```

#### Report Viewer
```bash
streamlit run Home.py
```

#### Simple UI
```bash
streamlit run ui.py
```

## 🧩 Components

### Core Components

#### 1. **HITL System**
- `hitl_main_app.py` - Main HITL Streamlit application
- `hitl_state.py` - State management for HITL workflow
- `hitl_ui.py` - UI components for checkpoints

#### 2. **Orchestration**
- `orchestrator.py` - Main workflow orchestrator
- `planner.py` - LLM-powered analysis planning
- `executor.py` - Executes specialized analyses
- `state.py` - System state definition

#### 3. **Analysis & Insights**
- `anl_funcs.py` - Specialized analytics functions
- `insights.py` - Insight generation (Llama + Gemini)
- `visualizer.py` - Chart generation
- `autoviz_charts.py` - AutoViz integration

#### 4. **Reporting**
- `report.py` - HTML report generation
- `report_html.py` - Report HTML utilities
- `summarize.py` - Summary generation

#### 5. **Chatbot**
- `chatbot_session.py` - Stateful Gemini chatbot
- `chatbot_agent.py` - Chatbot agent implementation
- `chatbot_config.py` - Chatbot configuration

#### 6. **UI Applications**
- `main_app.py` - Multi-tab Streamlit app
- `ui.py` - Simple query interface
- `Home.py` - Report viewer
- `deploy.py` - Deployment utilities

### Specialized Analytics Functions

The system includes 6 specialized analysis functions:

1. **Time Series Decomposition** (`time_series_decomposition`)
   - Trend, seasonal, and residual components
   - Requires: date column, value column

2. **Cohort Analysis** (`cohort_analysis`)
   - Customer retention over time
   - Requires: customer column, date column

3. **Customer Segmentation** (`customer_segmentation`)
   - RFM analysis and K-means clustering
   - Requires: customer column, feature columns

4. **Correlation Network Analysis** (`correlation_network_analysis`)
   - Variable relationships and dependencies
   - Requires: numeric columns

5. **Anomaly Detection** (`anomaly_detection`)
   - Outlier identification using Isolation Forest
   - Requires: numeric columns

6. **Distribution Comparison** (`distribution_comparison`)
   - Statistical group comparisons
   - Requires: target column, group column

### Chart Types

The system generates 10+ chart types:

- **Correlation Heatmap** - Variable relationships
- **Time Series Line** - Temporal trends
- **Bar Chart** - Categorical comparisons
- **Histogram** - Distribution analysis
- **Box Plot** - Outlier and quartile analysis
- **Scatter Plot** - Two-variable relationships
- **Violin Plot** - Distribution shapes by groups
- **Bubble Chart** - Three-dimensional relationships
- **Treemap** - Hierarchical data
- **Pie Chart** - Proportional data

## 🔄 HITL Workflow

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ 1. Load Dataset                                          │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Generate Analysis Plan (LLM)                         │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ ⏸️ CHECKPOINT 1: Analysis Plan Review                    │
│    - User selects analyses to execute                    │
│    - User approves/rejects plan                          │
└──────────────────┬───────────────────────────────────────┘
                   │ (Approved)
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Execute Approved Analyses                             │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Generate Charts                                       │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ ⏸️ CHECKPOINT 2: Chart Review                             │
│    - User reviews each chart                             │
│    - User approves/rejects individual charts             │
└──────────────────┬───────────────────────────────────────┘
                   │ (At least one approved)
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Generate Insights (Llama + Gemini)                    │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ ⏸️ CHECKPOINT 3: Insight Review                           │
│    - User reviews insights                               │
│    - User can EDIT insights directly                     │
│    - User approves/rejects insights                      │
└──────────────────┬───────────────────────────────────────┘
                   │ (At least one approved)
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Generate Report Preview                               │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ ⏸️ CHECKPOINT 4: Final Report Approval                   │
│    - User reviews report preview                         │
│    - User approves/rejects report                        │
└──────────────────┬───────────────────────────────────────┘
                   │ (Approved)
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 7. Generate Final Report (from approved content only)    │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ ✅ COMPLETED                                             │
└─────────────────────────────────────────────────────────┘
```

### Checkpoint Details

#### Checkpoint 1: Analysis Plan Review
- **Purpose**: Review and select which analyses to execute
- **Actions**: 
  - View proposed specialized analyses
  - Select analyses via checkboxes
  - Add optional feedback
  - Approve or reject the plan
- **Requirement**: At least one analysis must be selected

#### Checkpoint 2: Chart Review
- **Purpose**: Review and approve individual charts
- **Actions**:
  - View each generated chart (PNG/HTML)
  - Approve or reject individual charts
  - Add feedback per chart
  - Continue to insights when ready
- **Requirement**: At least one chart must be approved

#### Checkpoint 3: Insight Review
- **Purpose**: Review, edit, and approve insights
- **Actions**:
  - View generated insights (from Llama and/or Gemini)
  - **Edit insights directly** in text areas
  - Approve or reject insights
  - Save edits before approval
- **Requirement**: At least one insight must be approved
- **Special Feature**: Human edits override AI output

#### Checkpoint 4: Final Report Approval
- **Purpose**: Review and approve final report
- **Actions**:
  - View report preview
  - See summary of approved content
  - Add optional feedback
  - Approve or reject report
- **Requirement**: Report must be approved to generate final output

## 💬 Chatbot Features

### Stateful Gemini Chatbot

The `chatbot_session.py` provides an interactive chatbot that:

- **Loads Charts Once**: All charts are loaded at session start
- **Maintains Context**: Conversation history is preserved
- **Visual Analysis**: Uses Gemini Vision to analyze charts
- **Business Insights**: Answers questions based on chart data

### Usage

```bash
python chatbot_session.py
```

### Example Queries

- "What are the top 3 categories by sales?"
- "Show me the profit trends over time"
- "Which region has the highest sales?"
- "What anomalies do you see in the data?"

### Features

- Persistent chat session
- Chart-based analysis
- Natural language queries
- Context-aware responses

## 🔑 API Keys Setup

### Groq API Key (Required)

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Add to `.env`:
   ```env
   GROQ_API_KEY=your_key_here
   ```

### Google Gemini API Key (Optional)

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Create a new API key
4. Add to `.env`:
   ```env
   GEMINI_API_KEY=your_key_here
   ```

**Note**: Gemini API is optional but recommended for VLM (Vision Language Model) insights from charts.

## 📊 Examples

### Example 1: Basic HITL Workflow

```python
# 1. Start the HITL app
streamlit run hitl_main_app.py

# 2. In the UI:
#    - Load Superstore.csv
#    - Click "Start HITL Analysis Workflow"
#    - Review and approve at each checkpoint
#    - Download final report
```

### Example 2: Automated Analysis

```python
# Run automated analysis (no checkpoints)
python main.py

# Output:
# - Charts in charts/ directory
# - analysis_report.txt
# - interactive_analysis_report.html
```

### Example 3: Chatbot Session

```python
# Start chatbot
python chatbot_session.py

# Example conversation:
# > What are the sales trends?
# < [Gemini analyzes charts and responds]
# > Which category is most profitable?
# < [Gemini provides insights based on charts]
```

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Errors

**Problem**: `GROQ_API_KEY not found`

**Solution**:
- Verify `.env` file exists in root directory
- Check API key is correctly formatted (no quotes, no spaces)
- Restart the application after adding keys

#### 2. Chart Generation Fails

**Problem**: No charts generated

**Solution**:
- Check that data file is loaded correctly
- Verify numeric columns exist for analysis
- Check console for specific error messages
- Ensure `charts/` directory is writable

#### 3. HITL Workflow Not Starting

**Problem**: Workflow doesn't start after clicking button

**Solution**:
- Check that dataset is loaded (should show shape in UI)
- Verify GROQ_API_KEY is set
- Check browser console for errors
- Try refreshing the page

#### 4. Checkpoint Not Advancing

**Problem**: Can't proceed past a checkpoint

**Solution**:
- Ensure at least one item is approved
- Check approval status in sidebar
- For chart/insight checkpoints, click "Continue" button
- Verify no errors in console

#### 5. Insights Not Generating

**Problem**: No insights appear

**Solution**:
- Verify charts were approved in previous checkpoint
- Check that GEMINI_API_KEY is set (for VLM insights)
- Ensure analysis results exist
- Check console for API errors

#### 6. Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep streamlit
pip list | grep pandas
```

### Debug Mode

Enable debug logging:

```python
# In Python scripts, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Write tests for new features
- Update documentation

## 📄 License

This project is provided as-is for educational and commercial use. Ensure all API keys are kept secure and not committed to version control.

## 🙏 Acknowledgments

- **Groq** for LLM API access
- **Google Gemini** for Vision Language Model capabilities
- **Streamlit** for the web framework
- **Plotly** for interactive visualizations
- **AutoViz** for automated chart generation

## 📞 Support

For issues, questions, or contributions:

- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

## 🔮 Future Enhancements

- [ ] Multi-dataset support
- [ ] Custom analysis function registration
- [ ] Export to PDF
- [ ] Real-time collaboration
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment guides

---

**Made with ❤️ for intelligent data analysis**

