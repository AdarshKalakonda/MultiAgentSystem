# DataStory

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B35?logo=chainlink)
![Claude](https://img.shields.io/badge/Powered%20by-Claude%20Sonnet%20%2F%20Haiku-8B5CF6?logo=anthropic)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Charts-Plotly-3F4F75?logo=plotly)

**DataStory** turns a raw data file into a publication-ready analytical report
in seconds. Upload a CSV, Excel, Parquet, or JSON file and a team of specialised
AI agents — built on LangGraph and Claude — will profile your data, run
statistical tests, generate interactive visualisations, detect anomalies, and
write the narrative that ties everything together.

---

## Architecture

```
                         ┌─────────────┐
                         │ orchestrator │  validates input, seeds state
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │   profiler  │  loads data, builds column schema
                         └──────┬──────┘
                                │  Send() fan-out
            ┌───────────────────┼────────────────────┐
            │                   │                    │
   ┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
   │  statistician   │ │   viz_builder   │ │anomaly_detector │
   │  (Insights)     │ │  (PlotlySpecs)  │ │  (Anomalies)    │
   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                   │                    │
            └───────────────────┼────────────────────┘
                                │  fan-in
                         ┌──────▼──────┐
                         │   narrator  │  synthesises HTML report
                         └─────────────┘
```

Parallel branches run concurrently via LangGraph's `Send()` API; results
merge automatically through `operator.add` reducers on list-typed state fields.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph, LangChain-Anthropic |
| LLM backbone | Claude Sonnet 4 (analysis) · Claude Haiku 4.5 (fast ops) |
| Data processing | pandas, NumPy, SciPy, statsmodels, scikit-learn |
| Visualisation | Plotly |
| Report generation | Jinja2, WeasyPrint (PDF) |
| Data I/O | SQLAlchemy, openpyxl |
| Validation | Pydantic v2 |
| UI | Streamlit |
| Observability | LangSmith |

---

## Quickstart

```bash
# 1. Clone / navigate to the project
cd MultiAgentSystem/DataStory

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API keys
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY (and optionally LANGSMITH_API_KEY)

# 5. (Coming soon) Launch the Streamlit app
streamlit run app.py
```

---

## Project Structure

```
DataStory/
├── config.py              # Env-var loading, model constants
├── graph.py               # LangGraph StateGraph wiring
├── requirements.txt
├── .env.example
├── state/
│   └── schema.py          # Pydantic v2 models + DataStoryState TypedDict
├── agents/
│   ├── orchestrator.py    # Input validation, state seeding
│   ├── profiler.py        # Column-level dataset profiling
│   ├── statistician.py    # Correlations, hypothesis tests, insights
│   ├── viz_builder.py     # LLM-directed Plotly chart generation
│   ├── anomaly.py         # IQR / Isolation Forest anomaly detection
│   └── narrator.py        # HTML report synthesis
└── tools/
    ├── data_loader.py     # Multi-format file → DataFrame
    ├── code_executor.py   # Sandboxed pandas/Plotly execution
    └── chart_tools.py     # Figure serialisation & HTML embedding
```

---

## Status

> **Scaffold complete — implementation in progress.**
> All nodes, models, and graph wiring are in place.
> Each agent file contains a detailed `TODO` comment describing
> the full implementation plan.
