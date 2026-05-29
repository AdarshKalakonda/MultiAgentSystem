# DataStory — Project Context for Claude Code

## What this project is

DataStory is a **multi-agent AI data analysis system** that accepts a tabular
data file (CSV / Excel / Parquet / JSON) and produces a rich HTML report
containing statistical insights, Plotly visualisations, and anomaly findings —
all narrated in plain English by Claude.

## Architecture

```
User uploads file
       │
       ▼
 orchestrator          validates input, seeds metadata
       │
       ▼
   profiler            loads data, builds SchemaResult (column types, stats)
       │
   ┌───┴──────────────┐
   │                  │                  │
statistician     viz_builder      anomaly_detector    ← run in PARALLEL
(Insights)       (PlotlySpecs)    (Anomalies)
   │                  │                  │
   └──────────────────┴──────────────────┘
                       │
                       ▼
                   narrator            synthesises → HTML report
```

The three parallel branches are launched with LangGraph's `Send()` API.
Results merge back via `operator.add` reducers on list-typed state fields.

## Key files

| Path | Purpose |
|---|---|
| `graph.py` | LangGraph StateGraph — the wiring of all 6 nodes |
| `state/schema.py` | Pydantic v2 models + `DataStoryState` TypedDict |
| `config.py` | `.env` loading, API keys, model IDs |
| `agents/` | One module per LangGraph node |
| `tools/` | Reusable helpers (loader, executor, chart utils) |
| `app.py` | (not yet created) Streamlit front-end |
| `templates/` | (not yet created) Jinja2 HTML report template |

## State schema

`DataStoryState` (TypedDict in `state/schema.py`):

- `df_path: str` — absolute path to the uploaded file
- `schema_info: Optional[SchemaResult]` — produced by profiler
- `insights: Annotated[list[Insight], operator.add]` — produced by statistician
- `chart_specs: Annotated[list[PlotlySpec], operator.add]` — produced by viz_builder
- `anomalies: Annotated[list[Anomaly], operator.add]` — produced by anomaly_detector
- `report_html: Optional[str]` — produced by narrator
- `error_log: Annotated[list[str], operator.add]` — errors from all nodes
- `metadata: dict` — file info, timestamps, user query

## Models

| Constant | Value | Used by |
|---|---|---|
| `MODEL_FAST` | `claude-haiku-4-5-20251001` | orchestrator, profiler |
| `MODEL_SMART` | `claude-sonnet-4-20250514` | statistician, viz_builder, anomaly_detector, narrator |

## Conventions

- **Never hardcode API keys** — always read from `config.py`.
- **Pydantic v2** — use `model_validator(mode="after")` and `field_validator`.
- **Type hints everywhere** — Python 3.11+ syntax (`list[X]` not `List[X]`).
- **Docstrings** — every public function has a Google-style docstring.
- **No comments explaining what code does** — only non-obvious WHY comments.
- List fields on `DataStoryState` use `Annotated[list[X], operator.add]` so
  parallel branches merge without overwriting each other.

## Next implementation steps

1. Implement `tools/data_loader.py` — multi-format file loading.
2. Implement `tools/code_executor.py` — sandboxed execution environment.
3. Implement `agents/profiler.py` — column profiling with pandas.
4. Implement `agents/statistician.py` — correlation + hypothesis testing.
5. Implement `agents/viz_builder.py` — LLM-directed chart generation.
6. Implement `agents/anomaly.py` — IQR + Isolation Forest detection.
7. Implement `agents/narrator.py` — Jinja2 HTML report rendering.
8. Create `templates/report.html.j2` — HTML report template.
9. Create `app.py` — Streamlit UI (file upload → graph.invoke → render report).

## Running the project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# edit .env with your ANTHROPIC_API_KEY

# 3. (Future) Launch Streamlit
streamlit run app.py
```

## Dependency notes

- **weasyprint** requires GTK on Windows — see WeasyPrint docs for the
  Windows binary installer if PDF export is needed.
- **langsmith** tracing is opt-in; set `LANGSMITH_API_KEY` in `.env` to enable.
