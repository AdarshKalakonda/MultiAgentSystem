"""
DataStory — Streamlit UI

Three-page flow managed entirely via st.session_state:
  "upload"  → file upload / sample picker
  "running" → live execution with st.status + progress bar
  "report"  → four-tab report viewer + download buttons
"""

from __future__ import annotations

import sys
import tempfile
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import ANTHROPIC_API_KEY, MODEL_FAST          # noqa: E402
from graph import app as lg_app                            # noqa: E402
from state.schema import DataStoryState                    # noqa: E402

_ROOT       = Path(__file__).parent
_SAMPLE_DIR = _ROOT / "sample_data"
_OUTPUT_DIR = _ROOT / "outputs"
_LIST_FIELDS = {"insights", "chart_specs", "anomalies", "error_log"}

# One shared executor — survives the session (module-level, not pickled)
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="DataStory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inline CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Chrome ── */
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 900px;}

/* ── Hero ── */
.ds-hero {text-align:center; padding:3rem 1rem 1.5rem;}
.ds-hero h1 {
    font-size:2.6rem; font-weight:800; color:#1e2a3a;
    letter-spacing:-0.02em; margin-bottom:.5rem;
}
.ds-hero p {font-size:1.1rem; color:#64748b; margin:0;}

/* ── Sample-dataset buttons ── */
.ds-sample-card {
    border:2px solid #e2e8f0; border-radius:12px;
    padding:1.1rem .8rem; text-align:center;
    background:#fff; transition:border-color .15s;
}
.ds-sample-card:hover {border-color:#534AB7;}

/* ── Progress bar ── */
.stProgress > div > div > div > div {background:#534AB7 !important;}

/* ── Severity badges ── */
.sev-critical {background:#fee2e2;color:#dc2626;border-radius:4px;
               padding:1px 7px;font-size:.78rem;font-weight:700;}
.sev-high     {background:#fef3c7;color:#b45309;border-radius:4px;
               padding:1px 7px;font-size:.78rem;font-weight:700;}
.sev-medium   {background:#dbeafe;color:#1d4ed8;border-radius:4px;
               padding:1px 7px;font-size:.78rem;font-weight:700;}
.sev-low      {background:#d1fae5;color:#065f46;border-radius:4px;
               padding:1px 7px;font-size:.78rem;font-weight:700;}

/* ── Running-page spinner cards ── */
.agent-card {
    border:1px solid #e2e8f0; border-radius:10px;
    padding:1rem; text-align:center; background:#fafafa;
}
.agent-card.done {background:#f0fdf4; border-color:#86efac;}
.agent-card.running {background:#eff6ff; border-color:#93c5fd;}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    for key, default in {
        "page":             "upload",
        "df_path":          None,
        "dataset_name":     None,
        "depth":            "Full (sonnet)",
        "pipeline_future":  None,
        "start_time":       None,
        "result":           None,
        "chat_history":     [],
    }.items():
        st.session_state.setdefault(key, default)

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_pdf(html: str) -> bytes | None:
    """Convert HTML to PDF via WeasyPrint; returns None if GTK libs missing."""
    try:
        from weasyprint import HTML          # noqa: PLC0415
        return HTML(string=html).write_pdf()
    except Exception:
        return None


def _chat_context(result: dict) -> str:
    schema   = result.get("schema_info")
    insights = result.get("insights", [])
    anomalies = result.get("anomalies", [])
    meta     = result.get("metadata", {})
    lines = [
        f"File: {Path(result.get('df_path', '')).name}",
        f"Shape: {schema.row_count}r x {schema.column_count}c" if schema else "",
        f"Domain: {meta.get('domain', '?')}",
        f"Description: {meta.get('dataset_description', '')}",
        "",
        "Key insights:",
        *[f"  [{i.severity.upper()}] {i.title}: {i.description[:120]}"
          for i in insights[:5]],
    ]
    if anomalies:
        lines += ["", "Anomalies:"]
        lines += [f"  [{a.severity.upper()}] {a.description}"
                  for a in anomalies[:3]]
    return "\n".join(filter(lambda x: x is not None, lines))


def _chat_reply(question: str, context: str, history: list[dict]) -> str:
    from langchain_anthropic import ChatAnthropic          # noqa: PLC0415
    from langchain_core.messages import (                  # noqa: PLC0415
        AIMessage, HumanMessage, SystemMessage,
    )
    try:
        llm = ChatAnthropic(
            model=MODEL_FAST, api_key=ANTHROPIC_API_KEY,
            temperature=0.3, max_tokens=512,
        )
        msgs = [SystemMessage(
            content=("You are a concise data analyst. Answer questions about "
                     "the dataset using the context below.\n\n" + context)
        )]
        for m in history[-6:]:
            msgs.append(HumanMessage(content=m["content"])
                        if m["role"] == "user"
                        else AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=question))
        resp = llm.invoke(msgs)
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as exc:
        return f"Sorry, could not answer: {str(exc)[:120]}"


def _go(page: str, **kwargs: object) -> None:
    """Navigate to a page, setting extra session keys from kwargs."""
    st.session_state["page"] = page
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()


def _save_upload(uploaded) -> str:
    """Write an uploaded file to a temp path and return it."""
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, prefix="datastory_upload_"
    ) as fh:
        fh.write(uploaded.getbuffer())
        return fh.name


def _run_pipeline(df_path: str) -> dict:
    initial: DataStoryState = {          # type: ignore[assignment]
        "df_path":    df_path,
        "schema_info": None,
        "insights":   [],
        "chart_specs":[],
        "anomalies":  [],
        "report_html": None,
        "error_log":  [],
        "metadata":   {},
    }
    return lg_app.invoke(initial, config={"recursion_limit": 50})


# ── Page 1: Upload ────────────────────────────────────────────────────────────

def show_upload_page() -> None:
    st.markdown("""
    <div class="ds-hero">
      <h1>📊 DataStory</h1>
      <p>Upload any dataset. Get a boardroom-ready report in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop your file here — CSV, Excel, SQLite",
        type=["csv", "xlsx", "xls", "db", "sqlite"],
        label_visibility="collapsed",
    )
    if uploaded:
        tmp_path = _save_upload(uploaded)
        st.session_state["df_path"]      = tmp_path
        st.session_state["dataset_name"] = uploaded.name
        st.success(f"Ready: **{uploaded.name}**")

    # ── Sample dataset buttons ────────────────────────────────────────────────
    st.markdown("##### Or start with a sample dataset")
    col1, col2, col3 = st.columns(3)
    samples = [
        (col1, "📦 Sales data",      "sales"),
        (col2, "👥 HR data",         "hr"),
        (col3, "🛒 E-commerce data", "ecommerce"),
    ]
    for col, label, name in samples:
        with col:
            if st.button(label, use_container_width=True):
                path = _SAMPLE_DIR / f"{name}.csv"
                st.session_state["df_path"]      = str(path)
                st.session_state["dataset_name"] = f"{name}.csv"
                st.session_state["pipeline_future"] = None
                st.session_state["start_time"]   = None
                st.session_state["result"]       = None
                st.session_state["chat_history"] = []
                _go("running")

    # ── Analysis depth ────────────────────────────────────────────────────────
    st.markdown("")
    depth = st.radio(
        "Analysis depth",
        ["Full (sonnet)", "Quick (haiku only)"],
        horizontal=True,
        help="Full uses Claude Sonnet for richer insights; Quick uses Haiku for faster results.",
    )
    st.session_state["depth"] = depth

    # ── Generate button ───────────────────────────────────────────────────────
    has_file = bool(st.session_state.get("df_path"))
    st.markdown("")
    if st.button(
        "Generate Report →",
        type="primary",
        disabled=not has_file,
        use_container_width=False,
    ):
        st.session_state["pipeline_future"] = None
        st.session_state["start_time"]      = None
        st.session_state["result"]          = None
        st.session_state["chat_history"]    = []
        _go("running")


# ── Page 2: Live execution ────────────────────────────────────────────────────

def show_running_page() -> None:
    df_path      = st.session_state.get("df_path", "")
    dataset_name = st.session_state.get("dataset_name", "dataset")

    # ── Start pipeline (once per "running" visit) ─────────────────────────────
    if st.session_state.get("pipeline_future") is None:
        future = _EXECUTOR.submit(_run_pipeline, df_path)
        st.session_state["pipeline_future"] = future
        st.session_state["start_time"]      = time.time()

    future    = st.session_state["pipeline_future"]
    elapsed   = time.time() - (st.session_state.get("start_time") or time.time())

    # ── Pipeline done? ────────────────────────────────────────────────────────
    if future.done():
        try:
            result = future.result()
            st.session_state["result"]          = result
            st.session_state["pipeline_future"] = None
            _go("report")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            if st.button("← Try again"):
                st.session_state["pipeline_future"] = None
                _go("upload")
        return

    # ── Progress bar ─────────────────────────────────────────────────────────
    # Derive estimated stage from wall-clock time (tuned to observed ~40 s run)
    if elapsed < 6:
        stage, pct = "profiling",  max(5,  int(elapsed / 6  * 20))
    elif elapsed < 30:
        stage, pct = "parallel",   20 + int((elapsed - 6)  / 24 * 50)
    elif elapsed < 50:
        stage, pct = "narrating",  70 + int((elapsed - 30) / 20 * 25)
    else:
        stage, pct = "narrating",  95

    st.markdown("### Analysing your data...")
    st.progress(pct / 100, text=f"{pct}% complete")
    st.markdown("")

    # ── Stage 1: Profiler ─────────────────────────────────────────────────────
    profiler_done = stage != "profiling"
    with st.status(
        "Data loaded and profiled" if profiler_done else "Loading and profiling data...",
        state="complete" if profiler_done else "running",
        expanded=not profiler_done,
    ) as s_prof:
        if not profiler_done:
            st.write(f"Reading `{dataset_name}`  —  computing column statistics...")
        else:
            s_prof.update(
                label="Data loaded and profiled",
                state="complete",
                expanded=False,
            )
            st.caption(f"File: `{dataset_name}`")

    # ── Stage 2: Three parallel agents ───────────────────────────────────────
    if stage in ("parallel", "narrating"):
        parallel_done = stage == "narrating"
        with st.status(
            "Analysis complete" if parallel_done else "Running analysis in parallel...",
            state="complete" if parallel_done else "running",
            expanded=not parallel_done,
        ) as s_par:
            if parallel_done:
                s_par.update(
                    label="Analysis complete",
                    state="complete",
                    expanded=False,
                )
            c1, c2, c3 = st.columns(3)
            icon = "✅" if parallel_done else "⏳"
            with c1:
                done_cls = "done" if parallel_done else "running"
                st.markdown(
                    f'<div class="agent-card {done_cls}">'
                    f'{icon}<br><strong>📊 Finding patterns</strong>'
                    f'<br><small>Stats &amp; correlations</small></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="agent-card {done_cls}">'
                    f'{icon}<br><strong>📈 Building charts</strong>'
                    f'<br><small>Plotly visualizations</small></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="agent-card {done_cls}">'
                    f'{icon}<br><strong>🔍 Detecting anomalies</strong>'
                    f'<br><small>Outlier detection</small></div>',
                    unsafe_allow_html=True,
                )

    # ── Stage 3: Narrator ─────────────────────────────────────────────────────
    if stage == "narrating":
        with st.status("Writing your report...", state="running", expanded=True):
            st.write("Generating executive summary and narrative...")

    # ── Elapsed time hint ─────────────────────────────────────────────────────
    st.caption(f"Elapsed: {elapsed:.0f}s — typical run ~40 s")

    # ── Poll every second ─────────────────────────────────────────────────────
    time.sleep(1)
    st.rerun()


# ── Page 3: Report viewer ─────────────────────────────────────────────────────

def show_report_page() -> None:
    result      = st.session_state.get("result") or {}
    report_html = result.get("report_html") or ""
    insights    = result.get("insights",   [])
    chart_specs = result.get("chart_specs",[])
    anomalies   = result.get("anomalies",  [])
    metadata    = result.get("metadata",   {})
    dataset_name = st.session_state.get("dataset_name", "report")

    # ── Header ────────────────────────────────────────────────────────────────
    col_h, col_b = st.columns([3, 1])
    with col_h:
        st.markdown(
            f"## 📊 {dataset_name}"
            f"  <small style='color:#94a3b8;font-size:.75em'>"
            f"domain: {metadata.get('domain','?')} &nbsp;|&nbsp; "
            f"{len(insights)} insights &nbsp;|&nbsp; "
            f"{len(chart_specs)} charts &nbsp;|&nbsp; "
            f"{len(anomalies)} anomalies</small>",
            unsafe_allow_html=True,
        )
    with col_b:
        if st.button("← New report", use_container_width=True):
            _go("upload")

    # ── Download buttons ──────────────────────────────────────────────────────
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "⬇ Download HTML report",
            data=report_html.encode("utf-8"),
            file_name=f"datastory_{Path(dataset_name).stem}.html",
            mime="text/html",
            use_container_width=True,
        )
    with dcol2:
        pdf_bytes = _try_pdf(report_html)
        if pdf_bytes:
            st.download_button(
                "⬇ Download PDF",
                data=pdf_bytes,
                file_name=f"datastory_{Path(dataset_name).stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.button(
                "⬇ Download PDF (needs GTK)",
                disabled=True,
                use_container_width=True,
                help="WeasyPrint requires GTK libraries. See WeasyPrint installation docs.",
            )

    st.divider()

    # ── Four tabs ─────────────────────────────────────────────────────────────
    tab_report, tab_charts, tab_anomalies, tab_ask = st.tabs(
        ["📋 Report", "📊 Charts", "⚠️ Anomalies", "💬 Ask"]
    )

    # ── Tab 1: Full HTML report ───────────────────────────────────────────────
    with tab_report:
        if report_html:
            components.html(report_html, height=800, scrolling=True)
        else:
            st.warning("Report HTML not available.")

    # ── Tab 2: Interactive charts ─────────────────────────────────────────────
    with tab_charts:
        if not chart_specs:
            st.info("No charts were generated.")
        else:
            for spec in chart_specs:
                try:
                    fig = pio.from_json(spec.figure_json)
                    st.subheader(spec.title)
                    st.plotly_chart(fig, use_container_width=True)
                    if spec.insight:
                        st.caption(spec.insight)
                    st.divider()
                except Exception as exc:
                    st.error(f"Could not render '{spec.title}': {exc}")

    # ── Tab 3: Anomalies dataframe ────────────────────────────────────────────
    with tab_anomalies:
        if not anomalies:
            st.success("No anomalies or data-quality issues were detected.")
        else:
            sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_a  = sorted(anomalies, key=lambda a: sev_order.get(a.severity, 4))
            rows = [
                {
                    "Severity":    a.severity.capitalize(),
                    "Description": a.description,
                    "Rows":        len(a.affected_rows),
                    "Sample rows": str(a.affected_rows[:5])[1:-1]
                                   + (" ..." if len(a.affected_rows) > 5 else ""),
                    "Explanation": a.explanation,
                }
                for a in sorted_a
            ]
            df_a = pd.DataFrame(rows)

            sev_bg = {
                "Critical": "background-color:#fee2e2",
                "High":     "background-color:#fef3c7",
                "Medium":   "background-color:#dbeafe",
                "Low":      "background-color:#d1fae5",
            }

            def _color_row(row: pd.Series) -> list[str]:
                style = sev_bg.get(row["Severity"], "")
                return [style] * len(row)

            st.dataframe(
                df_a.style.apply(_color_row, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Severity":    st.column_config.TextColumn(width="small"),
                    "Rows":        st.column_config.NumberColumn(width="small"),
                    "Sample rows": st.column_config.TextColumn(width="medium"),
                    "Explanation": st.column_config.TextColumn(width="large"),
                },
            )

    # ── Tab 4: Chat Q&A ───────────────────────────────────────────────────────
    with tab_ask:
        ctx = _chat_context(result)

        if not st.session_state["chat_history"]:
            st.markdown(
                "_Ask anything about your data — the AI has read the full report._"
            )

        # Render history
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input("Ask a question about your data..."):
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state["chat_history"].append(
                {"role": "user", "content": prompt}
            )

            # Get + show response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = _chat_reply(prompt, ctx, st.session_state["chat_history"])
                st.markdown(answer)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

_page = st.session_state.get("page", "upload")

if _page == "upload":
    show_upload_page()
elif _page == "running":
    show_running_page()
elif _page == "report":
    show_report_page()
else:
    st.session_state["page"] = "upload"
    st.rerun()
