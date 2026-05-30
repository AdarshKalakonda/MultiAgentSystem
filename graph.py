"""
LangGraph StateGraph for DataStory.

Execution flow
──────────────
profiler ──┬──► statistician    ──┐
           ├──► viz_builder      ├──► narrator → END
           └──► anomaly_detector ──┘

The three parallel branches are launched via the Send() API after the
profiler completes, so they run concurrently in separate threads.
Their list-typed state fields merge automatically thanks to the
operator.add reducers defined in DataStoryState.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agents.anomaly import run_anomaly_detector
from agents.narrator import run_narrator
from agents.profiler import run_profiler
from agents.statistician import run_statistician
from agents.viz_builder import run_viz_builder
from state.schema import DataStoryState


# ── Routing — parallel fan-out after profiler ─────────────────────────────────

def _route_after_profiler(state: DataStoryState) -> list[Send]:
    """
    Dispatch to all three analysis agents in parallel.

    Each Send carries the full current state (including schema_info
    produced by the profiler) so every parallel branch has everything
    it needs to operate independently.
    """
    return [
        Send("statistician",    state),
        Send("viz_builder",     state),
        Send("anomaly_detector", state),
    ]


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and compile the DataStory StateGraph.

    Nodes
    ─────
    profiler         — loads data, produces SchemaResult
    statistician     — correlation, trends → Insight list
    viz_builder      — Plotly chart specs
    anomaly_detector — IQR / z-score / IsolationForest → Anomaly list
    narrator         — HTML report synthesis

    Returns the compiled graph; callers should use the module-level
    ``app`` object rather than calling this function directly.
    """
    workflow = StateGraph(DataStoryState)

    # ── Register nodes ────────────────────────────────────────────────────
    workflow.add_node("profiler",          run_profiler)
    workflow.add_node("statistician",      run_statistician)
    workflow.add_node("viz_builder",       run_viz_builder)
    workflow.add_node("anomaly_detector",  run_anomaly_detector)
    workflow.add_node("narrator",          run_narrator)

    # ── Entry point ───────────────────────────────────────────────────────
    workflow.set_entry_point("profiler")

    # ── Parallel fan-out via Send() ───────────────────────────────────────
    workflow.add_conditional_edges(
        "profiler",
        _route_after_profiler,
        ["statistician", "viz_builder", "anomaly_detector"],
    )

    # ── Fan-in: all three branches converge on narrator ───────────────────
    workflow.add_edge("statistician",      "narrator")
    workflow.add_edge("viz_builder",       "narrator")
    workflow.add_edge("anomaly_detector",  "narrator")

    # ── Terminal edge ─────────────────────────────────────────────────────
    workflow.add_edge("narrator", END)

    return workflow


# Module-level compiled app — import this in app.py, tests, and scripts
app = build_graph().compile()
