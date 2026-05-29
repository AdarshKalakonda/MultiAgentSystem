"""
LangGraph StateGraph for DataStory.

Execution flow
──────────────
orchestrator → profiler ──┬──► statistician ──┐
                           ├──► viz_builder    ├──► narrator → END
                           └──► anomaly_detector ┘

The three parallel branches are launched via the Send() API so they
run concurrently; their list-typed state fields merge automatically
thanks to the operator.add reducers defined in DataStoryState.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agents.anomaly import run_anomaly_detector
from agents.narrator import run_narrator
from agents.orchestrator import run_orchestrator
from agents.profiler import run_profiler
from agents.statistician import run_statistician
from agents.viz_builder import run_viz_builder
from state.schema import DataStoryState


# ── Routing / fan-out ─────────────────────────────────────────────────────────

def _route_after_profiler(state: DataStoryState) -> list[Send]:
    """
    Fan out to three independent analysis agents in parallel.

    Each Send carries the full current state so every agent has access
    to schema_info produced by the profiler.
    """
    return [
        Send("statistician", state),
        Send("viz_builder", state),
        Send("anomaly_detector", state),
    ]


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Assemble and return the compiled DataStory LangGraph application."""
    workflow = StateGraph(DataStoryState)

    # Register nodes
    workflow.add_node("orchestrator", run_orchestrator)
    workflow.add_node("profiler", run_profiler)
    workflow.add_node("statistician", run_statistician)
    workflow.add_node("viz_builder", run_viz_builder)
    workflow.add_node("anomaly_detector", run_anomaly_detector)
    workflow.add_node("narrator", run_narrator)

    # Sequential spine
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "profiler")

    # Parallel fan-out via Send() after profiler
    workflow.add_conditional_edges(
        "profiler",
        _route_after_profiler,
        ["statistician", "viz_builder", "anomaly_detector"],
    )

    # Fan-in: all three parallel branches converge on narrator
    workflow.add_edge("statistician", "narrator")
    workflow.add_edge("viz_builder", "narrator")
    workflow.add_edge("anomaly_detector", "narrator")

    # Narrator is the terminal node
    workflow.add_edge("narrator", END)

    return workflow


# Module-level compiled app — import this from app.py / tests
app = build_graph().compile()
