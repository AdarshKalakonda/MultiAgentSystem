"""Statistician agent — derives insights from schema and raw data."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from config import ANTHROPIC_API_KEY, MODEL_SMART
from state.schema import DataStoryState


def run_statistician(state: DataStoryState) -> dict:
    """
    Run statistical analysis and return a list of Insight objects.

    Responsibilities
    ────────────────
    1. Read state['schema_info'] to understand column types and distributions.
    2. Run correlation analysis (numeric columns), chi-square tests
       (categorical pairs), and trend detection (datetime + numeric).
    3. Call MODEL_SMART with structured output to interpret findings and
       produce ranked Insight objects (title, description, severity, column_refs).
    4. Return insights as a state update — uses operator.add reducer.

    Args:
        state: Must contain schema_info from profiler.

    Returns:
        Partial state dict with insights list appended.
    """
    # TODO: implement full statistician logic
    # - Load df from state["df_path"] (use tools.data_loader)
    # - Compute correlation matrix, top correlated pairs
    # - Run scipy.stats tests relevant to data types
    # - Use statsmodels for trend / seasonality detection on time-series cols
    # - Format findings as prompt context → call MODEL_SMART with tool_use
    #   to emit structured Insight objects
    # - Return {"insights": [Insight(...), ...], "error_log": []}

    _llm = ChatAnthropic(  # noqa: F841
        model=MODEL_SMART,
        api_key=ANTHROPIC_API_KEY,
    )

    return {
        "insights": [],   # replace with list[Insight]
        "error_log": [],
    }
