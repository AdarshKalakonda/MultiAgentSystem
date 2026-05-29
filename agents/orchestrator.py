"""Orchestrator agent — validates inputs and initialises graph state."""

from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic

from config import ANTHROPIC_API_KEY, MODEL_FAST
from state.schema import DataStoryState


def run_orchestrator(state: DataStoryState) -> dict:
    """
    Entry-point node.

    Responsibilities
    ────────────────
    1. Validate that df_path exists and is readable.
    2. Seed metadata (file name, size, start timestamp).
    3. Return a partial state update — downstream nodes fill the rest.

    Args:
        state: Current graph state (df_path must be set by the caller).

    Returns:
        Partial state dict with validated metadata and empty collections.
    """
    # TODO: implement full orchestrator logic
    # - Validate df_path is a real, readable file
    # - Detect file type (csv / xlsx / parquet / json)
    # - Seed metadata dict with file_name, file_size_mb, started_at
    # - Optionally call MODEL_FAST to produce a natural-language summary
    #   of the user's analysis goal from state.get("user_query", "")
    # - Return partial state update

    _llm = ChatAnthropic(  # noqa: F841 — used in TODO implementation
        model=MODEL_FAST,
        api_key=ANTHROPIC_API_KEY,
    )

    df_path = state.get("df_path", "")
    path = Path(df_path)

    metadata: dict = dict(state.get("metadata") or {})
    metadata["file_name"] = path.name
    metadata["file_exists"] = path.exists()

    errors: list[str] = []
    if not path.exists():
        errors.append(f"File not found: {df_path}")

    return {
        "metadata": metadata,
        "insights": [],
        "chart_specs": [],
        "anomalies": [],
        "error_log": errors,
        "report_html": None,
        "schema_info": None,
    }
