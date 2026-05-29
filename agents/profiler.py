"""Profiler agent — loads data and builds a full column-level schema."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from config import ANTHROPIC_API_KEY, MODEL_FAST
from state.schema import DataStoryState


def run_profiler(state: DataStoryState) -> dict:
    """
    Profile the dataset at state['df_path'] and populate schema_info.

    Responsibilities
    ────────────────
    1. Load the file via tools.data_loader.load_dataframe.
    2. Compute per-column stats: dtype, null %, unique count, sample values,
       descriptive stats (mean/std/min/max for numeric, top-k for categorical).
    3. Build a SchemaResult and return it as a state update.
    4. Append any load errors to error_log.

    Args:
        state: Must contain a valid df_path (validated by orchestrator).

    Returns:
        Partial state dict with schema_info populated.
    """
    # TODO: implement full profiler logic
    # - Call load_dataframe(state["df_path"]) → pd.DataFrame
    # - Iterate over df.columns and build ColumnInfo for each
    # - Construct SchemaResult(row_count, column_count, columns, memory_usage_mb)
    # - Optionally call MODEL_FAST to generate a one-sentence column description
    # - Return {"schema_info": schema_result, "error_log": []}

    _llm = ChatAnthropic(  # noqa: F841
        model=MODEL_FAST,
        api_key=ANTHROPIC_API_KEY,
    )

    return {
        "schema_info": None,  # replace with SchemaResult instance
        "error_log": [],
    }
