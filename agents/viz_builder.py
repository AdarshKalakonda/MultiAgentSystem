"""VizBuilder agent — generates Plotly chart specifications."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from config import ANTHROPIC_API_KEY, MODEL_SMART
from state.schema import DataStoryState


def run_viz_builder(state: DataStoryState) -> dict:
    """
    Generate a curated set of Plotly chart specifications.

    Responsibilities
    ────────────────
    1. Inspect schema_info to identify chart-worthy column combinations:
       - Numeric distributions → histograms / box plots
       - Numeric vs numeric → scatter / heatmap
       - Categorical vs numeric → bar / violin
       - Datetime + numeric → line chart
    2. Call MODEL_SMART to select the most impactful charts and write
       Plotly Express / graph_objects code.
    3. Execute the generated code via tools.code_executor to produce
       actual figure objects, then serialise to JSON.
    4. Return PlotlySpec objects — uses operator.add reducer.

    Args:
        state: Must contain schema_info and df_path.

    Returns:
        Partial state dict with chart_specs list appended.
    """
    # TODO: implement full viz_builder logic
    # - Analyse schema_info.columns to identify candidate chart pairs
    # - Build a prompt describing the dataset and ask MODEL_SMART to
    #   return a JSON list of {chart_type, x_col, y_col, title, rationale}
    # - For each chart spec, generate Plotly code and execute via
    #   tools.code_executor.execute_pandas_code
    # - Serialise the resulting fig to JSON (plotly.io.to_json)
    # - Construct PlotlySpec objects and return them
    # - Return {"chart_specs": [PlotlySpec(...), ...], "error_log": []}

    _llm = ChatAnthropic(  # noqa: F841
        model=MODEL_SMART,
        api_key=ANTHROPIC_API_KEY,
    )

    return {
        "chart_specs": [],  # replace with list[PlotlySpec]
        "error_log": [],
    }
