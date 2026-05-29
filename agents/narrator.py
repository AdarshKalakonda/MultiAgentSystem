"""Narrator agent — synthesises all findings into an HTML report."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from config import ANTHROPIC_API_KEY, MODEL_SMART
from state.schema import DataStoryState


def run_narrator(state: DataStoryState) -> dict:
    """
    Write the final DataStory report by combining all agent outputs.

    Responsibilities
    ────────────────
    1. Receive merged insights, chart_specs, and anomalies from the
       three parallel branches (already reduced into the state).
    2. Call MODEL_SMART with a rich context prompt to generate
       a coherent narrative that connects findings.
    3. Render an HTML report using Jinja2 templates, embedding
       Plotly charts as inline JSON (plotly.js CDN).
    4. Optionally export to PDF via WeasyPrint.
    5. Write report_html into the state.

    Args:
        state: Contains schema_info, insights, chart_specs, anomalies.

    Returns:
        Partial state dict with report_html populated.
    """
    # TODO: implement full narrator logic
    # - Sort insights and anomalies by severity (critical → low)
    # - Build a structured prompt: dataset summary, top insights, anomalies,
    #   chart descriptions → ask MODEL_SMART for executive narrative prose
    # - Load Jinja2 template from templates/report.html.j2
    # - Render template with: narrative, chart_specs (plotly JSON),
    #   insights table, anomalies table, schema summary
    # - Optionally call WeasyPrint to produce a PDF alongside the HTML
    # - Return {"report_html": html_string, "error_log": []}

    _llm = ChatAnthropic(  # noqa: F841
        model=MODEL_SMART,
        api_key=ANTHROPIC_API_KEY,
    )

    return {
        "report_html": None,  # replace with rendered HTML string
        "error_log": [],
    }
