"""Chart helpers — serialisation, rendering, and format conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_plotly_figure(figure_json: str) -> Any:
    """
    Deserialise a Plotly figure from its JSON representation.

    Args:
        figure_json: JSON string produced by plotly.io.to_json(fig).

    Returns:
        plotly.graph_objects.Figure instance.

    Raises:
        ValueError: If figure_json cannot be parsed or is not a valid figure.
    """
    # TODO: implement figure deserialisation
    # - json.loads(figure_json) → dict
    # - plotly.io.from_json(figure_json) → go.Figure
    # - Return the Figure

    raise NotImplementedError("render_plotly_figure is not yet implemented")


def save_figure_json(figure: Any, output_path: str | Path) -> str:
    """
    Serialise a Plotly figure to JSON and optionally write it to disk.

    Args:
        figure: plotly.graph_objects.Figure to serialise.
        output_path: File path where the JSON should be written.
                     Parent directories are created if needed.

    Returns:
        The JSON string (also written to output_path).
    """
    # TODO: implement figure serialisation
    # - import plotly.io as pio
    # - json_str = pio.to_json(figure)
    # - Ensure parent dir exists, write file
    # - Return json_str

    raise NotImplementedError("save_figure_json is not yet implemented")


def figure_to_html_div(figure_json: str, *, div_id: str = "chart") -> str:
    """
    Produce a self-contained HTML <div> with embedded Plotly JSON.

    The returned string is safe to inject into a Jinja2 template;
    it relies on the plotly.js CDN script loaded in the outer page.

    Args:
        figure_json: JSON string of the figure.
        div_id: HTML id attribute for the chart container.

    Returns:
        HTML string containing the <div> and inline <script>.
    """
    # TODO: implement HTML div generation
    # - Validate figure_json is parseable JSON
    # - Return:
    #   <div id="{div_id}"></div>
    #   <script>Plotly.newPlot('{div_id}', {data}, {layout});</script>

    data = json.loads(figure_json)
    chart_data = json.dumps(data.get("data", []))
    layout = json.dumps(data.get("layout", {}))
    return (
        f'<div id="{div_id}"></div>\n'
        f"<script>\n"
        f"  Plotly.newPlot('{div_id}', {chart_data}, {layout});\n"
        f"</script>"
    )
