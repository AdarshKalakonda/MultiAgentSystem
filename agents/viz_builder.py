"""
VizBuilder agent — generates Plotly chart specifications in-process.

Design note
───────────
Charts are built directly with plotly.graph_objects in the host process —
NOT via code_executor.  Plotly Figure objects are not serialisable across
subprocess boundaries, and fig.to_json() produces the canonical wire format
that PlotlySpec stores.

Chart-selection rules (up to 4 charts total)
─────────────────────────────────────────────
1. Datetime + numeric  → line chart  (time-series)
2. Categorical × top-2 numeric cols → bar chart  (mean per category)
3. Two numerics correlated |r| ≥ 0.25 → scatter with OLS trendline
4. First numeric col   → histogram   (distribution)

Columns flagged is_interesting by the profiler are promoted to front.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

import plotly.graph_objects as go

from state.schema import DataStoryState, Insight, PlotlySpec, SchemaResult
from tools.data_loader import load_dataframe


# ── Chart palette & layout ────────────────────────────────────────────────────

_PALETTE = ["#3B82F6", "#F97316", "#10B981", "#8B5CF6", "#EF4444", "#F59E0B"]
_PRIMARY  = _PALETTE[0]
_ACCENT   = _PALETTE[1]

_BASE_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial, sans-serif", size=12, color="#374151"),
    title_font=dict(size=14, color="#111827"),
    margin=dict(l=60, r=30, t=55, b=60),
    hoverlabel=dict(bgcolor="white", font_size=12),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#E5E7EB",
        borderwidth=1,
    ),
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="#F3F4F6",
    gridwidth=1,
    zeroline=False,
    linecolor="#E5E7EB",
    linewidth=1,
)


def _fmt_label(col: str) -> str:
    """'revenue_usd' → 'Revenue Usd'."""
    return col.replace("_", " ").title()


# ── Column-type helpers ───────────────────────────────────────────────────────

def _is_numeric(dtype: str) -> bool:
    d = dtype.lower()
    return any(t in d for t in ("int", "float", "complex"))


def _is_datetime(dtype: str) -> bool:
    return "datetime" in dtype.lower()


def _is_categorical(dtype: str) -> bool:
    """Everything that is not numeric or datetime (includes bool, str, object)."""
    return not (_is_numeric(dtype) or _is_datetime(dtype))


# ── Chart-config dataclass ────────────────────────────────────────────────────

@dataclass
class _ChartConfig:
    chart_type: str          # "line" | "bar" | "scatter" | "histogram"
    title: str
    x_col: str
    y_col: str = ""          # empty for histogram
    insight_text: str = ""   # optional annotation text for PlotlySpec.insight


# ── Chart-selection logic ─────────────────────────────────────────────────────

def _pick_configs(schema: SchemaResult, df: pd.DataFrame) -> list[_ChartConfig]:
    """Return up to 4 chart configs ranked by expected analytical value."""
    interesting = {c.name for c in schema.columns if c.stats.get("is_interesting", False)}

    def _rank(name: str) -> int:
        return 0 if name in interesting else 1

    num_cols  = sorted(
        [c.name for c in schema.columns if _is_numeric(c.dtype)],
        key=_rank,
    )
    dt_cols   = [c.name for c in schema.columns if _is_datetime(c.dtype)]
    cat_cols  = sorted(
        [c.name for c in schema.columns if _is_categorical(c.dtype)],
        key=_rank,
    )

    configs: list[_ChartConfig] = []

    # Rule 1 — Time series: datetime + most-interesting numeric
    if dt_cols and num_cols:
        dt, num = dt_cols[0], num_cols[0]
        configs.append(_ChartConfig(
            chart_type="line",
            title=f"{_fmt_label(num)} Over Time",
            x_col=dt,
            y_col=num,
            insight_text=f"Trend of {_fmt_label(num)} across the observation period.",
        ))

    # Rule 2 — Bar charts: up to 2 categorical columns × top numeric
    for cat in cat_cols[:2]:
        if len(configs) >= 3:
            break
        num = num_cols[0] if num_cols else None
        if num:
            configs.append(_ChartConfig(
                chart_type="bar",
                title=f"Mean {_fmt_label(num)} by {_fmt_label(cat)}",
                x_col=cat,
                y_col=num,
                insight_text=f"Average {_fmt_label(num)} broken down by {_fmt_label(cat)}.",
            ))

    # Rule 3 — Scatter: two most-correlated numeric columns
    if len(num_cols) >= 2 and len(configs) < 4:
        x_c, y_c = num_cols[0], num_cols[1]
        try:
            r = float(df[[x_c, y_c]].dropna().corr().iloc[0, 1])
        except Exception:
            r = 0.0
        if abs(r) >= 0.25:
            configs.append(_ChartConfig(
                chart_type="scatter",
                title=f"{_fmt_label(x_c)} vs {_fmt_label(y_c)}",
                x_col=x_c,
                y_col=y_c,
                insight_text=f"Correlation r={r:.2f} between {_fmt_label(x_c)} and {_fmt_label(y_c)}.",
            ))

    # Rule 4 — Histogram: distribution of first numeric
    if num_cols and len(configs) < 4:
        num = num_cols[0]
        configs.append(_ChartConfig(
            chart_type="histogram",
            title=f"Distribution of {_fmt_label(num)}",
            x_col=num,
            insight_text=f"Frequency distribution of {_fmt_label(num)}.",
        ))

    return configs[:4]


# ── Individual chart builders ─────────────────────────────────────────────────

def _apply_layout(fig: go.Figure, title: str, x_label: str, y_label: str) -> None:
    fig.update_layout(
        title=dict(text=title, x=0.04),
        xaxis=dict(title=x_label, **_AXIS_STYLE),
        yaxis=dict(title=y_label, **_AXIS_STYLE),
        **_BASE_LAYOUT,
    )


def _make_line(df: pd.DataFrame, cfg: _ChartConfig) -> go.Figure:
    df_s = df.sort_values(cfg.x_col)
    fig = go.Figure(
        go.Scatter(
            x=df_s[cfg.x_col],
            y=df_s[cfg.y_col],
            mode="lines+markers",
            name=_fmt_label(cfg.y_col),
            line=dict(color=_PRIMARY, width=2),
            marker=dict(color=_PRIMARY, size=5),
        )
    )
    _apply_layout(fig, cfg.title, _fmt_label(cfg.x_col), _fmt_label(cfg.y_col))
    return fig


def _make_bar(df: pd.DataFrame, cfg: _ChartConfig) -> go.Figure:
    grouped = (
        df.groupby(cfg.x_col)[cfg.y_col]
        .mean()
        .reset_index()
        .sort_values(cfg.y_col, ascending=False)
    )
    fig = go.Figure(
        go.Bar(
            x=grouped[cfg.x_col].astype(str),
            y=grouped[cfg.y_col].round(2),
            marker_color=_PALETTE[: len(grouped)],
            text=grouped[cfg.y_col].round(1),
            textposition="outside",
        )
    )
    _apply_layout(fig, cfg.title, _fmt_label(cfg.x_col), f"Mean {_fmt_label(cfg.y_col)}")
    fig.update_layout(showlegend=False)
    return fig


def _make_scatter(df: pd.DataFrame, cfg: _ChartConfig) -> go.Figure:
    clean = df[[cfg.x_col, cfg.y_col]].dropna()
    x_vals = clean[cfg.x_col].values.astype(float)
    y_vals = clean[cfg.y_col].values.astype(float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name="Data",
            marker=dict(color=_PRIMARY, size=7, opacity=0.7,
                        line=dict(color="white", width=0.5)),
        )
    )

    # OLS trendline
    if len(x_vals) >= 2:
        slope, intercept, r_val, *_ = linregress(x_vals, y_vals)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"OLS  r={r_val:.2f}",
                line=dict(color=_ACCENT, width=2, dash="dash"),
            )
        )

    _apply_layout(fig, cfg.title, _fmt_label(cfg.x_col), _fmt_label(cfg.y_col))
    return fig


def _make_histogram(df: pd.DataFrame, cfg: _ChartConfig) -> go.Figure:
    vals = df[cfg.x_col].dropna()
    fig = go.Figure(
        go.Histogram(
            x=vals,
            nbinsx=min(20, max(5, len(vals) // 3)),
            marker_color=_PRIMARY,
            opacity=0.85,
        )
    )
    _apply_layout(fig, cfg.title, _fmt_label(cfg.x_col), "Count")
    fig.update_layout(showlegend=False, bargap=0.05)
    return fig


def _build_chart(df: pd.DataFrame, cfg: _ChartConfig) -> go.Figure:
    dispatch = {
        "line":      _make_line,
        "bar":       _make_bar,
        "scatter":   _make_scatter,
        "histogram": _make_histogram,
    }
    builder = dispatch.get(cfg.chart_type)
    if builder is None:
        raise ValueError(f"Unknown chart type: {cfg.chart_type!r}")
    return builder(df, cfg)


# ── Public node function ───────────────────────────────────────────────────────

def run_viz_builder(state: DataStoryState) -> dict:
    """
    Generate up to 4 Plotly chart specs from schema_info.

    Charts are produced in-process (no subprocess) because Plotly Figure
    objects cannot be serialised across code_executor boundaries.

    Args:
        state: Must contain df_path and schema_info (set by profiler).

    Returns:
        Partial state update: {"chart_specs": [...], "error_log": [...]}.
    """
    schema: SchemaResult | None = state.get("schema_info")  # type: ignore[assignment]
    df_path: str = state.get("df_path", "")
    errors: list[str] = []

    if schema is None:
        return {
            "chart_specs": [],
            "error_log": ["[viz_builder] schema_info is None — profiler may have failed"],
        }

    # ── 1. Load DataFrame ─────────────────────────────────────────────────────
    try:
        df = load_dataframe(df_path)["df"]
    except Exception as exc:
        return {"chart_specs": [], "error_log": [f"[viz_builder] load error: {exc}"]}

    # ── 2. Decide which charts to make ────────────────────────────────────────
    configs = _pick_configs(schema, df)
    if not configs:
        return {"chart_specs": [], "error_log": errors}

    # ── 3. Build each chart, serialise, wrap in PlotlySpec ────────────────────
    chart_specs: list[PlotlySpec] = []
    for cfg in configs:
        try:
            fig = _build_chart(df, cfg)
            figure_json = fig.to_json()
            chart_specs.append(
                PlotlySpec(
                    chart_type=cfg.chart_type,
                    title=cfg.title,
                    figure_json=figure_json,
                    insight=cfg.insight_text or None,
                )
            )
        except Exception as exc:
            errors.append(
                f"[viz_builder] chart '{cfg.title}' failed: {exc}\n"
                f"{traceback.format_exc()}"
            )

    return {"chart_specs": chart_specs, "error_log": errors}


# ── Standalone smoke-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    from state.schema import ColumnInfo
    from tools.data_loader import load_sample_dataset

    dataset = sys.argv[1] if len(sys.argv) > 1 else "sales"
    print(f"Running viz_builder on sample dataset: '{dataset}'\n")

    load_res = load_sample_dataset(dataset)
    df = load_res["df"]
    sample_path = Path(__file__).parent.parent / "sample_data" / f"{dataset}.csv"

    _cols = [
        ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            null_count=int(df[col].isna().sum()),
            null_pct=round(float(df[col].isna().mean()) * 100, 4),
            unique_count=int(df[col].nunique()),
            sample_values=[str(v) for v in df[col].dropna().head(3).tolist()],
        )
        for col in df.columns
    ]
    schema = SchemaResult(
        row_count=len(df),
        column_count=len(df.columns),
        columns=_cols,
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / 1e6, 6),
        file_path=str(sample_path),
    )

    _state: dict = {
        "df_path": str(sample_path),
        "schema_info": schema,
        "insights": [], "chart_specs": [], "anomalies": [],
        "report_html": None, "error_log": [], "metadata": {},
    }

    result = run_viz_builder(_state)  # type: ignore[arg-type]

    if result["error_log"]:
        print("Errors / warnings:")
        for e in result["error_log"]:
            print(f"  {e}")
        print()

    specs = result["chart_specs"]
    print(f"Chart specs generated: {len(specs)}\n")
    for i, spec in enumerate(specs, 1):
        fig_size = len(spec.figure_json)
        print(f"  [{i}] {spec.chart_type:10s} | {spec.title}")
        print(f"        figure_json size: {fig_size:,} bytes")
        print(f"        insight: {(spec.insight or '')[:80]}")
