"""Pydantic v2 data models and LangGraph state schema for DataStory."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import TypedDict


# ── Column-level metadata ─────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    """Profile for a single DataFrame column."""

    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any] = []
    stats: dict[str, Any] = {}

    @field_validator("null_pct")
    @classmethod
    def clamp_pct(cls, v: float) -> float:
        """Ensure percentage is in [0.0, 100.0]."""
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"null_pct must be between 0 and 100, got {v}")
        return round(v, 4)


class SchemaResult(BaseModel):
    """Full dataset profile produced by the Profiler agent."""

    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    memory_usage_mb: float
    file_path: str = ""

    @model_validator(mode="after")
    def check_column_count(self) -> "SchemaResult":
        if len(self.columns) != self.column_count:
            raise ValueError(
                f"column_count={self.column_count} does not match "
                f"len(columns)={len(self.columns)}"
            )
        return self


# ── Analytical output models ──────────────────────────────────────────────────

SeverityLevel = Literal["low", "medium", "high", "critical"]


class Insight(BaseModel):
    """A single analytical finding from the Statistician agent."""

    title: str
    description: str
    severity: SeverityLevel
    column_refs: list[str] = []

    @field_validator("title", "description")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty or whitespace")
        return v.strip()


class PlotlySpec(BaseModel):
    """A Plotly chart specification produced by the VizBuilder agent."""

    chart_type: str
    title: str
    figure_json: str  # JSON-serialised plotly.graph_objects.Figure
    insight: Optional[str] = None

    @field_validator("figure_json")
    @classmethod
    def valid_json(cls, v: str) -> str:
        import json
        try:
            json.loads(v)
        except json.JSONDecodeError as exc:
            raise ValueError(f"figure_json is not valid JSON: {exc}") from exc
        return v


class Anomaly(BaseModel):
    """An anomaly or data-quality issue found by the AnomalyDetector agent."""

    description: str
    affected_rows: list[int]
    severity: SeverityLevel
    explanation: str

    @model_validator(mode="after")
    def rows_non_negative(self) -> "Anomaly":
        if any(r < 0 for r in self.affected_rows):
            raise ValueError("affected_rows must contain non-negative indices")
        return self


# ── LangGraph state ───────────────────────────────────────────────────────────

class DataStoryState(TypedDict):
    """
    Shared mutable state threaded through the LangGraph.

    List fields use operator.add as the reducer so parallel branches
    (statistician, viz_builder, anomaly_detector) can append results
    without overwriting each other.
    """

    df_path: str
    schema_info: Optional[SchemaResult]
    insights: Annotated[list[Insight], operator.add]
    chart_specs: Annotated[list[PlotlySpec], operator.add]
    anomalies: Annotated[list[Anomaly], operator.add]
    report_html: Optional[str]
    error_log: Annotated[list[str], operator.add]
    metadata: dict[str, Any]
