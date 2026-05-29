"""AnomalyDetector agent — finds outliers and data-quality issues."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from config import ANTHROPIC_API_KEY, MODEL_SMART
from state.schema import DataStoryState


def run_anomaly_detector(state: DataStoryState) -> dict:
    """
    Detect anomalies, outliers, and data-quality issues in the dataset.

    Responsibilities
    ────────────────
    1. Statistical outlier detection (IQR, z-score) for numeric columns.
    2. Isolation Forest for multivariate outliers (scikit-learn).
    3. Data-quality checks: impossible values, format inconsistencies,
       duplicate rows, referential integrity issues.
    4. Call MODEL_SMART to interpret findings and produce ranked Anomaly
       objects with human-readable explanations.

    Args:
        state: Must contain schema_info and df_path.

    Returns:
        Partial state dict with anomalies list appended.
    """
    # TODO: implement full anomaly_detector logic
    # - Load df from state["df_path"]
    # - For each numeric column: flag rows outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    # - Run sklearn.ensemble.IsolationForest on all numeric columns combined
    # - Check for: duplicate rows, mixed-type columns, out-of-range dates,
    #   negative values in columns that should be positive, etc.
    # - Prompt MODEL_SMART with findings → emit structured Anomaly objects
    # - Return {"anomalies": [Anomaly(...), ...], "error_log": []}

    _llm = ChatAnthropic(  # noqa: F841
        model=MODEL_SMART,
        api_key=ANTHROPIC_API_KEY,
    )

    return {
        "anomalies": [],  # replace with list[Anomaly]
        "error_log": [],
    }
