"""
AnomalyDetector agent — flags outliers and data-quality issues.

Pipeline
────────
1. Load the DataFrame from state["df_path"].
2. Run _OUTLIER_CODE inside execute_pandas_code (subprocess):
   - IQR rule (1.5×IQR) per numeric column
   - Z-score (|z| > 3) per numeric column
   - Duplicate row detection
   - Null-clustering check (are nulls concentrated at start/end?)
3. Run Isolation Forest IN-PROCESS on all numeric columns
   (sklearn is heavy to import in the sandbox; in-process is faster).
4. Merge all findings, call MODEL_SMART to explain each cluster in
   plain English with a severity score and business-impact statement.
5. Return {"anomalies": [...], "error_log": [...]}.
"""

from __future__ import annotations

import json
import re
import textwrap
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import ANTHROPIC_API_KEY, MODEL_SMART
from state.schema import Anomaly, DataStoryState, SchemaResult
from tools.code_executor import execute_pandas_code
from tools.data_loader import load_dataframe


# ── Outlier compute script (runs in sandbox subprocess) ───────────────────────
# Pre-injected: df, pd, np, stats (scipy.stats), json, math

_OUTLIER_CODE = textwrap.dedent("""\
    import math

    def _int_list(idx):
        return [int(i) for i in idx]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    result = {
        "iqr_outliers":   {},
        "zscore_outliers":{},
        "duplicate_rows": int(df.duplicated().sum()),
        "null_patterns":  {},
    }

    # -- IQR outliers ----------------------------------------------------------
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr     = q3 - q1
        lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask    = (df[col] < lo) | (df[col] > hi)
        idx     = _int_list(df.index[mask & df[col].notna()])
        result["iqr_outliers"][col] = {
            "count": len(idx),
            "indices": idx[:50],          # cap to avoid huge JSON
            "lower_bound": round(lo, 4),
            "upper_bound": round(hi, 4),
        }

    # -- Z-score outliers (|z| > 3) -------------------------------------------
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 4 or s.std() == 0:
            result["zscore_outliers"][col] = {"count": 0, "indices": []}
            continue
        z    = np.abs((s - s.mean()) / s.std())
        idx  = _int_list(z[z > 3].index)
        result["zscore_outliers"][col] = {"count": len(idx), "indices": idx[:50]}

    # -- Null-pattern analysis -------------------------------------------------
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        if n_null == 0:
            continue
        n      = len(df)
        seg    = max(1, n // 5)
        head_n = int(df.head(seg)[col].isna().sum())
        tail_n = int(df.tail(seg)[col].isna().sum())
        clustered = (head_n > n_null * 0.5) or (tail_n > n_null * 0.5)
        result["null_patterns"][col] = {
            "null_count":  n_null,
            "null_pct":    round(n_null / n * 100, 2),
            "clustered":   clustered,
            "head_nulls":  head_n,
            "tail_nulls":  tail_n,
        }

    print(json.dumps(result))
""")


# ── Isolation Forest (in-process) ────────────────────────────────────────────

def _run_isolation_forest(df: pd.DataFrame, numeric_cols: list[str]) -> list[int]:
    """
    Fit an Isolation Forest on all numeric columns and return the row
    indices (in the original DataFrame index) flagged as anomalies.

    Returns an empty list if there are fewer than 2 numeric columns or
    fewer than 10 non-null rows.
    """
    from sklearn.ensemble import IsolationForest          # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler      # noqa: PLC0415

    if len(numeric_cols) < 2:
        return []

    X = df[numeric_cols].dropna()
    if len(X) < 10:
        return []

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    preds = iso.fit_predict(X_scaled)

    return [int(i) for i in X.index[preds == -1].tolist()]


# ── LLM helpers ───────────────────────────────────────────────────────────────

_SYS = (
    "You are a data-quality analyst writing findings for a business report. "
    "Interpret the provided anomaly statistics and return a JSON array of "
    "anomaly objects. Focus on findings with real business impact; skip "
    "trivial or purely statistical artefacts. "
    "Return ONLY a valid JSON array — no prose, no markdown fences."
)

_ANOMALY_SCHEMA = (
    '[{"description":"<short title, max 12 words>",'
    '"affected_rows":[<int>],'
    '"severity":"<low|medium|high|critical>",'
    '"explanation":"<2-3 sentences: what it is, why it matters, what to check>"}]'
)


def _format_findings(
    stats: dict,
    if_outliers: list[int],
    schema: SchemaResult,
) -> str:
    """Render all anomaly findings as a compact human-readable summary."""
    lines: list[str] = []

    # IQR
    for col, data in stats.get("iqr_outliers", {}).items():
        if data["count"] > 0:
            lines.append(
                f"IQR outliers in '{col}': {data['count']} rows "
                f"{data['indices'][:6]} | bounds=[{data['lower_bound']}, {data['upper_bound']}]"
            )

    # Z-score
    for col, data in stats.get("zscore_outliers", {}).items():
        if data["count"] > 0:
            lines.append(
                f"Z-score (|z|>3) in '{col}': {data['count']} rows "
                f"{data['indices'][:6]}"
            )

    # Isolation Forest
    if if_outliers:
        lines.append(
            f"Isolation Forest (multivariate): {len(if_outliers)} rows "
            f"{if_outliers[:6]}"
        )

    # Duplicates
    dups = stats.get("duplicate_rows", 0)
    if dups:
        lines.append(f"Duplicate rows: {dups}")

    # Null patterns
    for col, data in stats.get("null_patterns", {}).items():
        pattern = "CLUSTERED (non-random)" if data["clustered"] else "scattered"
        lines.append(
            f"Nulls in '{col}': {data['null_count']} ({data['null_pct']}%) — {pattern}"
        )

    return "\n".join(lines) if lines else "No anomalies detected."


def _build_prompt(findings_text: str, schema: SchemaResult) -> str:
    col_descs = {c.name: c.stats.get("description") or c.name for c in schema.columns}
    lines = [
        f"Dataset : {Path(schema.file_path).name}",
        f"Shape   : {schema.row_count} rows x {schema.column_count} cols",
        "",
        "Column descriptions:",
        *[f"  {n}: {d}" for n, d in col_descs.items()],
        "",
        "Anomaly findings:",
        findings_text,
        "",
        "Return anomaly objects matching EXACTLY this schema (0 objects if nothing significant):",
        _ANOMALY_SCHEMA,
    ]
    return "\n".join(lines)


def _extract_json_list(text: str) -> list[dict]:
    stripped = text.strip()
    for attempt in (
        stripped,
        re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL),
        re.search(r"\[.*\]", stripped, re.DOTALL),
    ):
        src = attempt.group(1).strip() if hasattr(attempt, "group") else (attempt or "")
        if not src:
            continue
        try:
            parsed = json.loads(src)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def _parse_anomalies(raw_list: list[Any]) -> tuple[list[Anomaly], list[str]]:
    anomalies: list[Anomaly] = []
    errors: list[str] = []
    valid_severities = {"low", "medium", "high", "critical"}

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            severity = str(item.get("severity", "medium")).lower()
            if severity not in valid_severities:
                severity = "medium"
            rows = [int(r) for r in item.get("affected_rows", []) if r >= 0]
            anomalies.append(
                Anomaly(
                    description=str(item.get("description", "Anomaly detected")).strip(),
                    affected_rows=rows,
                    severity=severity,  # type: ignore[arg-type]
                    explanation=str(item.get("explanation", "")).strip(),
                )
            )
        except Exception as exc:
            errors.append(f"[anomaly] parse error: {exc} | item={item}")

    return anomalies, errors


# ── Public node function ───────────────────────────────────────────────────────

def run_anomaly_detector(state: DataStoryState) -> dict:
    """
    Detect anomalies using statistical rules, Isolation Forest, and an
    LLM explanation step.

    Args:
        state: Must contain df_path and schema_info (set by profiler).

    Returns:
        Partial state update: {"anomalies": [...], "error_log": [...]}.
    """
    schema: SchemaResult | None = state.get("schema_info")  # type: ignore[assignment]
    df_path: str = state.get("df_path", "")
    errors: list[str] = []

    if schema is None:
        return {
            "anomalies": [],
            "error_log": ["[anomaly] schema_info is None — profiler may have failed"],
        }

    # ── 1. Load DataFrame ─────────────────────────────────────────────────────
    try:
        df = load_dataframe(df_path)["df"]
    except Exception as exc:
        return {"anomalies": [], "error_log": [f"[anomaly] load error: {exc}"]}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # ── 2. Statistical checks (subprocess) ───────────────────────────────────
    exec_result = execute_pandas_code(_OUTLIER_CODE, df, timeout=30)
    if not exec_result["success"]:
        errors.append(f"[anomaly] outlier compute failed: {exec_result['error']}")
        stats_data: dict = {}
    else:
        stats_data = exec_result.get("result") or {}

    # ── 3. Isolation Forest (in-process) ─────────────────────────────────────
    if_outliers: list[int] = []
    try:
        if_outliers = _run_isolation_forest(df, numeric_cols)
    except Exception as exc:
        errors.append(f"[anomaly] Isolation Forest failed (non-fatal): {exc}")

    # ── 4. Build LLM prompt and get explanations ──────────────────────────────
    if not stats_data and not if_outliers:
        return {"anomalies": [], "error_log": errors}

    try:
        findings_text = _format_findings(stats_data, if_outliers, schema)
        prompt = _build_prompt(findings_text, schema)
        llm = ChatAnthropic(
            model=MODEL_SMART,
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=2048,
        )
        resp = llm.invoke(
            [SystemMessage(content=_SYS), HumanMessage(content=prompt)]
        )
        raw_text = resp.content if isinstance(resp.content, str) else str(resp.content)
        raw_list = _extract_json_list(raw_text)
    except Exception as exc:
        errors.append(f"[anomaly] LLM error: {exc}\n{traceback.format_exc()}")
        return {"anomalies": [], "error_log": errors}

    # ── 5. Validate Anomaly objects ───────────────────────────────────────────
    anomalies, parse_errors = _parse_anomalies(raw_list)
    errors.extend(parse_errors)

    return {"anomalies": anomalies, "error_log": errors}


# ── Standalone smoke-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from state.schema import ColumnInfo
    from tools.data_loader import load_sample_dataset

    dataset = sys.argv[1] if len(sys.argv) > 1 else "sales"
    print(f"Running anomaly detector on sample dataset: '{dataset}'\n")

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

    result = run_anomaly_detector(_state)  # type: ignore[arg-type]

    if result["error_log"]:
        print("Errors / warnings:")
        for e in result["error_log"]:
            print(f"  {e}")
        print()

    anomalies = result["anomalies"]
    print(f"Anomalies found: {len(anomalies)}\n")
    for i, a in enumerate(anomalies, 1):
        print(f"[{i}] {a.description}  [{a.severity.upper()}]")
        print(f"     Affected rows: {a.affected_rows[:8]}{'...' if len(a.affected_rows) > 8 else ''}")
        print(f"     {a.explanation}\n")
