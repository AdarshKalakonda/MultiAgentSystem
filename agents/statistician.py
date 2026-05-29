"""
Statistician agent — computes numerical statistics and converts them to Insights.

Pipeline
────────
1. Load the DataFrame from state["df_path"].
2. Run _COMPUTE_CODE inside execute_pandas_code (subprocess):
   - Pearson correlation matrix + top-3 correlated pairs
   - Descriptive stats (mean / median / std / skew / kurtosis) per numeric column
   - Value counts (top-5) for categorical columns
   - Linear trend (slope, R², p-value) per numeric column on the datetime axis
3. Pass the statistics JSON to MODEL_SMART (Sonnet) and parse the
   response into validated Insight objects.
4. Return {"insights": [...], "error_log": [...]}.
"""

from __future__ import annotations

import json
import re
import textwrap
import traceback
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import ANTHROPIC_API_KEY, MODEL_SMART
from state.schema import DataStoryState, Insight, SchemaResult
from tools.code_executor import execute_pandas_code
from tools.data_loader import load_dataframe


# ── Compute script (runs inside the sandbox subprocess) ───────────────────────
# Pre-injected names available: df, pd, np, stats (scipy.stats), json, math

_COMPUTE_CODE = textwrap.dedent("""\
    import itertools
    import math

    def _clean(v):
        \"\"\"Replace NaN / Inf with None for JSON safety.\"\"\"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_clean(x) for x in v]
        return v

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    result = {}

    # -- Descriptive stats per numeric column ----------------------------------
    desc = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        desc[col] = {
            "mean":     round(float(s.mean()), 6),
            "median":   round(float(s.median()), 6),
            "std":      round(float(s.std()), 6),
            "min":      round(float(s.min()), 6),
            "max":      round(float(s.max()), 6),
            "skew":     round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        }
    result["numeric_stats"] = desc

    # -- Categorical value counts (top 5) -------------------------------------
    cat_stats = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=True).head(5)
        cat_stats[col] = {str(k): int(v) for k, v in vc.items()}
    result["categorical_stats"] = cat_stats

    # -- Pearson correlation matrix + top-3 pairs -----------------------------
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(method="pearson")
        result["correlation_matrix"] = {
            c: {c2: round(float(v), 4) for c2, v in corr[c].items()}
            for c in corr.columns
        }
        pairs = [
            {
                "col1": c1, "col2": c2,
                "r":     round(float(corr.loc[c1, c2]), 4),
                "abs_r": abs(round(float(corr.loc[c1, c2]), 4)),
            }
            for c1, c2 in itertools.combinations(numeric_cols, 2)
            if pd.notna(corr.loc[c1, c2])
        ]
        result["top_correlations"] = sorted(pairs, key=lambda x: -x["abs_r"])[:3]
    else:
        result["correlation_matrix"] = {}
        result["top_correlations"] = []

    # -- Trend analysis via linregress on datetime axis -----------------------
    trends = {}
    if dt_cols and numeric_cols:
        dt_col = dt_cols[0]
        df_s = df.dropna(subset=[dt_col]).sort_values(dt_col)
        t_days = (
            (df_s[dt_col] - df_s[dt_col].min()).dt.total_seconds() / 86400
        ).values
        for col in numeric_cols:
            y = df_s[col].values.astype(float)
            mask = ~np.isnan(y)
            if mask.sum() >= 3:
                slope, _, r_val, p_val, _ = stats.linregress(t_days[mask], y[mask])
                trends[col] = {
                    "slope_per_day": round(float(slope), 6),
                    "r_squared":     round(float(r_val ** 2), 4),
                    "p_value":       round(float(p_val), 4),
                    "direction":     "upward" if slope > 0 else "downward",
                    "significant":   bool(p_val < 0.05),
                }
    result["trends"] = trends

    print(json.dumps(_clean(result)))
""")


# ── LLM helpers ───────────────────────────────────────────────────────────────

_SYS = (
    "You are a senior data analyst writing insights for a business report. "
    "Analyse the provided statistics and return a JSON array of insight objects. "
    "Focus on findings that are actionable, surprising, or have business impact. "
    "Return ONLY a valid JSON array — no prose, no markdown fences."
)

_INSIGHT_SCHEMA = (
    '[{"title":"<max 10 words>","description":"<2-3 sentences, plain English, business impact>",'
    '"severity":"<low|medium|high|critical>","column_refs":["<col>"]}]'
)


def _extract_json_list(text: str) -> list[dict]:
    """Extract a JSON array from a plain or markdown-wrapped LLM response."""
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


def _build_prompt(stats: dict, schema: SchemaResult) -> str:
    col_descs = {
        c.name: c.stats.get("description") or c.name
        for c in schema.columns
    }
    lines = [
        f"Dataset : {Path(schema.file_path).name}",
        f"Shape   : {schema.row_count} rows x {schema.column_count} cols",
        "",
        "Column descriptions:",
        *[f"  {n}: {d}" for n, d in col_descs.items()],
        "",
        "Computed statistics (truncated to 4 000 chars):",
        json.dumps(stats, indent=2)[:4000],
        "",
        "Return 3–6 insight objects matching EXACTLY this schema:",
        _INSIGHT_SCHEMA,
    ]
    return "\n".join(lines)


def _parse_insights(raw_list: list[Any]) -> tuple[list[Insight], list[str]]:
    insights: list[Insight] = []
    errors: list[str] = []
    valid_severities = {"low", "medium", "high", "critical"}

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            severity = str(item.get("severity", "medium")).lower()
            if severity not in valid_severities:
                severity = "medium"
            insights.append(
                Insight(
                    title=str(item.get("title", "Untitled")).strip(),
                    description=str(item.get("description", "")).strip(),
                    severity=severity,  # type: ignore[arg-type]
                    column_refs=[str(c) for c in item.get("column_refs", [])],
                )
            )
        except Exception as exc:
            errors.append(f"[statistician] insight parse: {exc} | item={item}")

    return insights, errors


# ── Public node function ───────────────────────────────────────────────────────

def run_statistician(state: DataStoryState) -> dict:
    """
    Compute statistics via a sandboxed subprocess, then interpret them with
    MODEL_SMART into a ranked list of Insight objects.

    Args:
        state: Must contain df_path and schema_info (set by profiler).

    Returns:
        Partial state update: {"insights": [...], "error_log": [...]}.
    """
    schema: SchemaResult | None = state.get("schema_info")  # type: ignore[assignment]
    df_path: str = state.get("df_path", "")
    errors: list[str] = []

    if schema is None:
        return {
            "insights": [],
            "error_log": ["[statistician] schema_info is None — profiler may have failed"],
        }

    # ── 1. Load DataFrame ─────────────────────────────────────────────────────
    try:
        df = load_dataframe(df_path)["df"]
    except Exception as exc:
        return {"insights": [], "error_log": [f"[statistician] load error: {exc}"]}

    # ── 2. Compute statistics (subprocess) ────────────────────────────────────
    exec_result = execute_pandas_code(_COMPUTE_CODE, df, timeout=30)
    if not exec_result["success"]:
        errors.append(f"[statistician] compute failed: {exec_result['error']}")
        computed: dict = {}
    else:
        computed = exec_result.get("result") or {}

    if not computed:
        return {"insights": [], "error_log": errors}

    # ── 3. LLM interpretation ─────────────────────────────────────────────────
    try:
        llm = ChatAnthropic(
            model=MODEL_SMART,
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=2048,
        )
        prompt = _build_prompt(computed, schema)
        resp = llm.invoke(
            [SystemMessage(content=_SYS), HumanMessage(content=prompt)]
        )
        raw_text = resp.content if isinstance(resp.content, str) else str(resp.content)
        raw_list = _extract_json_list(raw_text)
    except Exception as exc:
        errors.append(f"[statistician] LLM error: {exc}\n{traceback.format_exc()}")
        return {"insights": [], "error_log": errors}

    # ── 4. Validate and build Insight objects ─────────────────────────────────
    insights, parse_errors = _parse_insights(raw_list)
    errors.extend(parse_errors)

    return {"insights": insights, "error_log": errors}


# ── Standalone smoke-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    from state.schema import ColumnInfo
    from tools.data_loader import load_sample_dataset

    dataset = sys.argv[1] if len(sys.argv) > 1 else "sales"
    print(f"Running statistician on sample dataset: '{dataset}'\n")

    load_res = load_sample_dataset(dataset)
    df = load_res["df"]
    sample_path = Path(__file__).parent.parent / "sample_data" / f"{dataset}.csv"

    # Build a minimal SchemaResult (no LLM enrichment needed for this test)
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

    result = run_statistician(_state)  # type: ignore[arg-type]

    if result["error_log"]:
        print("Errors / warnings:")
        for e in result["error_log"]:
            print(f"  {e}")
        print()

    insights = result["insights"]
    print(f"Insights ({len(insights)} total):\n")
    for i, ins in enumerate(insights, 1):
        print(f"[{i}] {ins.title}  [{ins.severity.upper()}]")
        print(f"     {ins.description}")
        print(f"     Columns: {ins.column_refs}\n")
