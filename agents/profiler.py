"""Profiler agent — loads data and computes a full column-level schema."""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import ANTHROPIC_API_KEY, MODEL_FAST
from state.schema import ColumnInfo, DataStoryState, SchemaResult
from tools.data_loader import load_dataframe


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _to_scalar(val: Any) -> Any:
    """Coerce numpy / pandas types to JSON-safe Python primitives."""
    import numpy as np  # noqa: PLC0415

    if val is None:
        return None
    if isinstance(val, pd.NaT.__class__):
        return None
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return None if np.isnan(val) else float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _safe_list(values: list[Any]) -> list[Any]:
    """Apply _to_scalar to every element in a list."""
    return [_to_scalar(v) for v in values]


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Parse JSON from a plain string or a markdown code fence.

    Tries three strategies in order:
    1. Direct json.loads on the stripped text.
    2. Content inside a ```json ... ``` fence.
    3. Largest {...} block found via regex.

    Returns an empty dict if all three fail.
    """
    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return {}


# ── Pandas statistics ─────────────────────────────────────────────────────────

def _is_categorical(series: pd.Series) -> bool:
    """Return True if the column holds text / categorical data.

    Works across pandas 2 (object dtype) and pandas 4 (str / StringDtype)
    by checking what the column is NOT rather than what it is.
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def _compute_column_stats(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Compute per-column and dataset-level stats using only pandas.

    Args:
        df: Loaded DataFrame.

    Returns:
        (col_stats, dataset_meta) where col_stats is a list of raw dicts
        (one per column) and dataset_meta holds dataset-level aggregates.
    """
    col_stats: list[dict] = []

    for col in df.columns:
        s = df[col]
        n = len(s)

        null_count = int(s.isna().sum())
        null_pct = round(null_count / n * 100, 4) if n > 0 else 0.0
        unique_count = int(s.nunique(dropna=True))

        # 3 reproducible non-null sample values
        sample_values = _safe_list(s.dropna().head(3).tolist())

        # Top-3 most frequent values (robust across all dtypes)
        try:
            top_3 = _safe_list(
                s.value_counts(dropna=True).head(3).index.tolist()
            )
        except Exception:
            top_3 = []

        stats: dict[str, Any] = {"top_3_values": top_3}

        if pd.api.types.is_numeric_dtype(s) and not s.isna().all():
            stats["min"] = _to_scalar(s.min())
            stats["max"] = _to_scalar(s.max())
            stats["mean"] = round(float(s.mean()), 6)
            stats["std"] = round(float(s.std()), 6)
            stats["median"] = _to_scalar(s.median())

        if pd.api.types.is_datetime64_any_dtype(s) and not s.isna().all():
            stats["min_date"] = _to_scalar(s.min())
            stats["max_date"] = _to_scalar(s.max())
            stats["date_range_days"] = int((s.max() - s.min()).days)

        col_stats.append(
            {
                "name": col,
                "dtype": str(s.dtype),
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_count": unique_count,
                "sample_values": sample_values,
                "stats": stats,
            }
        )

    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 6)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    cat_cols = [c for c in df.columns if _is_categorical(df[c])]

    dataset_meta = {
        "row_count": len(df),
        "col_count": len(df.columns),
        "memory_mb": memory_mb,
        "has_datetime_cols": len(datetime_cols) > 0,
        "datetime_col_names": datetime_cols,
        "numeric_col_names": numeric_cols,
        "categorical_col_names": cat_cols,
    }

    return col_stats, dataset_meta


# ── LLM enrichment ────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a data analyst. Analyse the dataset summary and return a single "
    "JSON object with no surrounding text or markdown fences."
)

_JSON_SCHEMA = """\
{
  "domain": "<sales|hr|finance|ecommerce|logistics|healthcare|education|other>",
  "domain_confidence": <0.0-1.0>,
  "dataset_description": "<one sentence: what does this dataset contain?>",
  "interesting_columns": ["<col>", "<col>"],
  "string_date_columns": ["<col>"],
  "column_descriptions": {
    "<col_name>": "<one sentence plain-English description>"
  }
}"""


def _build_prompt(df_path: str, col_stats: list[dict], meta: dict) -> str:
    """Build a compact dataset summary to send to the LLM."""
    lines = [
        f"File   : {Path(df_path).name}",
        f"Shape  : {meta['row_count']} rows × {meta['col_count']} columns",
        f"Memory : {meta['memory_mb']} MB",
        "",
        "Columns (name | dtype | null% | unique | samples | extra):",
    ]

    for c in col_stats:
        parts = [
            f"  {c['name']}",
            f"dtype={c['dtype']}",
            f"null={c['null_pct']}%",
            f"unique={c['unique_count']}",
            f"samples={c['sample_values']}",
        ]
        s = c["stats"]
        if "min" in s:
            parts.append(f"range=[{s['min']}, {s['max']}] mean={s['mean']}")
        if "min_date" in s:
            parts.append(f"dates=[{s['min_date'][:10]}, {s['max_date'][:10]}]")
        if s.get("top_3_values"):
            parts.append(f"top3={s['top_3_values']}")
        lines.append("  " + "  |  ".join(parts))

    lines += [
        "",
        "Return EXACTLY this JSON structure (all keys required):",
        _JSON_SCHEMA,
    ]
    return "\n".join(lines)


def _call_llm(df_path: str, col_stats: list[dict], meta: dict) -> dict:
    """
    Call Claude haiku to enrich the schema with domain and descriptions.

    This call is best-effort: any exception returns {"_llm_error": <msg>}
    so the pipeline always continues with at least the pandas stats.
    """
    try:
        llm = ChatAnthropic(
            model=MODEL_FAST,
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=1024,
        )
        prompt = _build_prompt(df_path, col_stats, meta)
        response = llm.invoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        )
        raw: str = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        parsed = _extract_json(raw)
        if not parsed:
            return {"_llm_error": f"Could not parse JSON from response: {raw[:200]}"}
        return parsed
    except Exception as exc:  # noqa: BLE001
        return {"_llm_error": str(exc)}


# ── SchemaResult assembly ─────────────────────────────────────────────────────

def _assemble_schema(
    df_path: str,
    col_stats: list[dict],
    dataset_meta: dict,
    llm_data: dict,
) -> SchemaResult:
    """
    Merge pandas stats and LLM enrichment into a validated SchemaResult.

    LLM fields that are absent or of wrong type fall back to safe defaults.
    """
    descriptions: dict[str, str] = {}
    if isinstance(llm_data.get("column_descriptions"), dict):
        descriptions = {
            k: str(v) for k, v in llm_data["column_descriptions"].items()
        }

    string_date_cols: set[str] = set(
        llm_data.get("string_date_columns", [])
        if isinstance(llm_data.get("string_date_columns"), list)
        else []
    )
    interesting_cols: set[str] = set(
        llm_data.get("interesting_columns", [])
        if isinstance(llm_data.get("interesting_columns"), list)
        else []
    )

    columns: list[ColumnInfo] = []
    for c in col_stats:
        enriched = dict(c["stats"])
        enriched["description"] = descriptions.get(c["name"], "")
        enriched["is_string_date"] = c["name"] in string_date_cols
        enriched["is_interesting"] = c["name"] in interesting_cols

        columns.append(
            ColumnInfo(
                name=c["name"],
                dtype=c["dtype"],
                null_count=c["null_count"],
                null_pct=c["null_pct"],
                unique_count=c["unique_count"],
                sample_values=c["sample_values"],
                stats=enriched,
            )
        )

    return SchemaResult(
        row_count=dataset_meta["row_count"],
        column_count=dataset_meta["col_count"],
        columns=columns,
        memory_usage_mb=dataset_meta["memory_mb"],
        file_path=str(df_path),
    )


# ── Public node function ──────────────────────────────────────────────────────

def run_profiler(state: DataStoryState) -> dict:
    """
    Profile the dataset at state['df_path'] and return schema_info.

    Execution steps
    ───────────────
    1. Load the file with tools.data_loader.load_dataframe.
    2. Compute per-column and dataset-level stats with pandas only.
    3. Call Claude haiku (best-effort) to identify domain, write column
       descriptions, flag date-as-string columns, and rank interest.
    4. Merge pandas stats + LLM output into a validated SchemaResult.
    5. Write domain / interesting_columns / column meta into state metadata.

    Any individual step failure is caught; the pipeline always gets at
    least the pandas-computed schema even if the LLM call is unavailable.

    Args:
        state: Must contain df_path set by the orchestrator.

    Returns:
        Partial state update: schema_info, metadata (extended), error_log.
    """
    df_path: str = state.get("df_path", "")
    metadata: dict = dict(state.get("metadata") or {})
    errors: list[str] = []

    # ── Step 1: load ──────────────────────────────────────────────────────────
    try:
        load_result = load_dataframe(df_path)
        df: pd.DataFrame = load_result["df"]
    except FileNotFoundError as exc:
        errors.append(f"[profiler] file not found: {exc}")
        return {"schema_info": None, "metadata": metadata, "error_log": errors}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[profiler] load failed: {exc}")
        return {"schema_info": None, "metadata": metadata, "error_log": errors}

    # ── Step 2: pandas stats ──────────────────────────────────────────────────
    try:
        col_stats, dataset_meta = _compute_column_stats(df)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[profiler] stats computation failed: {exc}\n{traceback.format_exc()}")
        return {"schema_info": None, "metadata": metadata, "error_log": errors}

    # ── Step 3: LLM enrichment (best-effort) ──────────────────────────────────
    llm_data: dict = {}
    try:
        llm_data = _call_llm(df_path, col_stats, dataset_meta)
        if "_llm_error" in llm_data:
            errors.append(f"[profiler] LLM warning (non-fatal): {llm_data['_llm_error']}")
            llm_data = {}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[profiler] LLM error (non-fatal): {exc}")

    # ── Step 4: assemble SchemaResult ─────────────────────────────────────────
    try:
        schema_result = _assemble_schema(df_path, col_stats, dataset_meta, llm_data)
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"[profiler] schema assembly failed: {exc}\n{traceback.format_exc()}"
        )
        return {"schema_info": None, "metadata": metadata, "error_log": errors}

    # ── Step 5: extend metadata ───────────────────────────────────────────────
    metadata.update(
        {
            "domain": llm_data.get("domain", "unknown"),
            "domain_confidence": float(llm_data.get("domain_confidence", 0.0)),
            "dataset_description": llm_data.get("dataset_description", ""),
            "interesting_columns": list(llm_data.get("interesting_columns") or []),
            "string_date_columns": list(llm_data.get("string_date_columns") or []),
            "has_datetime_cols": dataset_meta["has_datetime_cols"],
            "datetime_col_names": dataset_meta["datetime_col_names"],
            "numeric_col_names": dataset_meta["numeric_col_names"],
            "categorical_col_names": dataset_meta["categorical_col_names"],
            "memory_mb": dataset_meta["memory_mb"],
        }
    )

    return {
        "schema_info": schema_result,
        "metadata": metadata,
        "error_log": errors,
    }


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target is None:
        target = str(Path(__file__).parent.parent / "sample_data" / "sales.csv")

    print(f"Running profiler on: {target}\n")

    _state: dict = {
        "df_path": target,
        "schema_info": None,
        "insights": [],
        "chart_specs": [],
        "anomalies": [],
        "report_html": None,
        "error_log": [],
        "metadata": {},
    }

    result = run_profiler(_state)  # type: ignore[arg-type]

    if result.get("error_log"):
        print("Errors / warnings:")
        for e in result["error_log"]:
            print(f"  {e}")
        print()

    schema: SchemaResult | None = result.get("schema_info")
    if schema is None:
        print("Profiler returned no schema_info — see errors above.")
        sys.exit(1)

    print("=== SchemaResult ===")
    print(schema.model_dump_json(indent=2))

    print("\n=== Metadata written to state ===")
    meta = result.get("metadata", {})
    print(json.dumps(
        {k: v for k, v in meta.items()},
        indent=2,
        default=str,
    ))
