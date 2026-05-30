"""
End-to-end LangGraph pipeline test for DataStory.

Runs the full graph on sample_data/sales.csv, prints per-node timing as
events arrive from the stream, analyses the parallel fan-out window, saves
the HTML report to outputs/graph_test_report.html, and opens it in the
default browser.

Usage
─────
    python test_graph.py [dataset]          # default: sales

    python test_graph.py hr
    python test_graph.py ecommerce
"""

from __future__ import annotations

import sys
import time
import webbrowser
from pathlib import Path

# ── Resolve project root so imports work when run from any CWD ───────────────
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

from graph import app                        # noqa: E402 — needs sys.path set first
from state.schema import DataStoryState      # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────────

_PARALLEL_NODES = {"statistician", "viz_builder", "anomaly_detector"}

# Fields that use operator.add reducers — must be accumulated, not replaced
_LIST_FIELDS = {"insights", "chart_specs", "anomalies", "error_log"}

_NODE_LABELS = {
    "profiler":          "profiler         ",
    "statistician":      "statistician     ",
    "viz_builder":       "viz_builder      ",
    "anomaly_detector":  "anomaly_detector ",
    "narrator":          "narrator         ",
}


# ── State accumulator ─────────────────────────────────────────────────────────

def _merge(base: dict, updates: dict) -> dict:
    """
    Apply a node's partial state update onto the accumulated state.

    List fields (operator.add reducers) are concatenated; all other
    fields are overwritten.
    """
    result = dict(base)
    for k, v in updates.items():
        if k in _LIST_FIELDS and isinstance(v, list):
            result[k] = list(result.get(k) or []) + v
        else:
            result[k] = v
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    dataset    = sys.argv[1] if len(sys.argv) > 1 else "sales"
    input_path = _PROJECT_ROOT / "sample_data" / f"{dataset}.csv"
    output_dir = _PROJECT_ROOT / "outputs"
    output_path = output_dir / "graph_test_report.html"

    if not input_path.exists():
        available = sorted(p.stem for p in (_PROJECT_ROOT / "sample_data").glob("*.csv"))
        print(f"ERROR: '{input_path}' not found.  Available: {', '.join(available)}")
        sys.exit(1)

    print("=" * 62)
    print("  DataStory - LangGraph Pipeline Test")
    print("=" * 62)
    print(f"  Dataset : {input_path}")
    print(f"  Output  : {output_path}")
    print()

    # ── Build initial state ───────────────────────────────────────────────
    initial_state: DataStoryState = {           # type: ignore[assignment]
        "df_path":    str(input_path),
        "schema_info": None,
        "insights":   [],
        "chart_specs":[],
        "anomalies":  [],
        "report_html": None,
        "error_log":  [],
        "metadata":   {},
    }

    # ── Stream the graph and collect per-node timing ──────────────────────
    accumulated = dict(initial_state)
    node_completion: dict[str, float] = {}      # node → wall-clock seconds
    t_start = time.perf_counter()

    print("  Node completions (wall-clock from pipeline start):")
    print("  " + "-" * 58)

    for chunk in app.stream(
        initial_state,
        config={"recursion_limit": 50},
    ):
        t_now = time.perf_counter() - t_start

        for node_name, node_output in chunk.items():
            node_completion[node_name] = t_now
            accumulated = _merge(accumulated, node_output)
            label = _NODE_LABELS.get(node_name, f"{node_name:18s}")

            # Build a brief status annotation
            if node_name == "profiler":
                schema = accumulated.get("schema_info")
                if schema:
                    ann = f"{schema.row_count}r x {schema.column_count}c  domain={accumulated.get('metadata', {}).get('domain', '?')}"
                else:
                    ann = "schema_info=None (profiler error)"
            elif node_name == "statistician":
                ann = f"{len(accumulated.get('insights', []))} insights"
            elif node_name == "viz_builder":
                ann = f"{len(accumulated.get('chart_specs', []))} charts"
            elif node_name == "anomaly_detector":
                ann = f"{len(accumulated.get('anomalies', []))} anomalies"
            elif node_name == "narrator":
                report_kb = len(accumulated.get("report_html") or "") // 1024
                ann = f"report {report_kb} KB"
            else:
                ann = ""

            print(f"  [{t_now:6.1f}s]  {label}  {ann}")

            # Print fan-out banner as soon as profiler finishes
            if node_name == "profiler":
                print(f"           >> fan-out: statistician + viz_builder + anomaly_detector")
                print(f"              running in parallel  ...")

    t_total = time.perf_counter() - t_start

    # ── Parallel execution analysis ───────────────────────────────────────
    if "profiler" in node_completion and _PARALLEL_NODES.issubset(node_completion):
        t_fanout = node_completion["profiler"]
        par_end_times = {n: node_completion[n] for n in _PARALLEL_NODES}
        t_fanin = max(par_end_times.values())

        wall_parallel   = t_fanin - t_fanout
        sequential_est  = sum(t - t_fanout for t in par_end_times.values())

        print()
        print("  " + "-" * 58)
        print("  Parallel fan-out analysis:")
        print(f"    Fan-out start (profiler done) : t = {t_fanout:.1f}s")
        for n in sorted(_PARALLEL_NODES, key=lambda x: par_end_times[x]):
            delta = par_end_times[n] - t_fanout
            print(f"    {n:26s}: t = {par_end_times[n]:.1f}s  (+{delta:.1f}s after fan-out)")
        print(f"    Parallel wall-clock           : {wall_parallel:.1f}s")
        print(f"    Sequential estimate           : {sequential_est:.1f}s")
        if wall_parallel > 0:
            speedup = sequential_est / wall_parallel
            print(f"    Speedup (estimated)           : {speedup:.1f}x")

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print("  " + "=" * 58)
    print("  Pipeline complete")
    print(f"    Total runtime : {t_total:.1f}s")
    print(f"    Insights      : {len(accumulated.get('insights', []))}")
    print(f"    Charts        : {len(accumulated.get('chart_specs', []))}")
    print(f"    Anomalies     : {len(accumulated.get('anomalies', []))}")

    # ── Save report ───────────────────────────────────────────────────────
    report_html: str = accumulated.get("report_html") or ""
    output_dir.mkdir(exist_ok=True)

    if report_html:
        output_path.write_text(report_html, encoding="utf-8")
        print(f"    Report saved  : {output_path}  ({len(report_html)//1024} KB)")
    else:
        # Fall back to the narrator's default output if HTML wasn't captured
        default = _PROJECT_ROOT / "outputs" / "report.html"
        if default.exists():
            import shutil
            shutil.copy(default, output_path)
            print(f"    Report copied : {default} -> {output_path}")
        else:
            print("    ERROR: no report HTML found in state or on disk")
            sys.exit(1)

    # ── Error summary ─────────────────────────────────────────────────────
    errors = [e for e in accumulated.get("error_log", []) if e]
    if errors:
        print(f"\n  {len(errors)} warning(s) / error(s):")
        for e in errors[:6]:
            print(f"    {e[:110]}")
        if len(errors) > 6:
            print(f"    ... ({len(errors) - 6} more)")

    # ── Open browser ──────────────────────────────────────────────────────
    print(f"\n  Opening report in browser: {output_path}")
    webbrowser.open(output_path.as_uri())


if __name__ == "__main__":
    main()
