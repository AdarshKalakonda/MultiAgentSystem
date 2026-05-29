"""
Narrator agent — final node that synthesises all agent outputs into a
self-contained HTML report.

Pipeline
────────
1. Sort insights (critical → low) and anomalies by severity.
2. Build a compact context dict from state.
3. Call MODEL_SMART to generate three narrative sections as JSON:
   executive_summary, key_findings, next_steps.
4. Render the full HTML report in-process (inline CSS, no frameworks).
5. Embed each chart with plotly.io.from_json + fig.to_html(full_html=False).
6. Write to outputs/report.html and store report_html in state.
"""

from __future__ import annotations

import html
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import plotly.io as pio
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import ANTHROPIC_API_KEY, MODEL_SMART
from state.schema import Anomaly, DataStoryState, Insight, PlotlySpec, SchemaResult


# ── Severity ordering ─────────────────────────────────────────────────────────

_SEV = {"critical": 0, "high": 1, "medium": 2, "low": 3}

def _sev_key(obj: Insight | Anomaly) -> int:
    return _SEV.get(obj.severity, 4)


# ── Inline CSS (no external stylesheets) ─────────────────────────────────────

_CSS = """\
:root{--navy:#0F172A;--navy2:#1E293B;--bg:#F8FAFC;--card:#ffffff;
--b200:#E2E8F0;--b100:#F1F5F9;--t600:#475569;--t700:#334155;--t800:#1E293B;
--shadow:0 1px 3px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.04);--r:12px}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
'Helvetica Neue',Arial,sans-serif;font-size:15px;line-height:1.6;
color:var(--t800);background:var(--bg)}
/* header */
.dh{background:var(--navy);color:#fff;padding:28px 0}
.dh-i{max-width:1120px;margin:0 auto;padding:0 32px;display:flex;
justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px}
.dh-brand{font-size:10px;font-weight:700;letter-spacing:.14em;
text-transform:uppercase;color:#64748B;display:block;margin-bottom:6px}
.dh h1{font-size:24px;font-weight:700;margin:0;color:#fff}
.dh-meta{display:flex;align-items:center;flex-wrap:wrap;gap:8px}
.dh-ts{font-size:12px;color:#64748B;margin-left:4px}
/* badges */
.b{display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;
font-size:12px;font-weight:600;white-space:nowrap}
.b-info{background:#1E3A5F;color:#93C5FD}
.b-dom{background:#2D1B69;color:#C4B5FD}
.b-critical{background:#FEE2E2;color:#DC2626}
.b-high{background:#FEF3C7;color:#B45309}
.b-medium{background:#DBEAFE;color:#1D4ED8}
.b-low{background:#D1FAE5;color:#065F46}
.b-cnt{background:var(--b100);color:#64748B;padding:2px 8px;
border-radius:10px;font-size:12px;font-weight:600;margin-left:8px}
/* layout */
.dm{max-width:1120px;margin:32px auto;padding:0 32px;
display:flex;flex-direction:column;gap:24px}
/* cards */
.dc{background:var(--card);border:1px solid var(--b200);
border-radius:var(--r);padding:28px 32px;box-shadow:var(--shadow)}
.dc h2{font-size:17px;font-weight:700;color:var(--navy);margin:0 0 20px;
padding-bottom:14px;border-bottom:2px solid var(--b100);
display:flex;align-items:center}
/* overview stats */
.do{display:flex;gap:36px;flex-wrap:wrap}
.do-lbl{font-size:11px;font-weight:700;text-transform:uppercase;
letter-spacing:.07em;color:#94A3B8;margin-bottom:4px}
.do-val{font-size:26px;font-weight:700;color:var(--navy)}
.do-sub{font-size:13px;color:#64748B;margin-top:2px}
/* prose */
.dp p{margin:0 0 10px;color:var(--t600);font-size:15px;line-height:1.75}
.dp p:last-child{margin-bottom:0}
/* findings */
.df-list{list-style:none;padding:0;margin:0}
.df-item{display:flex;gap:16px;padding:16px 0;border-bottom:1px solid var(--b100)}
.df-item:last-child{border-bottom:none}
.df-num{min-width:28px;height:28px;border-radius:50%;background:var(--navy);
color:#fff;font-size:12px;font-weight:700;display:flex;align-items:center;
justify-content:center;flex-shrink:0;margin-top:2px}
.df-title{font-weight:600;color:var(--navy);margin-bottom:4px;font-size:15px}
.df-text{font-size:14px;color:var(--t600);line-height:1.55;margin:0}
/* charts */
.cg{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}
.cw{border:1px solid var(--b200);border-radius:8px;overflow:hidden}
.cc{font-size:12px;color:#64748B;padding:7px 14px;
background:var(--b100);border-top:1px solid var(--b200)}
/* anomalies */
.at{width:100%;border-collapse:collapse;font-size:14px}
.at th{background:var(--b100);padding:10px 16px;text-align:left;font-weight:600;
color:var(--t700);border-bottom:2px solid var(--b200)}
.at td{padding:12px 16px;border-bottom:1px solid var(--b100);
color:var(--t600);vertical-align:top}
.at tr:last-child td{border-bottom:none}
.at-exp{font-size:13px;color:#94A3B8;margin-top:5px;line-height:1.5}
.at-rows{font-family:'Courier New',monospace;font-size:12px;color:#94A3B8}
.ae{text-align:center;padding:40px 20px;color:#94A3B8;font-size:14px}
/* steps */
.sl{list-style:none;padding:0;margin:0}
.si{display:flex;gap:14px;align-items:flex-start;padding:12px 0;
border-bottom:1px solid var(--b100)}
.si:last-child{border-bottom:none}
.sb{min-width:24px;height:24px;border-radius:50%;background:#EFF6FF;
color:#2563EB;font-size:12px;font-weight:700;display:flex;align-items:center;
justify-content:center;flex-shrink:0;margin-top:1px}
.st{font-size:14px;color:var(--t600);line-height:1.55;padding-top:2px}
/* footer */
.ft{text-align:center;padding:32px;color:#94A3B8;font-size:13px;margin-top:8px}
.ft-sub{margin-top:4px;opacity:.7}"""


# ── HTML building blocks ──────────────────────────────────────────────────────

def _e(text: Any) -> str:
    """HTML-escape any value."""
    return html.escape(str(text))


def _badge(severity: str) -> str:
    cls = {"critical": "b-critical", "high": "b-high",
           "medium": "b-medium", "low": "b-low"}.get(severity, "b-info")
    return f'<span class="b {cls}">{_e(severity)}</span>'


def _render_header(ctx: dict) -> str:
    name  = _e(ctx["dataset_name"])
    ts    = _e(ctx["timestamp"])
    rows  = f"{ctx['row_count']:,}"
    cols  = str(ctx["col_count"])
    domain = _e(ctx.get("domain", "unknown"))
    return (
        '<header class="dh"><div class="dh-i">'
        '<div><span class="dh-brand">DataStory</span>'
        f'<h1>{name}</h1></div>'
        '<div class="dh-meta">'
        f'<span class="b b-info">{rows} rows</span>'
        f'<span class="b b-info">{cols} columns</span>'
        f'<span class="b b-dom">{domain}</span>'
        f'<span class="dh-ts">{ts}</span>'
        '</div></div></header>'
    )


def _render_overview(ctx: dict) -> str:
    schema: SchemaResult = ctx["schema"]
    desc   = _e(ctx.get("dataset_description", ""))
    num_c  = ", ".join(_e(c) for c in ctx.get("numeric_col_names", []))
    cat_c  = ", ".join(_e(c) for c in ctx.get("categorical_col_names", []))
    dt_rng = _e(ctx.get("datetime_range", ""))
    mem    = f"{schema.memory_usage_mb:.3f}"

    stats_html = (
        '<div class="do">'
        f'<div><div class="do-lbl">Rows</div>'
        f'<div class="do-val">{ctx["row_count"]:,}</div></div>'
        f'<div><div class="do-lbl">Columns</div>'
        f'<div class="do-val">{ctx["col_count"]}</div></div>'
        f'<div><div class="do-lbl">Memory</div>'
        f'<div class="do-val">{mem}</div>'
        f'<div class="do-sub">MB (deep)</div></div>'
        '</div>'
    )
    meta_rows = ""
    if num_c:
        meta_rows += (
            '<tr><td style="color:#94A3B8;font-size:13px;width:160px'
            ';padding:6px 0">Numeric columns</td>'
            f'<td style="font-size:13px;color:#475569">{num_c}</td></tr>'
        )
    if cat_c:
        meta_rows += (
            '<tr><td style="color:#94A3B8;font-size:13px;padding:6px 0">'
            'Categorical columns</td>'
            f'<td style="font-size:13px;color:#475569">{cat_c}</td></tr>'
        )
    if dt_rng:
        meta_rows += (
            '<tr><td style="color:#94A3B8;font-size:13px;padding:6px 0">'
            'Date range</td>'
            f'<td style="font-size:13px;color:#475569">{dt_rng}</td></tr>'
        )
    meta_table = (
        f'<table style="border-collapse:collapse;margin-top:16px">'
        f'{meta_rows}</table>'
    ) if meta_rows else ""

    desc_p = f'<p style="margin:16px 0 0;color:#475569;font-size:14px">{desc}</p>' if desc else ""

    return (
        '<section class="dc">'
        '<h2>Dataset Overview</h2>'
        + stats_html + meta_table + desc_p
        + '</section>'
    )


def _render_exec_summary(ctx: dict) -> str:
    summary_html = ctx.get("exec_summary_html", "")
    if not summary_html:
        summary_html = "<p>No executive summary available.</p>"
    return (
        '<section class="dc">'
        '<h2>Executive Summary</h2>'
        f'<div class="dp">{summary_html}</div>'
        '</section>'
    )


def _render_findings(ctx: dict) -> str:
    findings = ctx.get("llm_findings", [])
    insights = ctx.get("insights", [])
    count    = len(insights)

    items = []
    if findings:
        # LLM-structured findings
        for i, f in enumerate(findings[:6], 1):
            title = _e(f.get("title", "Finding"))
            body  = _e(f.get("body", ""))
            items.append(
                f'<li class="df-item">'
                f'<div class="df-num">{i}</div>'
                f'<div><div class="df-title">{title}</div>'
                f'<p class="df-text">{body}</p></div></li>'
            )
    else:
        # Fallback: render insights directly
        for i, ins in enumerate(insights[:6], 1):
            items.append(
                f'<li class="df-item">'
                f'<div class="df-num">{i}</div>'
                f'<div>'
                f'<div class="df-title">{_e(ins.title)}'
                f' {_badge(ins.severity)}</div>'
                f'<p class="df-text">{_e(ins.description)}</p>'
                f'</div></li>'
            )

    list_html = (
        f'<ol class="df-list">{"".join(items)}</ol>'
        if items else
        '<p style="color:#94A3B8;font-size:14px">No insights were generated.</p>'
    )

    return (
        '<section class="dc">'
        f'<h2>Key Findings<span class="b-cnt">{count}</span></h2>'
        + list_html
        + '</section>'
    )


def _render_charts(ctx: dict) -> str:
    specs: list[PlotlySpec] = ctx.get("chart_specs", [])
    count = len(specs)
    if not specs:
        return (
            '<section class="dc">'
            '<h2>Visualizations</h2>'
            '<p style="color:#94A3B8;text-align:center;padding:32px;font-size:14px">'
            'No charts were generated.</p></section>'
        )

    chart_divs = []
    for spec in specs:
        try:
            fig = pio.from_json(spec.figure_json)
            chart_inner = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True, "displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )
            caption = (
                f'<div class="cc">{_e(spec.insight)}</div>'
                if spec.insight else ""
            )
            chart_divs.append(f'<div class="cw">{chart_inner}{caption}</div>')
        except Exception as exc:
            chart_divs.append(
                f'<div class="cw" style="padding:20px;color:#94A3B8;font-size:13px">'
                f'Chart unavailable: {_e(str(exc))}</div>'
            )

    return (
        '<section class="dc">'
        f'<h2>Visualizations<span class="b-cnt">{count}</span></h2>'
        f'<div class="cg">{"".join(chart_divs)}</div>'
        '</section>'
    )


def _render_anomalies(ctx: dict) -> str:
    anomalies: list[Anomaly] = ctx.get("anomalies", [])
    count = len(anomalies)

    if not anomalies:
        body = (
            '<div class="ae">'
            '<div style="font-size:36px;margin-bottom:8px">&#10003;</div>'
            'No anomalies or data-quality issues detected.'
            '</div>'
        )
    else:
        rows = []
        for a in anomalies:
            row_preview = str(a.affected_rows[:6])[1:-1]
            if len(a.affected_rows) > 6:
                row_preview += f" … +{len(a.affected_rows) - 6} more"
            rows.append(
                f'<tr><td>{_badge(a.severity)}</td>'
                f'<td><strong>{_e(a.description)}</strong>'
                f'<div class="at-exp">{_e(a.explanation)}</div></td>'
                f'<td class="at-rows">{_e(row_preview)}</td></tr>'
            )
        body = (
            '<table class="at">'
            '<thead><tr>'
            '<th style="width:100px">Severity</th>'
            '<th>Finding</th>'
            '<th style="width:160px">Affected Rows</th>'
            '</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )

    return (
        '<section class="dc">'
        f'<h2>Data Quality &amp; Anomalies<span class="b-cnt">{count}</span></h2>'
        + body
        + '</section>'
    )


def _render_next_steps(ctx: dict) -> str:
    steps: list[str] = ctx.get("next_steps", [])
    if not steps:
        return ""

    items = "".join(
        f'<li class="si">'
        f'<div class="sb">{i}</div>'
        f'<div class="st">{_e(s)}</div>'
        f'</li>'
        for i, s in enumerate(steps[:5], 1)
    )
    return (
        '<section class="dc">'
        '<h2>Recommended Next Steps</h2>'
        f'<ul class="sl">{items}</ul>'
        '</section>'
    )


def _render_footer(ctx: dict) -> str:
    ts = _e(ctx["timestamp"])
    return (
        f'<footer class="ft">'
        f'Generated by <strong>DataStory</strong> &middot; {ts}'
        f'<div class="ft-sub">Powered by Claude AI &middot; Plotly &middot; pandas</div>'
        f'</footer>'
    )


# ── Full page assembly ────────────────────────────────────────────────────────

def _full_page(title: str, body: str) -> str:
    # Build <head> by concatenation so CSS braces are not interpreted as format vars
    head = (
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"<title>DataStory — {_e(title)}</title>\n"
        "<style>\n"
        + _CSS
        + "\n</style>\n"
        '<script src="https://cdn.plot.ly/plotly-latest.min.js" '
        'charset="utf-8"></script>\n'
        "</head>"
    )
    return "<!DOCTYPE html>\n<html lang=\"en\">\n" + head + "\n<body>\n" + body + "\n</body>\n</html>"


def _build_html(ctx: dict) -> str:
    body = "\n".join([
        _render_header(ctx),
        '<main class="dm">',
        _render_overview(ctx),
        _render_exec_summary(ctx),
        _render_findings(ctx),
        _render_charts(ctx),
        _render_anomalies(ctx),
        _render_next_steps(ctx),
        "</main>",
        _render_footer(ctx),
    ])
    return _full_page(ctx["dataset_name"], body)


# ── LLM narrative generation ──────────────────────────────────────────────────

_SYS = (
    "You are a senior data analyst writing a business report for non-technical "
    "stakeholders. Use plain language; avoid statistical jargon. "
    "Return ONLY valid JSON — no markdown, no prose outside the JSON."
)

_RESP_SCHEMA = """\
{
  "executive_summary": "<2-3 sentence plain-text overview: what the dataset contains and the single most important finding>",
  "key_findings": [
    {"title": "<max 10 words>", "body": "<2 sentences: the finding and its business implication>"}
  ],
  "next_steps": [
    "<actionable recommendation — one sentence each>"
  ],
  "plain_summary": "<3 bullet points separated by newline, each starting with •>"
}"""


def _extract_json_dict(text: str) -> dict:
    stripped = text.strip()
    for src in (
        stripped,
        (re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL) or None),
        (re.search(r"\{.*\}", stripped, re.DOTALL) or None),
    ):
        if src is None:
            continue
        candidate = src.group(1).strip() if hasattr(src, "group") else src
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _build_narrator_prompt(ctx: dict) -> str:
    insights: list[Insight] = ctx.get("insights", [])
    anomalies: list[Anomaly] = ctx.get("anomalies", [])

    ins_lines = "\n".join(
        f"  [{ins.severity.upper()}] {ins.title}: {ins.description[:200]}"
        for ins in insights[:6]
    ) or "  (none)"

    ano_lines = "\n".join(
        f"  [{a.severity.upper()}] {a.description}: {a.explanation[:200]}"
        for a in anomalies[:3]
    ) or "  (none)"

    interesting = ", ".join(ctx.get("interesting_columns", []))

    lines = [
        f"Dataset    : {ctx['dataset_name']}",
        f"Domain     : {ctx.get('domain', 'unknown')}",
        f"Shape      : {ctx['row_count']:,} rows x {ctx['col_count']} columns",
        f"Description: {ctx.get('dataset_description', '')}",
        f"Interesting: {interesting or '—'}",
        "",
        "Insights:",
        ins_lines,
        "",
        "Anomalies:",
        ano_lines,
        "",
        "Return EXACTLY this JSON (key_findings: 3-5 items, next_steps: 3 items):",
        _RESP_SCHEMA,
    ]
    return "\n".join(lines)


def _call_llm(ctx: dict) -> dict:
    """Call MODEL_SMART for narrative sections. Returns {} on any failure."""
    try:
        llm = ChatAnthropic(
            model=MODEL_SMART,
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=2048,
        )
        prompt = _build_narrator_prompt(ctx)
        resp = llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=prompt)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        return _extract_json_dict(raw)
    except Exception:
        return {}


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(state: DataStoryState) -> dict:
    """Extract everything the HTML renderer needs from the full state."""
    schema: SchemaResult | None = state.get("schema_info")   # type: ignore[assignment]
    metadata: dict = dict(state.get("metadata") or {})
    insights: list[Insight] = sorted(state.get("insights", []), key=_sev_key)
    chart_specs: list[PlotlySpec] = list(state.get("chart_specs", []))
    anomalies: list[Anomaly] = sorted(state.get("anomalies", []), key=_sev_key)

    dataset_name = (
        Path(state.get("df_path", "unknown")).name if state.get("df_path") else "Dataset"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build datetime range string if available
    dt_range = ""
    if schema:
        for col in schema.columns:
            if "datetime" in col.dtype.lower():
                min_d = col.stats.get("min_date", "")
                max_d = col.stats.get("max_date", "")
                if min_d and max_d:
                    dt_range = f"{str(min_d)[:10]} to {str(max_d)[:10]}"
                    break

    return {
        "dataset_name":         dataset_name,
        "timestamp":            timestamp,
        "row_count":            schema.row_count if schema else 0,
        "col_count":            schema.column_count if schema else 0,
        "schema":               schema,
        "domain":               metadata.get("domain", "unknown"),
        "dataset_description":  metadata.get("dataset_description", ""),
        "numeric_col_names":    metadata.get("numeric_col_names", []),
        "categorical_col_names":metadata.get("categorical_col_names", []),
        "interesting_columns":  metadata.get("interesting_columns", []),
        "datetime_range":       dt_range,
        "insights":             insights,
        "chart_specs":          chart_specs,
        "anomalies":            anomalies,
        # LLM sections filled in after the API call
        "exec_summary_html":    "",
        "llm_findings":         [],
        "next_steps":           [],
    }


# ── Public node function ──────────────────────────────────────────────────────

def run_narrator(state: DataStoryState) -> dict:
    """
    Generate a full HTML report from schema_info, insights, chart_specs,
    and anomalies.  The only external dependency is the Plotly CDN script
    tag — all CSS is inline.

    Args:
        state: Must contain all fields populated by prior agents.

    Returns:
        {"report_html": str, "metadata": {...}, "error_log": [...]}.
    """
    errors: list[str] = []
    metadata: dict = dict(state.get("metadata") or {})

    # ── 1. Build context ──────────────────────────────────────────────────────
    try:
        ctx = _build_context(state)
    except Exception as exc:
        errors.append(f"[narrator] context build failed: {exc}\n{traceback.format_exc()}")
        return {"report_html": None, "metadata": metadata, "error_log": errors}

    # ── 2. LLM narrative (best-effort) ────────────────────────────────────────
    llm_data: dict = {}
    try:
        llm_data = _call_llm(ctx)
        if not llm_data:
            errors.append("[narrator] LLM returned no usable content — using fallback prose")
    except Exception as exc:
        errors.append(f"[narrator] LLM error (non-fatal): {exc}")

    # Inject LLM sections into context
    raw_summary = llm_data.get("executive_summary", "")
    ctx["exec_summary_html"] = (
        f"<p>{_e(raw_summary)}</p>" if raw_summary else
        f"<p>{_e(ctx.get('dataset_description', ''))}</p>"
    )
    ctx["llm_findings"] = (
        llm_data.get("key_findings", [])
        if isinstance(llm_data.get("key_findings"), list) else []
    )
    ctx["next_steps"] = (
        llm_data.get("next_steps", [])
        if isinstance(llm_data.get("next_steps"), list) else []
    )

    # ── 3. Build HTML ─────────────────────────────────────────────────────────
    try:
        report_html = _build_html(ctx)
    except Exception as exc:
        errors.append(f"[narrator] HTML build failed: {exc}\n{traceback.format_exc()}")
        return {"report_html": None, "metadata": metadata, "error_log": errors}

    # ── 4. Write to outputs/ ──────────────────────────────────────────────────
    output_path: Path | None = None
    try:
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "report.html"
        output_path.write_text(report_html, encoding="utf-8")
    except Exception as exc:
        errors.append(f"[narrator] file write failed (non-fatal): {exc}")

    # ── 5. Update metadata ────────────────────────────────────────────────────
    plain_summary = llm_data.get("plain_summary", "")
    if not plain_summary and ctx["insights"]:
        ins = ctx["insights"]
        plain_summary = "\n".join([
            f"• {ins[0].title}" if len(ins) > 0 else "",
            f"• {ins[1].title}" if len(ins) > 1 else "",
            f"• {len(ctx['anomalies'])} anomaly group(s) detected",
        ]).strip()

    metadata.update({
        "summary":      plain_summary,
        "report_path":  str(output_path) if output_path else "",
        "charts_count": len(ctx["chart_specs"]),
        "insights_count": len(ctx["insights"]),
        "anomalies_count": len(ctx["anomalies"]),
    })

    return {
        "report_html": report_html,
        "metadata":    metadata,
        "error_log":   errors,
    }


# ── Full-pipeline smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import webbrowser

    from agents.anomaly import run_anomaly_detector
    from agents.profiler import run_profiler
    from agents.statistician import run_statistician
    from agents.viz_builder import run_viz_builder

    dataset = sys.argv[1] if len(sys.argv) > 1 else "sales"
    sample_path = Path(__file__).parent.parent / "sample_data" / f"{dataset}.csv"

    print(f"DataStory full pipeline on: {sample_path}")
    print("=" * 60)

    state: dict = {
        "df_path":    str(sample_path),
        "schema_info": None,
        "insights":   [],
        "chart_specs":[],
        "anomalies":  [],
        "report_html": None,
        "error_log":  [],
        "metadata":   {},
    }

    # 1. Profiler
    print("\n[1/5] Profiler ...")
    r = run_profiler(state)  # type: ignore[arg-type]
    state["schema_info"] = r.get("schema_info")
    state["metadata"]    = r.get("metadata") or {}
    state["error_log"]  += r.get("error_log", [])
    schema = state["schema_info"]
    print(f"      {schema.row_count if schema else 'FAIL'} rows, "
          f"{schema.column_count if schema else '?'} cols, "
          f"domain={state['metadata'].get('domain', '?')}")

    # 2. Statistician
    print("\n[2/5] Statistician ...")
    r = run_statistician(state)  # type: ignore[arg-type]
    state["insights"]   = r.get("insights", [])
    state["error_log"] += r.get("error_log", [])
    print(f"      {len(state['insights'])} insights")

    # 3. Viz builder
    print("\n[3/5] Viz builder ...")
    r = run_viz_builder(state)  # type: ignore[arg-type]
    state["chart_specs"] = r.get("chart_specs", [])
    state["error_log"]  += r.get("error_log", [])
    print(f"      {len(state['chart_specs'])} charts")

    # 4. Anomaly detector
    print("\n[4/5] Anomaly detector ...")
    r = run_anomaly_detector(state)  # type: ignore[arg-type]
    state["anomalies"]  = r.get("anomalies", [])
    state["error_log"] += r.get("error_log", [])
    print(f"      {len(state['anomalies'])} anomalies")

    # 5. Narrator
    print("\n[5/5] Narrator (building report) ...")
    r = run_narrator(state)  # type: ignore[arg-type]
    state["report_html"] = r.get("report_html")
    state["error_log"]  += r.get("error_log", [])

    # Print any errors
    all_errors = [e for e in state["error_log"] if e]
    if all_errors:
        print("\nErrors / warnings:")
        for e in all_errors:
            print(f"  {e[:140]}")

    # Summary
    meta = r.get("metadata", {})
    print("\n" + "=" * 60)
    print("Report summary:")
    print(f"  Insights  : {meta.get('insights_count', 0)}")
    print(f"  Charts    : {meta.get('charts_count', 0)}")
    print(f"  Anomalies : {meta.get('anomalies_count', 0)}")
    print(f"  Path      : {meta.get('report_path', 'N/A')}")
    if meta.get("summary"):
        print("\nPlain summary:")
        for line in meta["summary"].split("\n"):
            print(f"  {line}")

    # Open browser
    report_path = meta.get("report_path", "")
    if report_path and Path(report_path).exists():
        print(f"\nOpening: {report_path}")
        webbrowser.open(f"file:///{report_path}")
    else:
        print("\nERROR: Report file not found.")
        sys.exit(1)
