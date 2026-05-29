"""
Sandboxed Python executor for LLM-generated pandas / Plotly snippets.

Isolation model
───────────────
Each call spawns a fresh subprocess via sys.executable so that:
  - Crashed or infinite-looping code cannot affect the host process.
  - A hard wall-clock timeout (subprocess.communicate + Popen.kill)
    is enforced even if the code monkey-patches threading or signals.
  - Outbound network calls are blocked at socket level inside the sandbox.

Contract for user code
──────────────────────
The last statement in `code` MUST be:

    print(json.dumps(result))

where `result` is any JSON-serialisable dict.  Anything printed before
that line is ignored.  The executor parses only the LAST line of stdout
as JSON; this tolerates debug prints earlier in the script.

Pre-injected names (available without import in user code)
──────────────────────────────────────────────────────────
  df            — the pandas DataFrame passed to execute_pandas_code
  pd            — pandas
  np            — numpy
  stats         — scipy.stats
  px            — plotly.express
  go            — plotly.graph_objects
  pio           — plotly.io
  json          — standard library json
  math, re,     — standard library utilities
  datetime, Path
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TypedDict

import pandas as pd


# ── Return type ────────────────────────────────────────────────────────────────

class ExecResult(TypedDict):
    """Typed return value from execute_pandas_code."""

    success: bool
    result: dict | None   # parsed JSON output from the script
    error: str | None     # stderr or parse error message
    runtime_ms: int       # wall-clock time in milliseconds


# ── Prelude injected at the top of every generated script ─────────────────────

# {csv_path} is substituted at runtime with the actual temp-file path.
_PRELUDE_TEMPLATE = """\
import json
import math
import os
import re
import socket
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    px = go = pio = None  # type: ignore[assignment]

warnings.filterwarnings("ignore")

# Block outbound network calls by replacing the DNS resolver.
# urlopen / requests / httpx all call getaddrinfo before connecting;
# raising here prevents any hostname-based TCP connection without needing
# OS-level firewall rules.  setdefaulttimeout(0) as an extra layer.
def _no_network(*_args, **_kwargs):
    raise PermissionError("[Sandbox] Network access is disabled.")
socket.getaddrinfo = _no_network
socket.setdefaulttimeout(0)

df = pd.read_csv({csv_path!r})
"""


# ── Private helpers ────────────────────────────────────────────────────────────

def _write_temp_csv(df: pd.DataFrame) -> str:
    """
    Persist df to a named temp CSV file and return its path.

    Uses delete=False because Windows locks the file while it is open;
    the caller is responsible for unlinking the path in a finally block.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="datastory_exec_",
        delete=False,
        encoding="utf-8",
        newline="",
    ) as fh:
        df.to_csv(fh, index=False)
        return fh.name


def _write_temp_script(script: str) -> str:
    """
    Write the full Python script to a named temp .py file.

    Using a file rather than -c avoids argument-length limits on Windows
    and keeps the subprocess call simple.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="datastory_script_",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(script)
        return fh.name


def _build_script(csv_path: str, code: str) -> str:
    """Concatenate the data prelude and the user-supplied code."""
    prelude = _PRELUDE_TEMPLATE.format(csv_path=csv_path)
    return prelude + "\n" + code


def _parse_output(stdout: str, stderr: str, returncode: int) -> ExecResult:
    """
    Interpret subprocess stdout/stderr into an ExecResult.

    Only the LAST non-empty line of stdout is parsed as JSON so that
    earlier debug prints do not break the protocol.
    """
    if returncode != 0:
        # Return the full traceback, capped at 2 000 chars.
        # Truncating only the middle keeps both the user's line and the
        # final error type / message visible.
        tb = stderr.strip() or "(no stderr)"
        if len(tb) > 2000:
            tb = tb[:900] + "\n... (truncated) ...\n" + tb[-900:]
        return ExecResult(success=False, result=None, error=tb, runtime_ms=0)

    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return ExecResult(
            success=False,
            result=None,
            error="Script produced no output. Did you forget print(json.dumps(result))?",
            runtime_ms=0,
        )

    last_line = lines[-1]
    try:
        parsed = json.loads(last_line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
        return ExecResult(success=True, result=parsed, error=None, runtime_ms=0)
    except (json.JSONDecodeError, ValueError) as exc:
        preview = last_line[:200]
        return ExecResult(
            success=False,
            result=None,
            error=f"Could not parse JSON from last output line: {exc}\n  Got: {preview!r}",
            runtime_ms=0,
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def execute_pandas_code(
    code: str,
    df: pd.DataFrame,
    timeout: int = 30,
) -> ExecResult:
    """
    Execute a pandas / Plotly code snippet in an isolated subprocess.

    The DataFrame `df` is serialised to a temp CSV, injected into the
    subprocess via a prelude, and cleaned up regardless of outcome.

    Args:
        code: Python source to execute.  Must end with:
              ``print(json.dumps(result))``
              where ``result`` is a JSON-serialisable dict.
        df: DataFrame made available as ``df`` inside the script.
            Passed via a temp CSV — dtypes may be re-inferred on load.
        timeout: Wall-clock seconds before the subprocess is killed.
                 Default 30 s.

    Returns:
        ExecResult with keys:
          success    — True only if the script ran, exited 0, and printed JSON.
          result     — Parsed dict from the script's last print, or None.
          error      — Stderr / parse error message, or None on success.
          runtime_ms — Elapsed wall-clock time in milliseconds.

    Notes:
        - Network calls are blocked inside the sandbox (socket timeout = 0).
        - The temp CSV and temp .py file are always deleted in a finally block.
        - On Windows, delete=False is required for NamedTemporaryFile.
    """
    csv_path: str = ""
    script_path: str = ""
    start = time.monotonic()

    try:
        # ── 1. Persist DataFrame ───────────────────────────────────────────
        csv_path = _write_temp_csv(df)

        # ── 2. Build and persist script ────────────────────────────────────
        script = _build_script(csv_path, code)
        script_path = _write_temp_script(script)

        # ── 3. Spawn subprocess ────────────────────────────────────────────
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # drain pipes to avoid deadlock
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return ExecResult(
                success=False,
                result=None,
                error=f"Execution timed out after {timeout}s",
                runtime_ms=elapsed_ms,
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # ── 4. Parse and return ────────────────────────────────────────────
        result = _parse_output(stdout, stderr, proc.returncode)
        result["runtime_ms"] = elapsed_ms
        return result

    except Exception as exc:  # noqa: BLE001  — unexpected host-side error
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ExecResult(
            success=False,
            result=None,
            error=f"Executor internal error: {exc}",
            runtime_ms=elapsed_ms,
        )

    finally:
        # Always clean up temp files — never leave them on disk
        for path in (csv_path, script_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


# ── Standalone tests ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    _sample_df = pd.DataFrame(
        {
            "product": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"],
            "revenue": [1299.99, 749.99, 449.99, 329.99, 79.99],
            "units": [1, 3, 2, 4, 10],
        }
    )

    # ── Test 1: happy path ─────────────────────────────────────────────────
    print("=" * 60)
    print("Test 1: Happy path - mean revenue + correlation")
    print("=" * 60)
    happy_code = textwrap.dedent("""\
        mean_rev   = float(df["revenue"].mean())
        max_rev    = float(df["revenue"].max())
        corr       = float(df["revenue"].corr(df["units"]))
        result = {
            "mean_revenue": round(mean_rev, 2),
            "max_revenue":  round(max_rev,  2),
            "rev_units_corr": round(corr, 4),
            "row_count":    int(len(df)),
        }
        print(json.dumps(result))
    """)
    out = execute_pandas_code(happy_code, _sample_df)
    print(f"  success    : {out['success']}")
    print(f"  result     : {out['result']}")
    print(f"  error      : {out['error']}")
    print(f"  runtime_ms : {out['runtime_ms']}")

    # ── Test 2: syntax error ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Test 2: Syntax error - missing closing parenthesis")
    print("=" * 60)
    bad_syntax = textwrap.dedent("""\
        result = {"mean": float(df["revenue"].mean(}
        print(json.dumps(result))
    """)
    out = execute_pandas_code(bad_syntax, _sample_df)
    print(f"  success    : {out['success']}")
    print(f"  result     : {out['result']}")
    print(f"  error      : {out['error'][:120] if out['error'] else None}")
    print(f"  runtime_ms : {out['runtime_ms']}")

    # ── Test 3: runtime error (KeyError) ──────────────────────────────────
    print()
    print("=" * 60)
    print("Test 3: Runtime error - nonexistent column")
    print("=" * 60)
    bad_runtime = textwrap.dedent("""\
        result = {"v": float(df["nonexistent"].mean())}
        print(json.dumps(result))
    """)
    out = execute_pandas_code(bad_runtime, _sample_df)
    print(f"  success    : {out['success']}")
    print(f"  result     : {out['result']}")
    print(f"  error      : {out['error'][:120] if out['error'] else None}")
    print(f"  runtime_ms : {out['runtime_ms']}")

    # ── Test 4: network call is blocked ───────────────────────────────────
    print()
    print("=" * 60)
    print("Test 4: Network call should be blocked (DNS patched)")
    print("=" * 60)
    net_code = textwrap.dedent("""\
        import urllib.request
        try:
            urllib.request.urlopen("http://example.com", timeout=2)
            result = {"network_allowed": True}
        except Exception as exc:
            result = {"network_allowed": False, "error_type": type(exc).__name__}
        print(json.dumps(result))
    """)
    out = execute_pandas_code(net_code, _sample_df)
    print(f"  success    : {out['success']}")
    print(f"  result     : {out['result']}")
    print(f"  runtime_ms : {out['runtime_ms']}")
