"""Sandboxed code executor for LLM-generated pandas / Plotly snippets."""

from __future__ import annotations

from typing import Any


def execute_pandas_code(
    code: str,
    context: dict[str, Any],
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """
    Execute a pandas / Plotly code snippet in a restricted namespace.

    The execution environment is intentionally restricted: only pandas,
    numpy, plotly, and scipy are available.  No file system access,
    no network calls, no subprocess.

    Args:
        code: Python source code to execute.
        context: Variables injected into the execution namespace
                 (e.g., {"df": dataframe}).
        timeout_seconds: Hard wall-clock limit.

    Returns:
        The local namespace after execution, so callers can extract
        named outputs (e.g., result["fig"]).

    Raises:
        TimeoutError: If execution exceeds timeout_seconds.
        RuntimeError: If execution raises an unhandled exception.
    """
    # TODO: implement sandboxed execution
    # - Build a restricted globals dict:
    #   {"pd": pandas, "np": numpy, "px": plotly.express,
    #    "go": plotly.graph_objects, "stats": scipy.stats}
    # - Merge with caller-supplied context
    # - Use concurrent.futures.ThreadPoolExecutor with timeout
    # - Capture stdout / stderr for debugging
    # - Return the local namespace on success
    # - Wrap exceptions in RuntimeError with sanitised message

    raise NotImplementedError("execute_pandas_code is not yet implemented")
