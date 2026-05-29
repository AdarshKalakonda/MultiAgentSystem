"""Central configuration — all settings loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve .env relative to this file so the app works from any CWD
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _require(key: str) -> str:
    """Return env var or raise a clear error at startup."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


# ── API keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = _require("ANTHROPIC_API_KEY")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")  # optional

# ── Model identifiers ─────────────────────────────────────────────────────────
MODEL_FAST: str = os.getenv("MODEL_FAST", "claude-haiku-4-5-20251001")
MODEL_SMART: str = os.getenv("MODEL_SMART", "claude-sonnet-4-6")

# ── LangSmith tracing (opt-in) ────────────────────────────────────────────────
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "DataStory")

if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
