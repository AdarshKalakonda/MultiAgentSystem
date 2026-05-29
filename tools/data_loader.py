"""Multi-format data loader — CSV, Excel, SQLite, and PostgreSQL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, TypedDict, Union

import pandas as pd


# ── Directory for bundled sample datasets ─────────────────────────────────────
_SAMPLE_DATA_DIR = Path(__file__).parent.parent / "sample_data"

# ── Extension / scheme registries ─────────────────────────────────────────────
_CSV_EXTENSIONS: frozenset[str] = frozenset({".csv", ".tsv"})
_EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls"})
_SQLITE_EXTENSIONS: frozenset[str] = frozenset({".db", ".sqlite", ".sqlite3"})

# Covers bare "postgres://" aliases and common driver variants
_PG_SCHEMES: frozenset[str] = frozenset({
    "postgres",
    "postgresql",
    "postgresql+psycopg2",
    "postgresql+pg8000",
    "postgresql+asyncpg",
})


# ── Typed return value ────────────────────────────────────────────────────────

class LoadResult(TypedDict):
    """Structured return from every load_* function in this module."""

    df: pd.DataFrame
    source_name: str   # file basename or masked connection string
    row_count: int
    col_count: int
    file_size_kb: float


# ── Private helpers ───────────────────────────────────────────────────────────

def _file_size_kb(path: Path) -> float:
    try:
        return round(path.stat().st_size / 1024, 3)
    except OSError:
        return 0.0


def _make_result(
    df: pd.DataFrame,
    source_name: str,
    file_size_kb: float = 0.0,
) -> LoadResult:
    return LoadResult(
        df=df,
        source_name=source_name,
        row_count=len(df),
        col_count=len(df.columns),
        file_size_kb=file_size_kb,
    )


def _detect_format(source: str) -> str:
    """Return one of: 'csv', 'excel', 'sqlite', 'postgres'.

    The PostgreSQL scheme check must happen before Path parsing so that
    a connection string like "postgresql://host/db" is never mistaken for
    a local path with an unknown extension.

    Raises:
        ValueError: Unrecognised format.
    """
    if "://" in source:
        scheme = source.split("://")[0].lower()
        if scheme in _PG_SCHEMES:
            return "postgres"

    suffix = Path(source).suffix.lower()
    if suffix in _CSV_EXTENSIONS:
        return "csv"
    if suffix in _EXCEL_EXTENSIONS:
        return "excel"
    if suffix in _SQLITE_EXTENSIONS:
        return "sqlite"

    supported_exts = sorted(_CSV_EXTENSIONS | _EXCEL_EXTENSIONS | _SQLITE_EXTENSIONS)
    raise ValueError(
        f"Unsupported source format for '{source}'.\n"
        f"  Supported file extensions : {supported_exts}\n"
        f"  For databases             : postgresql://user:pass@host/db  "
        f"or a SQLite path (.db / .sqlite / .sqlite3)"
    )


def _load_csv(path: Path, nrows: Optional[int]) -> pd.DataFrame:
    """Read CSV/TSV with utf-8; fall back to latin-1 on encoding errors."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    kwargs: dict = {"sep": sep, "low_memory": False}
    if nrows is not None:
        kwargs["nrows"] = nrows

    try:
        df = pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", **kwargs)

    # Strip stray whitespace from string-typed columns.
    # Comparing dtype == object avoids the select_dtypes("object") deprecation
    # introduced in pandas 2.2+ where "str" and "object" were split.
    str_cols = df.columns[df.dtypes == object]
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Opportunistically parse columns whose names suggest dates
    for col in df.columns:
        if any(hint in col.lower() for hint in ("date", "time", "created", "updated")):
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass

    return df


def _load_excel(
    path: Path,
    sheet_name: Union[str, int],
    nrows: Optional[int],
) -> pd.DataFrame:
    kwargs: dict = {"sheet_name": sheet_name, "engine": "openpyxl"}
    if nrows is not None:
        kwargs["nrows"] = nrows
    return pd.read_excel(path, **kwargs)


def _load_sqlite(
    path: Path,
    table_name: Optional[str],
    query: Optional[str],
    nrows: Optional[int],
) -> pd.DataFrame:
    # Imported lazily — SQLAlchemy not required for file-only usage
    from sqlalchemy import create_engine, inspect, text  # noqa: PLC0415

    engine = create_engine(f"sqlite:///{path.as_posix()}")

    if query:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    if not table_name:
        tables = inspect(engine).get_table_names()
        if not tables:
            raise ValueError(f"SQLite database '{path}' has no tables.")
        table_name = tables[0]

    # Double-quote the identifier; nrows translated to LIMIT
    sql = f'SELECT * FROM "{table_name}"'
    if nrows is not None:
        sql += f" LIMIT {int(nrows)}"

    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def _load_postgres(
    conn_string: str,
    table_name: Optional[str],
    query: Optional[str],
    nrows: Optional[int],
) -> pd.DataFrame:
    from sqlalchemy import create_engine, text  # noqa: PLC0415

    engine = create_engine(conn_string)

    if query:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    if not table_name:
        raise ValueError(
            "Provide table_name or query when loading from PostgreSQL."
        )

    sql = f'SELECT * FROM "{table_name}"'
    if nrows is not None:
        sql += f" LIMIT {int(nrows)}"

    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataframe(
    source: Union[str, Path],
    *,
    table_name: Optional[str] = None,
    query: Optional[str] = None,
    sheet_name: Union[str, int] = 0,
    nrows: Optional[int] = None,
) -> LoadResult:
    """Load tabular data from a file or database into a pandas DataFrame.

    Supported sources
    ─────────────────
    • CSV / TSV (.csv, .tsv)    — auto utf-8 → latin-1 fallback
    • Excel (.xlsx, .xls)       — reads first sheet by default
    • SQLite (.db, .sqlite, .sqlite3) — auto-selects first table if
      neither table_name nor query is provided
    • PostgreSQL                — pass a full connection string as source

    Args:
        source: File path (str or Path) or PostgreSQL connection string.
        table_name: Table to read from a DB source. Auto-detected for
                    SQLite; required for PostgreSQL unless query is set.
        query: Raw SQL that overrides table_name for DB sources.
        sheet_name: Excel sheet name or zero-based index (default: 0).
        nrows: Read at most this many rows (file sources and DB sources
               without an explicit query).

    Returns:
        LoadResult with keys: df, source_name, row_count, col_count,
        file_size_kb.

    Raises:
        FileNotFoundError: The file or SQLite path does not exist.
        ValueError: Unsupported format or missing required parameters.
    """
    source_str = str(source)
    fmt = _detect_format(source_str)

    if fmt == "csv":
        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = _load_csv(path, nrows)
        return _make_result(df, path.name, _file_size_kb(path))

    if fmt == "excel":
        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = _load_excel(path, sheet_name, nrows)
        return _make_result(df, path.name, _file_size_kb(path))

    if fmt == "sqlite":
        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        df = _load_sqlite(path, table_name, query, nrows)
        label = table_name or (f"query:{query[:40]}…" if query else "auto")
        return _make_result(df, f"{path.name}:{label}", _file_size_kb(path))

    # fmt == "postgres"
    df = _load_postgres(source_str, table_name, query, nrows)
    # Mask credentials so the source_name is safe to log
    safe_name = re.sub(r"://[^@]+@", "://<credentials>@", source_str)
    return _make_result(df, safe_name, 0.0)


def load_sample_dataset(name: str) -> LoadResult:
    """Load a named built-in sample dataset from sample_data/.

    Available datasets
    ──────────────────
    • sales      — 50 rows: date, product, region, revenue, units
    • hr         — 50 rows: employee_id, department, salary, tenure, attrition
    • ecommerce  — 50 rows: order_id, category, price, returns, rating

    Args:
        name: Dataset name without extension (e.g. ``"sales"``).

    Returns:
        LoadResult for the requested dataset.

    Raises:
        ValueError: Dataset not found; lists available names.
    """
    path = _SAMPLE_DATA_DIR / f"{name}.csv"
    if not path.exists():
        available = sorted(p.stem for p in _SAMPLE_DATA_DIR.glob("*.csv"))
        raise ValueError(
            f"Sample dataset '{name}' not found in {_SAMPLE_DATA_DIR}.\n"
            f"Available: {', '.join(available) if available else '(none — run scaffold first)'}"
        )
    return load_dataframe(path)


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "sales"
    print(f"Loading sample dataset: '{name}' ...\n")

    result = load_sample_dataset(name)

    print(
        f"  rows       : {result['row_count']}\n"
        f"  cols       : {result['col_count']}\n"
        f"  size       : {result['file_size_kb']:.1f} KB\n"
        f"  source     : {result['source_name']}\n"
    )
    print("-- First 5 rows " + "-" * 43)
    print(result["df"].head(5).to_string(index=False))
    print("\n-- Column dtypes " + "-" * 42)
    print(result["df"].dtypes.to_string())
    print("\n-- Descriptive stats (numeric columns) " + "-" * 20)
    print(result["df"].describe().to_string())
