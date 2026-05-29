"""Data loader — reads CSV, Excel, Parquet, and JSON into DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}


def load_dataframe(
    file_path: str | Path,
    *,
    sheet_name: Optional[str | int] = 0,
    encoding: str = "utf-8",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a tabular file into a pandas DataFrame.

    Supports: .csv, .xlsx / .xls (via openpyxl), .parquet, .json / .jsonl.

    Args:
        file_path: Absolute or relative path to the data file.
        sheet_name: Excel sheet name or index (ignored for non-Excel).
        encoding: Character encoding for text formats.
        nrows: If set, read only the first N rows (useful for large files).

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    # TODO: implement full data_loader logic
    # - Resolve and validate path
    # - Dispatch to pd.read_csv / read_excel / read_parquet / read_json
    # - Detect and parse datetime columns automatically
    # - Strip leading/trailing whitespace from string columns
    # - Return cleaned DataFrame

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    raise NotImplementedError("load_dataframe is not yet implemented")
