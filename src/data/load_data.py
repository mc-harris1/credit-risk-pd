# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_loans(path: Path) -> pd.DataFrame:
    """Load raw loan data from a CSV file.

    Args:
        path: Full path to the raw CSV file

    Returns:
        DataFrame containing the raw loan data
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    if not path.is_file():
        raise RuntimeError(f"Expected a file but got: {path}")

    if path.suffix.lower() != ".csv":
        raise RuntimeError(f"Unsupported file type: {path.suffix} (expected .csv)")

    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise RuntimeError(f"Loaded empty dataframe from: {path}") from exc

    if df.empty:
        raise RuntimeError(f"Loaded empty dataframe from: {path}")

    return df
