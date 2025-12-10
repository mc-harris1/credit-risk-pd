# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

import os

import pandas as pd

from src.config import RAW_DATA_DIR


def load_raw_loans(filename: str, subdirectory: str = "") -> pd.DataFrame:
    """Load raw loan data from the data/raw directory.

    Args:
        filename: Name of the CSV file to load
        subdirectory: Optional subdirectory within data/raw

    Returns:
        DataFrame containing the raw loan data
    """
    if subdirectory:
        raw_path = os.path.join(RAW_DATA_DIR, subdirectory, filename)
    else:
        raw_path = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(raw_path):
        msg = f"Raw data file not found: {raw_path}"
        raise FileNotFoundError(msg)

    return pd.read_csv(raw_path, low_memory=False)
