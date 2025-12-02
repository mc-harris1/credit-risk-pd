from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_DIR


def load_raw_loans(filename: str) -> pd.DataFrame:
    """Load raw loans data from data/raw."""
    path = RAW_DATA_DIR / filename
    if not path.exists():
        msg = f"Raw data file not found: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)
