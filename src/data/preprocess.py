# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

# =============================================================================
# EDA-driven constants (make these mirror your 01_eda.ipynb decisions)
# =============================================================================

DEFAULT_OUTPUT_FILE = "loans_cleaned.parquet"
DEFAULT_PATTERN_CONTAINS = "_2007_to_2018Q4"

# Column normalization (if upstream schemas vary)
RENAME_COLUMNS: dict[str, str] = {
    "loan_amount": "loan_amnt",
    "annual_income": "annual_inc",
}

# Target definition (should eventually be read from ARTIFACTS_DIR JSON)
STATUS_TO_DEFAULT: dict[str, int] = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Late (16-30 days)": 1,
}
TARGET_COL = "default"

# "Keep set" — the minimal canonical set for downstream feature engineering
# IMPORTANT: includes TARGET_COL so downstream steps can rely on it.
CANONICAL_COLUMNS: list[str] = [
    "loan_amnt",
    "annual_inc",
    "int_rate",
    "term",
    "dti",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "issue_d",
    TARGET_COL,
]

# Columns required to exist for this script to have meaning
REQUIRED_COLUMNS: list[str] = [
    "loan_amnt",
    "annual_inc",
    "int_rate",
    "term",
    "issue_d",
    TARGET_COL,
]


# =============================================================================
# Config + CLI
# =============================================================================


@dataclass(frozen=True)
class PreprocessConfig:
    raw_dir: Path
    interim_dir: Path
    output_file: str
    pattern_contains: str
    overwrite: bool


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess LendingClub raw CSVs into an interim parquet dataset."
    )
    p.add_argument("--raw-dir", type=Path, default=Path(RAW_DATA_DIR))
    p.add_argument("--interim-dir", type=Path, default=Path(INTERIM_DATA_DIR))
    p.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--pattern-contains", type=str, default=DEFAULT_PATTERN_CONTAINS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
