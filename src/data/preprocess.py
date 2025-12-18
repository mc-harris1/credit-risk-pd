# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.data.load_data import load_raw_loans

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

# "Keep set" — the minimal canonical set for downstream feature engineering
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
]

# Columns required to exist for this script to have meaning
REQUIRED_COLUMNS: list[str] = [
    "loan_amnt",
    "annual_inc",
    "int_rate",
    "term",
    "issue_d",
]

# Target definition (should eventually be read from ARTIFACTS_DIR JSON)
STATUS_TO_DEFAULT: dict[str, int] = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Late (16-30 days)": 1,
}
TARGET_COL = "default"


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


# =============================================================================
# File discovery + cleanup
# =============================================================================


def discover_input_files(raw_dir: Path, pattern_contains: str) -> list[Path]:
    if not raw_dir.exists():
        raise RuntimeError(f"Raw data directory does not exist: {raw_dir}")
    if not raw_dir.is_dir():
        raise RuntimeError(f"Raw data path is not a directory: {raw_dir}")

    files = sorted(raw_dir.iterdir(), key=lambda p: p.name)
    matched = [
        p
        for p in files
        if p.is_file() and p.suffix.lower() == ".csv" and pattern_contains in p.name
    ]
    return matched


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def cleanup_output_targets(interim_dir: Path, output_file: str) -> None:
    out_path = interim_dir / output_file
    safe_unlink(out_path)
    safe_unlink(out_path.with_suffix(out_path.suffix + ".gz"))


# =============================================================================
# Validation helpers
# =============================================================================


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def log_step(name: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    LOGGER.info(
        "%s | rows: %d -> %d | cols: %d -> %d",
        name,
        before.shape[0],
        after.shape[0],
        before.shape[1],
        after.shape[1],
    )


# =============================================================================
# Core transforms (pure functions)
# =============================================================================

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def drop_fully_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME_COLUMNS)


def select_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the intersection to avoid hard failure on missing optional cols.
    cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    if not cols:
        raise RuntimeError("No canonical columns found; raw schema mismatch.")
    return df[cols].copy()


def coerce_numeric_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "int_rate" in df.columns:
        if df["int_rate"].dtype == object:
            df["int_rate"] = df["int_rate"].astype(str).str.rstrip("%")
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    for col in ("loan_amnt", "annual_inc", "dti"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def drop_rows_missing_critical(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    require_columns(df, present)
    return df.dropna(subset=present)


def filter_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    if "loan_amnt" in df.columns:
        mask &= df["loan_amnt"] > 0
    if "annual_inc" in df.columns:
        mask &= df["annual_inc"] > 0
    if "int_rate" in df.columns:
        mask &= (df["int_rate"] > 0) & (df["int_rate"] <= 100)
    if "dti" in df.columns:
        mask &= (df["dti"] >= 0) & (df["dti"] <= 100)

    return df.loc[mask].copy()


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["loan_status"])
    df = df.copy()
    df[TARGET_COL] = df["loan_status"].map(STATUS_TO_DEFAULT)
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("No rows remaining after preprocessing.")
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("Input dataframe is empty before preprocessing.")

    pipeline: list[tuple[str, Transform]] = [
        ("drop_fully_empty_rows", drop_fully_empty_rows),
        ("normalize_column_names", normalize_column_names),
        ("define_target", define_target),
        ("select_canonical_columns", select_canonical_columns),
        ("coerce_numeric_fields", coerce_numeric_fields),
        ("drop_rows_missing_critical", drop_rows_missing_critical),
        ("filter_impossible_values", filter_impossible_values),
        ("finalize", finalize),
    ]

    cur = df
    for name, fn in pipeline:
        before = cur
        cur = fn(cur)
        log_step(name, before, cur)

    return cur


# =============================================================================
# IO helpers
# =============================================================================


def load_and_concat(files: list[Path]) -> pd.DataFrame:
    if not files:
        raise RuntimeError("No matching data files found in raw data directory.")

    dfs: list[pd.DataFrame] = []
    for path in files:
        temp_df = load_raw_loans(path)  # accepts full Path in your current usage
        if temp_df is None or temp_df.empty:
            raise RuntimeError(f"Loaded empty dataframe from: {path}")
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        raise RuntimeError("Concatenated dataframe is empty after loading all files.")
    return df


# =============================================================================
# Runner
# =============================================================================


def run(cfg: PreprocessConfig) -> Path:
    files = discover_input_files(cfg.raw_dir, cfg.pattern_contains)
    if not files:
        raise RuntimeError(
            f"No raw .csv files found in {cfg.raw_dir} containing '{cfg.pattern_contains}'."
        )

    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.interim_dir / cfg.output_file

    if out_path.exists() and not cfg.overwrite:
        raise RuntimeError(
            f"Output already exists: {out_path}. Re-run with --overwrite to replace it."
        )

    if cfg.overwrite:
        cleanup_output_targets(cfg.interim_dir, cfg.output_file)

    LOGGER.info("Discovered %d input files under %s", len(files), cfg.raw_dir)

    LOGGER.info("Loading raw files...")
    df_raw = load_and_concat(files)
    LOGGER.info("Raw shape: rows=%d cols=%d", df_raw.shape[0], df_raw.shape[1])

    LOGGER.info("Preprocessing dataframe...")
    df = preprocess_dataframe(df_raw)

    LOGGER.info("Writing parquet: %s", out_path)
    df.to_parquet(out_path, index=False)

    LOGGER.info("Saved loans to %s (rows=%d, cols=%d)", out_path, df.shape[0], df.shape[1])
    return out_path


def main() -> None:
    args = build_argparser().parse_args()
    setup_logging(args.log_level)

    cfg = PreprocessConfig(
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        output_file=args.output_file,
        pattern_contains=args.pattern_contains,
        overwrite=args.overwrite,
    )

    run(cfg)


if __name__ == "__main__":
    main()
