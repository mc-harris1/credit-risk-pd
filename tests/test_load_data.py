from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data.load_data import load_raw_loans


def test_load_raw_loans_raises_if_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Raw data file not found"):
        load_raw_loans(missing)


def test_load_raw_loans_raises_if_not_a_file(tmp_path: Path) -> None:
    d = tmp_path / "a_directory"
    d.mkdir()
    with pytest.raises(RuntimeError, match="Expected a file"):
        load_raw_loans(d)


def test_load_raw_loans_raises_if_wrong_suffix(tmp_path: Path) -> None:
    p = tmp_path / "loans.parquet"
    p.write_bytes(b"not really parquet")
    with pytest.raises(RuntimeError, match="Unsupported file type"):
        load_raw_loans(p)


def test_load_raw_loans_raises_if_empty_dataframe(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Loaded empty dataframe"):
        load_raw_loans(p)


def test_load_raw_loans_reads_csv(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("loan_amnt,annual_inc\n1000,75000\n2000,120000\n", encoding="utf-8")

    df = load_raw_loans(p)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["loan_amnt", "annual_inc"]
    assert df["loan_amnt"].tolist() == [1000, 2000]
