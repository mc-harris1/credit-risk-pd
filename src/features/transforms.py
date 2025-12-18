# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

"""Feature transformation functions shared across training and serving.

Design principles:
- Pure / deterministic where possible
- Explicit separation of:
    - FIT (learns parameters from data; train-time only)
    - APPLY (uses learned parameters; serving-time safe)
- No target engineering here (target must be created upstream in preprocessing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

pd.set_option("future.no_silent_downcasting", True)

# =============================================================================
# Domain / deterministic transforms (safe for serving)
# =============================================================================


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic domain feature engineering.

    NOTE:
      - This function must be consistent between training and serving.
      - It must NOT create or modify the target label.
    """
    df = df.copy()

    # loan_to_income ratio (%)
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        safe_annual_inc = df["annual_inc"].replace(0, pd.NA)
        df["loan_to_income"] = ((df["loan_amnt"] / safe_annual_inc) * 100).round(4)

    # Extract term in months: " 36 months" -> 36
    if "term" in df.columns:
        # term is often object; be defensive
        term_num = df["term"].astype("string").str.extract(r"(\d+)")[0]
        df["term_months"] = pd.to_numeric(term_num, errors="coerce").astype("Int64")

    # Convert grade to numeric: A->1, ..., G->7
    if "grade" in df.columns:
        g = df["grade"].astype("string").str.extract(r"([A-G])")[0]
        df["grade_numeric"] = g.replace(
            {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        ).astype("Int64")

    # Convert sub_grade to numeric: A1->11, A5->15, ..., G5->75
    if "sub_grade" in df.columns:
        letter = df["sub_grade"].astype("string").str.extract(r"([A-G])")[0]
        number = df["sub_grade"].astype("string").str.extract(r"(\d+)")[0]
        letter_num = letter.replace({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7})
        number_num = pd.to_numeric(number, errors="coerce")
        df["sub_grade_numeric"] = (
            letter_num.astype("Int64") * 10 + number_num.astype("Int64")
        ).astype("Int64")

    return df


def derive_datetime_parts(
    df: pd.DataFrame,
    col: str,
    parts: Iterable[str] = ("year", "month", "quarter"),
    errors: str = "coerce",
    drop_raw: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Derive datetime parts from a date-like column (string or datetime).

    Returns:
      (df, created_cols)
    """
    df = df.copy()
    parsed = pd.to_datetime(df[col], errors=errors, format="mixed")  # type: ignore[arg-type]
    created: List[str] = []

    parts_set = set(parts)

    if "year" in parts_set:
        name = f"{col}_year"
        df[name] = parsed.dt.year.astype("Int64")
        created.append(name)
    if "month" in parts_set:
        name = f"{col}_month"
        df[name] = parsed.dt.month.astype("Int64")
        created.append(name)
    if "quarter" in parts_set:
        name = f"{col}_quarter"
        df[name] = parsed.dt.quarter.astype("Int64")
        created.append(name)

    if drop_raw and col in df.columns:
        df = df.drop(columns=[col])

    return df, created


def months_since(series: pd.Series, reference_date: pd.Timestamp) -> pd.Series:
    """Floor month difference: reference_date - series."""
    s = pd.to_datetime(series, errors="coerce", format="mixed")
    year_diff = reference_date.year - s.dt.year
    month_diff = reference_date.month - s.dt.month
    return (year_diff * 12 + month_diff).astype("Int64")


def add_loan_age_months(
    df: pd.DataFrame,
    issue_col: str = "issue_d",
    reference_date: pd.Timestamp | None = None,
    clip_min: int = 0,
) -> pd.DataFrame:
    """
    Add `loan_age_months` as months between issue date and a reference date.

    Serving-safe if `reference_date` is provided by the caller (prediction-time).
    If reference_date is None, we infer max(issue_d) from the provided df (train-time convenience).
    """
    df = df.copy()
    issue = pd.to_datetime(df[issue_col], errors="coerce", format="mixed")

    ref = reference_date
    if ref is None:
        ref = issue.dropna().max()
        if pd.isna(ref):
            raise ValueError("Cannot infer reference_date: issue dates are all NaT.")

    df["loan_age_months"] = months_since(issue, ref).clip(lower=clip_min)
    return df


# =============================================================================
# Encoders / learned transforms (fit vs apply)
# =============================================================================


@dataclass(frozen=True)
class FrequencyEncoder:
    """Learned frequency encoder (fit on train, apply on any split)."""

    mapping: Dict[Any, float]
    prior: float  # fallback frequency for unknowns

    @classmethod
    def fit(cls, s: pd.Series) -> "FrequencyEncoder":
        s2 = s.copy()
        freq = (s2.value_counts(dropna=False) / len(s2)).to_dict()
        prior = float(np.mean(list(freq.values()))) if freq else 0.0
        return cls(mapping=freq, prior=prior)

    def transform(self, s: pd.Series) -> pd.Series:
        return s.map(self.mapping).fillna(self.prior).astype(float)


@dataclass(frozen=True)
class OneHotEncoderSpec:
    """A minimal, dependency-light one-hot spec for consistent columns at serving."""

    columns: List[str]  # final dummy columns

    @classmethod
    def fit(cls, s: pd.Series, prefix: str, dummy_na: bool = True) -> "OneHotEncoderSpec":
        d = pd.get_dummies(s, prefix=prefix, dummy_na=dummy_na)
        cols = [str(c) for c in d.columns]
        return cls(columns=cols)

    def transform(self, s: pd.Series, prefix: str, dummy_na: bool = True) -> pd.DataFrame:
        d = pd.get_dummies(s, prefix=prefix, dummy_na=dummy_na)
        d.columns = [str(c) for c in d.columns]
        # align to training columns
        for c in self.columns:
            if c not in d.columns:
                d[c] = 0
        return d[self.columns]


def oof_target_mean_encode(
    df: pd.DataFrame,
    col: str,
    target: str,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.Series:
    """
    Out-of-fold target mean encoding (train-time only).

    This returns an encoding for the *same df* using only out-of-fold statistics,
    reducing target leakage during training.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not present for OOF encoding.")

    y = df[target].astype(float)
    prior = float(y.mean())
    enc = pd.Series(index=df.index, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x = df[col]

    for tr_idx, va_idx in kf.split(df):
        tr = df.iloc[tr_idx]
        means = tr.groupby(col)[target].mean()
        enc.iloc[va_idx] = x.iloc[va_idx].map(means).fillna(prior).astype(float)

    return enc


# =============================================================================
# Numeric plans (contract-driven)
# =============================================================================


def apply_numeric_plan(
    df: pd.DataFrame,
    col: str,
    planned_transformation: str = "none",
    shift: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply a numeric transformation plan.

    Supported:
      - planned_transformation == "log" -> adds {col}__log1p = log1p(clip(x+shift, 0, inf))

    Returns:
      (df, output_cols) where output_cols includes original col and any derived cols.
    """
    df = df.copy()
    out_cols: List[str] = []

    s = pd.to_numeric(df[col], errors="coerce")

    if planned_transformation == "log":
        transformed = np.log1p(np.clip(s + float(shift), a_min=0.0, a_max=None))
        new_col = f"{col}__log1p"
        df[new_col] = transformed
        out_cols.append(new_col)

    out_cols.append(col)
    return df, out_cols
