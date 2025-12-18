# src/models/datasets.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.features.transforms import add_domain_features


@dataclass(frozen=True)
class DataContract:
    """
    A single source of truth for downstream scripts (train/eval/explain).

    If loaded from a manifest, these values are populated from it.
    """

    features_path: Path
    target_col: str
    date_col: str
    train_fraction: float
    split_strategy: str = "time_based"
    cutoff_date: Optional[str] = None  # reserved for future use (e.g., explicit cutoff)

    # Optional references (useful for observability and future checks)
    feature_spec_path: Optional[Path] = None
    target_definition_path: Optional[Path] = None
    column_roles_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    manifest_payload: Optional[Dict[str, Any]] = None


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data_contract(
    feature_manifest_path: Optional[Path] = None,
    *,
    features_path: Optional[Path] = None,
    target_col: str = "default",
    date_col: str = "issue_d",
    train_fraction: float = 0.80,
) -> DataContract:
    """
    Manifest-first contract loader.

    If feature_manifest_path is provided:
      - features_path comes from manifest["features_path"] or manifest["output_path"]
      - target_col from manifest["target_col"] (or legacy manifest["target"])
      - date_col from manifest["date_col"]
      - train_fraction from manifest["split"]["train_fraction"]

    Otherwise, uses explicit args (with a sensible default features_path).
    """
    if feature_manifest_path:
        payload = read_json(feature_manifest_path)

        fp = payload.get("features_path") or payload.get("output_path")
        if not fp:
            raise KeyError("Manifest missing 'features_path' or 'output_path'.")

        split = payload.get("split", {}) if isinstance(payload.get("split", {}), dict) else {}
        return DataContract(
            features_path=Path(fp),
            target_col=payload.get("target_col") or payload.get("target") or target_col,
            date_col=payload.get("date_col") or date_col,
            train_fraction=float(split.get("train_fraction", train_fraction)),
            split_strategy=str(split.get("strategy", "time_based")),
            cutoff_date=split.get("cutoff_date"),
            feature_spec_path=Path(payload["feature_spec_path"])
            if payload.get("feature_spec_path")
            else None,
            target_definition_path=Path(payload["target_definition_path"])
            if payload.get("target_definition_path")
            else None,
            column_roles_path=Path(payload["column_roles_path"])
            if payload.get("column_roles_path")
            else None,
            manifest_path=feature_manifest_path,
            manifest_payload=payload,
        )

    # no manifest
    fp = features_path or (PROCESSED_DATA_DIR / "engineered_features_v1.parquet")
    return DataContract(
        features_path=Path(fp),
        target_col=target_col,
        date_col=date_col,
        train_fraction=float(train_fraction),
        split_strategy="time_based",
        cutoff_date=None,
        manifest_path=None,
        manifest_payload=None,
    )


def load_engineered_features(
    contract: DataContract,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load engineered feature matrix and return:
      X: all columns except target (date still included here)
      y: target
      dates: parsed date series
    """
    if not contract.features_path.exists():
        raise FileNotFoundError(f"Features file not found: {contract.features_path}")

    df = pd.read_parquet(contract.features_path)
    if df.empty:
        raise ValueError("Loaded features dataframe is empty.")

    if contract.target_col not in df.columns:
        raise KeyError(f"Missing target column: {contract.target_col}")

    if contract.date_col not in df.columns:
        raise KeyError(f"Missing date column: {contract.date_col}")

    y = df[contract.target_col].astype(int)
    dates = pd.to_datetime(df[contract.date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Invalid values found in {contract.date_col}")

    X = df.drop(columns=[contract.target_col]).copy()
    return X, y, dates


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    *,
    train_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Deterministic time split:
      - sort by dates ascending
      - first train_fraction => train
      - remaining => test/val
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")

    order = np.argsort(dates.to_numpy())
    Xs = X.iloc[order].reset_index(drop=True)
    ys = y.iloc[order].reset_index(drop=True)

    cut = int(len(Xs) * train_fraction)
    if cut <= 0 or cut >= len(Xs):
        raise ValueError("Invalid train_fraction for time split")

    return (
        Xs.iloc[:cut].copy(),
        Xs.iloc[cut:].copy(),
        ys.iloc[:cut].copy(),
        ys.iloc[cut:].copy(),
    )


def apply_domain_features_and_drop_date(
    X: pd.DataFrame,
    *,
    date_col: str,
) -> pd.DataFrame:
    """
    03_mp ordering helper:
      - apply domain features
      - drop date column from model inputs
    """
    X2 = add_domain_features(X.copy())
    if date_col in X2.columns:
        X2 = X2.drop(columns=[date_col])
    return X2


def prepare_time_split_xy(
    contract: DataContract,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DataContract]:
    """
    One-call convenience for train/eval/explain.

    Steps (03_mp style):
      load -> time split -> domain features -> drop date col

    Returns:
      X_train, X_holdout, y_train, y_holdout, contract
    """
    if contract.split_strategy != "time_based":
        raise ValueError(f"Unsupported split strategy: {contract.split_strategy}")

    X, y, dates = load_engineered_features(contract)
    X_train, X_holdout, y_train, y_holdout = time_based_split(
        X, y, dates, train_fraction=contract.train_fraction
    )

    X_train = apply_domain_features_and_drop_date(X_train, date_col=contract.date_col)
    X_holdout = apply_domain_features_and_drop_date(X_holdout, date_col=contract.date_col)

    return X_train, X_holdout, y_train, y_holdout, contract
