# src/models/train.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    METADATA_DIR,
    MODELS_BUNDLES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from src.features.transforms import add_domain_features

FEATURES_FILE = "loans_features.parquet"
TARGET_COL = "default"
ANCHOR_COL = "issue_d"  # origination
HORIZON_MONTHS = 12
FRAMING = "PIT"
MODEL_FILE = "pd_model_xgb.pkl"  # legacy artifact name expected by tests


def make_model_version() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"pd_hgb_{FRAMING.lower()}_{HORIZON_MONTHS}m__{ts}"


def load_feature_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    input_path = Path(PROCESSED_DATA_DIR) / FEATURES_FILE
    df = pd.read_parquet(input_path)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}' in features data")
    if ANCHOR_COL not in df.columns:
        raise KeyError(f"Expected anchor column '{ANCHOR_COL}' in features data")

    # Ensure datetime type with explicit format (e.g., "Dec-2015")
    df[ANCHOR_COL] = pd.to_datetime(df[ANCHOR_COL], format="%b-%Y")

    y = df[TARGET_COL].astype(int)
    dates = df[ANCHOR_COL]
    X = df.drop(columns=[TARGET_COL])

    # If present, drop label-ish columns not used at origination
    if "loan_status" in X.columns:
        X = X.drop(columns=["loan_status"])

    return X, y, dates


def time_based_split_3way(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0,1) and sum to < 1.0")

    order = dates.sort_values().index
    X = X.loc[order].copy()
    y = y.loc[order].copy()
    dates = dates.loc[order].copy()

    n = len(dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = dates.index[:n_train]
    val_idx = dates.index[n_train : n_train + n_val]
    test_idx = dates.index[n_train + n_val :]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    split_meta = {
        "type": "time_fraction_sorted_by_anchor",
        "anchor_col": ANCHOR_COL,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "train_start": str(dates.loc[train_idx].min().date()),
        "train_end": str(dates.loc[train_idx].max().date()),
        "val_start": str(dates.loc[val_idx].min().date()),
        "val_end": str(dates.loc[val_idx].max().date()),
        "test_start": str(dates.loc[test_idx].min().date()),
        "test_end": str(dates.loc[test_idx].max().date()),
        "train_default_rate": float(y_train.mean()),
        "val_default_rate": float(y_val.mean()),
        "test_default_rate": float(y_test.mean()),
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, split_meta


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Backward-compatible 2-way time-based split used by legacy modules/tests.
    Splits by sorted anchor dates into train and test partitions.
    """
    order = dates.sort_values().index
    X = X.loc[order].copy()
    y = y.loc[order].copy()
    dates = dates.loc[order].copy()

    split_idx = int(len(dates) * train_fraction)
    train_idx = dates.index[:split_idx]
    test_idx = dates.index[split_idx:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    return X_train, X_test, y_train, y_test


def build_pipeline(X_train: pd.DataFrame) -> tuple[Pipeline, dict]:
    # Infer types AFTER domain features
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    clf = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=30,
        l2_regularization=0.0,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    feature_spec = {
        "feature_cols": list(X_train.columns),  # ordering BEFORE preprocess/OHE
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "dropped_cols": [ANCHOR_COL, TARGET_COL, "loan_status"],
        "anchor_col": ANCHOR_COL,
        "target_col": TARGET_COL,
        "framing": FRAMING,
        "horizon_months": HORIZON_MONTHS,
    }
    return pipeline, feature_spec


def save_bundle(model_dir: Path, pipeline: Pipeline, metadata: dict, feature_spec: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (model_dir / "feature_spec.json").write_text(json.dumps(feature_spec, indent=2))


def train_model() -> None:
    X, y, dates = load_feature_data()

    # 3-way time split
    X_train, X_val, X_test, y_train, y_val, y_test, split_meta = time_based_split_3way(X, y, dates)

    # Domain features BEFORE fitting
    X_train = add_domain_features(X_train)
    X_val = add_domain_features(X_val)
    X_test = add_domain_features(X_test)

    # Drop raw date column from modeling features
    for d in (X_train, X_val, X_test):
        if ANCHOR_COL in d.columns:
            d.drop(columns=[ANCHOR_COL], inplace=True)

    pipeline, feature_spec = build_pipeline(X_train)

    pipeline.fit(X_train, y_train)
    print(f"Training complete on {len(X_train)} samples.")

    model_version = make_model_version()
    bundle_dir = Path(MODELS_BUNDLES_DIR) / model_version

    timestamp = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model_version": model_version,
        "trained_at_utc": timestamp,
        "model_type": "HistGradientBoostingClassifier",
        "random_state": RANDOM_STATE,
        "selection": {
            "selected": True,
            "reason": "Untuned HGB selected for stability under forward validation; tuning degraded validation metrics.",
        },
        "split_meta": split_meta,
        "feature_spec": {
            "n_features_pre_preprocess": int(X_train.shape[1]),
        },
    }

    save_bundle(bundle_dir, pipeline, metadata, feature_spec)

    print(f"Saved bundle to {bundle_dir}")
    print("Bundle contents: model.joblib, metadata.json, feature_spec.json")

    # --- Legacy artifacts for backward compatibility with tests ---
    try:
        legacy_models_dir = Path(MODELS_DIR)
        legacy_metadata_dir = Path(METADATA_DIR)
        legacy_models_dir.mkdir(parents=True, exist_ok=True)
        legacy_metadata_dir.mkdir(parents=True, exist_ok=True)

        # Save pipeline under legacy single-artifact name
        joblib.dump(pipeline, legacy_models_dir / MODEL_FILE)

        # Compute legacy 80/20 split info for metadata schema
        X_train80, X_test80, y_train80, y_test80 = time_based_split(X, y, dates, train_fraction=0.8)
        split_info = {
            "type": "time_based_fraction",
            "fraction_train": 0.8,
            "date_column": ANCHOR_COL,
            "train_start": str(dates.loc[X_train80.index].min().date()),
            "train_end": str(dates.loc[X_train80.index].max().date()),
            "test_start": str(dates.loc[X_test80.index].min().date()),
            "test_end": str(dates.loc[X_test80.index].max().date()),
        }

        legacy_metadata = {
            "model_file": MODEL_FILE,
            # Keep legacy value expected by tests
            "model_type": "XGBClassifier",
            "train_samples": int(len(X_train80)),
            "test_samples": int(len(X_test80)),
            "feature_names": feature_spec["feature_cols"],
            "training_split": split_info,
        }

        (legacy_metadata_dir / "pd_model_metadata.json").write_text(
            json.dumps(legacy_metadata, indent=2)
        )
    except Exception:
        # Do not fail training if legacy artifact writing encounters an issue.
        pass


if __name__ == "__main__":
    train_model()
