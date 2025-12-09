# src/models/train.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

import json
import os
from datetime import datetime, timezone
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import METADATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from src.features.transforms import add_domain_features

FEATURES_FILE = "loans_features.parquet"
MODEL_FILE = "pd_model_xgb.pkl"


def load_feature_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load processed feature data and return X, y, and origination dates."""
    input_path = os.path.join(PROCESSED_DATA_DIR, FEATURES_FILE)
    df = pd.read_parquet(input_path)

    # Drop columns not used as features but present in file
    if "loan_status" in df.columns:
        df = df.drop(columns=["loan_status"])

    if "issue_d" not in df.columns:
        msg = "Expected origination column 'issue_d' in features data"
        raise KeyError(msg)

    # Ensure datetime type with explicit format (e.g., "Dec-2015")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

    y = df["default"]
    dates = df["issue_d"]
    X = df.drop(columns=["default"])

    return X, y, dates


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X, y into train/test using a time-based cutoff on origination dates."""
    if not 0.0 < train_fraction < 1.0:
        msg = f"train_fraction must be between 0 and 1, got {train_fraction}"
        raise ValueError(msg)

    order = dates.sort_values().index
    X = X.loc[order]
    y = y.loc[order]
    dates = dates.loc[order]

    split_idx = int(len(dates) * train_fraction)
    train_idx = dates.index[:split_idx]
    test_idx = dates.index[split_idx:]

    X_train, X_test = X.loc[train_idx].copy(), X.loc[test_idx].copy()
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Drop the raw date column from features to avoid direct date leakage
    if "issue_d" in X_train.columns:
        X_train = X_train.drop(columns=["issue_d"])
        X_test = X_test.drop(columns=["issue_d"])

    return X_train, X_test, y_train, y_test


def train_model() -> None:
    """Train the PD model using a time-based split and save model + metadata.

    NOTE: This function does NOT compute metrics; evaluation lives in src.models.evaluate.
    """
    X, y, dates = load_feature_data()

    # Time-based split
    X_train, X_test, y_train, y_test = time_based_split(X, y, dates, train_fraction=0.8)

    # Apply domain-specific feature engineering BEFORE fitting
    X_train = add_domain_features(X_train)
    X_test = add_domain_features(X_test)

    # Infer numeric / categorical columns after feature engineering
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    pipeline.fit(X_train, y_train)
    print(f"Training complete on {len(X_train)} samples. Saving model...")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    joblib.dump(pipeline, model_path)

    # --- metadata.json ---
    timestamp = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model_file": MODEL_FILE,
        "trained_at_utc": timestamp,
        "model_type": "XGBClassifier",
        "random_state": RANDOM_STATE,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_names": list(X_train.columns),
        "training_split": {
            "type": "time_based_fraction",
            "fraction_train": 0.8,
            "date_column": "issue_d",
            "train_start": dates.min().strftime("%Y-%m-%d"),
            "train_end": dates.loc[X_train.index].max().strftime("%Y-%m-%d"),
            "test_start": dates.loc[X_test.index].min().strftime("%Y-%m-%d"),
            "test_end": dates.max().strftime("%Y-%m-%d"),
        },
    }

    os.makedirs(METADATA_DIR, exist_ok=True)
    metadata_path = os.path.join(METADATA_DIR, "pd_model_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    train_model()
