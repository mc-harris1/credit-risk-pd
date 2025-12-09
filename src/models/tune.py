# src/models/tune.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

import json
import os
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import METADATA_DIR, RANDOM_STATE
from src.features.transforms import add_domain_features
from src.models.train import (
    FEATURES_FILE,
    load_feature_data,
    time_based_split,
)


def hyperparam_grid() -> Dict[str, List]:
    """Define a small, sensible hyperparameter grid for XGBoost."""
    return {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load features and return time-based train/validation sets."""
    X, y, dates = load_feature_data()
    X_train, X_val, y_train, y_val = time_based_split(X, y, dates, train_fraction=0.8)

    # Apply domain-specific feature engineering once here
    X_train = add_domain_features(X_train)
    X_val = add_domain_features(X_val)

    # Drop the date column if still present
    if "issue_d" in X_train.columns:
        X_train = X_train.drop(columns=["issue_d"])
        X_val = X_val.drop(columns=["issue_d"])

    return X_train, X_val, y_train, y_val


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer based on training data schema."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor


def tune_model() -> None:
    """
    Run a simple grid search over XGBoost hyperparameters using a time-based split.

    Results:
    - models/metadata/hparam_search_results.json   (list of {params, roc_auc})
    - models/metadata/best_params.json             ({"best_params": {...}, "best_roc_auc": ...})
    """
    print(f"Loading features from {FEATURES_FILE} for tuning...")
    X_train, X_val, y_train, y_val = prepare_data()
    preprocessor = build_preprocessor(X_train)

    grid = hyperparam_grid()
    keys = list(grid.keys())

    results = []
    best_score = -np.inf
    best_params = None

    print("Starting hyperparameter search...")
    for values in product(*grid.values()):
        params = dict(zip(keys, values, strict=True))

        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", clf),
            ]
        )

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)

        result = {"params": params, "roc_auc": float(auc)}
        results.append(result)
        print(f"Params: {params} -> ROC-AUC: {auc:.4f}")

        if auc > best_score:
            best_score = auc
            best_params = params

    os.makedirs(METADATA_DIR, exist_ok=True)

    search_path = os.path.join(METADATA_DIR, "hparam_search_results.json")
    with open(search_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved hyperparameter search results to {search_path}")

    best_path = os.path.join(METADATA_DIR, "best_params.json")
    payload = {"best_params": best_params, "best_roc_auc": float(best_score)}
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved best_params.json to {best_path}")
    print(f"Best params: {best_params}")
    print(f"Best ROC-AUC: {best_score:.4f}")


if __name__ == "__main__":
    tune_model()
