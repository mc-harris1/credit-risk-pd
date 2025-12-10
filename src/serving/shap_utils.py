"""
SHAP utilities for per-instance explanations.

This module:
- Loads the trained pipeline (preprocess + XGBoost model)
- Builds a TreeExplainer with a background sample
- Provides explain_instance() to return top-N feature contributions
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import joblib
import pandas as pd
import shap

from src.config import MODELS_DIR
from src.features.transforms import add_domain_features
from src.models.train import (
    MODEL_FILE,
    load_feature_data,
    time_based_split,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load the trained sklearn Pipeline (preprocess + model) once per process."""
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        msg = f"Model artifact not found at {model_path}. Train the model first."
        raise FileNotFoundError(msg)

    pipeline = joblib.load(model_path)
    if "preprocess" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        msg = "Expected pipeline with 'preprocess' and 'model' steps."
        raise ValueError(msg)

    return pipeline


@lru_cache(maxsize=1)
def _build_explainer(num_background: int = 1000):
    """
    Build and cache a SHAP TreeExplainer.

    Uses a background sample drawn from the time-based training split, then
    transformed through the same preprocessor used at train time.
    """
    pipeline = _load_pipeline()
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # Load feature data and use training portion as background
    X, y, dates = load_feature_data()
    X_train, _, _, _ = time_based_split(X, y, dates, train_fraction=0.8)

    # Apply domain feature engineering
    X_train = add_domain_features(X_train)

    # Drop date column if present
    if "issue_d" in X_train.columns:
        X_train = X_train.drop(columns=["issue_d"])

    # Sample background rows
    if len(X_train) > num_background:
        X_bg = X_train.sample(num_background, random_state=0)
    else:
        X_bg = X_train.copy()

    # Transform into model feature space
    X_bg_trans = preprocessor.transform(X_bg)

    # Convert to dense array if sparse (SHAP TreeExplainer has issues with sparse matrices)
    if hasattr(X_bg_trans, "toarray"):
        X_bg_trans = X_bg_trans.toarray()

    # Get feature names after preprocessing (handles OHE)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"f_{i}" for i in range(X_bg_trans.shape[1])]

    explainer = shap.TreeExplainer(
        model,
        data=X_bg_trans,
        feature_names=feature_names,
        feature_perturbation="interventional",
    )
    return explainer, feature_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_instance(
    raw_row: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """
    Compute SHAP-based feature contributions for a single instance.

    Parameters
    ----------
    raw_row:
        A single-row DataFrame containing the *raw input features* in the same
        format used at training time (before preprocessing). This is typically
        built from the LoanApplication payload in the API layer.

    top_n:
        Number of top features (by absolute SHAP value) to return.

    Returns
    -------
    List[Dict[str, float]]:
        List of dictionaries like:
        [
            {"feature": "loan_to_income", "shap_value": -0.12},
            {"feature": "grade_C", "shap_value": 0.08},
            ...
        ]
    """
    if not isinstance(raw_row, pd.DataFrame):
        raise TypeError("raw_row must be a pandas DataFrame with a single row.")

    if len(raw_row) != 1:
        raise ValueError(f"Expected raw_row with exactly 1 row, got {len(raw_row)}")

    pipeline = _load_pipeline()
    explainer, feature_names = _build_explainer()

    preprocessor = pipeline.named_steps["preprocess"]

    # Apply domain feature engineering
    X_row = add_domain_features(raw_row.copy())

    # Drop date column if present (API typically won't send it, but be safe)
    if "issue_d" in X_row.columns:
        X_row = X_row.drop(columns=["issue_d"])

    # Transform into model feature space
    X_trans = preprocessor.transform(X_row)

    # Convert to dense array if sparse (SHAP TreeExplainer has issues with sparse matrices)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Compute SHAP values
    shap_values = explainer.shap_values(X_trans)

    # For binary classification, SHAP may return a list [class0, class1]; pick class1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Single row → index 0
    shap_row = shap_values[0]

    # Pair feature names with SHAP values
    contributions = [
        {"feature": name, "shap_value": float(val)}
        for name, val in zip(feature_names, shap_row, strict=True)
    ]

    # Sort by absolute contribution magnitude
    contributions_sorted = sorted(
        contributions,
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )

    return contributions_sorted[:top_n]
