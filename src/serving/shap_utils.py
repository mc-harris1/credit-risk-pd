"""
SHAP utilities for per-instance explanations.

Design goals:
- Serving-safe: no dependency on training scripts or full dataset loading.
- Bundle-based: uses PD_MODEL_DIR (same as API) to locate model + background.
- Best-effort: SHAP is optional and may be disabled in production.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import shap

from src.features.transforms import add_domain_features
from src.serving.registry import enforce_feature_contract, load_bundle

logger = logging.getLogger(__name__)

# Optional: if present in the bundle, we use it as background for SHAP
DEFAULT_BG_FILENAME = "shap_background.parquet"


@lru_cache(maxsize=1)
def _load_bundle_cached():
    return load_bundle()


@lru_cache(maxsize=1)
def _load_background_df(bg_filename: str = DEFAULT_BG_FILENAME, n: int = 500) -> pd.DataFrame:
    bundle = _load_bundle_cached()
    bg_path = Path(bundle.model_dir) / bg_filename

    if not bg_path.exists():
        raise FileNotFoundError(
            f"SHAP background not found at {bg_path}. "
            "Either add it to the model bundle at training time or disable SHAP."
        )

    df_bg = pd.read_parquet(bg_path)
    if len(df_bg) > n:
        df_bg = df_bg.sample(n, random_state=0)

    # Apply same domain features + contract enforcement as inference
    df_bg = add_domain_features(df_bg)
    if "issue_d" in df_bg.columns:
        df_bg = df_bg.drop(columns=["issue_d"])

    df_bg = enforce_feature_contract(df_bg, bundle.feature_spec)
    return df_bg


def _predict_proba_1(model, X: pd.DataFrame):
    """Predict probability of class 1 for SHAP explainers."""
    return model.predict_proba(X)[:, 1]


@lru_cache(maxsize=1)
def _build_explainer() -> Tuple[shap.Explainer, List[str]]:
    """
    Build and cache a SHAP explainer.

    Notes:
    - TreeExplainer is great for XGBoost; may not be reliable for sklearn HGB.
    - We use shap.Explainer with a prediction function for broad compatibility.
    """
    bundle = _load_bundle_cached()
    model = bundle.model

    X_bg = _load_background_df()
    feature_names = list(X_bg.columns)

    # Use the unified high-level API for broad compatibility.
    explainer = shap.Explainer(
        lambda X: _predict_proba_1(model, pd.DataFrame(X, columns=feature_names)), X_bg
    )

    return explainer, feature_names


def explain_instance(raw_row: pd.DataFrame, top_n: int = 5) -> List[Dict[str, float]]:
    """
    Compute SHAP feature contributions for a single instance (best-effort).

    raw_row: single-row DataFrame of *raw input features* (pre-domain-features).
    """
    if not isinstance(raw_row, pd.DataFrame):
        raise TypeError("raw_row must be a pandas DataFrame with a single row.")
    if len(raw_row) != 1:
        raise ValueError(f"Expected raw_row with exactly 1 row, got {len(raw_row)}")

    bundle = _load_bundle_cached()

    # Prepare row as in inference
    X_row = add_domain_features(raw_row.copy())
    if "issue_d" in X_row.columns:
        X_row = X_row.drop(columns=["issue_d"])

    X_row = enforce_feature_contract(X_row, bundle.feature_spec)

    explainer, feature_names = _build_explainer()

    # Compute SHAP values
    shap_values = explainer(X_row)

    # shap_values.values shape: (n_samples, n_features)
    vals = shap_values.values[0]

    contributions = [
        {"feature": str(name), "shap_value": float(val)}
        for name, val in zip(feature_names, vals, strict=True)
    ]

    contributions_sorted = sorted(contributions, key=lambda d: abs(d["shap_value"]), reverse=True)
    return contributions_sorted[:top_n]
