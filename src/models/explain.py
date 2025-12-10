# src/models/explain.py
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse

from src.config import METADATA_DIR, MODELS_DIR
from src.features.transforms import add_domain_features
from src.models.train import (
    MODEL_FILE,
    load_feature_data,
    time_based_split,
)


def _prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load feature data and return X_test, y_test for SHAP analysis."""
    X, y, dates = load_feature_data()
    # use same time-based split as training/eval
    _, X_test, _, y_test = time_based_split(X, y, dates, train_fraction=0.8)

    # domain features (same as train)
    X_test = add_domain_features(X_test)

    # drop raw date column if present
    if "issue_d" in X_test.columns:
        X_test = X_test.drop(columns=["issue_d"])

    return X_test, y_test


def _load_pipeline():
    model_path = Path(MODELS_DIR) / MODEL_FILE
    if not model_path.exists():
        msg = f"Model artifact not found at {model_path}. Train the model first."
        raise FileNotFoundError(msg)
    return joblib.load(model_path)


def _build_explainer(pipeline, X_background: pd.DataFrame) -> shap.Explainer:
    """Build a TreeExplainer on the fitted XGBoost model using transformed features."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    # transform background data into model space
    X_bg_trans = preprocessor.transform(X_background)

    # Convert sparse matrix to dense for SHAP compatibility
    if sparse.issparse(X_bg_trans):
        X_bg_trans = X_bg_trans.toarray()

    # feature names after preprocessing (handles OHE)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # fallback: generic names
        feature_names = [f"f_{i}" for i in range(X_bg_trans.shape[1])]

    explainer = shap.TreeExplainer(
        model,
        data=X_bg_trans,
        feature_names=feature_names,
        feature_perturbation="interventional",
    )
    return explainer


def run_shap_global(num_background: int = 1000, num_samples: int = 2000) -> None:
    """Compute global SHAP feature importance and save plots + JSON summary."""
    os.makedirs(METADATA_DIR, exist_ok=True)

    X_test, y_test = _prepare_data()
    pipeline = _load_pipeline()
    preprocessor = pipeline.named_steps["preprocess"]

    # choose background subset from test (or you could use train)
    if len(X_test) > num_background:
        X_bg = X_test.sample(num_background, random_state=0)
    else:
        X_bg = X_test.copy()

    explainer = _build_explainer(pipeline, X_bg)

    # sample points for global explanation
    if len(X_test) > num_samples:
        X_sample = X_test.sample(num_samples, random_state=1)
        # y_sample = y_test.loc[X_sample.index]
    else:
        X_sample = X_test
        # y_sample = y_test

    X_sample_trans = preprocessor.transform(X_sample)

    # Convert sparse matrix to dense for SHAP compatibility
    if sparse.issparse(X_sample_trans):
        X_sample_trans = X_sample_trans.toarray()

    shap_values = explainer(X_sample_trans)

    # SHAP can return Explanation object or array; extract values
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    # SHAP can return list for multiclass; we want positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # feature names after preprocessing
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"f_{i}" for i in range(X_sample_trans.shape[1])]

    # --- Global importance JSON: mean |SHAP| per feature ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        [
            {"feature": name, "mean_abs_shap": float(val)}
            for name, val in zip(feature_names, mean_abs, strict=True)
        ],
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )

    importance_path = os.path.join(METADATA_DIR, "shap_global_importance.json")
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump({"importance": importance}, f, indent=2)
    print(f"Saved SHAP global importance to {importance_path}")

    # --- Summary bar plot ---
    shap.summary_plot(
        shap_values,
        X_sample_trans,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    bar_path = os.path.join(METADATA_DIR, "shap_summary_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary bar plot to {bar_path}")

    # --- Beeswarm plot ---
    shap.summary_plot(
        shap_values,
        X_sample_trans,
        feature_names=feature_names,
        show=False,
    )
    beeswarm_path = os.path.join(METADATA_DIR, "shap_summary_beeswarm.png")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP beeswarm plot to {beeswarm_path}")


if __name__ == "__main__":
    run_shap_global()
