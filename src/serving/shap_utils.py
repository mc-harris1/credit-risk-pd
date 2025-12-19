# src/serving/shap_utils.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap


class ShapError(Exception):
    """Raised when SHAP computation fails."""


def _unwrap_for_shap(model: Any) -> tuple[Any, Any]:
    """
    Unwrap common sklearn wrappers so SHAP can explain the underlying estimator.

    Supports:
      - CalibratedClassifierCV(cv='prefit', estimator=...)
      - FrozenEstimator(estimator=...)
      - Pipeline(steps=[..., ('clf', ...)])
    Returns:
      (pipeline_or_estimator, final_estimator)
    """
    est = model

    # CalibratedClassifierCV(estimator=...)
    if hasattr(est, "estimator") and "CalibratedClassifierCV" in type(est).__name__:
        est = est.estimator

    # FrozenEstimator(estimator=...)
    if hasattr(est, "estimator") and "FrozenEstimator" in type(est).__name__:
        est = est.estimator

    # Pipeline(... -> clf)
    if hasattr(est, "named_steps"):
        clf = est.named_steps.get("clf")
        if clf is None:
            # fallback: last step
            try:
                clf = est.steps[-1][1]
            except Exception:
                clf = est
        return est, clf

    return est, est


def compute_shap_for_single_row(model: Any, df_row: pd.DataFrame) -> Tuple[float, np.ndarray]:
    """
    Compute SHAP for a single row.

    Expects df_row is a 1-row DataFrame with the same columns used by your pipeline.

    Behavior:
      - unwrap CalibratedClassifierCV/FrozenEstimator/Pipeline to get the underlying tree estimator
      - if an "imputer" step exists in the pipeline, transform the row before SHAP
      - use TreeExplainer on the underlying estimator

    Returns:
      (base_value, shap_vector) for class 1 when applicable.
    """
    if not hasattr(df_row, "shape") or df_row.shape[0] != 1:
        raise ShapError("df_row must be a single-row DataFrame")

    try:
        pipe_or_est, clf = _unwrap_for_shap(model)

        feature_names = list(df_row.columns)

        # Apply pipeline imputer if present
        X = df_row
        if hasattr(pipe_or_est, "named_steps") and "imputer" in pipe_or_est.named_steps:
            imputer = pipe_or_est.named_steps["imputer"]
            X_np = imputer.transform(df_row)
            X = pd.DataFrame(X_np, columns=feature_names)

        explainer = shap.TreeExplainer(clf, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X)
        expected = explainer.expected_value

        # Binary outputs: list[class0, class1] or a single array depending on model/SHAP
        if isinstance(shap_vals, list):
            shap_vec = np.asarray(shap_vals[1])[0]
            base_value = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
        else:
            shap_vec = np.asarray(shap_vals)[0]
            base_value = expected

        # Scalarize base_value
        if isinstance(base_value, (list, np.ndarray)):
            base_arr = np.asarray(base_value).ravel()
            base_value_f = float(base_arr[0]) if base_arr.size else 0.0
        else:
            base_value_f = float(base_value)  # type: ignore

        return base_value_f, np.asarray(shap_vec, dtype=float)

    except Exception as e:
        raise ShapError(f"Failed to compute SHAP values: {e}") from e


def format_top_attributions(
    df_row: pd.DataFrame,
    shap_vec: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Format SHAP values into a list of dicts compatible with FeatureAttribution:
      {"feature": str, "value": Any, "shap_value": float}

    Sorts by absolute SHAP magnitude descending and returns top_k.

    Args:
      df_row: 1-row DataFrame of features (same order as shap_vec)
      shap_vec: 1D array of SHAP values aligned to df_row columns
      top_k: number of features to return
    """
    if not hasattr(df_row, "shape") or df_row.shape[0] != 1:
        raise ShapError("df_row must be a single-row DataFrame")
    if shap_vec is None:
        raise ShapError("shap_vec is None")

    cols = list(df_row.columns)

    shap_arr = np.asarray(shap_vec).ravel()
    if shap_arr.shape[0] != len(cols):
        raise ShapError(
            f"shap_vec length ({shap_arr.shape[0]}) does not match number of features ({len(cols)})"
        )

    values = df_row.iloc[0].to_dict()

    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(cols):
        rows.append(
            {
                "feature": str(c),
                "value": values.get(c),
                "shap_value": float(shap_arr[i]),
            }
        )

    rows.sort(key=lambda r: abs(float(r["shap_value"])), reverse=True)
    return rows[: max(1, int(top_k))]
