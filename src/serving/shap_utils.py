from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ShapError(RuntimeError):
    pass


def _jsonable(x: Any) -> Any:
    """
    Convert common numpy/pandas scalars to JSON-serializable python types.
    """
    if x is None:
        return None
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    # pandas NA / NaT
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def compute_shap_for_single_row(
    model: Any, X_row: pd.DataFrame
) -> Tuple[Optional[float], np.ndarray]:
    """
    Returns (base_value, shap_values_vector).

    Tries TreeExplainer first (best for HGB/XGBoost/LightGBM-like),
    falls back to generic Explainer.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        raise ShapError("shap is not installed or failed to import") from e

    # TreeExplainer tends to be fastest/most stable for tree models
    explainer = None
    base_value = None

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_row)
        base_value = explainer.expected_value
    except Exception:
        explainer = shap.Explainer(model, X_row)
        exp = explainer(X_row)
        sv = exp.values
        base_value = getattr(exp, "base_values", None)

    # Normalize shapes:
    # - For binary: could be [n_samples, n_features] or [2][n_samples, n_features]
    if isinstance(sv, list):
        # take class-1 by convention
        sv_arr = np.asarray(sv[1])[0]
    else:
        sv_np = np.asarray(sv)
        if sv_np.ndim == 3:
            # (n_samples, n_classes, n_features) or similar
            sv_arr = sv_np[0, 1, :]
        else:
            sv_arr = sv_np[0]

    # base_value can be scalar / list / array
    base = None
    if base_value is not None:
        if isinstance(base_value, (list, tuple, np.ndarray)):
            bv = np.asarray(base_value)
            if bv.ndim == 0:
                base = float(bv)
            elif bv.ndim == 1 and bv.shape[0] >= 2:
                base = float(bv[1])
            else:
                base = float(bv.ravel()[0])
        elif not callable(base_value):
            base = float(base_value)

    return base, sv_arr


def format_top_attributions(
    X_row: pd.DataFrame,
    shap_values: np.ndarray,
    top_k: int = 15,
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts:
      {feature, value, shap_value}
    sorted by absolute impact descending.
    """
    features = list(X_row.columns)
    values = X_row.iloc[0].tolist()
    pairs = []
    for f, v, sv in zip(features, values, shap_values, strict=False):
        pairs.append(
            {
                "feature": str(f),
                "value": _jsonable(v),
                "shap_value": float(sv),
                "abs": float(abs(sv)),
            }
        )
    pairs.sort(key=lambda d: d["abs"], reverse=True)
    for d in pairs:
        d.pop("abs", None)
    return pairs[:top_k]
