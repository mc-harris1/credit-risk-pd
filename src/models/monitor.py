# src/models/monitor.py
"""
Offline drift / health checks for the PD model.

This script:
- Uses the same time-based split as training/evaluation
- Treats the training period as "baseline" and the holdout as "recent"
- Computes:
    - Default rate drift
    - Summary stats for numeric features
    - PSI (Population Stability Index) per numeric feature
- Writes:
    - models/metadata/drift_report.json
    - models/metadata/drift_numeric_summary.csv   (optional but handy)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import METADATA_DIR
from src.models.datasets import (
    DataContract,
    apply_domain_features_and_drop_date,
    load_data_contract,
    load_engineered_features,
    time_based_split,
)

# ---------------------------------------------------------------------------
# PSI utilities
# ---------------------------------------------------------------------------


def _compute_psi_for_feature(
    base: pd.Series,
    new: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI) for a single numeric feature.

    PSI = sum( (p_i - q_i) * ln(p_i / q_i) )

    base: "training" / baseline period
    new:  "recent"   / monitoring period
    """
    base = base.replace([np.inf, -np.inf], np.nan).dropna()
    new = new.replace([np.inf, -np.inf], np.nan).dropna()

    if base.empty or new.empty:
        return float("nan")

    # Define bins based on baseline distribution (quantiles)
    # guard against too few unique values
    quantiles = np.linspace(0, 1, n_bins + 1)
    try:
        cuts = np.unique(np.nanquantile(base, quantiles))
    except Exception:
        # degenerate distribution
        return float("nan")

    if len(cuts) <= 2:
        # Not enough unique bins
        return float("nan")

    # Bin both distributions using baseline cuts
    base_counts, _ = np.histogram(base, bins=cuts)
    new_counts, _ = np.histogram(new, bins=cuts)

    # Convert to proportions with small epsilon to avoid zero divisions
    eps = 1e-6
    base_props = base_counts / max(base_counts.sum(), eps)
    new_props = new_counts / max(new_counts.sum(), eps)

    base_props = np.clip(base_props, eps, 1.0)
    new_props = np.clip(new_props, eps, 1.0)

    psi = float(
        np.sum(
            (base_props - new_props) * np.log(base_props / new_props),
        )
    )
    return psi


def _numeric_feature_summary(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compute simple summary stats for all numeric columns in df."""
    summary: Dict[str, Dict[str, float]] = {}
    num_cols = df.select_dtypes(include=["number", "bool"]).columns

    for col in num_cols:
        col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if col_data.empty:
            continue

        summary[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std(ddof=1)) if len(col_data) > 1 else 0.0,
            "min": float(col_data.min()),
            "max": float(col_data.max()),
        }
    return summary


# ---------------------------------------------------------------------------
# Main monitor logic
# ---------------------------------------------------------------------------


def load_feature_data(contract: DataContract | None = None):
    """Compatibility shim: load engineered features + target + dates."""
    contract = contract or load_data_contract()
    return load_engineered_features(contract)


def _prepare_baseline_and_recent() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load feature data and split into baseline (train) and recent (test)
    using the same time-based split as training/evaluation.
    """
    X, y, dates = load_feature_data()
    X_train, X_recent, y_train, y_recent = time_based_split(X, y, dates, train_fraction=0.8)

    # Apply domain features + drop date to match training pipeline
    X_train = apply_domain_features_and_drop_date(X_train, date_col="issue_d")
    X_recent = apply_domain_features_and_drop_date(X_recent, date_col="issue_d")

    return X_train, X_recent, y_train, y_recent


def run_drift_checks() -> None:
    """
    Run basic drift / health checks offline and write a JSON summary + CSV.

    The report includes:
    - Default rate drift
    - PSI per numeric feature
    - Summary stats for numeric features by period
    """
    os.makedirs(METADATA_DIR, exist_ok=True)

    X_train, X_recent, y_train, y_recent = _prepare_baseline_and_recent()

    # 1) Default-rate drift
    default_rate_train = float(y_train.mean())
    default_rate_recent = float(y_recent.mean())
    default_rate_change = default_rate_recent - default_rate_train

    # 2) Numeric feature summaries
    train_summary = _numeric_feature_summary(X_train)
    recent_summary = _numeric_feature_summary(X_recent)

    # 3) PSI per numeric feature
    num_cols = sorted(set(train_summary.keys()) & set(recent_summary.keys()))
    psi_scores: Dict[str, float] = {}
    for col in num_cols:
        psi_scores[col] = _compute_psi_for_feature(X_train[col], X_recent[col])

    # 4) Assemble JSON report
    drift_report = {
        "default_rate": {
            "train": default_rate_train,
            "recent": default_rate_recent,
            "delta": default_rate_change,
        },
        "psi": psi_scores,
        "numeric_summary": {
            "train": train_summary,
            "recent": recent_summary,
        },
    }

    # classify PSI magnitudes (common rules-of-thumb)
    psi_buckets: Dict[str, str] = {}
    for feature, psi in psi_scores.items():
        if math.isnan(psi):
            label = "nan"
        elif psi < 0.1:
            label = "no_or_negligible_shift"
        elif psi < 0.25:
            label = "moderate_shift"
        else:
            label = "significant_shift"
        psi_buckets[feature] = label
    drift_report["psi_labels"] = psi_buckets

    # 5) Write JSON
    json_path = Path(METADATA_DIR) / "drift_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2)
    print(f"Saved drift report to {json_path}")

    # 6) Optional: write tabular CSV for numeric features
    rows: List[Dict[str, float | str]] = []
    for col in num_cols:
        row = {
            "feature": col,
            "psi": psi_scores[col],
            "psi_label": psi_buckets[col],
            "train_mean": train_summary[col]["mean"],
            "train_std": train_summary[col]["std"],
            "train_min": train_summary[col]["min"],
            "train_max": train_summary[col]["max"],
            "recent_mean": recent_summary[col]["mean"],
            "recent_std": recent_summary[col]["std"],
            "recent_min": recent_summary[col]["min"],
            "recent_max": recent_summary[col]["max"],
        }
        rows.append(row)

    if rows:
        df_out = pd.DataFrame(rows)
        csv_path = Path(METADATA_DIR) / "drift_numeric_summary.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"Saved numeric drift summary to {csv_path}")


if __name__ == "__main__":
    run_drift_checks()
