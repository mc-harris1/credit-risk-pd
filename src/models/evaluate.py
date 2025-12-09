# src/models/evaluate.py
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import METADATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR

FEATURES_FILE = "loans_features.parquet"
MODEL_FILE = "pd_model_xgb.pkl"
ORIGINATION_COL = "issue_d"  # keep in sync with train.py


def load_feature_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load processed feature data and return X, y, and origination dates."""
    input_path = os.path.join(PROCESSED_DATA_DIR, FEATURES_FILE)
    df = pd.read_parquet(input_path)

    if "loan_status" in df.columns:
        df = df.drop(columns=["loan_status"])

    if ORIGINATION_COL not in df.columns:
        msg = f"Expected origination column '{ORIGINATION_COL}' in features data"
        raise KeyError(msg)

    df[ORIGINATION_COL] = pd.to_datetime(df[ORIGINATION_COL], format="%b-%Y")

    y = df["default"]
    dates = df[ORIGINATION_COL]
    X = df.drop(columns=["default"])

    return X, y, dates


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    order = dates.sort_values().index
    X = X.loc[order]
    y = y.loc[order]
    dates = dates.loc[order]

    split_idx = int(len(dates) * train_fraction)
    train_idx = dates.index[:split_idx]
    test_idx = dates.index[split_idx:]

    X_train, X_test = X.loc[train_idx].copy(), X.loc[test_idx].copy()
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    if ORIGINATION_COL in X_train.columns:
        X_train = X_train.drop(columns=[ORIGINATION_COL])
        X_test = X_test.drop(columns=[ORIGINATION_COL])

    return X_train, X_test, y_train, y_test


def evaluate_model() -> None:
    """Evaluate the trained model on a time-based holdout and save metrics + plots."""
    X, y, dates = load_feature_data()

    _, X_test, _, y_test = time_based_split(X, y, dates, train_fraction=0.8)

    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        msg = f"Model file not found at {model_path}. Train the model first."
        raise FileNotFoundError(msg)

    model = joblib.load(model_path)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Core metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": 0.5,
    }

    os.makedirs(METADATA_DIR, exist_ok=True)
    metrics_path = Path(METADATA_DIR) / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation metrics to {metrics_path}")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = Path(METADATA_DIR) / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    # --- Precision-Recall curve ---
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_path = Path(METADATA_DIR) / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()
    print(f"Saved PR curve to {pr_path}")

    # --- Calibration curve ---
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    calib_path = Path(METADATA_DIR) / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(calib_path, dpi=200)
    plt.close()
    print(f"Saved calibration curve to {calib_path}")


if __name__ == "__main__":
    evaluate_model()
