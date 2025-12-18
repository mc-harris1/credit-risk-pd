# src/models/evaluate.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import ARTIFACTS_DIR, RANDOM_STATE
from src.models.datasets import DataContract, load_data_contract, prepare_time_split_xy

LOGGER = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class EvalConfig:
    # Manifest-first input
    feature_manifest_path: Optional[Path] = None

    # Fallbacks (only used if manifest not provided)
    features_path: Optional[Path] = None
    target_col: str = "default"
    date_col: str = "issue_d"
    train_fraction: float = 0.80

    # Model input
    train_run_dir: Optional[Path] = None
    model_path: Optional[Path] = None

    # Output
    out_dir: Path = Path(ARTIFACTS_DIR) / "runs" / "eval" / datetime.now().strftime("%Y%m%d_%H%M%S")

    threshold: float = 0.50
    calib_bins: int = 10
    calib_strategy: str = "quantile"

    log_level: str = "INFO"
    random_state: int = RANDOM_STATE


# -----------------------------
# Logging / IO
# -----------------------------


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# -----------------------------
# Model loading
# -----------------------------


def resolve_model_path(cfg: EvalConfig) -> Path:
    if cfg.model_path:
        return cfg.model_path
    if cfg.train_run_dir:
        candidate = cfg.train_run_dir / "model.joblib"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"model.joblib not found in train_run_dir: {cfg.train_run_dir}")
    raise ValueError("Provide MODEL_PATH or TRAIN_RUN_DIR")


# -----------------------------
# Contract resolution
# -----------------------------


def resolve_contract(cfg: EvalConfig) -> DataContract:
    return load_data_contract(
        feature_manifest_path=cfg.feature_manifest_path,
        features_path=cfg.features_path,
        target_col=cfg.target_col,
        date_col=cfg.date_col,
        train_fraction=cfg.train_fraction,
    )


# -----------------------------
# Evaluation
# -----------------------------


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    ensure_dir(cfg.out_dir)

    contract = resolve_contract(cfg)
    model_path = resolve_model_path(cfg)
    model = joblib.load(model_path)

    # One call: load -> time split -> domain features -> drop date col
    _, X_test, _, y_test, contract = prepare_time_split_xy(contract)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= cfg.threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "threshold": float(cfg.threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    metrics_path = cfg.out_dir / "metrics.json"
    write_json(
        metrics_path,
        {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "data_contract": {
                "feature_manifest_path": str(contract.manifest_path)
                if contract.manifest_path
                else None,
                "features_path": str(contract.features_path),
                "target_col": contract.target_col,
                "date_col": contract.date_col,
                "split": {
                    "strategy": contract.split_strategy,
                    "train_fraction": float(contract.train_fraction),
                    "cutoff_date": contract.cutoff_date,
                },
            },
            "model": {
                "model_path": str(model_path),
                "train_run_dir": str(cfg.train_run_dir) if cfg.train_run_dir else None,
            },
            "metrics": metrics,
            "linked_manifest_preview": {
                "feature_spec_path": str(contract.feature_spec_path)
                if contract.feature_spec_path
                else None,
                "target_definition_path": str(contract.target_definition_path)
                if contract.target_definition_path
                else None,
                "column_roles_path": str(contract.column_roles_path)
                if contract.column_roles_path
                else None,
            },
        },
    )

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.tight_layout()
    roc_path = cfg.out_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP={pr_auc:.3f}")
    plt.legend()
    plt.tight_layout()
    pr_path = cfg.out_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # Calibration
    prob_true, prob_pred = calibration_curve(
        y_test,
        y_prob,
        n_bins=cfg.calib_bins,
        strategy=cfg.calib_strategy,  # type: ignore
    )
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "k--")
    plt.tight_layout()
    calib_path = cfg.out_dir / "calibration_curve.png"
    plt.savefig(calib_path, dpi=200)
    plt.close()

    write_json(
        cfg.out_dir / "manifest.json",
        {
            "metrics": str(metrics_path),
            "roc_curve": str(roc_path),
            "pr_curve": str(pr_path),
            "calibration_curve": str(calib_path),
            "feature_manifest_path": str(contract.manifest_path)
            if contract.manifest_path
            else None,
            "features_path": str(contract.features_path),
            "model_path": str(model_path),
        },
    )

    LOGGER.info("Saved evaluation artifacts to %s", cfg.out_dir)
    return metrics


# -----------------------------
# CLI
# -----------------------------


def _env_path(name: str) -> Optional[Path]:
    v = os.getenv(name)
    return Path(v) if v else None


def main() -> None:
    cfg = EvalConfig(
        feature_manifest_path=_env_path("FEATURE_MANIFEST_PATH"),
        features_path=_env_path("FEATURES_PATH"),
        target_col=os.getenv("TARGET_COL", "default"),
        date_col=os.getenv("DATE_COL", "issue_d"),
        train_fraction=float(os.getenv("TRAIN_FRACTION", "0.8")),
        train_run_dir=_env_path("TRAIN_RUN_DIR"),
        model_path=_env_path("MODEL_PATH"),
        out_dir=_env_path("EVAL_OUT_DIR") or EvalConfig().out_dir,
        threshold=float(os.getenv("THRESHOLD", "0.5")),
        calib_bins=int(os.getenv("CALIB_BINS", "10")),
        calib_strategy=os.getenv("CALIB_STRATEGY", "quantile"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        random_state=int(os.getenv("RANDOM_STATE", str(RANDOM_STATE))),
    )
    setup_logging(cfg.log_level)
    evaluate(cfg)


if __name__ == "__main__":
    main()
