# src/models/train.py
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
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACTS_DIR, RANDOM_STATE
from src.models.datasets import DataContract, load_data_contract, prepare_time_split_xy

LOGGER = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class TrainConfig:
    # Manifest-first input
    feature_manifest_path: Optional[Path] = None

    # Fallbacks (only used if manifest not provided)
    features_path: Optional[Path] = None
    target_col: str = "default"
    date_col: str = "issue_d"
    train_fraction: float = 0.80

    # Output
    out_dir: Path = (
        Path(ARTIFACTS_DIR) / "runs" / "train" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Model params
    C: float = 1.0
    max_iter: int = 2000
    class_weight: Optional[str] = "balanced"
    random_state: int = RANDOM_STATE

    log_level: str = "INFO"


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
# Pipeline / model
# -----------------------------


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)],
        remainder="drop",
    )


def build_pipeline(cfg: TrainConfig, X_train: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X_train)
    model = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        solver="lbfgs",
        n_jobs=None,
    )
    return Pipeline(steps=[("preprocess", pre), ("model", model)])


# -----------------------------
# Metrics
# -----------------------------


def evaluate_binary(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_prob)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "brier": float(brier),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# -----------------------------
# Orchestration
# -----------------------------


def resolve_contract(cfg: TrainConfig) -> DataContract:
    return load_data_contract(
        feature_manifest_path=cfg.feature_manifest_path,
        features_path=cfg.features_path,
        target_col=cfg.target_col,
        date_col=cfg.date_col,
        train_fraction=cfg.train_fraction,
    )


def train(cfg: TrainConfig) -> Dict[str, Any]:
    ensure_dir(cfg.out_dir)

    contract = resolve_contract(cfg)

    # One call: load -> time split -> domain features -> drop date col
    X_train, X_val, y_train, y_val, contract = prepare_time_split_xy(contract)

    pipe = build_pipeline(cfg, X_train)
    LOGGER.info("Fitting LogisticRegression...")
    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_val = pipe.predict_proba(X_val)[:, 1]

    train_metrics = evaluate_binary(y_train.to_numpy(), p_train, threshold=0.5)
    val_metrics = evaluate_binary(y_val.to_numpy(), p_val, threshold=0.5)

    model_path = cfg.out_dir / "model.joblib"
    metrics_path = cfg.out_dir / "metrics.json"
    manifest_path = cfg.out_dir / "manifest.json"

    joblib.dump(pipe, model_path)

    payload = {
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
        "data": {
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_val": float(y_val.mean()),
        },
        "model": {
            "type": "LogisticRegression",
            "params": {
                "C": cfg.C,
                "max_iter": cfg.max_iter,
                "class_weight": cfg.class_weight,
                "random_state": cfg.random_state,
            },
        },
        "metrics": {"train": train_metrics, "val": val_metrics},
        "artifacts": {"model_path": str(model_path), "metrics_path": str(metrics_path)},
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
    }

    write_json(metrics_path, payload)
    write_json(
        manifest_path,
        {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "feature_manifest_path": str(contract.manifest_path)
            if contract.manifest_path
            else None,
            "features_path": str(contract.features_path),
        },
    )

    LOGGER.info("Saved model: %s", model_path)
    LOGGER.info("Saved metrics: %s", metrics_path)
    return payload


# -----------------------------
# CLI
# -----------------------------


def _env_path(name: str) -> Optional[Path]:
    v = os.getenv(name)
    return Path(v) if v else None


def main() -> None:
    cfg = TrainConfig(
        feature_manifest_path=_env_path("FEATURE_MANIFEST_PATH"),
        features_path=_env_path("FEATURES_PATH"),
        target_col=os.getenv("TARGET_COL", "default"),
        date_col=os.getenv("DATE_COL", "issue_d"),
        train_fraction=float(os.getenv("TRAIN_FRACTION", "0.8")),
        out_dir=_env_path("TRAIN_OUT_DIR") or TrainConfig().out_dir,
        C=float(os.getenv("LOGREG_C", "1.0")),
        max_iter=int(os.getenv("MAX_ITER", "2000")),
        class_weight=os.getenv("CLASS_WEIGHT", "balanced"),
        random_state=int(os.getenv("RANDOM_STATE", str(RANDOM_STATE))),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    setup_logging(cfg.log_level)
    train(cfg)


if __name__ == "__main__":
    main()
