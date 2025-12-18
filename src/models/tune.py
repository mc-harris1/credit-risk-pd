# src/models/tune.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import METADATA_DIR, RANDOM_STATE
from src.models.datasets import (
    apply_domain_features_and_drop_date,
    load_data_contract,
    load_engineered_features,
    time_based_split,
)

LOGGER = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class TuneConfig:
    # Data contract
    feature_manifest_path: Optional[Path] = None
    features_path: Optional[Path] = None
    target_col: str = "default"
    date_col: str = "issue_d"
    train_fraction: float = 0.80

    # Output
    out_dir: Path = Path(METADATA_DIR) / "tuning" / datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reproducibility
    random_state: int = RANDOM_STATE

    # xgb fixed bits
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    tree_method: str = "hist"
    n_jobs: int = -1

    # logging
    log_level: str = "INFO"


# -----------------------------
# IO
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
# Search space + data
# -----------------------------


def hyperparam_grid() -> Dict[str, List[Any]]:
    """Small, sensible hyperparameter grid for XGBoost."""
    return {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


def prepare_data(cfg: TuneConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load engineered features and split/time-order consistently with train/eval."""

    contract = load_data_contract(
        feature_manifest_path=cfg.feature_manifest_path,
        features_path=cfg.features_path,
        target_col=cfg.target_col,
        date_col=cfg.date_col,
        train_fraction=cfg.train_fraction,
    )
    X, y, dates = load_engineered_features(contract)
    X_train, X_val, y_train, y_val = time_based_split(
        X, y, dates, train_fraction=cfg.train_fraction
    )

    X_train = apply_domain_features_and_drop_date(X_train, date_col=contract.date_col)
    X_val = apply_domain_features_and_drop_date(X_val, date_col=contract.date_col)

    return X_train, X_val, y_train, y_val


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing based on training schema."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def build_pipeline(
    preprocessor: ColumnTransformer,
    cfg: TuneConfig,
    params: Dict[str, Any],
) -> Pipeline:
    clf = XGBClassifier(
        objective=cfg.objective,
        eval_metric=cfg.eval_metric,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        tree_method=cfg.tree_method,
        **params,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )


# -----------------------------
# Tuning core
# -----------------------------


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TuneConfig,
    grid: Dict[str, List[Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    keys = list(grid.keys())

    results: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    LOGGER.info(
        "Starting hyperparameter search (%d combos)...",
        int(np.prod([len(v) for v in grid.values()])),
    )
    for values in product(*grid.values()):
        params = dict(zip(keys, values, strict=True))

        pipe = build_pipeline(build_preprocessor(X_train), cfg, params)

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, y_proba))

        results.append({"params": params, "roc_auc": auc})
        LOGGER.info("Params: %s -> ROC-AUC: %.4f", params, auc)

        if auc > best_score:
            best_score = auc
            best_params = params

    best = {
        "model_type": "xgb",
        "params": best_params,
        "metric": "roc_auc",
        "best_score": float(best_score),
    }
    return results, best


def write_tuning_artifacts(
    cfg: TuneConfig, results: List[Dict[str, Any]], best: Dict[str, Any]
) -> None:
    ensure_dir(cfg.out_dir)

    results_path = cfg.out_dir / "cv_results.json"
    best_path = cfg.out_dir / "best_params.json"
    manifest_path = cfg.out_dir / "manifest.json"

    write_json(results_path, {"results": results})
    write_json(best_path, best)

    write_json(
        manifest_path,
        {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "out_dir": str(cfg.out_dir),
            "results": str(results_path),
            "best_params": str(best_path),
        },
    )

    LOGGER.info("Saved cv results: %s", results_path)
    LOGGER.info("Saved best params: %s", best_path)
    LOGGER.info("Saved manifest: %s", manifest_path)


def _env_path(name: str) -> Optional[Path]:
    v = os.getenv(name)
    return Path(v) if v else None


def tune() -> Dict[str, Any]:
    cfg = TuneConfig(
        feature_manifest_path=_env_path("FEATURE_MANIFEST_PATH"),
        features_path=_env_path("FEATURES_PATH"),
        target_col=os.getenv("TARGET_COL", "default"),
        date_col=os.getenv("DATE_COL", "issue_d"),
        train_fraction=float(os.getenv("TRAIN_FRACTION", "0.8")),
        out_dir=_env_path("TUNE_OUT_DIR") or TuneConfig().out_dir,
        random_state=int(os.getenv("RANDOM_STATE", str(RANDOM_STATE))),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
    setup_logging(cfg.log_level)

    X_train, X_val, y_train, y_val = prepare_data(cfg)
    grid = hyperparam_grid()

    results, best = run_grid_search(X_train, y_train, X_val, y_val, cfg, grid)
    write_tuning_artifacts(cfg, results, best)

    LOGGER.info("Best params: %s", best["params"])
    LOGGER.info("Best ROC-AUC: %.4f", best["best_score"])
    return best


if __name__ == "__main__":
    tune()
