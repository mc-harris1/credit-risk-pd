# src/models/explain.py
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
import numpy as np
import shap
from scipy import sparse
from sklearn.pipeline import Pipeline

from src.config import ARTIFACTS_DIR, RANDOM_STATE
from src.models.datasets import DataContract, load_data_contract, prepare_time_split_xy

LOGGER = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class ExplainConfig:
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
    out_dir: Path = (
        Path(ARTIFACTS_DIR) / "runs" / "explain" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # SHAP sampling
    num_background: int = 1000
    num_samples: int = 2000

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


def resolve_model_path(cfg: ExplainConfig) -> Path:
    if cfg.model_path:
        return cfg.model_path
    if cfg.train_run_dir:
        candidate = cfg.train_run_dir / "model.joblib"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"model.joblib not found in train_run_dir: {cfg.train_run_dir}")
    raise ValueError("Provide MODEL_PATH or TRAIN_RUN_DIR")


def load_pipeline(model_path: Path) -> Pipeline:
    obj = joblib.load(model_path)
    if not isinstance(obj, Pipeline):
        raise TypeError("Loaded model artifact is not a sklearn Pipeline")
    return obj


# -----------------------------
# Contract resolution
# -----------------------------


def resolve_contract(cfg: ExplainConfig) -> DataContract:
    return load_data_contract(
        feature_manifest_path=cfg.feature_manifest_path,
        features_path=cfg.features_path,
        target_col=cfg.target_col,
        date_col=cfg.date_col,
        train_fraction=cfg.train_fraction,
    )


# -----------------------------
# SHAP helpers
# -----------------------------


def _to_dense(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _feature_names(pre) -> list[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        return []


# -----------------------------
# Explain
# -----------------------------


def run_shap_global(cfg: ExplainConfig) -> Dict[str, Any]:
    ensure_dir(cfg.out_dir)

    contract = resolve_contract(cfg)
    model_path = resolve_model_path(cfg)
    pipeline = load_pipeline(model_path)

    # One call: load -> time split -> domain features -> drop date col
    _, X_test, _, y_test, contract = prepare_time_split_xy(contract)

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    rng = np.random.default_rng(cfg.random_state)

    if len(X_test) > cfg.num_background:
        bg_idx = rng.choice(len(X_test), size=cfg.num_background, replace=False)
        X_bg = X_test.iloc[bg_idx]
    else:
        X_bg = X_test

    X_bg_trans = _to_dense(pre.transform(X_bg))

    names = _feature_names(pre)
    if not names:
        names = [f"f_{i}" for i in range(X_bg_trans.shape[1])]

    # Linear-friendly SHAP (LogReg baseline)
    explainer = shap.LinearExplainer(model, X_bg_trans, feature_names=names)

    if len(X_test) > cfg.num_samples:
        smp_idx = rng.choice(len(X_test), size=cfg.num_samples, replace=False)
        X_s = X_test.iloc[smp_idx]
    else:
        X_s = X_test

    X_s_trans = _to_dense(pre.transform(X_s))
    shap_exp = explainer(X_s_trans)
    values = np.asarray(shap_exp.values)

    # Global importance
    mean_abs = np.abs(values).mean(axis=0)
    importance = sorted(
        [{"feature": n, "mean_abs_shap": float(v)} for n, v in zip(names, mean_abs, strict=True)],
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )

    importance_path = cfg.out_dir / "shap_global_importance.json"
    write_json(importance_path, {"importance": importance})

    # Plots
    bar_path = cfg.out_dir / "shap_summary_bar.png"
    beeswarm_path = cfg.out_dir / "shap_summary_beeswarm.png"

    shap.summary_plot(values, X_s_trans, feature_names=names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(values, X_s_trans, feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Manifest
    manifest_path = cfg.out_dir / "manifest.json"
    write_json(
        manifest_path,
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
            "shap": {"num_background": int(len(X_bg)), "num_samples": int(len(X_s))},
            "artifacts": {
                "importance_json": str(importance_path),
                "summary_bar": str(bar_path),
                "summary_beeswarm": str(beeswarm_path),
            },
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

    LOGGER.info("Saved SHAP artifacts to %s", cfg.out_dir)
    return {"importance_top10": importance[:10]}


# -----------------------------
# CLI
# -----------------------------


def _env_path(name: str) -> Optional[Path]:
    v = os.getenv(name)
    return Path(v) if v else None


def main() -> None:
    cfg = ExplainConfig(
        feature_manifest_path=_env_path("FEATURE_MANIFEST_PATH"),
        features_path=_env_path("FEATURES_PATH"),
        target_col=os.getenv("TARGET_COL", "default"),
        date_col=os.getenv("DATE_COL", "issue_d"),
        train_fraction=float(os.getenv("TRAIN_FRACTION", "0.8")),
        train_run_dir=_env_path("TRAIN_RUN_DIR"),
        model_path=_env_path("MODEL_PATH"),
        out_dir=_env_path("EXPLAIN_OUT_DIR") or ExplainConfig().out_dir,
        num_background=int(os.getenv("NUM_BACKGROUND", "1000")),
        num_samples=int(os.getenv("NUM_SAMPLES", "2000")),
        random_state=int(os.getenv("RANDOM_STATE", str(RANDOM_STATE))),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
    setup_logging(cfg.log_level)
    run_shap_global(cfg)


if __name__ == "__main__":
    main()
