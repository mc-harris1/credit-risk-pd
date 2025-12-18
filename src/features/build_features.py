# src/features/build_features.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.features.transforms import add_domain_features

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildFeaturesConfig:
    input_path: Path = INTERIM_DATA_DIR / "loans_cleaned.parquet"
    output_path: Path = PROCESSED_DATA_DIR / "engineered_features_v1.parquet"

    # Artifacts
    artifacts_dir: Path = Path(ARTIFACTS_DIR)
    feature_spec_path: Path = Path(ARTIFACTS_DIR) / "feature_spec_v1.json"
    target_definition_path: Path = Path(ARTIFACTS_DIR) / "target_definition_v1.json"
    column_roles_path: Path = Path(ARTIFACTS_DIR) / "column_roles_v1.json"

    # Contract fields (manifest-first)
    manifest_path: Path = Path(ARTIFACTS_DIR) / "engineered_features_manifest_v1.json"
    manifest_version: str = "engineered_features_manifest_v1"

    target_col: str = "default"
    date_col: str = "issue_d"
    id_cols: Tuple[str, ...] = ("loan_id",)

    # Split defaults (cutoff_date can be filled later by train)
    split_strategy: str = "time_based"
    train_fraction: float = 0.80

    log_level: str = "INFO"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324 (non-crypto use)


def _schema_md5(df: pd.DataFrame) -> str:
    parts = [f"{c}:{str(df[c].dtype)}" for c in df.columns]
    return _md5_text("|".join(parts))


def _head_md5(df: pd.DataFrame, n: int = 50) -> str:
    head = df.head(n).to_csv(index=False)
    return _md5_text(head)


def _try_rel(path: Path, base: Path) -> Optional[str]:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return None


def _infer_model_features(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    id_cols: Tuple[str, ...],
) -> List[str]:
    drop = {target_col, date_col, *id_cols}
    return [c for c in df.columns if c not in drop]


def build_engineered_features(cfg: BuildFeaturesConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")

    df = pd.read_parquet(cfg.input_path)
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    df = add_domain_features(df)

    # Ensure target exists (robustness for CI + future preprocess tweaks)
    if cfg.target_col not in df.columns:
        if "loan_status" in df.columns and cfg.target_col == "default":
            default_statuses = {
                "Charged Off",
                "Default",
                "Late (31-120 days)",
                "In Grace Period",
                "Late (16-30 days)",
            }
            df[cfg.target_col] = df["loan_status"].isin(default_statuses).astype(int)
            LOGGER.info("Derived target '%s' from loan_status mapping.", cfg.target_col)
        else:
            raise KeyError(f"Expected target column '{cfg.target_col}' to exist after FE.")

    if cfg.date_col not in df.columns:
        raise KeyError(f"Expected date column '{cfg.date_col}' to exist after FE.")

    extra_meta: Dict[str, Any] = {
        "fitted_encoder_summaries": {"one_hot": {}, "frequency": {}},
    }
    return df, extra_meta


def write_engineered_features_and_manifest(cfg: BuildFeaturesConfig) -> Dict[str, Any]:
    ensure_dir(cfg.output_path.parent)

    engineered_df, extra_meta = build_engineered_features(cfg)
    engineered_df.to_parquet(cfg.output_path, index=False)

    model_features = _infer_model_features(engineered_df, cfg.target_col, cfg.date_col, cfg.id_cols)

    payload: Dict[str, Any] = {
        "manifest_version": cfg.manifest_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Paths
        "input_path": str(cfg.input_path),
        "input_path_rel": _try_rel(cfg.input_path, DATA_DIR),
        "output_path": str(cfg.output_path),
        "features_path": str(cfg.output_path),
        "output_path_rel": _try_rel(cfg.output_path, DATA_DIR),
        "features_path_rel": _try_rel(cfg.output_path, DATA_DIR),
        # Contract fields
        "target_col": cfg.target_col,
        "date_col": cfg.date_col,
        "id_cols": list(cfg.id_cols),
        # Linked specs
        "feature_spec_path": str(cfg.feature_spec_path),
        "feature_spec_path_rel": _try_rel(cfg.feature_spec_path, DATA_DIR),
        "target_definition_path": str(cfg.target_definition_path)
        if cfg.target_definition_path.exists()
        else None,
        "target_definition_path_rel": _try_rel(cfg.target_definition_path, DATA_DIR)
        if cfg.target_definition_path.exists()
        else None,
        "column_roles_path": str(cfg.column_roles_path) if cfg.column_roles_path.exists() else None,
        "column_roles_path_rel": _try_rel(cfg.column_roles_path, DATA_DIR)
        if cfg.column_roles_path.exists()
        else None,
        # Split skeleton
        "split": {
            "strategy": cfg.split_strategy,
            "train_fraction": float(cfg.train_fraction),
            "date_col": cfg.date_col,
            "cutoff_date": None,
        },
        # Dataset shape
        "n_rows": int(engineered_df.shape[0]),
        "n_cols": int(engineered_df.shape[1]),
        # Drift guards
        "output_head_md5": _head_md5(engineered_df),
        "output_schema_md5": _schema_md5(engineered_df),
        "feature_list_md5": _md5_text("|".join(model_features)),
        # Helpful peeks
        "created_cols_sample": model_features[:25],
        # Extra metadata hook
        **extra_meta,
    }

    write_json(cfg.manifest_path, payload)
    LOGGER.info("Wrote engineered features: %s", cfg.output_path)
    LOGGER.info("Wrote manifest: %s", cfg.manifest_path)
    return payload


def _env_path(name: str) -> Optional[Path]:
    v = os.getenv(name)
    return Path(v) if v else None


def main() -> None:
    cfg = BuildFeaturesConfig(
        input_path=_env_path("INPUT_PATH") or BuildFeaturesConfig().input_path,
        output_path=_env_path("OUTPUT_PATH") or BuildFeaturesConfig().output_path,
        manifest_path=_env_path("FEATURE_MANIFEST_OUT") or BuildFeaturesConfig().manifest_path,
        target_col=os.getenv("TARGET_COL", "default"),
        date_col=os.getenv("DATE_COL", "issue_d"),
        train_fraction=float(os.getenv("TRAIN_FRACTION", "0.8")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    setup_logging(cfg.log_level)
    write_engineered_features_and_manifest(cfg)


if __name__ == "__main__":
    main()
