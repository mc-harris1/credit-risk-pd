# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Legacy output dirs (keep for backward compatibility during refactor)
MODELS_ARTIFACTS_DIR: Path = PROJECT_ROOT / "models" / "artifacts"
MODELS_METADATA_DIR: Path = PROJECT_ROOT / "models" / "metadata"

# Backward-compatible aliases expected by older modules/tests
# These names were previously exported and are kept to avoid import errors.
MODELS_DIR: Path = PROJECT_ROOT / "models"
METADATA_DIR: Path = MODELS_METADATA_DIR

# New canonical output dir (recommended)
MODELS_BUNDLES_DIR: Path = PROJECT_ROOT / "models" / "bundles"

# Reproducibility
RANDOM_STATE: int = 42

# Deployment/environment overrides
PD_MODEL_DIR_ENV = "PD_MODEL_DIR"
LOG_LEVEL_ENV = "LOG_LEVEL"


def get_model_dir(default: Optional[Path] = None) -> Path:
    """
    Resolve the model bundle directory to use for inference.

    Priority:
      1) PD_MODEL_DIR environment variable
      2) `default` arg (if provided)
      3) Latest bundle under models/bundles
      4) Fallback to legacy artifacts dir (last resort)
    """
    env_val = os.getenv(PD_MODEL_DIR_ENV)
    if env_val:
        p = Path(env_val).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"{PD_MODEL_DIR_ENV} is set but does not exist: {p}")
        return p

    if default is not None:
        p = default.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Default model_dir does not exist: {p}")
        return p

    if MODELS_BUNDLES_DIR.exists():
        bundles = sorted([d for d in MODELS_BUNDLES_DIR.iterdir() if d.is_dir()])
        if bundles:
            return bundles[-1]

    # fallback (kept only to avoid breaking things during migration)
    return MODELS_ARTIFACTS_DIR
