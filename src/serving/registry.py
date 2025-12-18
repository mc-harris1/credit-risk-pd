# src/serving/registry.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.config import get_model_dir

logger = logging.getLogger(__name__)

MODEL_FILENAME_DEFAULT = "model.joblib"
FEATURE_SPEC_DEFAULT = "feature_spec.json"
METADATA_DEFAULT = "metadata.json"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    model_dir: Path
    metadata: Dict[str, Any]
    feature_spec: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_bundle(
    model_dir: Optional[str | Path] = None,
    *,
    model_filename: str = MODEL_FILENAME_DEFAULT,
    feature_spec_filename: str = FEATURE_SPEC_DEFAULT,
    metadata_filename: str = METADATA_DEFAULT,
) -> ModelBundle:
    """
    Load the model bundle from:
      - explicit model_dir, or
      - PD_MODEL_DIR env var, or
      - latest models/bundles/* directory.
    """
    resolved_dir = get_model_dir(Path(model_dir) if model_dir else None)
    resolved_dir = Path(resolved_dir).resolve()

    model_path = resolved_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    feature_spec = _read_json(resolved_dir / feature_spec_filename)
    metadata = _read_json(resolved_dir / metadata_filename)

    logger.info("Loaded model bundle from %s", resolved_dir)
    return ModelBundle(
        model=model, model_dir=resolved_dir, metadata=metadata, feature_spec=feature_spec
    )


def enforce_feature_contract(df: pd.DataFrame, feature_spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Enforce required feature columns and ordering if present in feature_spec.
    Expects feature_spec to contain 'feature_cols' or 'feature_columns' (list[str]).
    """
    cols = None
    for k in ("feature_cols", "feature_columns", "features"):
        if k in feature_spec and isinstance(feature_spec[k], list):
            cols = feature_spec[k]
            break

    if not cols:
        # No contract present, return as-is (still useful for backward compat)
        return df

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing[:25]}")

    # reorder and drop extras
    return df[cols].copy()
