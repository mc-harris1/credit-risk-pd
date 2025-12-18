# src/models/load.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass(frozen=True)
class ModelingInputs:
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    target: str
    feature_cols: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]
    engineered_cols: List[str]
    metadata: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path.resolve()}")
    with open(path, "r") as f:
        return json.load(f)


def _ensure_list(d: Dict[str, Any], key: str) -> List[str]:
    v = d.get(key, [])
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"Expected '{key}' to be a list in spec, got {type(v)}")
    bad = [x for x in v if not isinstance(x, str) or not x.strip()]
    if bad:
        raise ValueError(f"Spec '{key}' contains non-string/empty entries (sample): {bad[:10]}")
    return v


def _infer_target(spec: Dict[str, Any]) -> str:
    # support a few common keys at top level
    for k in ("target", "target_variable", "label", "y_col"):
        if k in spec and isinstance(spec[k], str) and spec[k].strip():
            return spec[k]

    # check nested location: spec["exclusions"]["target"]
    if "exclusions" in spec and isinstance(spec["exclusions"], dict):
        target = spec["exclusions"].get("target")
        if isinstance(target, str) and target.strip():
            return target

    raise ValueError(
        "Could not infer target column from feature spec. "
        "Expected one of keys: target, target_variable, label, y_col "
        "(at top level or under exclusions.target)."
    )


def _validate_schema(df: pd.DataFrame, required_cols: List[str], *, strict_extra: bool) -> None:
    df_cols = set(df.columns)
    req = set(required_cols)

    missing = sorted(list(req - df_cols))
    extra = sorted(list(df_cols - req))

    if missing:
        raise ValueError(
            "Schema mismatch: required columns are missing from FE dataset.\n"
            f"Missing (sample): {missing[:50]}"
        )
    if strict_extra and extra:
        raise ValueError(
            "Schema mismatch: FE dataset has unexpected extra columns (strict_extra=True).\n"
            f"Extra (sample): {extra[:50]}"
        )


def load_modeling_inputs(
    artifacts_dir: str | Path = "../data/_artifacts_preview",
    fe_parquet_name: str = "engineered_features.parquet",
    feature_spec_name: str = "feature_spec_v1.json",
    *,
    strict_extra_columns: bool = False,
    drop_datetime_from_X: bool = True,
) -> ModelingInputs:
    """
    Loads the FE dataset produced by 02_fe.ipynb and returns modeling-ready inputs.

    Parameters
    ----------
    artifacts_dir:
        Directory containing FE artifacts.
    fe_parquet_name:
        Parquet filename produced by feature engineering (should include target).
    feature_spec_name:
        Feature spec JSON produced by feature engineering.
    strict_extra_columns:
        If True, fails if FE parquet contains columns not declared in the feature spec.
        If False (default), allows extras but still fails if required cols are missing.
    drop_datetime_from_X:
        If True (default), datetime columns are excluded from X (common for baseline models).
        Keep them if you're doing explicit datetime feature extraction downstream.
    """
    artifacts_dir = Path(artifacts_dir)

    fe_path = artifacts_dir / fe_parquet_name
    spec_path = artifacts_dir / feature_spec_name

    if not fe_path.exists():
        raise FileNotFoundError(f"Missing FE parquet: {fe_path.resolve()}")
    spec = _read_json(spec_path)

    target = _infer_target(spec)

    # Extract column lists from spec
    # First try top-level keys for backward compatibility
    numeric_cols = _ensure_list(spec, "numerical_cols")
    categorical_cols = _ensure_list(spec, "categorical_cols")
    datetime_cols = _ensure_list(spec, "datetime_cols")
    engineered_cols = _ensure_list(spec, "engineered_cols")

    # If empty, try extracting from nested "features" dict (newer format)
    if not numeric_cols and "features" in spec and "numerical" in spec["features"]:
        numeric_cols = sorted(list(spec["features"]["numerical"].keys()))

    if not categorical_cols and "features" in spec and "categorical" in spec["features"]:
        categorical_cols = sorted(list(spec["features"]["categorical"].keys()))

    if not datetime_cols and "features" in spec and "datetime" in spec["features"]:
        datetime_cols = sorted(list(spec["features"]["datetime"].keys()))

    # Some specs use "features" explicitly at top level; if present, we respect it.
    if "feature_cols" in spec and isinstance(spec["feature_cols"], list):
        feature_cols = _ensure_list(spec, "feature_cols")
    else:
        feature_cols = sorted(
            list(set(numeric_cols + categorical_cols + datetime_cols + engineered_cols))
        )

    # --- Load data
    df = pd.read_parquet(fe_path)

    # --- Contract checks
    required_cols = feature_cols + [target]
    _validate_schema(df, required_cols, strict_extra=strict_extra_columns)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in FE dataset.")

    # --- Build X/y
    X = df[feature_cols].copy()
    y = df[target].copy()

    if drop_datetime_from_X and datetime_cols:
        # Only drop those datetime cols that are actually in feature_cols
        dt_to_drop = [c for c in datetime_cols if c in X.columns]
        X = X.drop(columns=dt_to_drop)

    # basic sanity checks
    if y.isna().any():
        raise ValueError("Target contains missing values. Fix in EDA/FE before modeling.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Row mismatch between X and y.")

    metadata = {
        "artifacts_dir": str(artifacts_dir.resolve()),
        "fe_path": str(fe_path.resolve()),
        "spec_path": str(spec_path.resolve()),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "drop_datetime_from_X": drop_datetime_from_X,
        "strict_extra_columns": strict_extra_columns,
    }

    return ModelingInputs(
        df=df,
        X=X,
        y=y,
        target=target,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        engineered_cols=engineered_cols,
        metadata=metadata,
    )


# Optional CLI usage: python -m src.modeling.load_modeling_inputs
if __name__ == "__main__":
    mi = load_modeling_inputs()
    print("✅ Loaded modeling inputs")
    print(mi.metadata)
    print(f"X: {mi.X.shape} | y: {mi.y.shape} | target={mi.target}")
