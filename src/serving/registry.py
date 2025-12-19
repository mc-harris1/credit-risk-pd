from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd


@dataclass(frozen=True)
class ModelBundle:
    bundle_dir: Path
    model_path: Path
    metadata_path: Path
    feature_spec_path: Optional[Path]
    version: str


class ModelRegistryError(RuntimeError):
    pass


def _default_bundles_dir() -> Path:
    # Repo-root relative default: models/bundles
    return Path(os.getenv("MODEL_BUNDLES_DIR", "models/bundles")).resolve()


def _pick_latest_bundle_dir(bundles_dir: Path) -> Path:
    if not bundles_dir.exists():
        raise ModelRegistryError(f"Bundles dir not found: {bundles_dir}")

    candidates = [p for p in bundles_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise ModelRegistryError(f"No model bundles found in: {bundles_dir}")

    # Most bundles are timestamped; lexical sort typically works.
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def _pick_active_bundle_dir(bundles_dir: Path) -> Path:
    # If explicitly set, prefer it
    active = os.getenv("MODEL_BUNDLE_NAME")
    if active:
        p = bundles_dir / active
        if not p.exists():
            raise ModelRegistryError(f"MODEL_BUNDLE_NAME={active} but bundle not found at {p}")
        return p
    return _pick_latest_bundle_dir(bundles_dir)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class ModelRegistry:
    """
    Loads a trained model bundle + (optionally) feature spec and provides
    predict() and dataframe construction helpers.
    """

    def __init__(self, bundles_dir: Optional[Path] = None) -> None:
        self._bundles_dir = (bundles_dir or _default_bundles_dir()).resolve()
        self._bundle: Optional[ModelBundle] = None
        self._model: Any = None
        self._metadata: Dict[str, Any] = {}
        self._feature_spec: Optional[Dict[str, Any]] = None

    @property
    def bundles_dir(self) -> Path:
        return self._bundles_dir

    def load(self) -> None:
        bundle_dir = _pick_active_bundle_dir(self._bundles_dir)
        model_path = bundle_dir / "model.joblib"
        metadata_path = bundle_dir / "metadata.json"
        feature_spec_path = bundle_dir / "feature_spec_v1.json"

        if not model_path.exists():
            raise ModelRegistryError(f"Missing model file: {model_path}")
        if not metadata_path.exists():
            raise ModelRegistryError(f"Missing metadata file: {metadata_path}")

        self._model = joblib.load(model_path)
        self._metadata = _load_json(metadata_path)

        fs = None
        if feature_spec_path.exists():
            fs = _load_json(feature_spec_path)

        self._feature_spec = fs
        self._bundle = ModelBundle(
            bundle_dir=bundle_dir,
            model_path=model_path,
            metadata_path=metadata_path,
            feature_spec_path=feature_spec_path if feature_spec_path.exists() else None,
            version=bundle_dir.name,
        )

    def is_loaded(self) -> bool:
        return self._model is not None and self._bundle is not None

    def version(self) -> Optional[str]:
        return self._bundle.version if self._bundle else None

    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def feature_spec(self) -> Optional[Dict[str, Any]]:
        return dict(self._feature_spec) if self._feature_spec else None

    def to_frame(self, application_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert request payload into a single-row DataFrame.
        If a feature spec exists, order/align to spec where possible.
        """
        df = pd.DataFrame([application_dict])

        fs = self._feature_spec
        if fs:
            # Common pattern: fs["features"] is a list of feature names
            feature_list = None
            if isinstance(fs.get("features"), list):
                feature_list = fs["features"]
            elif isinstance(fs.get("input_features"), list):
                feature_list = fs["input_features"]

            if feature_list:
                # Add any missing features as NaN and order columns
                for col in feature_list:
                    if col not in df.columns:
                        df[col] = pd.NA
                df = df[feature_list]

        return df

    def predict_pd(self, df: pd.DataFrame) -> float:
        """
        Return probability of default.
        Works with sklearn-like estimators having predict_proba.
        """
        if not self.is_loaded():
            self.load()

        model = self._model
        if not hasattr(model, "predict_proba"):
            raise ModelRegistryError("Loaded model does not implement predict_proba()")

        proba = model.predict_proba(df)
        # Assume binary classifier with class-1 as default
        pd_val = float(proba[0, 1])
        return pd_val

    def predict_from_payload(self, application_dict: Dict[str, Any]) -> Tuple[float, pd.DataFrame]:
        df = self.to_frame(application_dict)
        pd_val = self.predict_pd(df)
        return pd_val, df


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    reg = ModelRegistry()
    # Lazy-load (but you can uncomment to eager load):
    # reg.load()
    return reg
