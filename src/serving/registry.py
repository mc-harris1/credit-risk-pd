from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
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
        self._feature_mapper: Optional[FeatureMapper] = None

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

        if hasattr(self._model, "feature_names_in_"):
            self._feature_mapper = FeatureMapper(list(self._model.feature_names_in_))
        else:
            self._feature_mapper = None

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
        """
        Accept raw payload from the API, map it to the model's expected
        preprocessed feature columns, then predict.
        """
        # Ensure model is loaded before inspecting feature names
        if not self.is_loaded():
            self.load()

        df_raw = self.to_frame(application_dict)

        if self._feature_mapper is not None:
            df = self._feature_mapper.transform(df_raw)
        else:
            df = df_raw

        pd_val = self.predict_pd(df)
        return pd_val, df

    # --- Helpers ---------------------------------------------------------

    def _preprocess_raw(self, df_raw: pd.DataFrame, expected_feats: np.ndarray) -> pd.DataFrame:
        """
        Map raw request features to the engineered feature space the model expects.
        This covers:
          - annual_inc__log1p
          - one-hot grade, sub_grade, home_ownership, term
          - passes through dti and annual_inc
          - fills missing engineered fields with 0/NaN (imputer will handle)
        """
        expected = list(expected_feats)
        out = pd.DataFrame(np.zeros((len(df_raw), len(expected))), columns=expected)

        def set_col(col, series):
            if col in out.columns:
                out[col] = series

        raw = df_raw.iloc[0]

        # Numeric direct
        set_col("annual_inc", pd.Series([raw.get("annual_inc", np.nan)]))
        set_col("dti", pd.Series([raw.get("dti", np.nan)]))
        set_col("loan_amnt", pd.Series([raw.get("loan_amnt", np.nan)]))  # add this

        # Derived numeric
        if "annual_inc__log1p" in out.columns:
            ann = raw.get("annual_inc", np.nan)
            out["annual_inc__log1p"] = np.log1p(ann) if pd.notnull(ann) else np.nan

        # Grade one-hot
        grade_val = str(raw.get("grade", "")).strip()
        grade_cols = [c for c in expected if c.startswith("grade_")]
        if grade_cols:
            if grade_val and f"grade_{grade_val}" in out.columns:
                out[f"grade_{grade_val}"] = 1
            elif "grade_<NA>" in out.columns:
                out["grade_<NA>"] = 1

        # Sub-grade one-hot
        sub_grade_val = str(raw.get("sub_grade", "")).strip()
        sub_grade_cols = [c for c in expected if c.startswith("sub_grade_")]
        if sub_grade_cols:
            if sub_grade_val and f"sub_grade_{sub_grade_val}" in out.columns:
                out[f"sub_grade_{sub_grade_val}"] = 1

        # Home ownership one-hot
        ho_val = str(raw.get("home_ownership", "")).strip()
        ho_cols = [c for c in expected if c.startswith("home_ownership_")]
        if ho_cols:
            if ho_val and f"home_ownership_{ho_val}" in out.columns:
                out[f"home_ownership_{ho_val}"] = 1

        # Term one-hot
        term_val = str(raw.get("term", "")).strip()
        term_cols = [c for c in expected if c.startswith("term_")]
        if term_cols:
            if term_val and f"term_{term_val}" in out.columns:
                out[f"term_{term_val}"] = 1

        # emp_length__target_mean – we cannot derive; leave NaN for imputer
        if "emp_length__target_mean" in out.columns:
            out["emp_length__target_mean"] = np.nan

        # Additional engineered features seen in model
        if "sub_grade__target_mean" in out.columns:
            out["sub_grade__target_mean"] = 0.0  # no encoder at inference; imputer handles
        if "loan_age_months" in out.columns:
            out["loan_age_months"] = 0.0  # not provided by API; set to 0
        if "int_rate" in out.columns:
            out["int_rate"] = 0.0  # not provided by API; set to 0

        return out


class FeatureMapper:
    """
    Lightweight, stateless mapper to turn raw API inputs into the exact
    engineered feature set expected by the trained model.
    """

    def __init__(self, expected_features: list[str]) -> None:
        self.expected = expected_features
        self.grade_cols = [c for c in expected_features if c.startswith("grade_")]
        self.sub_grade_cols = [c for c in expected_features if c.startswith("sub_grade_")]
        self.home_ownership_cols = [c for c in expected_features if c.startswith("home_ownership_")]
        self.term_cols = [c for c in expected_features if c.startswith("term_")]

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.zeros((len(df_raw), len(self.expected))), columns=self.expected)
        raw = df_raw.iloc[0]

        def set_col(col, val):
            if col in out.columns:
                out[col] = val

        # Numeric pass-through
        set_col("annual_inc", pd.Series([raw.get("annual_inc", np.nan)]))
        set_col("dti", pd.Series([raw.get("dti", np.nan)]))
        set_col("loan_amnt", pd.Series([raw.get("loan_amnt", np.nan)]))

        # Numeric derived
        if "annual_inc__log1p" in out.columns:
            ann = raw.get("annual_inc", np.nan)
            out["annual_inc__log1p"] = np.log1p(ann) if pd.notnull(ann) else np.nan

        # One-hot: grade
        gv = str(raw.get("grade", "")).strip()
        if gv and f"grade_{gv}" in out.columns:
            out[f"grade_{gv}"] = 1
        elif "grade_<NA>" in out.columns:
            out["grade_<NA>"] = 1

        # One-hot: sub_grade
        sg = str(raw.get("sub_grade", "")).strip()
        if sg and f"sub_grade_{sg}" in out.columns:
            out[f"sub_grade_{sg}"] = 1

        # One-hot: home_ownership
        ho = str(raw.get("home_ownership", "")).strip()
        if ho and f"home_ownership_{ho}" in out.columns:
            out[f"home_ownership_{ho}"] = 1

        # One-hot: term
        term = str(raw.get("term", "")).strip()
        if term and f"term_{term}" in out.columns:
            out[f"term_{term}"] = 1

        # Engineered placeholders (imputer will handle NaN/0)
        if "emp_length__target_mean" in out.columns:
            out["emp_length__target_mean"] = np.nan
        if "sub_grade__target_mean" in out.columns:
            out["sub_grade__target_mean"] = np.nan
        if "loan_age_months" in out.columns:
            out["loan_age_months"] = 0.0
        if "int_rate" in out.columns:
            out["int_rate"] = 0.0

        return out


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    reg = ModelRegistry()
    # Lazy-load (but you can uncomment to eager load):
    # reg.load()
    return reg
