# src/serving/api.py
from __future__ import annotations

import logging
import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features.transforms import add_domain_features
from src.serving.registry import load_bundle
from src.serving.schemas import FeatureContribution, LoanApplication, ScoreResponse
from src.serving.shap_utils import explain_instance

logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk PD Scoring API")

# In production, consider disabling SHAP by default to reduce latency/cost
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "false").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def _get_bundle():
    return load_bundle()  # uses PD_MODEL_DIR or latest bundle


@lru_cache(maxsize=1)
def _load_pipeline():
    """
    Backward-compatible loader used by tests; returns the fitted pipeline.
    """
    bundle = _get_bundle()
    return bundle.model


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _ = _load_pipeline()
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Health check failed: %s", exc)
        raise HTTPException(status_code=500, detail="Model not available") from exc


@app.post("/predict", response_model=ScoreResponse)
def predict(app_data: LoanApplication) -> ScoreResponse:
    # Build 1-row DataFrame from request
    df_row_raw = pd.DataFrame([app_data.model_dump()])

    # Apply domain features (required before inference)
    df_row = add_domain_features(df_row_raw)

    # Drop date column (not used as feature)
    if "issue_d" in df_row.columns:
        df_row = df_row.drop(columns=["issue_d"])

    try:
        pipeline = _load_pipeline()

        # Predict PD
        proba = pipeline.predict_proba(df_row)[:, 1][0]
        pd_value = float(proba)

        risk_band = "Low" if pd_value < 0.1 else "Medium" if pd_value < 0.3 else "High"

    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction error") from exc

    # SHAP explanations (best-effort, optionally disabled)
    top_factors: list[FeatureContribution] = []
    if ENABLE_SHAP:
        try:
            shap_contribs = explain_instance(df_row_raw, top_n=5)
            top_factors = [
                FeatureContribution(feature=str(item["feature"]), shap_value=item["shap_value"])
                for item in shap_contribs
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP explanation failed: %s", exc)
            top_factors = []

    return ScoreResponse(prob_default=pd_value, risk_band=risk_band, top_factors=top_factors)
