# src/serving/api.py
from __future__ import annotations

import logging
import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import MODELS_DIR
from src.features.transforms import add_domain_features
from src.models.train import MODEL_FILE
from src.serving.schemas import FeatureContribution, LoanApplication, ScoreResponse
from src.serving.shap_utils import explain_instance

logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk PD Scoring API")


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load the trained sklearn pipeline once per process."""
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        msg = f"Model artifact not found at {model_path}. Train the model first."
        logger.error(msg)
        raise FileNotFoundError(msg)

    pipeline = joblib.load(model_path)
    if "preprocess" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        msg = "Expected pipeline with 'preprocess' and 'model' steps."
        logger.error(msg)
        raise ValueError(msg)

    return pipeline


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    try:
        _ = _load_pipeline()
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Health check failed: %s", exc)
        raise HTTPException(status_code=500, detail="Model not available") from exc


@app.post("/predict", response_model=ScoreResponse)
def predict(app_data: LoanApplication) -> ScoreResponse:
    """
    Score a single loan application and return PD + SHAP top factors.

    Response is guaranteed to have:
    - pd: float
    - top_factors: list[FeatureContribution] (possibly empty)
    """
    # Build 1-row DataFrame from request
    df_row_raw = pd.DataFrame([app_data.model_dump()])

    # Apply domain features (required before pipeline prediction)
    df_row = add_domain_features(df_row_raw)

    # Drop date column (not used as feature)
    if "issue_d" in df_row.columns:
        df_row = df_row.drop(columns=["issue_d"])

    # Predict PD
    try:
        pipeline = _load_pipeline()
        proba = pipeline.predict_proba(df_row)[:, 1][0]
        pd_value = float(proba)
        risk_band = "Low" if pd_value < 0.1 else "Medium" if pd_value < 0.3 else "High"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction error") from exc

    # SHAP explanations (best-effort)
    top_factors: list[FeatureContribution] = []
    try:
        # Pass raw input to explain_instance (it applies domain features internally)
        shap_contribs = explain_instance(df_row_raw, top_n=5)
        top_factors = [
            FeatureContribution(feature=str(item["feature"]), shap_value=item["shap_value"])
            for item in shap_contribs
        ]
    except Exception as exc:  # noqa: BLE001
        # Do not fail the request; just log and return an empty list
        logger.warning("SHAP explanation failed: %s", exc)
        top_factors = []

    return ScoreResponse(prob_default=pd_value, risk_band=risk_band, top_factors=top_factors)
