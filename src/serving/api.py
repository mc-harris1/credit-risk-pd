import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MODELS_DIR
from src.features.transforms import add_domain_features  # noqa: F401 - needed for unpickling

MODEL_FILE = "pd_model_xgb.pkl"

app = FastAPI(title="Credit Risk PD API", version="0.1.0")

_model = None


class LoanApplication(BaseModel):
    # Add whatever fields your feature set expects.
    # These should line up with columns in your training data.
    loan_amnt: float
    annual_inc: float
    int_rate: float | None = None
    term: str | None = None
    loan_status: str | None = None
    dti: float | None = None
    grade: str | None = None
    sub_grade: str | None = None
    emp_length: str | None = None
    home_ownership: str | None = None
    issue_d: str | None = None  # e.g.,
    # You can extend this model as needed.


class ScoreResponse(BaseModel):
    prob_default: float
    risk_band: Literal["Low", "Medium", "High"]
    # Placeholder: in the future, add SHAP-based explanations.
    top_factors: list[dict] | None = None


def load_model():
    global _model
    if _model is None:
        model_path = os.path.join(MODELS_DIR, MODEL_FILE)
        if not os.path.exists(model_path):
            msg = f"Model file not found at {model_path}. Train the model first."
            raise RuntimeError(msg)
        _model = joblib.load(model_path)
    return _model


def _risk_band(prob_default: float) -> Literal["Low", "Medium", "High"]:
    if prob_default < 0.1:
        return "Low"
    if prob_default < 0.25:
        return "Medium"
    return "High"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(app_data: LoanApplication) -> ScoreResponse:
    model = load_model()

    # Convert incoming payload to a DataFrame for the model.
    X_row = pd.DataFrame([app_data.model_dump()])

    try:
        proba = model.predict_proba(X_row)[0][1]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    prob_default = float(proba)
    band = _risk_band(prob_default)

    return ScoreResponse(prob_default=prob_default, risk_band=band, top_factors=None)
