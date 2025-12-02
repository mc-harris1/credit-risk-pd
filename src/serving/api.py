from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MODELS_DIR

MODEL_FILE = "pd_model_xgb.pkl"

app = FastAPI(title="Credit Risk PD API", version="0.1.0")

_model = None


class LoanApplication(BaseModel):
    # Add whatever fields your feature set expects.
    # These should line up with columns in your training data.
    loan_amnt: float
    annual_inc: float
    dti: float | None = None
    term: str | None = None
    home_ownership: str | None = None
    grade: str | None = None
    # You can extend this model as needed.


class ScoreResponse(BaseModel):
    prob_default: float
    risk_band: Literal["Low", "Medium", "High"]
    # Placeholder: in the future, add SHAP-based explanations.
    top_factors: list[dict] | None = None


def load_model():
    global _model
    if _model is None:
        model_path = MODELS_DIR / MODEL_FILE
        if not model_path.exists():
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

    # Convert incoming payload to a single-row dict for the model.
    X_row = {k: v for k, v in app_data.model_dump().items()}

    try:
        proba = model.predict_proba([X_row])[0][1]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    prob_default = float(proba)
    band = _risk_band(prob_default)

    return ScoreResponse(prob_default=prob_default, risk_band=band, top_factors=None)
