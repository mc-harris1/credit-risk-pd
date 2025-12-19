from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoanApplication(BaseModel):
    """
    Raw request schema. Keep this aligned with your Streamlit UI + docs.

    NOTE: You can expand this to match your full feature spec.
    For MVP / tests, a small subset is usually enough.
    """

    model_config = ConfigDict(extra="forbid")

    loan_amnt: float = Field(..., gt=0)
    annual_inc: float = Field(..., gt=0)
    term: str = Field(..., description='e.g. "36 months" or "60 months"')
    home_ownership: str = Field(..., description="e.g. RENT, OWN, MORTGAGE")
    grade: str = Field(..., min_length=1, max_length=2, description="e.g. A, B, C")
    sub_grade: str = Field(..., min_length=2, max_length=3, description="e.g. B3")
    dti: float = Field(..., ge=0)

    @field_validator("term")
    @classmethod
    def normalize_term(cls, v: str) -> str:
        v2 = v.strip()
        # Allow "36 months" / "60 months" and also "36" / "60"
        if v2.isdigit():
            return f"{v2} months"
        return v2


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pd: float = Field(..., ge=0.0, le=1.0)
    model_version: Optional[str] = Field(default=None)

    risk_band: str = Field(default="Unknown")
    top_factors: List[FeatureAttribution] = Field(default_factory=list)

    # add this so we can see why top_factors is empty
    shap_error: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_version: Optional[str] = None


class ExplainRequest(BaseModel):
    """
    Explanation request.
    - top_k controls how many features to return (sorted by |impact|)
    - include_base_value is useful for additive explanations
    """

    model_config = ConfigDict(extra="forbid")

    application: LoanApplication
    top_k: int = Field(default=15, ge=1, le=200)
    include_base_value: bool = True


class FeatureAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    value: Any
    shap_value: float


class ExplainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pd: float = Field(..., ge=0.0, le=1.0)
    model_version: Optional[str] = None
    base_value: Optional[float] = None
    attributions: List[FeatureAttribution]
    meta: Dict[str, Any] = Field(default_factory=dict)
