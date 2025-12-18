# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from pydantic import BaseModel, Field
from typing_extensions import Literal


class LoanApplication(BaseModel):
    """
    Input schema for PD scoring at origination.

    Fields represent borrower- and loan-level information
    available at decision time (PIT).
    """

    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    annual_inc: float = Field(..., gt=0, description="Annual income")

    int_rate: float | None = Field(default=10.0, ge=0.0, le=50.0, description="Interest rate (%)")
    term: str | None = Field(default="36 months", description="Loan term (e.g., '36 months')")
    dti: float | None = Field(
        default=15.0, ge=0.0, le=100.0, description="Debt-to-income ratio (%)"
    )

    grade: str | None = Field(default="C", description="Credit grade")
    sub_grade: str | None = Field(default="C1", description="Credit sub-grade")
    emp_length: str | None = Field(default="5 years", description="Employment length")
    home_ownership: str | None = Field(default="RENT", description="Home ownership status")
    loan_status: str | None = Field(default=None, description="Loan status (ignored)")

    issue_d: str | None = Field(
        default=None,
        description="Origination month (YYYY-MM). Used for logging/monitoring only.",
    )

    class Config:
        extra = "forbid"  # fail fast on unexpected fields


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class ScoreResponse(BaseModel):
    prob_default: float = Field(..., ge=0.0, le=1.0)
    risk_band: Literal["Low", "Medium", "High"]
    top_factors: list[FeatureContribution] = Field(default_factory=list)
