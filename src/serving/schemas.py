# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    loan_amnt: float
    annual_inc: float
    int_rate: float | None = Field(default=10.0)  # Average interest rate
    term: str | None = Field(default="36 months")
    loan_status: str | None = Field(default="Current")
    dti: float | None = Field(default=15.0)
    grade: str | None = Field(default="C")
    sub_grade: str | None = Field(default="C1")
    emp_length: str | None = Field(default="5 years")
    home_ownership: str | None = Field(default="RENT")
    issue_d: str | None = Field(default="2020-01")
    # extend with whatever features you actually train on
