# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

"""Feature transformation functions shared across training and serving."""

import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific feature engineering transformations.

    This function must be consistent between training and serving.
    """
    df = df.copy()

    # loan_to_income ratio
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        safe_annual_inc = df["annual_inc"].replace(0, pd.NA)
        df["loan_to_income"] = ((df["loan_amnt"] / safe_annual_inc) * 100).round(4)

    # Extract term in months
    if "term" in df.columns:
        df["term_months"] = df["term"].str.extract(r"(\d+)")[0].astype(int)

    # Convert grade to numeric
    if "grade" in df.columns:
        df["grade_numeric"] = (
            df["grade"]
            .str.extract(r"([A-G])")[0]
            .replace({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7})
            .infer_objects(copy=False)
            .astype(int)
        )

    # Convert sub_grade to numeric
    if "sub_grade" in df.columns:
        df["sub_grade_numeric"] = df["sub_grade"].str.extract(r"([A-G])")[0].replace(
            {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        ).infer_objects(copy=False).astype(int) * 10 + df["sub_grade"].str.extract(r"(\d+)")[
            0
        ].astype(int)

    # Create binary default target variable from loan_status
    if "loan_status" in df.columns:
        default_statuses = ["Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"]
        df["default"] = df["loan_status"].isin(default_statuses).astype(int)

    return df
