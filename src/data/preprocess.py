import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.data.load_data import load_raw_loans

DEFAULT_OUTPUT_FILE = "loans_preprocessed.parquet"


def preprocess_loans(
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> None:
    dfs = []

    """Basic preprocessing stub: load, lightly clean, and save interim data."""
    for input_file in os.listdir(RAW_DATA_DIR):  # Look in raw data directory
        # if the file is of the .csv format and matches the following pattern - _2007_to_2018Q4.csv
        # Skip test.csv and other files that don't match the production data pattern
        if input_file.endswith(".csv") and "_2007_to_2018Q4" in input_file:
            temp_df = load_raw_loans(input_file)
            dfs.append(temp_df)

    if not dfs:
        raise ValueError("No matching data files found in raw data directory")

    df = pd.concat(dfs, ignore_index=True)

    # 1. Drop fully empty rows
    df = df.dropna(how="all")

    # 2. Standardize column names if needed (example for OpenIntro / LC style)
    df = df.rename(
        columns={
            "loan_amount": "loan_amnt",
            "annual_income": "annual_inc",
        }
    )

    # 3. Keep only columns that are required for modeling
    required_cols = [
        "loan_amnt",
        "annual_inc",
        "int_rate",
        "term",
        "loan_status",
        "dti",
        # add grade, sub_grade, emp_length, home_ownership, etc. as needed
    ]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df[existing_required].copy()

    # 4. Drop rows missing critical fields
    critical_cols = ["loan_amnt", "annual_inc", "int_rate", "term", "loan_status"]
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])

    # 5. Clean numeric fields & drop impossible values
    # Convert interest_rate from "13.56%" → 13.56 if needed
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].astype(str).str.rstrip("%").astype(float)

    # Cast numerics
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce")
    df["dti"] = pd.to_numeric(df["dti"], errors="coerce")

    # Drop any rows where these conversions failed
    df = df.dropna(subset=["loan_amnt", "annual_inc", "int_rate"])

    # Drop obviously impossible values
    mask_valid = (
        (df["loan_amnt"] > 0)
        & (df["annual_inc"] > 0)
        & (df["int_rate"] > 0)
        & (df["int_rate"] <= 100)
    )
    if "dti" in df:
        mask_valid &= (df["dti"] >= 0) & (df["dti"] <= 100)

    df = df[mask_valid].copy()

    # 6. Map loan_status → binary default target; drop ambiguous rows
    status_to_default = {
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1,
        "Late (31-120 days)": 1,
        # extend this mapping as needed based on dataset
    }

    df["default"] = df["loan_status"].map(status_to_default)

    # "Broken" target rows = statuses we didn't map → default = NaN
    df = df.dropna(subset=["default"])

    # Cast to int
    df["default"] = df["default"].astype(int)

    # ensure interim directory exists
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    output_path = os.path.join(INTERIM_DATA_DIR, output_file)
    df.to_parquet(output_path, index=False)
    print(f"Saved preprocessed loans to {output_path}")


if __name__ == "__main__":
    preprocess_loans()
