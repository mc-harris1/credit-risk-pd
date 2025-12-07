import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

DEFAULT_INPUT_FILE = "loans_preprocessed.parquet"
DEFAULT_OUTPUT_FILE = "loans_features.parquet"

pd.set_option("future.no_silent_downcasting", True)


def build_features(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> None:
    """Feature engineering stub: builds simple numeric features and a binary target."""
    input_path = os.path.join(INTERIM_DATA_DIR, input_file)
    df = pd.read_parquet(input_path)

    # Engineer simple features
    if "loan_amnt" in df and "annual_inc" in df:
        df["loan_to_income"] = ((df["loan_amnt"] / df["annual_inc"]) * 100).round(4)

    if "term" in df:
        df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)

    if "grade" in df:
        df["grade_numeric"] = (
            df["grade"]
            .str.extract(r"([A-G])")
            .replace({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7})
            .astype(int)
        )

    if "sub_grade" in df:
        df["sub_grade_numeric"] = df["sub_grade"].str.extract(r"([A-G])").replace(
            {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        ).astype(int) * 10 + df["sub_grade"].str.extract(r"(\d+)").astype(int)

    # Ensure target variable is binary: 1 for default, 0 for non-default
    if "loan_status" in df:
        df["default"] = (
            df["loan_status"].isin(["Charged Off", "Default", "Late (31-120 days)"]).astype(int)
        )

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, output_file)
    df.to_parquet(output_path, index=False)
    print(f"Saved feature dataset to {output_path}")


if __name__ == "__main__":
    build_features()
