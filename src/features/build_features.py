import pandas as pd

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

DEFAULT_INPUT_FILE = "loans_preprocessed.parquet"
DEFAULT_OUTPUT_FILE = "loans_features.parquet"


def build_features(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> None:
    """Feature engineering stub: builds simple numeric features and a binary target."""
    input_path = INTERIM_DATA_DIR / input_file
    df = pd.read_parquet(input_path)

    # TODO: replace these with real LendingClub-style feature engineering
    if "loan_amnt" in df and "annual_inc" in df:
        df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1e-3)

    # Stub target: you will replace with real logic (status -> default)
    if "loan_status" in df:
        df["default"] = df["loan_status"].isin(
            ["Charged Off", "Default", "Late (31-120 days)"]
        ).astype(int)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / output_file
    df.to_parquet(output_path, index=False)
    print(f"Saved feature dataset to {output_path}")


if __name__ == "__main__":
    build_features()
