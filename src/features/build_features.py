# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

import os

import pandas as pd

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.features.transforms import add_domain_features

DEFAULT_INPUT_FILE = "loans_preprocessed.parquet"
DEFAULT_OUTPUT_FILE = "loans_features.parquet"


def build_features(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> None:
    """Feature engineering: applies domain transformations and creates target variable."""
    input_path = os.path.join(INTERIM_DATA_DIR, input_file)
    df = pd.read_parquet(input_path)

    # Apply domain feature engineering
    df = add_domain_features(df)

    # Ensure target variable is binary: 1 for default, 0 for non-default
    if "default" not in df.columns:
        raise ValueError("Expected 'default' column; did you run preprocess first?")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, output_file)
    df.to_parquet(output_path, index=False)
    print(f"Saved feature dataset to {output_path}")


if __name__ == "__main__":
    build_features()
