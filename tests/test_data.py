import os

import pandas as pd
from src.config import INTERIM_DATA_DIR
from src.data.load_data import load_raw_loans
from src.data.preprocess import preprocess_loans

string = "test.csv"


def test_load_raw_loans():
    df = load_raw_loans(string)
    assert not df.empty


def test_preprocess_loans():
    output_filename = "test_loans_preprocessed.parquet"
    preprocess_loans(output_file=output_filename)

    output_path = os.path.join(INTERIM_DATA_DIR, output_filename)
    assert os.path.exists(output_path)

    df = pd.read_parquet(output_path)
    assert not df.empty
    assert "default" in df.columns

    os.remove(output_path)
