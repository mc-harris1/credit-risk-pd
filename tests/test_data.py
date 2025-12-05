import os
import tempfile

import pandas as pd
from src.config import INTERIM_DATA_DIR
from src.data.load_data import load_raw_loans
from src.data.preprocess import preprocess_loans

string = "test.csv"


def test_load_raw_loans():
    df = load_raw_loans(string)
    assert not df.empty


def test_preprocess_loans():
    """Test that preprocess_loans correctly concatenates multiple data files."""
    import src.data.preprocess as preprocess_module

    # Save original RAW_DATA_DIR
    original_raw_dir = preprocess_module.RAW_DATA_DIR

    # Create temporary mock data files to test concatenation
    temp_dir = tempfile.mkdtemp()
    try:
        # Create two mock CSV files that match the expected naming pattern
        accepted_file = os.path.join(temp_dir, "accepted_2007_to_2018Q4.csv")
        rejected_file = os.path.join(temp_dir, "rejected_2007_to_2018Q4.csv")

        # Create mock data with the required columns
        mock_data_1 = pd.DataFrame(
            {
                "loan_amnt": [5000, 10000],
                "annual_inc": [50000, 75000],
                "int_rate": ["10.5%", "12.0%"],
                "term": ["36 months", "60 months"],
                "loan_status": ["Fully Paid", "Charged Off"],
                "dti": [5.0, 8.0],
            }
        )

        mock_data_2 = pd.DataFrame(
            {
                "loan_amnt": [15000, 20000],
                "annual_inc": [100000, 120000],
                "int_rate": ["8.5%", "9.0%"],
                "term": ["36 months", "60 months"],
                "loan_status": ["Default", "Fully Paid"],
                "dti": [3.0, 6.0],
            }
        )

        mock_data_1.to_csv(accepted_file, index=False)
        mock_data_2.to_csv(rejected_file, index=False)

        # Monkey-patch RAW_DATA_DIR to point to our temp directory
        preprocess_module.RAW_DATA_DIR = temp_dir

        output_filename = "test_loans_preprocessed.parquet"
        preprocess_loans(output_file=output_filename)

        output_path = os.path.join(INTERIM_DATA_DIR, output_filename)
        assert os.path.exists(output_path)

        df = pd.read_parquet(output_path)
        assert not df.empty
        assert "default" in df.columns
        # Verify concatenation worked: we should have data from both files
        assert len(df) >= 2

        os.remove(output_path)
    finally:
        # Restore original RAW_DATA_DIR before cleaning up temp directory
        preprocess_module.RAW_DATA_DIR = original_raw_dir
        # Clean up temp directory and its contents
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
