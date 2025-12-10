import os
import tempfile

import pandas as pd


def test_load_raw_loans(monkeypatch):
    """Test that load_raw_loans correctly loads CSV files from the raw data directory."""
    import src.config as config_module
    import src.data.load_data as load_data_module
    from src.data.load_data import load_raw_loans

    # Create temporary directory for test isolation
    temp_dir = tempfile.mkdtemp()

    try:
        # Create a mock CSV file that matches the expected naming pattern
        accepted_file = os.path.join(temp_dir, "accepted_2007_to_2018Q4.csv")

        # Create mock data
        mock_data = pd.DataFrame(
            {
                "loan_amnt": [10000],
                "annual_inc": [75000],
                "int_rate": ["12.0%"],
                "term": ["36 months"],
                "loan_status": ["Fully Paid"],
                "dti": [8.0],
            }
        )
        mock_data.to_csv(accepted_file, index=False)

        # Patch the RAW_DATA_DIR in both modules
        monkeypatch.setattr(config_module, "RAW_DATA_DIR", temp_dir)
        monkeypatch.setattr(load_data_module, "RAW_DATA_DIR", temp_dir)

        # Test the load_raw_loans function
        df = load_raw_loans("accepted_2007_to_2018Q4.csv")

        assert not df.empty, "DataFrame is empty"
        assert list(df.columns) == list(mock_data.columns), (
            f"Columns do not match: {df.columns} vs {mock_data.columns}"
        )
        assert len(df) >= len(mock_data), f"Expected at least {len(mock_data)} rows, got {len(df)}"

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_preprocess_loans(monkeypatch):
    """Test that preprocess_loans correctly concatenates multiple data files."""
    import src.config as config_module
    import src.data.load_data as load_data_module
    import src.data.preprocess as preprocess_module
    from src.data.preprocess import preprocess_loans

    # Create temporary directories for test isolation
    temp_dir = tempfile.mkdtemp()
    temp_interim_dir = tempfile.mkdtemp()
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

        # Use monkeypatch for safer test isolation - patch all modules that use these directories
        monkeypatch.setattr(config_module, "RAW_DATA_DIR", temp_dir)
        monkeypatch.setattr(preprocess_module, "RAW_DATA_DIR", temp_dir)
        monkeypatch.setattr(load_data_module, "RAW_DATA_DIR", temp_dir)
        monkeypatch.setattr(config_module, "INTERIM_DATA_DIR", temp_interim_dir)
        monkeypatch.setattr(preprocess_module, "INTERIM_DATA_DIR", temp_interim_dir)

        output_filename = "test_loans_preprocessed.parquet"
        preprocess_loans(output_file=output_filename)

        output_path = os.path.join(temp_interim_dir, output_filename)
        assert os.path.exists(output_path), f"Output file not found at {output_path}"

        df = pd.read_parquet(output_path)
        assert not df.empty, "DataFrame is empty"
        assert "default" in df.columns, "Missing 'default' column"
        # Verify concatenation worked: we should have data from both files
        assert len(df) >= 2, f"Expected at least 2 rows, got {len(df)}"

    finally:
        # Clean up temp directories and their contents
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(temp_interim_dir, ignore_errors=True)
