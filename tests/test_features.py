import os
import shutil
import tempfile

import pandas as pd


def test_build_features():
    """Test that the build_features function is called, asserts the engineered features columns, and the default target exists."""
    import src.config as config_module
    import src.features.build_features as build_features_module

    # Save original directory paths from both modules
    original_processed_dir_config = config_module.PROCESSED_DATA_DIR
    original_processed_dir_features = build_features_module.PROCESSED_DATA_DIR
    original_interim_dir_config = config_module.INTERIM_DATA_DIR
    original_interim_dir_features = build_features_module.INTERIM_DATA_DIR

    # Create temporary mock data file to test feature engineering
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a mock parquet file that matches the expected input
        features_file = os.path.join(temp_dir, "loans_preprocessed.parquet")

        # Create a mock preprocessed data file
        mock_data = pd.DataFrame(
            {
                "loan_amnt": [5000, 10000],
                "annual_inc": [50000, 75000],
                "term": ["36 months", "60 months"],
                "grade": ["A", "B"],
                "sub_grade": ["A1", "B2"],
                "loan_status": ["Fully Paid", "Charged Off"],
            }
        )

        mock_data.to_parquet(features_file, index=False)

        # Monkey-patch directories in the modules to use temp_dir
        config_module.PROCESSED_DATA_DIR = temp_dir
        build_features_module.PROCESSED_DATA_DIR = temp_dir
        config_module.INTERIM_DATA_DIR = temp_dir
        build_features_module.INTERIM_DATA_DIR = temp_dir

        # Build features
        output_filename = "test_loans_features.parquet"
        build_features_module.build_features(
            input_file="loans_preprocessed.parquet",
            output_file=output_filename,
        )

        # Verify output file exists and has expected columns
        output_path = os.path.join(temp_dir, output_filename)
        assert os.path.exists(output_path)

        df = pd.read_parquet(output_path)
        assert not df.empty
        assert "loan_to_income" in df.columns
        assert "term_months" in df.columns
        assert "grade_numeric" in df.columns
        assert "sub_grade_numeric" in df.columns
        assert "default" in df.columns

        # Verify actual computed values
        # loan_to_income: 5000/50000*100 = 10.0, 10000/75000*100 = 13.3333
        assert df.loc[0, "loan_to_income"] == 10.0
        assert abs(df.loc[1, "loan_to_income"] - 13.3333) < 0.0001
        
        # term_months: "36 months" → 36, "60 months" → 60
        assert df.loc[0, "term_months"] == 36
        assert df.loc[1, "term_months"] == 60
        
        # grade_numeric: "A" → 1, "B" → 2
        assert df.loc[0, "grade_numeric"] == 1
        assert df.loc[1, "grade_numeric"] == 2
        
        # sub_grade_numeric: "A1" → 11, "B2" → 22
        assert df.loc[0, "sub_grade_numeric"] == 11
        assert df.loc[1, "sub_grade_numeric"] == 22
        
        # default: "Fully Paid" → 0, "Charged Off" → 1
        assert df.loc[0, "default"] == 0
        assert df.loc[1, "default"] == 1

    finally:
        # Restore original directory paths in all modules
        config_module.PROCESSED_DATA_DIR = original_processed_dir_config
        build_features_module.PROCESSED_DATA_DIR = original_processed_dir_features
        config_module.INTERIM_DATA_DIR = original_interim_dir_config
        build_features_module.INTERIM_DATA_DIR = original_interim_dir_features

        # Clean up temp directory and its contents
        shutil.rmtree(temp_dir, ignore_errors=True)
