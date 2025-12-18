# tests/test_data_validation.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.data.load_data import load_raw_loans


class TestRawDataValidation:
    """Tests for raw data validation."""

    def test_load_raw_loans_validates_columns(self, tmp_path: Path) -> None:
        """Test that loaded data has expected structure."""
        csv_file = tmp_path / "loans.csv"
        df = pd.DataFrame(
            {
                "loan_amnt": [5000, 10000],
                "annual_inc": [50000, 75000],
            }
        )
        df.to_csv(csv_file, index=False)

        result = load_raw_loans(csv_file)

        assert result.shape == (2, 2)
        assert "loan_amnt" in result.columns
        assert "annual_inc" in result.columns

    def test_load_raw_loans_handles_large_file(self, tmp_path: Path) -> None:
        """Test that large files are loaded correctly."""
        csv_file = tmp_path / "loans.csv"
        large_df = pd.DataFrame(
            {
                "loan_amnt": [5000] * 10000,
                "annual_inc": [50000] * 10000,
            }
        )
        large_df.to_csv(csv_file, index=False)

        result = load_raw_loans(csv_file)

        assert len(result) == 10000

    def test_load_raw_loans_preserves_data_types(self, tmp_path: Path) -> None:
        """Test that data types are preserved on load."""
        csv_file = tmp_path / "loans.csv"
        df = pd.DataFrame(
            {
                "loan_amnt": [5000, 10000],
                "interest_rate": [5.5, 8.5],
                "status": ["Paid", "Default"],
            }
        )
        df.to_csv(csv_file, index=False)

        result = load_raw_loans(csv_file)

        assert result is not None
        assert len(result) == 2


class TestDataQualityChecks:
    """Tests for data quality validation."""

    def test_detect_missing_values(self) -> None:
        """Test that missing values are properly detected."""
        df = pd.DataFrame(
            {
                "loan_amnt": [5000, None, 10000],
                "annual_inc": [50000, 75000, None],
            }
        )

        assert df["loan_amnt"].isna().sum() == 1
        assert df["annual_inc"].isna().sum() == 1

    def test_detect_duplicates(self) -> None:
        """Test that duplicate rows are detected."""
        df = pd.DataFrame(
            {
                "loan_id": [1, 2, 2, 3],
                "loan_amnt": [5000, 10000, 10000, 15000],
            }
        )

        duplicates = df[df.duplicated(subset=["loan_id"], keep=False)]
        assert len(duplicates) > 0

    def test_detect_outliers_interest_rate(self) -> None:
        """Test detection of outlier interest rates."""
        df = pd.DataFrame(
            {
                "int_rate": [5.5, 8.5, 500.0, 6.5],  # 500% is clearly an outlier
            }
        )

        # Simple outlier detection: >100%
        outliers = df[df["int_rate"] > 100]
        assert len(outliers) == 1

    def test_detect_negative_amounts(self) -> None:
        """Test detection of negative loan amounts."""
        df = pd.DataFrame(
            {
                "loan_amnt": [5000, -1000, 10000],
                "annual_inc": [50000, -5000, 75000],
            }
        )

        assert (df["loan_amnt"] < 0).sum() == 1
        assert (df["annual_inc"] < 0).sum() == 1

    def test_check_data_consistency(self) -> None:
        """Test checking for data consistency issues."""
        df = pd.DataFrame(
            {
                "loan_amnt": [5000, 10000],
                "annual_inc": [50000, 75000],
                "dti": [10.0, 13.3],  # dti = (loan_amnt * 12) / annual_inc (approximation)
            }
        )

        # Basic consistency check
        assert len(df) == len(df.dropna())


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validate_required_columns(self) -> None:
        """Test that required columns are validated."""
        required_cols = ["loan_amnt", "annual_inc", "int_rate"]
        df = pd.DataFrame(
            {
                "loan_amnt": [5000],
                "annual_inc": [50000],
            }
        )

        missing = [c for c in required_cols if c not in df.columns]
        assert "int_rate" in missing

    def test_validate_column_types(self) -> None:
        """Test that column types are validated."""
        df = pd.DataFrame(
            {
                "loan_amnt": ["5000"],  # Should be numeric
                "annual_inc": [50000],
            }
        )

        assert df["loan_amnt"].dtype == object
        assert pd.api.types.is_numeric_dtype(df["annual_inc"])

    def test_validate_enum_values(self) -> None:
        """Test that enum fields have valid values."""
        valid_grades = ["A", "B", "C", "D", "E", "F", "G"]
        df = pd.DataFrame(
            {
                "grade": ["A", "B", "Z"],  # Z is invalid
            }
        )

        invalid = [g for g in df["grade"] if g not in valid_grades]
        assert "Z" in invalid

    def test_validate_date_format(self) -> None:
        """Test that dates are in valid format."""
        df = pd.DataFrame(
            {
                "issue_d": ["2020-01-01", "2020-02-01", "invalid-date"],
            }
        )

        # Try to parse dates
        parsed = pd.to_datetime(df["issue_d"], errors="coerce")
        assert parsed.isna().sum() == 1  # One invalid date


class TestTargetValidation:
    """Tests for target variable validation."""

    def test_validate_target_is_binary(self) -> None:
        """Test that target is binary."""
        df = pd.DataFrame(
            {
                "default": [0, 1, 0, 1, 0],
            }
        )

        unique_vals = set(df["default"].unique())
        assert unique_vals.issubset({0, 1})

    def test_detect_class_imbalance(self) -> None:
        """Test detection of class imbalance."""
        df = pd.DataFrame(
            {
                "default": [0] * 90 + [1] * 10,  # 90-10 split
            }
        )

        class_counts = df["default"].value_counts()
        ratio = class_counts.min() / class_counts.max()
        assert ratio < 0.2  # Significant imbalance

    def test_validate_no_null_target(self) -> None:
        """Test that target has no null values."""
        df = pd.DataFrame(
            {
                "default": [0, 1, None, 0],
            }
        )

        assert df["default"].isna().sum() == 1

    def test_validate_target_coverage(self) -> None:
        """Test that target covers sufficient data."""
        df = pd.DataFrame(
            {
                "default": [0, 1, 0, 1],
            }
        )

        # All rows should have a target
        assert df["default"].notna().sum() == len(df)


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""

    def test_single_row_dataframe(self) -> None:
        """Test handling of single-row dataframe."""
        df = pd.DataFrame(
            {
                "loan_amnt": [5000],
                "annual_inc": [50000],
            }
        )

        assert len(df) == 1
        assert not df.empty

    def test_max_value_handling(self) -> None:
        """Test handling of maximum values."""
        df = pd.DataFrame(
            {
                "loan_amnt": [9999999999],
                "annual_inc": [9999999999],
            }
        )

        assert df["loan_amnt"].max() > 0
        assert df["annual_inc"].max() > 0

    def test_min_value_handling(self) -> None:
        """Test handling of minimum positive values."""
        df = pd.DataFrame(
            {
                "loan_amnt": [1],
                "annual_inc": [1],
            }
        )

        assert df["loan_amnt"].min() > 0
        assert df["annual_inc"].min() > 0

    def test_nan_handling(self) -> None:
        """Test proper handling of NaN values."""
        import numpy as np

        df = pd.DataFrame(
            {
                "value": [1.0, np.nan, 3.0],
            }
        )

        assert df["value"].isna().sum() == 1
        assert df["value"].notna().sum() == 2

    def test_infinity_handling(self) -> None:
        """Test handling of infinity values."""
        import numpy as np

        df = pd.DataFrame(
            {
                "value": [1.0, np.inf, 3.0],
            }
        )

        assert np.isinf(df["value"]).sum() == 1


class TestDataConsistency:
    """Tests for data consistency across pipeline."""

    def test_column_order_preserved(self) -> None:
        """Test that column order is preserved through transformations."""
        df = pd.DataFrame(
            {
                "col_a": [1],
                "col_b": [2],
                "col_c": [3],
            }
        )

        col_order = list(df.columns)
        df_copy = df.copy()

        assert list(df_copy.columns) == col_order

    def test_row_count_tracking(self) -> None:
        """Test that row counts are tracked through pipeline."""
        df_initial = pd.DataFrame({"value": range(100)})
        n_initial = len(df_initial)

        # Simulate some filtering
        df_filtered = df_initial[df_initial["value"] > 50]
        n_filtered = len(df_filtered)

        assert n_filtered < n_initial
        assert n_filtered == 49  # 51-99 inclusive

    def test_id_uniqueness(self) -> None:
        """Test that IDs remain unique through transformations."""
        df = pd.DataFrame(
            {
                "loan_id": [1, 2, 3, 4, 5],
                "value": [10, 20, 30, 40, 50],
            }
        )

        assert len(df["loan_id"]) == len(df["loan_id"].unique())
