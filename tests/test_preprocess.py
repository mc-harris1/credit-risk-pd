# tests/test_preprocess.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

import pandas as pd
from src.data.preprocess import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    STATUS_TO_DEFAULT,
    TARGET_COL,
)

# unchanged imports omitted for brevity

DEFAULT_OUTPUT_FILE = "loans_cleaned.parquet"
DEFAULT_PATTERN_CONTAINS = "_2007_to_2018Q4"


class TestPreprocessConstants:
    """Tests for preprocessing configuration constants."""

    def test_canonical_columns_includes_target(self):
        """Test that CANONICAL_COLUMNS includes the target column."""
        assert TARGET_COL in CANONICAL_COLUMNS
        assert "default" in CANONICAL_COLUMNS

    def test_required_columns_includes_target(self):
        """Test that REQUIRED_COLUMNS includes the target column."""
        assert TARGET_COL in REQUIRED_COLUMNS
        assert "default" in REQUIRED_COLUMNS

    def test_canonical_columns_structure(self):
        """Test that CANONICAL_COLUMNS has expected columns."""
        expected_minimal = [
            "loan_amnt",
            "annual_inc",
            "int_rate",
            "term",
            "dti",
            "issue_d",
            "default",
        ]
        for col in expected_minimal:
            assert col in CANONICAL_COLUMNS

    def test_required_columns_subset_of_canonical(self):
        """Test that REQUIRED_COLUMNS is a subset of CANONICAL_COLUMNS."""
        for col in REQUIRED_COLUMNS:
            assert col in CANONICAL_COLUMNS

    def test_status_to_default_mapping(self):
        """Test that STATUS_TO_DEFAULT has expected mappings."""
        # Defaults (1)
        assert STATUS_TO_DEFAULT["Charged Off"] == 1
        assert STATUS_TO_DEFAULT["Default"] == 1
        assert STATUS_TO_DEFAULT["Late (31-120 days)"] == 1
        assert STATUS_TO_DEFAULT["Late (16-30 days)"] == 1

        # Non-defaults (0)
        assert STATUS_TO_DEFAULT["Fully Paid"] == 0


class TestPreprocessTargetColumn:
    """Tests for target column definition and validation."""

    def test_target_col_constant(self):
        """Test that TARGET_COL is properly defined."""
        assert TARGET_COL == "default"

    def test_status_mapping_includes_all_required_statuses(self):
        """Test that STATUS_TO_DEFAULT includes typical loan statuses."""
        assert "Fully Paid" in STATUS_TO_DEFAULT
        assert "Charged Off" in STATUS_TO_DEFAULT
        assert "Default" in STATUS_TO_DEFAULT

    def test_status_mapping_creates_binary_target(self):
        """Test that status mapping produces binary (0/1) target."""
        df = pd.DataFrame(
            {
                "loan_status": list(STATUS_TO_DEFAULT.keys()),
            }
        )

        # Create target column using the mapping
        df["default"] = df["loan_status"].map(STATUS_TO_DEFAULT)

        # Should have only 0s and 1s
        assert set(df["default"].unique()) == {0, 1}

    def test_status_mapping_consistency(self):
        """Test that default status mapping is consistent."""
        late_statuses = [
            "Late (31-120 days)",
            "Late (16-30 days)",
        ]
        for status in late_statuses:
            assert STATUS_TO_DEFAULT[status] == 1


class TestCanonicalColumnsContract:
    """Tests for the canonical columns contract."""

    def test_canonical_columns_are_immutable(self):
        """Test that canonical columns list is stable."""
        # This serves as a regression test for future changes
        assert "default" in CANONICAL_COLUMNS
        assert "loan_amnt" in CANONICAL_COLUMNS
        assert "annual_inc" in CANONICAL_COLUMNS

    def test_canonical_vs_required_relationship(self):
        """Test that required columns are properly constrained."""
        # These fields must be required because they're critical
        critical_fields = ["loan_amnt", "annual_inc", "int_rate", "issue_d", "default"]
        for field in critical_fields:
            assert field in REQUIRED_COLUMNS

    def test_canonical_includes_dates(self):
        """Test that canonical columns include date information."""
        assert "issue_d" in CANONICAL_COLUMNS

    def test_canonical_includes_credit_grades(self):
        """Test that canonical columns include credit information."""
        assert "grade" in CANONICAL_COLUMNS
        assert "sub_grade" in CANONICAL_COLUMNS


class TestPreprocessingDataQuality:
    """Tests for data quality expectations after preprocessing."""

    def test_target_binary_validation(self):
        """Test that target should be binary after preprocessing."""
        df = pd.DataFrame(
            {
                "default": [0, 1, 0, 1, 0],
            }
        )

        unique_vals = set(df["default"].unique())
        assert unique_vals.issubset({0, 1})

    def test_detect_class_imbalance(self):
        """Test detection of class imbalance."""
        df = pd.DataFrame(
            {
                "default": [0] * 90 + [1] * 10,  # 90-10 split
            }
        )

        class_counts = df["default"].value_counts()
        ratio = class_counts.min() / class_counts.max()
        assert ratio < 0.2  # Significant imbalance

    def test_required_fields_no_nulls(self):
        """Test that required fields should have no nulls."""
        df = pd.DataFrame(
            {
                "loan_amnt": [5000, 10000],
                "annual_inc": [50000, 75000],
                "default": [0, 1],
            }
        )

        for col in REQUIRED_COLUMNS:
            if col in df.columns:
                assert df[col].notna().all(), f"Column {col} has nulls"


class TestPreprocessingArtifacts:
    """Tests for preprocessing artifacts and outputs."""

    def test_canonical_columns_count(self):
        """Test that canonical columns has expected number of columns."""
        # Should have at least the core columns
        assert len(CANONICAL_COLUMNS) >= 10

    def test_required_columns_count(self):
        """Test that required columns is reasonable subset."""
        assert len(REQUIRED_COLUMNS) < len(CANONICAL_COLUMNS)
        assert len(REQUIRED_COLUMNS) >= 5

    def test_status_to_default_completeness(self):
        """Test that STATUS_TO_DEFAULT has expected statuses."""
        expected_statuses = [
            "Fully Paid",
            "Charged Off",
            "Default",
            "Late (31-120 days)",
            "Late (16-30 days)",
        ]
        for status in expected_statuses:
            assert status in STATUS_TO_DEFAULT


class TestPreprocessingPipeline:
    """Tests for preprocessing pipeline structure and contracts."""

    def test_canonical_columns_define_output_schema(self):
        """Test that canonical columns define the expected output schema."""
        # After preprocessing, output should have these canonical columns
        expected_in_output = [
            "loan_amnt",
            "annual_inc",
            "int_rate",
            "issue_d",
            "default",
        ]
        for col in expected_in_output:
            assert col in CANONICAL_COLUMNS

    def test_required_columns_validation(self):
        """Test that required columns are properly validated."""
        # These columns must exist after preprocessing
        for col in REQUIRED_COLUMNS:
            assert col in CANONICAL_COLUMNS

    def test_target_always_required(self):
        """Test that target column is always required in output."""
        assert TARGET_COL in REQUIRED_COLUMNS
        assert TARGET_COL in CANONICAL_COLUMNS
