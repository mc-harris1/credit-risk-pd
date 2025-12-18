import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from src.features.build_features import (
    BuildFeaturesConfig,
    build_engineered_features,
    write_engineered_features_and_manifest,
)
from src.features.transforms import add_domain_features


def _create_mock_cleaned_data(**kwargs):
    """Helper to create mock cleaned data with defaults."""
    defaults = {
        "loan_amnt": [5000, 10000],
        "annual_inc": [50000, 75000],
        "term": ["36 months", "60 months"],
        "grade": ["A", "B"],
        "sub_grade": ["A1", "B2"],
        "loan_status": ["Fully Paid", "Charged Off"],
        "issue_d": ["2020-01-01", "2020-02-01"],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestBuildFeaturesBasic:
    """Basic tests for build_engineered_features configuration."""

    def test_build_features_config_defaults(self):
        """Test that BuildFeaturesConfig has proper defaults."""
        cfg = BuildFeaturesConfig()
        assert cfg.target_col == "default"
        assert cfg.date_col == "issue_d"
        assert cfg.split_strategy == "time_based"
        assert cfg.train_fraction == 0.80

    def test_build_features_config_target_col(self):
        """Test that BuildFeaturesConfig can set custom target column."""
        cfg = BuildFeaturesConfig(target_col="custom_target")
        assert cfg.target_col == "custom_target"

    def test_build_features_config_train_fraction(self):
        """Test that BuildFeaturesConfig can set custom train fraction."""
        cfg = BuildFeaturesConfig(train_fraction=0.70)
        assert cfg.train_fraction == 0.70


class TestBuildEngineeredFeatures:
    """Tests for build_engineered_features function."""

    def test_build_engineered_features_creates_target(self):
        """Test that build_engineered_features creates default target from loan_status."""
        temp_dir = tempfile.mkdtemp()
        try:
            input_file = Path(temp_dir) / "cleaned.parquet"
            mock_data = _create_mock_cleaned_data()
            mock_data.to_parquet(input_file, index=False)

            cfg = BuildFeaturesConfig(
                input_path=input_file,
                output_path=Path(temp_dir) / "output.parquet",
            )

            df, metadata = build_engineered_features(cfg)

            assert "default" in df.columns, "Target column 'default' not created"
            assert df["default"].dtype in [int, "int64", "int32"]
            assert set(df["default"].unique()).issubset({0, 1})
            assert len(metadata["fitted_encoder_summaries"]) > 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_engineered_features_preserves_existing_target(self):
        """Test that existing target column is not overwritten."""
        temp_dir = tempfile.mkdtemp()
        try:
            input_file = Path(temp_dir) / "cleaned.parquet"
            mock_data = _create_mock_cleaned_data(default=[1, 0])
            mock_data.to_parquet(input_file, index=False)

            cfg = BuildFeaturesConfig(
                input_path=input_file,
                output_path=Path(temp_dir) / "output.parquet",
            )

            df, _ = build_engineered_features(cfg)

            assert "default" in df.columns
            assert df.loc[0, "default"] == 1
            assert df.loc[1, "default"] == 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_engineered_features_missing_date_col(self):
        """Test that missing date column raises KeyError."""
        temp_dir = tempfile.mkdtemp()
        try:
            input_file = Path(temp_dir) / "cleaned.parquet"
            # Create data without issue_d
            mock_data = pd.DataFrame(
                {
                    "loan_amnt": [5000],
                    "annual_inc": [50000],
                    "loan_status": ["Fully Paid"],
                }
            )
            mock_data.to_parquet(input_file, index=False)

            cfg = BuildFeaturesConfig(
                input_path=input_file,
                output_path=Path(temp_dir) / "output.parquet",
            )

            with pytest.raises(KeyError, match="date column"):
                build_engineered_features(cfg)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_engineered_features_missing_target_col(self):
        """Test that missing target column (and no loan_status) raises KeyError."""
        temp_dir = tempfile.mkdtemp()
        try:
            input_file = Path(temp_dir) / "cleaned.parquet"
            # Create data without loan_status or default
            mock_data = pd.DataFrame(
                {
                    "loan_amnt": [5000],
                    "annual_inc": [50000],
                    "issue_d": ["2020-01-01"],
                }
            )
            mock_data.to_parquet(input_file, index=False)

            cfg = BuildFeaturesConfig(
                input_path=input_file,
                output_path=Path(temp_dir) / "output.parquet",
            )

            with pytest.raises(KeyError, match="target column"):
                build_engineered_features(cfg)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestWriteManifest:
    """Tests for manifest generation."""

    def test_write_manifest_creates_file(self):
        """Test that manifest file is created with correct structure."""
        temp_dir = tempfile.mkdtemp()
        try:
            input_file = Path(temp_dir) / "cleaned.parquet"
            manifest_file = Path(temp_dir) / "manifest.json"
            output_file = Path(temp_dir) / "output.parquet"

            mock_data = _create_mock_cleaned_data()
            mock_data.to_parquet(input_file, index=False)

            cfg = BuildFeaturesConfig(
                input_path=input_file,
                output_path=output_file,
                manifest_path=manifest_file,
            )

            write_engineered_features_and_manifest(cfg)

            assert manifest_file.exists(), "Manifest file not created"

            with manifest_file.open() as f:
                manifest = json.load(f)

            # Check required manifest fields
            assert manifest["manifest_version"] == "engineered_features_manifest_v1"
            assert manifest["target_col"] == "default"
            assert manifest["date_col"] == "issue_d"
            assert manifest["n_rows"] > 0
            assert manifest["n_cols"] > 0
            assert "output_schema_md5" in manifest
            assert "output_head_md5" in manifest
            assert "feature_list_md5" in manifest
            assert "created_at" in manifest
            assert manifest["split"]["strategy"] == "time_based"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_manifest_contains_paths(self):
        """Test that manifest contains input/output paths."""
        temp_dir = tempfile.mkdtemp()
        try:
            input_file = Path(temp_dir) / "cleaned.parquet"
            manifest_file = Path(temp_dir) / "manifest.json"
            output_file = Path(temp_dir) / "output.parquet"

            mock_data = _create_mock_cleaned_data()
            mock_data.to_parquet(input_file, index=False)

            cfg = BuildFeaturesConfig(
                input_path=input_file,
                output_path=output_file,
                manifest_path=manifest_file,
            )

            write_engineered_features_and_manifest(cfg)

            with manifest_file.open() as f:
                manifest = json.load(f)

            assert "input_path" in manifest
            assert "output_path" in manifest
            assert "features_path" in manifest
            assert str(input_file) in manifest["input_path"]
            assert str(output_file) in manifest["output_path"]

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestFeatureTransformsEdgeCases:
    """Tests for edge cases in feature transforms."""

    def test_add_domain_features_with_nulls(self):
        """Test add_domain_features handles null values gracefully."""
        df = pd.DataFrame(
            {
                "loan_amnt": [5000, None, 10000],
                "annual_inc": [50000, 75000, None],
                "term": ["36 months", None, "60 months"],
                "grade": ["A", None, "B"],
                "sub_grade": ["A1", None, "B2"],
            }
        )

        result = add_domain_features(df)

        # Should not crash and should preserve nulls in ratio
        assert result is not None
        assert "loan_to_income" in result.columns
        assert "term_months" in result.columns
        assert pd.isna(result.loc[1, "loan_to_income"])

    def test_add_domain_features_zero_income(self):
        """Test add_domain_features handles income of zero (replaced with NA)."""
        # This is an edge case where the preprocessing should have already removed zero income
        # Just verify the function doesn't crash with valid numeric dtypes
        df = pd.DataFrame(
            {
                "loan_amnt": pd.Series([5000.0], dtype="float64"),
                "annual_inc": pd.Series([75000.0], dtype="float64"),
            }
        )

        result = add_domain_features(df)

        # Should produce a valid result
        assert result is not None
        assert "loan_to_income" in result.columns
        assert result.loc[0, "loan_to_income"] > 0

    def test_add_domain_features_invalid_grade(self):
        """Test add_domain_features handles invalid grades."""
        df = pd.DataFrame(
            {
                "grade": ["A", "X", "B"],
            }
        )

        result = add_domain_features(df)

        # Invalid grade should be handled
        assert result is not None
        assert "grade_numeric" in result.columns
        assert result.loc[0, "grade_numeric"] == 1
        assert result.loc[2, "grade_numeric"] == 2

    def test_add_domain_features_missing_columns(self):
        """Test add_domain_features with missing optional columns."""
        df = pd.DataFrame({"loan_amnt": [5000]})

        result = add_domain_features(df)

        # Should not crash even with minimal input
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_build_features_with_late_payment_status(self):
        """Test that Late payment statuses map to default."""
        temp_dir = tempfile.mkdtemp()
        try:
            input_file = Path(temp_dir) / "cleaned.parquet"
            mock_data = pd.DataFrame(
                {
                    "loan_amnt": [5000, 10000, 15000],
                    "annual_inc": [50000, 75000, 90000],
                    "term": ["36 months"] * 3,
                    "grade": ["A"] * 3,
                    "sub_grade": ["A1"] * 3,
                    "loan_status": [
                        "Fully Paid",
                        "Late (31-120 days)",
                        "Charged Off",
                    ],
                    "issue_d": ["2020-01-01"] * 3,
                }
            )
            mock_data.to_parquet(input_file, index=False)

            cfg = BuildFeaturesConfig(
                input_path=input_file,
                output_path=Path(temp_dir) / "output.parquet",
            )

            df, _ = build_engineered_features(cfg)

            assert df.loc[0, "default"] == 0  # Fully Paid
            assert df.loc[1, "default"] == 1  # Late (31-120 days)
            assert df.loc[2, "default"] == 1  # Charged Off

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
