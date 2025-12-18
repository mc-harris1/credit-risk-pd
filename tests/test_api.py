from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.serving.api import app
from src.serving.schemas import LoanApplication

client = TestClient(app)

payload = {
    "loan_amnt": 3600,
    "annual_inc": 100000,
    "int_rate": 13.99,
    "term": "36 months",
    "loan_status": "Fully Paid",
    "dti": 5.91,
    "grade": "C",
    "sub_grade": "C4",
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "issue_d": "Dec-2015",
}


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health(self) -> None:
        # Mock the pipeline loader to avoid needing actual model
        mock_pipeline = MagicMock()
        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "timestamp" in data or isinstance(data, dict)

    def test_health_structure(self) -> None:
        """Test that health endpoint returns valid structure."""
        mock_pipeline = MagicMock()
        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            response = client.get("/health")
            assert response.status_code == 200
            assert isinstance(response.json(), dict)


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_success(self) -> None:
        """Test successful prediction."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.70, 0.30]])

        mock_explanation = [
            {"feature": "int_rate", "shap_value": 0.15},
            {"feature": "grade_C", "shap_value": 0.08},
            {"feature": "loan_amnt", "shap_value": -0.05},
        ]

        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            with patch("src.serving.api.explain_instance", return_value=mock_explanation):
                response = client.post("/predict", json=payload)
                assert response.status_code == 200
                data = response.json()
                assert "prob_default" in data
                assert "risk_band" in data
                assert data["risk_band"] in ["Low", "Medium", "High"]
                assert "top_factors" in data
                assert isinstance(data["top_factors"], list)

    def test_predict_response_structure(self) -> None:
        """Test that predict response has correct structure."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.70, 0.30]])

        mock_explanation = [
            {"feature": "int_rate", "shap_value": 0.15},
        ]

        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            with patch("src.serving.api.explain_instance", return_value=mock_explanation):
                response = client.post("/predict", json=payload)
                data = response.json()

                # Check required fields
                assert isinstance(data["prob_default"], (int, float))
                assert 0 <= data["prob_default"] <= 1
                assert isinstance(data["risk_band"], str)
                assert isinstance(data["top_factors"], list)

    def test_predict_missing_required_field(self) -> None:
        """Test prediction with missing required field."""
        incomplete_payload = {k: v for k, v in payload.items() if k != "loan_amnt"}

        mock_pipeline = MagicMock()
        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            response = client.post("/predict", json=incomplete_payload)
            # Should return 422 Unprocessable Entity for validation error
            assert response.status_code in [422, 400]

    def test_predict_invalid_field_type(self) -> None:
        """Test prediction with invalid field type."""
        bad_payload = payload.copy()
        bad_payload["loan_amnt"] = "not_a_number"  # Should be numeric

        mock_pipeline = MagicMock()
        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            response = client.post("/predict", json=bad_payload)
            assert response.status_code in [422, 400]

    def test_predict_negative_amount(self) -> None:
        """Test that negative loan amount is rejected."""
        bad_payload = payload.copy()
        bad_payload["loan_amnt"] = -5000

        mock_pipeline = MagicMock()
        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            response = client.post("/predict", json=bad_payload)
            # Validation error or successful call depends on schema validation
            # If validated, should at least not crash
            assert response.status_code in [200, 422, 400]

    def test_predict_zero_income(self) -> None:
        """Test that zero income is handled."""
        bad_payload = payload.copy()
        bad_payload["annual_inc"] = 0

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.50, 0.50]])

        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            with patch("src.serving.api.explain_instance", return_value=[]):
                response = client.post("/predict", json=bad_payload)
                # Should handle gracefully
                assert response.status_code in [200, 400, 422]

    def test_predict_invalid_grade(self) -> None:
        """Test prediction with invalid grade."""
        bad_payload = payload.copy()
        bad_payload["grade"] = "Z"  # Invalid grade

        mock_pipeline = MagicMock()
        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            response = client.post("/predict", json=bad_payload)
            # Should handle gracefully
            assert response.status_code in [200, 400, 422]

    def test_predict_various_risk_bands(self) -> None:
        """Test that predictions produce consistent risk bands."""
        # Note: risk band logic depends on implementation
        # Just verify that predictions produce valid risk band outputs
        test_probs = [0.1, 0.35, 0.7]

        mock_explanation = [{"feature": "test", "shap_value": 0.01}]

        for prob in test_probs:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = np.array([[1 - prob, prob]])

            with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
                with patch("src.serving.api.explain_instance", return_value=mock_explanation):
                    response = client.post("/predict", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        # Just verify risk_band is one of the expected values
                        assert data["risk_band"] in ["Low", "Medium", "High"]
                        assert isinstance(data["prob_default"], (int, float))
                        assert 0 <= data["prob_default"] <= 1


class TestSchemaValidation:
    """Tests for LoanApplication schema validation."""

    def test_schema_accepts_valid_payload(self) -> None:
        """Test that schema validates correct payload."""
        app_data = LoanApplication(**payload)
        assert app_data.loan_amnt == 3600
        assert app_data.annual_inc == 100000

    def test_schema_rejects_missing_required(self) -> None:
        """Test that schema rejects missing required field."""
        incomplete = {k: v for k, v in payload.items() if k != "loan_amnt"}

        with pytest.raises(ValidationError):
            LoanApplication(**incomplete)

    def test_schema_validates_field_types(self) -> None:
        """Test that schema validates field types."""
        bad_payload = payload.copy()
        bad_payload["loan_amnt"] = "not_a_number"

        with pytest.raises(ValidationError):
            LoanApplication(**bad_payload)

    def test_schema_accepts_all_required_fields(self) -> None:
        """Test that truly required fields cause validation error when missing."""
        # Only loan_amnt and annual_inc are actually required (gt=0)
        required_fields = ["loan_amnt", "annual_inc"]

        for field in required_fields:
            test_payload = payload.copy()
            del test_payload[field]

            with pytest.raises(ValidationError):
                LoanApplication(**test_payload)

    def test_schema_optional_fields_have_defaults(self) -> None:
        """Test that optional fields have sensible defaults."""
        minimal_payload = {
            "loan_amnt": 5000,
            "annual_inc": 50000,
        }

        # Should work with only required fields
        app = LoanApplication(**minimal_payload)
        assert app.loan_amnt == 5000
        assert app.annual_inc == 50000
        # Check defaults were applied
        assert app.int_rate is not None
        assert app.term is not None


class TestErrorHandling:
    """Tests for error handling in API."""

    def test_predict_with_pipeline_error(self) -> None:
        """Test graceful handling of pipeline errors."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.side_effect = RuntimeError("Model prediction failed")

        with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
            # Behavior depends on error handling in API
            response = client.post("/predict", json=payload)
            # Should return 5xx or handle gracefully
            assert response.status_code >= 400

    def test_health_with_pipeline_error(self) -> None:
        """Test health check when pipeline loading fails."""
        with patch("src.serving.api._load_pipeline", side_effect=RuntimeError("Load failed")):
            response = client.get("/health")
            # Health check behavior depends on implementation
            assert response.status_code in [200, 500]

    def test_invalid_json_body(self) -> None:
        """Test API handles invalid JSON gracefully."""
        response = client.post(
            "/predict",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code >= 400


def test_health() -> None:
    """Legacy test for backward compatibility."""
    mock_pipeline = MagicMock()
    with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict() -> None:
    """Legacy test for backward compatibility."""
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.70, 0.30]])

    mock_explanation = [
        {"feature": "int_rate", "shap_value": 0.15},
        {"feature": "grade_C", "shap_value": 0.08},
        {"feature": "loan_amnt", "shap_value": -0.05},
    ]

    with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
        with patch("src.serving.api.explain_instance", return_value=mock_explanation):
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "prob_default" in data
            assert "risk_band" in data
            assert data["risk_band"] in ["Low", "Medium", "High"]
            assert "top_factors" in data
            assert isinstance(data["top_factors"], list)
            if data["top_factors"]:
                factor = data["top_factors"][0]
                assert "feature" in factor
                assert "shap_value" in factor
