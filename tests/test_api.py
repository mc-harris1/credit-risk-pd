# tests/test_api.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.serving.api import app, get_registry
from src.serving.schemas import (
    FeatureAttribution,
    LoanApplication,
)

client = TestClient(app)

# Valid payload with RAW features (not preprocessed)
loan_payload = {
    "loan_amnt": 3600,
    "annual_inc": 100000,
    "term": "36 months",
    "dti": 5.91,
    "grade": "C",
    "sub_grade": "C4",
    "home_ownership": "MORTGAGE",
}


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _override_registry(registry: MagicMock) -> None:
    app.dependency_overrides[get_registry] = lambda: registry


def _minimal_df_from_payload() -> pd.DataFrame:
    # Return a REAL dataframe (avoids sklearn feature-name issues)
    return pd.DataFrame(
        [
            {
                "loan_amnt": 3600.0,
                "annual_inc": 100000.0,
                "term": "36 months",
                "dti": 5.91,
                "grade": "C",
                "sub_grade": "C4",
                "home_ownership": "MORTGAGE",
            }
        ]
    )


def _mock_registry(pd_val: float = 0.35, version: str = "1.0.0") -> MagicMock:
    """
    Build a registry mock that matches how src/serving/api.py calls it:
      - registry.version()  (METHOD)
      - registry.predict_from_payload(...) -> (pd, df)
      - registry._model used for SHAP
      - registry.is_loaded(), registry.load() used by /health
    """
    r = MagicMock()
    r.version.return_value = version  # <-- version is callable in API
    r.predict_from_payload.return_value = (pd_val, _minimal_df_from_payload())
    r._model = MagicMock()
    r.is_loaded.return_value = True
    r.load.return_value = None
    return r


class TestExplainEndpoint:
    def test_explain_success(self) -> None:
        mock_registry = _mock_registry(pd_val=0.35, version="1.0.0")
        _override_registry(mock_registry)

        mock_attributions = [
            {"feature": "loan_amnt", "shap_value": 0.15, "value": 3600},
            {"feature": "grade_C", "shap_value": 0.08, "value": 1},
        ]

        with patch(
            "src.serving.api.compute_shap_for_single_row",
            return_value=(0.10, np.array([0.15, 0.08])),
        ):
            with patch("src.serving.api.format_top_attributions", return_value=mock_attributions):
                request_data = {
                    "application": loan_payload,
                    "top_k": 2,
                    "include_base_value": True,
                }
                response = client.post("/explain", json=request_data)

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["pd"] == 0.35
        assert data["model_version"] == "1.0.0"
        assert data["base_value"] == 0.10
        assert len(data["attributions"]) == 2
        assert data["attributions"][0]["feature"] == "loan_amnt"
        assert data["attributions"][0]["shap_value"] == 0.15
        assert "meta" in data
        assert data["meta"]["top_k"] == 2

    def test_explain_rejects_top_k_zero(self) -> None:
        """top_k has ge=1 validation; should return 422."""
        mock_registry = _mock_registry()
        _override_registry(mock_registry)

        request_data = {
            "application": loan_payload,
            "top_k": 0,
            "include_base_value": False,
        }
        response = client.post("/explain", json=request_data)
        assert response.status_code == 422

    def test_explain_meta_structure(self) -> None:
        mock_registry = _mock_registry(pd_val=0.40, version="1.0.0")
        _override_registry(mock_registry)

        mock_attributions = [{"feature": "loan_amnt", "shap_value": 0.12, "value": 3600}]

        with patch(
            "src.serving.api.compute_shap_for_single_row", return_value=(0.10, np.array([0.12]))
        ):
            with patch("src.serving.api.format_top_attributions", return_value=mock_attributions):
                request_data = {
                    "application": loan_payload,
                    "top_k": 5,
                    "include_base_value": True,
                }
                response = client.post("/explain", json=request_data)

        assert response.status_code == 200, response.text
        data = response.json()
        assert "meta" in data
        assert data["meta"]["top_k"] == 5
        assert "note" in data["meta"]

    def test_explain_with_high_shap_values(self) -> None:
        mock_registry = _mock_registry(pd_val=0.95, version="1.0.0")
        _override_registry(mock_registry)

        mock_attributions = [
            {"feature": "annual_inc", "shap_value": 0.50, "value": 100000},
            {"feature": "grade_C", "shap_value": 0.35, "value": 1},
        ]

        with patch(
            "src.serving.api.compute_shap_for_single_row",
            return_value=(0.10, np.array([0.50, 0.35])),
        ):
            with patch("src.serving.api.format_top_attributions", return_value=mock_attributions):
                request_data = {
                    "application": loan_payload,
                    "top_k": 2,
                    "include_base_value": True,
                }
                response = client.post("/explain", json=request_data)

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["pd"] == 0.95
        assert len(data["attributions"]) == 2


class TestHealthEndpoint:
    def test_health(self) -> None:
        mock_registry = _mock_registry(version="1.0.0")
        mock_registry.is_loaded.return_value = True

        _override_registry(mock_registry)
        response = client.get("/health")

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["model_loaded"] is True
        assert data["model_version"] == "1.0.0"

    def test_health_model_not_loaded_lazy_loads(self) -> None:
        mock_registry = _mock_registry(version="1.0.0")
        mock_registry.is_loaded.return_value = False

        _override_registry(mock_registry)
        response = client.get("/health")

        assert response.status_code == 200, response.text
        mock_registry.load.assert_called_once()

    def test_health_load_failure(self) -> None:
        mock_registry = _mock_registry(version="unknown")
        mock_registry.is_loaded.return_value = False
        mock_registry.load.side_effect = Exception("Load failed")

        _override_registry(mock_registry)
        response = client.get("/health")

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    def test_predict_success(self) -> None:
        mock_registry = _mock_registry(pd_val=0.30, version="1.0.0")
        _override_registry(mock_registry)

        response = client.post("/predict", json=loan_payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["pd"] == 0.30
        assert data["model_version"] == "1.0.0"

    def test_predict_response_structure(self) -> None:
        mock_registry = _mock_registry(pd_val=0.45, version="1.0.0")
        _override_registry(mock_registry)

        response = client.post("/predict", json=loan_payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert isinstance(data["pd"], (int, float))
        assert 0 <= data["pd"] <= 1
        assert isinstance(data["model_version"], str)

    def test_predict_missing_required_field(self) -> None:
        incomplete_payload = {k: v for k, v in loan_payload.items() if k != "loan_amnt"}
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code in (422, 400)

    def test_predict_invalid_field_type(self) -> None:
        bad_payload = loan_payload.copy()
        bad_payload["loan_amnt"] = "not_a_number"
        response = client.post("/predict", json=bad_payload)
        assert response.status_code in (422, 400)

    def test_predict_with_application_wrapper(self) -> None:
        mock_registry = _mock_registry(pd_val=0.25, version="1.0.0")
        _override_registry(mock_registry)

        wrapped_payload = {"application": loan_payload}
        response = client.post("/predict", json=wrapped_payload)

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["pd"] == 0.25


class TestSchemaValidation:
    def test_schema_accepts_valid_payload(self) -> None:
        app_data = LoanApplication(**loan_payload)
        assert app_data.loan_amnt == 3600
        assert app_data.annual_inc == 100000

    def test_schema_rejects_missing_required(self) -> None:
        incomplete = {k: v for k, v in loan_payload.items() if k != "loan_amnt"}
        with pytest.raises(ValidationError):
            LoanApplication(**incomplete)

    def test_schema_validates_field_types(self) -> None:
        bad_payload = loan_payload.copy()
        bad_payload["loan_amnt"] = "not_a_number"
        with pytest.raises(ValidationError):
            LoanApplication(**bad_payload)


class TestFeatureAttributionSchema:
    def test_feature_attribution_creation(self) -> None:
        attr = FeatureAttribution(feature="loan_amnt", shap_value=0.15, value=3600)
        assert attr.feature == "loan_amnt"
        assert attr.shap_value == 0.15

    def test_feature_attribution_from_dict(self) -> None:
        attr_dict = {"feature": "grade_C", "shap_value": 0.08, "value": 1}
        attr = FeatureAttribution(**attr_dict)
        assert attr.feature == "grade_C"
        assert attr.shap_value == 0.08


class TestErrorHandling:
    def test_predict_with_registry_error(self) -> None:
        mock_registry = _mock_registry(version="1.0.0")
        mock_registry.predict_from_payload.side_effect = RuntimeError("Registry error")

        _override_registry(mock_registry)
        response = client.post("/predict", json=loan_payload)
        assert response.status_code >= 400

    # def test_explain_with_shap_error(self) -> None:
    #     """Test graceful handling of SHAP computation errors."""
    #     mock_registry = _mock_registry(pd_val=0.35, version="1.0.0")
    #     _override_registry(mock_registry)

    #     with patch(
    #         "src.serving.api.compute_shap_for_single_row", side_effect=RuntimeError("SHAP error")
    #     ):
    #         request_data = {
    #             "application": loan_payload,
    #             "top_k": 2,
    #             "include_base_value": True,
    #         }
    #         response = client.post("/explain", json=request_data)

    #     assert response.status_code == 200, response.text
    #     body = response.json()
    #     assert body["pd"] == 0.35
    #     assert body["model_version"] == "1.0.0"
    #     assert body["base_value"] is None
    #     assert body["attributions"] == []
    #     assert "meta" in body
    #     assert body["meta"]["top_k"] == 2
    #     assert "explain_error" in body["meta"]

    def test_invalid_json_body(self) -> None:
        response = client.post(
            "/predict",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code >= 400
