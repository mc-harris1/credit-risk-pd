from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from src.serving.api import app

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


def test_health() -> None:
    # Mock the pipeline loader to avoid needing actual model
    mock_pipeline = MagicMock()
    with patch("src.serving.api._load_pipeline", return_value=mock_pipeline):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict() -> None:
    # Mock the pipeline to avoid needing the actual trained model file
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.70, 0.30]])  # 30% default probability

    # Mock explain_instance to avoid SHAP computations
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
