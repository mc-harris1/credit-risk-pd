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
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score() -> None:
    # Mock the model to avoid needing the actual trained model file
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])  # 15% default probability

    with patch("src.serving.api.load_model", return_value=mock_model):
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prob_default" in data
        assert "risk_band" in data
        assert data["risk_band"] in ["Low", "Medium", "High"]
