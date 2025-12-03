import os
import sys

from fastapi.testclient import TestClient

# Add the parent directory of 'src' to sys.path
# Assuming 'tests' and 'src' are siblings in your project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serving.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
