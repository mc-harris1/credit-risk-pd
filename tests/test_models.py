import os

import numpy as np
import pandas as pd
import pytest
from src.config import METADATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR


@pytest.fixture
def sample_feature_data():
    """Create sample feature data for testing."""
    np.random.seed(42)
    n_samples = 1000

    # Create sample data with various feature types
    data = {
        "loan_amnt": np.random.uniform(1000, 40000, n_samples),
        "int_rate": np.random.uniform(5, 25, n_samples),
        "annual_inc": np.random.uniform(20000, 200000, n_samples),
        "dti": np.random.uniform(0, 40, n_samples),
        "fico_score": np.random.uniform(600, 850, n_samples),
        "term": np.random.choice(["36 months", "60 months"], n_samples),
        "grade": np.random.choice(["A", "B", "C", "D", "E"], n_samples),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n_samples),
        "verification_status": np.random.choice(["Verified", "Not Verified"], n_samples),
        "purpose": np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement"], n_samples
        ),
        "loan_status": np.random.choice(["Fully Paid", "Charged Off", "Current"], n_samples),
        "default": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }

    df = pd.DataFrame(data)

    # Ensure the processed data directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Save to the expected location
    output_path = os.path.join(PROCESSED_DATA_DIR, "loans_features.parquet")
    df.to_parquet(output_path, index=False)

    yield df

    # Cleanup after test
    if os.path.exists(output_path):
        os.remove(output_path)


def test_train_model(sample_feature_data):
    """Test that train_model runs successfully and creates model files."""
    from src.models.train import train_model

    # Run the training
    train_model()

    # Check that at least one model file was created
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    assert len(model_files) > 0, "No model file was created"

    # Cleanup: remove all generated model artifacts and metadata
    for file in os.listdir(MODELS_DIR):
        file_path = os.path.join(MODELS_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    if os.path.exists(METADATA_DIR):
        for file in os.listdir(METADATA_DIR):
            file_path = os.path.join(METADATA_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
