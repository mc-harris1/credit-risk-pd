def test_models():
    from src.models.train import train_model

    # Simply test that the train_model function runs without errors
    train_model()

    # Check that the model file was created
    import os

    from src.config import MODELS_DIR

    model_path = os.path.join(MODELS_DIR, "pd_model_xgb.pkl")
    assert os.path.exists(model_path), "Model file was not created"
