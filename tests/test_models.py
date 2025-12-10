# tests/test_models.py

import json
from datetime import datetime

import numpy as np
import pandas as pd
from src.models import evaluate as evaluate_module
from src.models import train as train_module
from src.models import tune as tune_module
from src.models.evaluate import evaluate_model
from src.models.train import train_model
from src.models.tune import tune_model


def _create_synthetic_features(path, filename="loans_features.parquet"):
    """Create a minimal synthetic loans_features.parquet for training/eval tests."""
    # 20 synthetic loans over 20 months
    n = 20
    dates = pd.date_range("2015-01-01", periods=n, freq="ME")

    df = pd.DataFrame(
        {
            "loan_amnt": np.linspace(5000, 15000, n),
            "annual_inc": np.linspace(40000, 90000, n),
            "dti": np.linspace(5, 30, n),
            "term": ["36 months"] * n,
            "home_ownership": ["RENT", "MORTGAGE"] * (n // 2),
            "grade": ["B", "C", "D", "E"] * 5,
            "sub_grade": ["B1", "B2", "C1", "C2", "D1"] * 4,
            "loan_status": ["Fully Paid", "Charged Off"] * (n // 2),
            # simple alternating default pattern
            "default": ([0, 1] * (n // 2)),
            "issue_d": dates,
            # Engineered features
            "term_months": [36] * n,
            "loan_to_income": np.linspace(5000, 15000, n) / np.linspace(40000, 90000, n),
            "grade_numeric": [2, 3, 4, 5] * 5,  # B=2, C=3, D=4, E=5
            "sub_grade_numeric": [2.1, 2.2, 3.1, 3.2, 4.1] * 4,
        }
    )

    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_bytes(b"")  # ensure file gets created
    df.to_parquet(path / filename, index=False)


def test_train_model(tmp_path, monkeypatch):
    """
    Train model on synthetic loans_features.parquet and verify that:
    - model artifact is created
    - metadata JSON is created with expected structure
    """
    # Arrange: point train module to temp dirs
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    metadata_dir = tmp_path / "metadata"

    monkeypatch.setattr(train_module, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(train_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(train_module, "METADATA_DIR", metadata_dir)

    # Create synthetic feature data
    _create_synthetic_features(processed_dir, filename=train_module.FEATURES_FILE)

    # Act
    train_model()

    # Assert: model artifact
    model_path = models_dir / train_module.MODEL_FILE
    assert model_path.exists(), "Expected trained model artifact to be created."

    # Assert: metadata JSON
    metadata_path = metadata_dir / "pd_model_metadata.json"
    assert metadata_path.exists(), "Expected pd_model_metadata.json to be created."

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    # Basic schema checks
    assert metadata["model_file"] == train_module.MODEL_FILE
    assert metadata["model_type"] == "XGBClassifier"
    assert metadata["train_samples"] > 0
    assert metadata["test_samples"] > 0
    assert isinstance(metadata["feature_names"], list)
    assert len(metadata["feature_names"]) > 0

    split_info = metadata["training_split"]
    assert split_info["type"] == "time_based_fraction"
    assert split_info["fraction_train"] == 0.8
    assert split_info["date_column"] == "issue_d"
    # sanity check parsed dates
    datetime.fromisoformat(split_info["train_start"])
    datetime.fromisoformat(split_info["train_end"])
    datetime.fromisoformat(split_info["test_start"])
    datetime.fromisoformat(split_info["test_end"])


def test_evaluate(tmp_path, monkeypatch):
    """
    Evaluate the trained model on a synthetic time-based split and verify that:
    - evaluation_metrics.json is created with expected keys
    - ROC, PR, and calibration plots are written
    """
    # Arrange: point BOTH train and evaluate modules to the same temp dirs
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    metadata_dir = tmp_path / "metadata"

    # Train module dirs
    monkeypatch.setattr(train_module, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(train_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(train_module, "METADATA_DIR", metadata_dir)

    # Evaluate module dirs
    monkeypatch.setattr(evaluate_module, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(evaluate_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(evaluate_module, "METADATA_DIR", metadata_dir)

    # Synthetic features (same helper)
    _create_synthetic_features(processed_dir, filename=train_module.FEATURES_FILE)

    # First train a model into the temp dirs
    train_model()

    # Act: run evaluation
    evaluate_model()

    # Assert: metrics JSON
    metrics_path = metadata_dir / "evaluation_metrics.json"
    assert metrics_path.exists(), "Expected evaluation_metrics.json to be created."

    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    for key in ("roc_auc", "pr_auc", "brier_score", "confusion_matrix", "threshold"):
        assert key in metrics, f"Missing key '{key}' in evaluation metrics."

    cm = metrics["confusion_matrix"]
    for key in ("tn", "fp", "fn", "tp"):
        assert key in cm, f"Missing key '{key}' in confusion_matrix."

    # Assert: evaluation plots
    roc_path = metadata_dir / "roc_curve.png"
    pr_path = metadata_dir / "pr_curve.png"
    calib_path = metadata_dir / "calibration_curve.png"

    assert roc_path.exists(), "Expected ROC curve plot to be created."
    assert pr_path.exists(), "Expected PR curve plot to be created."
    assert calib_path.exists(), "Expected calibration curve plot to be created."


def test_tune(tmp_path, monkeypatch):
    """
    Run hyperparameter tuning on synthetic loans_features.parquet and verify that:
    - hparam_search_results.json is created with expected structure
    - best_params.json is created with best params and ROC-AUC score
    """
    # Arrange: point train and tune modules to temp dirs
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"

    # tune() uses load_feature_data from train_module, so patch train_module
    monkeypatch.setattr(train_module, "PROCESSED_DATA_DIR", processed_dir)
    # tune() uses METADATA_DIR from config, patch both tune and config
    monkeypatch.setattr(tune_module, "METADATA_DIR", metadata_dir)

    # Create synthetic feature data
    _create_synthetic_features(processed_dir, filename="loans_features.parquet")

    # Act
    tune_model()

    # Assert: hparam_search_results.json
    hparam_results_path = metadata_dir / "hparam_search_results.json"
    assert hparam_results_path.exists(), "Expected hparam_search_results.json to be created."

    with hparam_results_path.open(encoding="utf-8") as f:
        results = json.load(f)

    assert isinstance(results, list), "Expected hparam_search_results to be a list."
    assert len(results) > 0, "Expected at least one hyperparameter combination to be evaluated."

    # Check structure of first result
    first_result = results[0]
    assert "params" in first_result, "Expected 'params' key in result."
    assert "roc_auc" in first_result, "Expected 'roc_auc' key in result."

    params = first_result["params"]
    for key in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        assert key in params, f"Expected '{key}' in params."

    # Assert: best_params.json
    best_params_path = metadata_dir / "best_params.json"
    assert best_params_path.exists(), "Expected best_params.json to be created."

    with best_params_path.open(encoding="utf-8") as f:
        best_results = json.load(f)

    assert "best_params" in best_results, "Expected 'best_params' key in best_results."
    assert "best_roc_auc" in best_results, "Expected 'best_roc_auc' key in best_results."

    best_params = best_results["best_params"]
    for key in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        assert key in best_params, f"Expected '{key}' in best_params."

    assert isinstance(best_results["best_roc_auc"], (int, float)), (
        "Expected best_roc_auc to be numeric."
    )


def test_explain_shap_global(tmp_path, monkeypatch):
    """
    Run SHAP global explanation on synthetic loans_features.parquet and verify that:
    - shap_global_importance.json is created with expected structure
    """
    # Arrange: point train and explain modules to temp dirs
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    metadata_dir = tmp_path / "metadata"

    # Patch train module (used by both train and explain)
    monkeypatch.setattr(train_module, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(train_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(train_module, "METADATA_DIR", metadata_dir)

    # Also patch explain module's imported constants
    from src.models import explain as explain_module

    monkeypatch.setattr(explain_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(explain_module, "METADATA_DIR", metadata_dir)

    # Create synthetic feature data
    _create_synthetic_features(processed_dir, filename=train_module.FEATURES_FILE)

    # First train a model into the temp dirs
    train_module.train_model()

    # Act: run SHAP global explanation
    from src.models.explain import run_shap_global

    run_shap_global(num_background=10, num_samples=10)

    # Assert: SHAP global importance JSON
    importance_path = metadata_dir / "shap_global_importance.json"
    assert importance_path.exists(), "Expected shap_global_importance.json to be created."

    with importance_path.open(encoding="utf-8") as f:
        data = json.load(f)

    assert "importance" in data, "Expected 'importance' key in JSON."
    importance = data["importance"]
    assert isinstance(importance, list), "Expected SHAP global importance to be a list."
    assert len(importance) > 0, "Expected at least one feature importance entry."

    first_entry = importance[0]
    assert "feature" in first_entry, "Expected 'feature' key in importance entry."
    assert "mean_abs_shap" in first_entry, "Expected 'mean_abs_shap' key in importance entry."
