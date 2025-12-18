# tests/test_models.py

import json

import numpy as np
import pandas as pd
from src.models.evaluate import EvalConfig, evaluate
from src.models.explain import ExplainConfig, run_shap_global
from src.models.train import TrainConfig, train


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


def test_train_model(tmp_path):
    """Train model end-to-end on synthetic data using the refactored pipeline."""

    processed_dir = tmp_path / "processed"
    features_path = processed_dir / "synthetic_features.parquet"
    out_dir = tmp_path / "train_run"

    _create_synthetic_features(processed_dir, filename=features_path.name)

    cfg = TrainConfig(features_path=features_path, out_dir=out_dir, log_level="INFO")

    train(cfg)

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"

    assert model_path.exists(), "Expected trained model artifact to be created."
    assert metrics_path.exists(), "Expected metrics.json to be created."

    with metrics_path.open(encoding="utf-8") as f:
        metrics_payload = json.load(f)

    assert metrics_payload["model"]["type"] == "LogisticRegression"
    assert metrics_payload["data"]["n_train"] > 0
    assert metrics_payload["data"]["n_val"] > 0
    assert "train" in metrics_payload["metrics"] and "val" in metrics_payload["metrics"]


def test_evaluate(tmp_path):
    """Evaluate a trained model and check evaluation artifacts."""

    processed_dir = tmp_path / "processed"
    features_path = processed_dir / "synthetic_features.parquet"
    train_out = tmp_path / "train_run"
    eval_out = tmp_path / "eval_run"

    _create_synthetic_features(processed_dir, filename=features_path.name)

    train_cfg = TrainConfig(features_path=features_path, out_dir=train_out, log_level="INFO")
    train(train_cfg)

    eval_cfg = EvalConfig(
        features_path=features_path, model_path=train_out / "model.joblib", out_dir=eval_out
    )
    metrics = evaluate(eval_cfg)

    metrics_path = eval_out / "metrics.json"
    assert metrics_path.exists(), "Expected metrics.json to be created."

    for key in ("roc_auc", "pr_auc", "brier", "confusion_matrix", "threshold"):
        assert key in metrics, f"Missing key '{key}' in evaluation metrics."

    cm = metrics["confusion_matrix"]
    for key in ("tn", "fp", "fn", "tp"):
        assert key in cm, f"Missing key '{key}' in confusion_matrix."

    for artifact in ("roc_curve.png", "pr_curve.png", "calibration_curve.png"):
        assert (eval_out / artifact).exists(), f"Expected {artifact} to be created."


def test_tune(tmp_path):
    """Temporarily skip until tuning utilities are ported to the refactored API."""
    import pytest

    pytest.skip("Hyperparameter tuning helpers not yet wired to refactored train pipeline.")


def test_explain_shap_global(tmp_path):
    """Run SHAP global explanation on synthetic data using the refactored pipeline."""

    processed_dir = tmp_path / "processed"
    features_path = processed_dir / "synthetic_features.parquet"
    train_out = tmp_path / "train_run"
    explain_out = tmp_path / "explain_run"

    _create_synthetic_features(processed_dir, filename=features_path.name)

    train_cfg = TrainConfig(features_path=features_path, out_dir=train_out, log_level="INFO")
    train(train_cfg)

    explain_cfg = ExplainConfig(
        features_path=features_path,
        model_path=train_out / "model.joblib",
        out_dir=explain_out,
        num_background=10,
        num_samples=10,
    )

    run_shap_global(explain_cfg)

    importance_path = explain_out / "shap_global_importance.json"
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
