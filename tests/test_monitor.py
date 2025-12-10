# tests/test_monitor.py
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from src.models import monitor as monitor_module
from src.models.monitor import run_drift_checks


def _make_synthetic_drift_data():
    """
    Create a synthetic dataset where 'dti' drifts between train and recent.

    - X: loan_amnt, annual_inc, dti
    - y: default indicator
    - dates: increasing over time
    """
    n = 200

    # Baseline (older loans): lower DTI, lower default
    n_train = int(n * 0.8)
    n_recent = n - n_train

    dti_train = np.random.normal(loc=10.0, scale=2.0, size=n_train)  # low DTI
    dti_recent = np.random.normal(loc=25.0, scale=2.0, size=n_recent)  # high DTI

    dti = np.concatenate([dti_train, dti_recent])

    loan_amnt = np.linspace(5000, 15000, n)
    annual_inc = np.linspace(40000, 90000, n)

    # Simple default pattern: higher DTI → more defaults
    y_train = (dti_train > 12).astype(int)
    y_recent = (dti_recent > 20).astype(int)
    y = np.concatenate([y_train, y_recent])

    dates = pd.date_range("2015-01-01", periods=n, freq="ME")

    df = pd.DataFrame(
        {
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "dti": dti,
            "issue_d": dates,
            "default": y,
        }
    )

    # X = all feature columns including date (monitor drops date internally)
    X = df.drop(columns=["default"])
    y_series = df["default"]
    dates_series = df["issue_d"]

    return X, y_series, dates_series


def test_run_drift_checks_with_synthetic_drift(tmp_path, monkeypatch):
    """
    Run drift checks on synthetic data where 'dti' clearly shifts upward
    in the recent period, and verify PSI + labels behave as expected.
    """

    # --- Arrange: synthetic data + monkeypatch train/monitor helpers ---

    X, y, dates = _make_synthetic_drift_data()

    def fake_load_feature_data():
        # match signature of train.load_feature_data()
        return X.copy(), y.copy(), dates.copy()

    def fake_time_based_split(X_in, y_in, dates_in, train_fraction=0.8):
        # simple time-based split by index
        n = len(dates_in)
        split_idx = int(n * train_fraction)
        train_idx = dates_in.index[:split_idx]
        recent_idx = dates_in.index[split_idx:]

        X_train = X_in.loc[train_idx].copy()
        X_recent = X_in.loc[recent_idx].copy()
        y_train = y_in.loc[train_idx].copy()
        y_recent = y_in.loc[recent_idx].copy()

        return X_train, X_recent, y_train, y_recent

    # Redirect METADATA_DIR to a temp directory
    metadata_dir = tmp_path / "metadata"
    monkeypatch.setattr(monitor_module, "METADATA_DIR", metadata_dir)

    # Patch load_feature_data and time_based_split used inside monitor
    monkeypatch.setattr(monitor_module, "load_feature_data", fake_load_feature_data)
    monkeypatch.setattr(monitor_module, "time_based_split", fake_time_based_split)

    # --- Act ---
    run_drift_checks()

    # --- Assert: JSON report exists and has expected structure ---
    drift_json = metadata_dir / "drift_report.json"
    assert drift_json.exists(), "Expected drift_report.json to be created."

    with drift_json.open(encoding="utf-8") as f:
        report = json.load(f)

    # Default-rate drift
    assert "default_rate" in report
    dr = report["default_rate"]
    assert "train" in dr and "recent" in dr and "delta" in dr
    # Sanity: recent default rate should be >= train default rate in this synthetic setup
    assert dr["recent"] >= dr["train"]

    # PSI section
    assert "psi" in report
    psi = report["psi"]
    assert "dti" in psi, "Expected PSI score for 'dti'."
    dti_psi = psi["dti"]
    assert isinstance(dti_psi, float)

    # PSI labels
    assert "psi_labels" in report
    psi_labels = report["psi_labels"]
    assert "dti" in psi_labels
    # We expect at least moderate_shift or significant_shift given the synthetic drift
    assert psi_labels["dti"] in {"moderate_shift", "significant_shift"}

    # Numeric summary should exist for both periods
    assert "numeric_summary" in report
    ns = report["numeric_summary"]
    assert "train" in ns and "recent" in ns
    assert "dti" in ns["train"] and "dti" in ns["recent"]
