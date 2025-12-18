# tests/test_guardrails.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

from __future__ import annotations

from pathlib import Path

import pytest
from src.config import ARTIFACTS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

# NOTE:
# These tests validate *guardrails* against your local repo artifacts.
# In CI environments where large data artifacts aren't present, we skip cleanly.
FEATURE_SPEC = Path(ARTIFACTS_DIR) / "feature_spec_v1.json"
ENGINEERED = Path(PROCESSED_DATA_DIR) / "engineered_features_v1.parquet"
INTERIM = Path(INTERIM_DATA_DIR) / "loans_cleaned.parquet"


def _require_files(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"Guardrail artifacts not present; skipping: {[str(p) for p in missing]}")


def test_guardrails_fast_validate() -> None:
    """
    Fast end-to-end guardrail validation (no model fitting, no permutation importance).

    Asserts:
      - feature spec hash matches freeze (if present)
      - engineered matrix contains target and contracted features
      - no leakage prefixes like loan_status_* in engineered matrix
      - anchor reconstruction + temporal split checks pass
    """
    _require_files(FEATURE_SPEC, ENGINEERED)

    from src.validate_pipeline import ValidateConfig, validate

    cfg = ValidateConfig(
        artifacts_dir=Path(ARTIFACTS_DIR),
        processed_dir=Path(PROCESSED_DATA_DIR),
        interim_dir=Path(INTERIM_DATA_DIR),
        feature_spec_file="feature_spec_v1.json",
        engineered_file="engineered_features_v1.parquet",
        interim_file=None,  # skip interim leakage firewall here (separate test below)
        forbidden_cols=(),  # not used since interim_file=None
        run_canary=False,
        run_perm_importance=False,
    )

    result = validate(cfg)

    assert result["target"] == "default"
    assert result["n_features_contracted"] > 0
    assert result["n_rows"] > 0


def test_guardrails_interim_leakage_firewall() -> None:
    """
    Ensures the interim dataset does not contain forbidden post-outcome/leakage columns.
    """
    _require_files(FEATURE_SPEC, ENGINEERED, INTERIM)

    from src.validate_pipeline import DEFAULT_FORBIDDEN_COLS, ValidateConfig, validate

    cfg = ValidateConfig(
        artifacts_dir=Path(ARTIFACTS_DIR),
        processed_dir=Path(PROCESSED_DATA_DIR),
        interim_dir=Path(INTERIM_DATA_DIR),
        feature_spec_file="feature_spec_v1.json",
        engineered_file="engineered_features_v1.parquet",
        interim_file="loans_cleaned.parquet",
        forbidden_cols=sorted(DEFAULT_FORBIDDEN_COLS),
        run_canary=False,
        run_perm_importance=False,
    )

    result = validate(cfg)
    assert result["n_rows"] > 0


@pytest.mark.slow
def test_guardrails_canary_baseline() -> None:
    """
    Canary baseline LR check to catch 'impossibly high' AUC (leakage) or huge overfit gaps.
    Marked slow because it fits a model.

    Run locally with:
      pytest -m slow
    """
    _require_files(FEATURE_SPEC, ENGINEERED)

    from src.validate_pipeline import ValidateConfig, validate

    cfg = ValidateConfig(
        artifacts_dir=Path(ARTIFACTS_DIR),
        processed_dir=Path(PROCESSED_DATA_DIR),
        interim_dir=Path(INTERIM_DATA_DIR),
        feature_spec_file="feature_spec_v1.json",
        engineered_file="engineered_features_v1.parquet",
        interim_file=None,
        forbidden_cols=(),
        run_canary=True,
        run_perm_importance=False,
    )

    result = validate(cfg)
    assert result["canary"] is not None
    assert 0.5 < result["canary"]["val_auc"] < 0.95
