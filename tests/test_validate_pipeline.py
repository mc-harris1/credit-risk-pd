# tests/test_validate_pipeline.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.validate_pipeline import (
    DEFAULT_FORBIDDEN_COLS,
    MAX_BASELINE_AUC,
    MAX_SINGLE_FEAT_DOMINANCE,
    ValidateConfig,
    ValidationError,
    assert_no_temporal_overlap,
    derive_contracted_feature_cols,
    make_anchor_timestamp,
    md5_file,
    run_canary_lr,
    run_permutation_importance_checks,
    time_based_split_indices,
    validate,
)

# ----------------------------
# Parquet helper (robust if pyarrow/fastparquet not installed)
# ----------------------------


def _write_df_as_parquet(path: Path, df: pd.DataFrame) -> None:
    """
    Writes a DataFrame to a .parquet path.
    If parquet engines aren't available, fall back to pickle + patch read_parquet.
    """
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # fallback: pickle
        pkl = path.with_suffix(".pkl")
        df.to_pickle(pkl)

        # write a tiny marker "parquet" file so validate() sees it exists
        path.write_bytes(b"PARQUET_FALLBACK_TO_PICKLE")


@pytest.fixture()
def patch_read_parquet_if_needed(monkeypatch):
    """
    If parquet engines aren't available, pandas.read_parquet will fail.
    Patch it to load our fallback .pkl next to the requested .parquet.
    """
    orig = pd.read_parquet

    def _patched(path, *args, **kwargs):
        p = Path(path)
        pkl = p.with_suffix(".pkl")
        if p.exists() and p.read_bytes() == b"PARQUET_FALLBACK_TO_PICKLE" and pkl.exists():
            return pd.read_pickle(pkl)
        return orig(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _patched)
    return _patched


# ----------------------------
# Spec + data builders
# ----------------------------


def _base_spec(target_name: str = "target") -> dict:
    """
    Minimal spec with all feature families represented so derive_contracted_feature_cols
    exercises datetime/engineered/categorical/numerical branches.
    """
    return {
        "freeze": {},  # filled by tests as needed
        "target": {"name": target_name, "allowed_values": [0, 1]},
        "features": {
            "datetime": {"issue_d": {"derived_features": ["issue_d_year", "issue_d_month"]}},
            "engineered": {
                "eng_feat_1": {},
                "eng_feat_2": {},
            },
            "categorical": {
                "grade": {"encoding_strategy": "one_hot"},
                "state": {"encoding_strategy": "target_mean"},
                "purpose": {"encoding_strategy": "freq"},
            },
            "numerical": {
                "loan_amnt": {"planned_transformation": "log"},
                "int_rate": {},
            },
        },
    }


def _engineered_df(n: int = 120, seed: int = 0, target_name: str = "target") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # time columns for anchor
    years = np.repeat([2018, 2019, 2020, 2021], repeats=n // 4)
    if len(years) < n:
        years = np.concatenate([years, np.repeat(2021, n - len(years))])
    months = (np.arange(n) % 12) + 1

    # numerical + transformed
    loan_amnt = rng.lognormal(mean=10, sigma=0.3, size=n)
    int_rate = rng.normal(0.12, 0.03, size=n)

    # categorical encodings (already engineered)
    grade_A = (rng.random(n) > 0.5).astype(int)
    grade_B = 1 - grade_A
    state_tm = rng.normal(0.0, 1.0, size=n)
    purpose_freq = rng.uniform(0.0, 1.0, size=n)

    # engineered
    eng1 = rng.normal(0, 1, size=n)
    eng2 = rng.normal(0, 1, size=n)

    # target: somewhat learnable but not "impossibly perfect"
    # (small signal so canary usually passes)
    logit = 0.4 * (int_rate - int_rate.mean()) + 0.1 * (eng1)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "issue_d_year": years.astype(int),
            "issue_d_month": months.astype(int),
            "eng_feat_1": eng1,
            "eng_feat_2": eng2,
            "grade_A": grade_A,
            "grade_B": grade_B,
            "state__target_mean": state_tm,
            "purpose__freq": purpose_freq,
            "loan_amnt": loan_amnt,
            "loan_amnt__log1p": np.log1p(loan_amnt),
            "int_rate": int_rate,
            target_name: y,
        }
    )
    return df


# ----------------------------
# Unit tests: contract derivation + temporal
# ----------------------------


def test_derive_contracted_feature_cols_includes_expected_columns():
    spec = _base_spec()
    df = _engineered_df()

    cols = derive_contracted_feature_cols(df, spec)

    # datetime derived
    assert "issue_d_year" in cols
    assert "issue_d_month" in cols

    # engineered
    assert "eng_feat_1" in cols
    assert "eng_feat_2" in cols

    # categorical: one-hot discovered from df prefix
    assert "grade_A" in cols
    assert "grade_B" in cols

    # categorical: scalar encodings
    assert "state__target_mean" in cols
    assert "purpose__freq" in cols

    # numerical + transformed
    assert "loan_amnt" in cols
    assert "loan_amnt__log1p" in cols
    assert "int_rate" in cols


def test_make_anchor_timestamp_prefers_year_month():
    df = _engineered_df(n=10)
    anchor = make_anchor_timestamp(df)
    assert pd.api.types.is_datetime64_any_dtype(anchor)
    # should be first-of-month
    assert set(anchor.dt.day.unique()) == {1}  # type: ignore[attr-defined]


def test_make_anchor_timestamp_fallback_year_quarter():
    df = pd.DataFrame(
        {
            "issue_d_year": [2020, 2020, 2021],
            "issue_d_quarter": [1, 2, 4],
        }
    )
    anchor = make_anchor_timestamp(df)
    assert anchor.iloc[0].month == 1
    assert anchor.iloc[1].month == 4
    assert anchor.iloc[2].month == 10


def test_make_anchor_timestamp_raises_if_missing():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValidationError, match="Cannot reconstruct anchor time"):
        make_anchor_timestamp(df)


def test_time_based_split_indices_sums_to_n():
    anchor = pd.to_datetime(
        pd.DataFrame({"year": [2020] * 20, "month": list(range(1, 13)) + [1] * 8, "day": 1})
    )
    tr, va, te = time_based_split_indices(anchor, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    assert len(tr) + len(va) + len(te) == len(anchor)
    assert len(set(tr).intersection(set(va))) == 0
    assert len(set(tr).intersection(set(te))) == 0
    assert len(set(va).intersection(set(te))) == 0


def test_time_based_split_indices_requires_sum_to_one():
    anchor = pd.to_datetime(pd.DataFrame({"year": [2020, 2020], "month": [1, 2], "day": 1}))
    with pytest.raises(ValidationError, match="fractions must sum to 1.0"):
        time_based_split_indices(anchor, train_frac=0.7, val_frac=0.2, test_frac=0.2)


def test_assert_no_temporal_overlap_passes_for_sorted_time():
    df = _engineered_df(n=80)
    anchor = make_anchor_timestamp(df)
    tr, va, te = time_based_split_indices(anchor)
    # should not raise on correctly sorted split-by-time
    assert_no_temporal_overlap(anchor, tr, va, te)


def test_assert_no_temporal_overlap_raises_when_overlap_constructed():
    # Create an anchor where "val" contains earlier items than "train"
    anchor = pd.to_datetime(
        pd.DataFrame(
            {
                "year": [2020] * 10,
                "month": [5, 6, 7, 8, 9, 1, 2, 3, 4, 10],  # earlier months appear later
                "day": 1,
            }
        )
    )
    # fabricate indices that cause overlap by time
    idx_train = np.array([0, 1, 2, 3, 4])  # months 5..9
    idx_val = np.array([5, 6])  # months 1..2 (earlier) => overlap
    idx_test = np.array([7, 8, 9])
    with pytest.raises(ValidationError, match="Temporal leakage: train overlaps validation"):
        assert_no_temporal_overlap(anchor, idx_train, idx_val, idx_test)


# ----------------------------
# Unit tests: canary + permutation importance
# ----------------------------


def test_run_canary_lr_passes_on_reasonable_signal():
    df = _engineered_df(n=200, seed=2)
    spec = _base_spec()
    contracted = derive_contracted_feature_cols(df, spec)
    X = df[contracted]
    y = df["target"].astype(int)
    anchor = make_anchor_timestamp(df)

    # Synthetic data can be noisy and sometimes produces a larger train/val gap.
    # This test verifies the routine runs end-to-end and returns expected keys,
    # not that the default guardrails always pass for random synthetic data.
    metrics = run_canary_lr(X, y, anchor, max_auc=0.99, max_gap=0.35)

    assert set(metrics.keys()) == {"train_auc", "val_auc", "auc_gap"}
    assert 0.0 <= metrics["train_auc"] <= 1.0
    assert 0.0 <= metrics["val_auc"] <= 1.0


def test_run_canary_lr_fails_if_val_auc_too_high():
    # Create blatant leakage feature => AUC ~ 1
    n = 200
    rng = np.random.default_rng(0)
    y = (rng.random(n) > 0.5).astype(int)
    df = pd.DataFrame(
        {
            "issue_d_year": np.repeat([2020, 2021, 2022, 2023], n // 4),
            "issue_d_month": (np.arange(n) % 12) + 1,
            # leakage: perfectly equals target
            "leak": y.astype(float),
            "noise": rng.normal(0, 1, n),
            "target": y,
        }
    )
    X = df[["leak", "noise"]]
    anchor = make_anchor_timestamp(df)

    with pytest.raises(ValidationError, match="val ROC AUC too high"):
        run_canary_lr(X, df["target"], anchor, max_auc=MAX_BASELINE_AUC)


def test_run_permutation_importance_checks_fails_on_single_feature_dominance():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(0, 1, n)
    y = (x1 > 0).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": rng.normal(0, 1, n)})

    model = LogisticRegression(max_iter=200).fit(X, y)

    with pytest.raises(ValidationError, match="Permutation importance dominance"):
        run_permutation_importance_checks(
            model,
            X,
            pd.Series(y),
            max_single=MAX_SINGLE_FEAT_DOMINANCE,
            max_time_share=999.0,  # don't trip the time-share check here
        )


def test_run_permutation_importance_checks_fails_on_time_feature_share():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    n = 300
    # Make time feature strongly predictive
    issue_d_year = rng.integers(2018, 2023, size=n)
    y = (issue_d_year >= 2021).astype(int)

    X = pd.DataFrame(
        {
            "issue_d_year": issue_d_year.astype(float),
            "issue_d_month": rng.integers(1, 13, size=n).astype(float),
            "other": rng.normal(0, 1, size=n),
        }
    )
    model = LogisticRegression(max_iter=250).fit(X, y)

    with pytest.raises(ValidationError, match="Time-feature dominance"):
        run_permutation_importance_checks(
            model,
            X,
            pd.Series(y),
            max_single=999.0,  # avoid single-feature dominance error
            max_time_share=0.01,  # force fail
        )


# ----------------------------
# Integration-ish tests: validate()
# ----------------------------


def test_validate_happy_path(tmp_path: Path, patch_read_parquet_if_needed, capsys):
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"
    artifacts.mkdir()
    processed.mkdir()
    interim.mkdir()

    spec = _base_spec(target_name="target")
    spec_path = artifacts / "feature_spec_v1.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # freeze hash matches
    spec["freeze"]["spec_hash_md5"] = md5_file(spec_path)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    df_eng = _engineered_df(n=200, seed=3, target_name="target")
    feat_path = processed / "engineered_features_v1.parquet"
    _write_df_as_parquet(feat_path, df_eng)

    df_interim = pd.DataFrame({"ok_col": [1, 2, 3]})
    interim_path = interim / "loans_cleaned.parquet"
    _write_df_as_parquet(interim_path, df_interim)

    cfg = ValidateConfig(
        artifacts_dir=artifacts,
        processed_dir=processed,
        interim_dir=interim,
        feature_spec_file=spec_path.name,
        engineered_file=feat_path.name,
        interim_file=interim_path.name,
        forbidden_cols=sorted(DEFAULT_FORBIDDEN_COLS),
        run_canary=False,
        run_perm_importance=False,
    )

    out = validate(cfg)
    assert out["target"] == "target"
    assert out["n_rows"] == 200
    assert out["n_features_contracted"] > 0
    captured = capsys.readouterr().out
    assert "VALIDATION PASSED" not in captured  # validate() doesn't print that, main() does


def test_validate_fails_on_missing_target(tmp_path: Path, patch_read_parquet_if_needed):
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    artifacts.mkdir()
    processed.mkdir()

    spec = _base_spec(target_name="target")
    spec_path = artifacts / "feature_spec_v1.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    df_eng = _engineered_df(n=50, seed=4, target_name="target").drop(columns=["target"])
    feat_path = processed / "engineered_features_v1.parquet"
    _write_df_as_parquet(feat_path, df_eng)

    cfg = ValidateConfig(
        artifacts_dir=artifacts,
        processed_dir=processed,
        interim_dir=tmp_path / "interim",
        feature_spec_file=spec_path.name,
        engineered_file=feat_path.name,
        interim_file=None,
        forbidden_cols=[],
        run_canary=False,
        run_perm_importance=False,
    )

    with pytest.raises(ValidationError, match="Target 'target' missing"):
        validate(cfg)


def test_validate_fails_on_missing_contracted_features(
    tmp_path: Path, patch_read_parquet_if_needed
):
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    artifacts.mkdir()
    processed.mkdir()

    spec = _base_spec(target_name="target")
    spec_path = artifacts / "feature_spec_v1.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    df_eng = _engineered_df(n=60, seed=5, target_name="target").drop(columns=["eng_feat_2"])
    feat_path = processed / "engineered_features_v1.parquet"
    _write_df_as_parquet(feat_path, df_eng)

    cfg = ValidateConfig(
        artifacts_dir=artifacts,
        processed_dir=processed,
        interim_dir=tmp_path / "interim",
        feature_spec_file=spec_path.name,
        engineered_file=feat_path.name,
        interim_file=None,
        forbidden_cols=[],
        run_canary=False,
        run_perm_importance=False,
    )

    with pytest.raises(ValidationError, match="Feature contract failed: missing"):
        validate(cfg)


def test_validate_fails_on_leakage_prefix_in_engineered_matrix(
    tmp_path: Path, patch_read_parquet_if_needed
):
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    artifacts.mkdir()
    processed.mkdir()

    spec = _base_spec(target_name="target")
    spec_path = artifacts / "feature_spec_v1.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    df_eng = _engineered_df(n=80, seed=6, target_name="target")
    df_eng["loan_status_something"] = 1  # forbidden leakage prefix
    feat_path = processed / "engineered_features_v1.parquet"
    _write_df_as_parquet(feat_path, df_eng)

    cfg = ValidateConfig(
        artifacts_dir=artifacts,
        processed_dir=processed,
        interim_dir=tmp_path / "interim",
        feature_spec_file=spec_path.name,
        engineered_file=feat_path.name,
        interim_file=None,
        forbidden_cols=[],
        run_canary=False,
        run_perm_importance=False,
    )

    with pytest.raises(ValidationError, match="Leakage detected in engineered matrix"):
        validate(cfg)


def test_validate_fails_leakage_firewall_in_interim(tmp_path: Path, patch_read_parquet_if_needed):
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"
    artifacts.mkdir()
    processed.mkdir()
    interim.mkdir()

    spec = _base_spec(target_name="target")
    spec_path = artifacts / "feature_spec_v1.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    df_eng = _engineered_df(n=70, seed=7, target_name="target")
    feat_path = processed / "engineered_features_v1.parquet"
    _write_df_as_parquet(feat_path, df_eng)

    # Put a forbidden leakage col into interim
    forbidden_col = sorted(DEFAULT_FORBIDDEN_COLS)[0]
    df_interim = pd.DataFrame({forbidden_col: [0, 1], "ok": [1, 2]})
    interim_path = interim / "loans_cleaned.parquet"
    _write_df_as_parquet(interim_path, df_interim)

    cfg = ValidateConfig(
        artifacts_dir=artifacts,
        processed_dir=processed,
        interim_dir=interim,
        feature_spec_file=spec_path.name,
        engineered_file=feat_path.name,
        interim_file=interim_path.name,
        forbidden_cols=sorted(DEFAULT_FORBIDDEN_COLS),
        run_canary=False,
        run_perm_importance=False,
    )

    with pytest.raises(ValidationError, match="Leakage firewall failed"):
        validate(cfg)


def test_validate_runs_canary_and_can_fail(tmp_path: Path, patch_read_parquet_if_needed):
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    artifacts.mkdir()
    processed.mkdir()

    spec = _base_spec(target_name="target")
    spec_path = artifacts / "feature_spec_v1.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    # engineered df with a leakage column that will be included by contract:
    # easiest: add it as an "engineered" spec key so contract includes it
    spec["features"]["engineered"]["leak_feat"] = {}
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    df_eng = _engineered_df(n=240, seed=8, target_name="target")
    df_eng["leak_feat"] = df_eng["target"].astype(float)  # perfect leakage
    feat_path = processed / "engineered_features_v1.parquet"
    _write_df_as_parquet(feat_path, df_eng)

    cfg = ValidateConfig(
        artifacts_dir=artifacts,
        processed_dir=processed,
        interim_dir=tmp_path / "interim",
        feature_spec_file=spec_path.name,
        engineered_file=feat_path.name,
        interim_file=None,
        forbidden_cols=[],
        run_canary=True,  # should trip
        run_perm_importance=False,
    )

    with pytest.raises(ValidationError, match="Canary failed: baseline val ROC AUC too high"):
        validate(cfg)
