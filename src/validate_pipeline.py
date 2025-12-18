# src/validate_pipeline.py
# Copyright (c) 2025 Mark Harris
# Licensed under the MIT License. See LICENSE file in the project root.

"""
End-to-end pipeline validation (guardrails) for the credit-risk PD project.

What this script validates:
1) Spec integrity
   - feature_spec_v1.json hash matches spec["freeze"]["spec_hash_md5"] (if present)
2) Data leakage firewall (optional interim input)
   - forbidden post-outcome / leakage columns are not present
3) Feature-matrix contract
   - target exists, binary, no missing
   - all contracted feature columns exist in engineered feature matrix
4) Temporal framing
   - anchor can be reconstructed from issue_d_year/month (or year/quarter)
   - time split has no overlap (train <= val <= test)
5) Canary modeling checks (optional)
   - baseline LR ROC AUC not “impossibly high”
   - overfit gap not excessive
6) (Optional) Permutation-importance dominance checks
   - no single feature dominates
   - time-derived features not dominating

Usage:
  python -m src.validate_pipeline
  python -m src.validate_pipeline --run-canary
  python -m src.validate_pipeline --run-canary --run-permutation-importance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

# ----------------------------
# Defaults (match your pipeline)
# ----------------------------
DEFAULT_FEATURE_SPEC = "feature_spec_v1.json"
DEFAULT_ENGINEERED = "engineered_features_v1.parquet"
DEFAULT_INTERIM = "loans_cleaned.parquet"


# ----------------------------
# Guardrail thresholds
# ----------------------------
MAX_BASELINE_AUC = 0.95  # canary: baseline too good => likely leakage
MAX_AUC_GAP = 0.10  # train - val
MAX_SINGLE_FEAT_DOMINANCE = 0.30  # permutation importance
MAX_TIME_FEAT_SHARE = 0.25  # sum importance of issue_d_*


# ----------------------------
# Leakage firewall columns
# (keep this conservative; spec exclusions already do a lot)
# ----------------------------
DEFAULT_FORBIDDEN_COLS = {
    "loan_status",
    "total_pymnt",
    "total_pymnt_inv",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "out_prncp",
    "out_prncp_inv",
    "next_pymnt_d",
    "last_credit_pull_d",
}


# =============================================================================
# Helpers
# =============================================================================


class ValidationError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def md5_file(path: Path) -> str:
    return md5_bytes(path.read_bytes())


def fail(msg: str) -> NoReturn:
    raise ValidationError(msg)


def info(msg: str) -> None:
    print(f"[validate] {msg}")


def warn(msg: str) -> None:
    print(f"[validate][WARN] {msg}")


# =============================================================================
# Contract derivation (same logic as 03_mp refactor)
# =============================================================================


def derive_contracted_feature_cols(df: pd.DataFrame, spec: Dict[str, Any]) -> List[str]:
    # 1) datetime derived features
    datetime_feats: List[str] = []
    for _col, cfg in spec.get("features", {}).get("datetime", {}).items():
        datetime_feats.extend(cfg.get("derived_features", []))

    # 2) engineered
    engineered_feats = list(spec.get("features", {}).get("engineered", {}).keys())

    # 3) categorical outputs
    categorical_cfg = spec.get("features", {}).get("categorical", {})
    cat_one_hot_cols: List[str] = []
    cat_scalar_cols: List[str] = []
    for col, cfg in categorical_cfg.items():
        strat = cfg.get("encoding_strategy")
        if strat == "one_hot":
            # built by get_dummies(prefix=col)
            cat_one_hot_cols.extend([c for c in df.columns if c.startswith(f"{col}_")])
        elif strat == "target_mean":
            cat_scalar_cols.append(f"{col}__target_mean")
        else:
            cat_scalar_cols.append(f"{col}__freq")

    # 4) numerical outputs
    numerical_cfg = spec.get("features", {}).get("numerical", {})
    num_cols = list(numerical_cfg.keys())
    num_transformed = []
    for col, cfg in numerical_cfg.items():
        if cfg.get("planned_transformation") == "log":
            num_transformed.append(f"{col}__log1p")

    cols = sorted(
        set(
            datetime_feats
            + engineered_feats
            + cat_one_hot_cols
            + cat_scalar_cols
            + num_cols
            + num_transformed
        )
    )
    return cols


def make_anchor_timestamp(df_all: pd.DataFrame) -> pd.Series:
    # Preferred: year + month
    if "issue_d_year" in df_all.columns and "issue_d_month" in df_all.columns:
        y = df_all["issue_d_year"].astype("Int64")
        m = df_all["issue_d_month"].astype("Int64")
        return pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": 1}), errors="raise")

    # Fallback: year + quarter
    if "issue_d_year" in df_all.columns and "issue_d_quarter" in df_all.columns:
        y = df_all["issue_d_year"].astype("Int64")
        q = df_all["issue_d_quarter"].astype("Int64")
        m = (q - 1) * 3 + 1
        return pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": 1}), errors="raise")

    fail("Cannot reconstruct anchor time: missing issue_d_year/month (or year/quarter).")


def time_based_split_indices(
    anchor: pd.Series, train_frac=0.70, val_frac=0.15, test_frac=0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        fail("train/val/test fractions must sum to 1.0")

    order = np.argsort(anchor.to_numpy())
    n = len(order)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    idx_train = order[:n_train]
    idx_val = order[n_train : n_train + n_val]
    idx_test = order[n_train + n_val :]
    return idx_train, idx_val, idx_test


def assert_no_temporal_overlap(
    anchor: pd.Series, idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray
) -> None:
    a_train_max = anchor.iloc[idx_train].max()
    a_val_min = anchor.iloc[idx_val].min()
    a_val_max = anchor.iloc[idx_val].max()
    a_test_min = anchor.iloc[idx_test].min()

    if a_train_max > a_val_min:
        fail(f"Temporal leakage: train overlaps validation ({a_train_max} > {a_val_min}).")
    if a_val_max > a_test_min:
        fail(f"Temporal leakage: validation overlaps test ({a_val_max} > {a_test_min}).")


# =============================================================================
# Canary model + permutation importance (optional)
# =============================================================================


def run_canary_lr(
    X: pd.DataFrame,
    y: pd.Series,
    anchor: pd.Series,
    max_auc: float = MAX_BASELINE_AUC,
    max_gap: float = MAX_AUC_GAP,
) -> Dict[str, float]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    idx_train, idx_val, _ = time_based_split_indices(anchor)
    Xtr, ytr = X.iloc[idx_train], y.iloc[idx_train]
    Xva, yva = X.iloc[idx_val], y.iloc[idx_val]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=400, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    model.fit(Xtr, ytr.to_numpy())
    ptr = model.predict_proba(Xtr)[:, 1]
    pva = model.predict_proba(Xva)[:, 1]

    train_auc = float(roc_auc_score(ytr, ptr))
    val_auc = float(roc_auc_score(yva, pva))
    gap = train_auc - val_auc

    if val_auc > max_auc:
        fail(
            f"Canary failed: baseline val ROC AUC too high ({val_auc:.3f} > {max_auc}). Possible leakage."
        )
    if gap > max_gap:
        fail(f"Canary failed: excessive overfit gap (train-val AUC gap={gap:.3f} > {max_gap}).")

    return {"train_auc": train_auc, "val_auc": val_auc, "auc_gap": float(gap)}


def run_permutation_importance_checks(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    max_single: float = MAX_SINGLE_FEAT_DOMINANCE,
    max_time_share: float = MAX_TIME_FEAT_SHARE,
) -> Dict[str, Any]:
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    imp = pd.DataFrame(
        {
            "feature": X_val.columns,
            "importance_mean": perm["importances_mean"],
            "importance_std": perm["importances_std"],
        }
    ).sort_values("importance_mean", ascending=False)

    top = imp.iloc[0]
    if float(top["importance_mean"]) > max_single:
        fail(
            f"Permutation importance dominance: '{top['feature']}' importance={top['importance_mean']:.3f} > {max_single}"
        )

    time_feats = [c for c in imp["feature"] if c.startswith("issue_d_")]
    time_share = float(imp.loc[imp["feature"].isin(time_feats), "importance_mean"].sum())
    if time_share > max_time_share:
        fail(f"Time-feature dominance: sum(issue_d_*)={time_share:.3f} > {max_time_share}")

    return {
        "top_feature": str(top["feature"]),
        "top_importance": float(top["importance_mean"]),
        "time_feature_importance_sum": time_share,
        "n_features": int(len(imp)),
    }


# =============================================================================
# Main validation routine
# =============================================================================


@dataclass(frozen=True)
class ValidateConfig:
    artifacts_dir: Path
    processed_dir: Path
    interim_dir: Path
    feature_spec_file: str
    engineered_file: str
    interim_file: str | None
    forbidden_cols: Sequence[str]
    run_canary: bool
    run_perm_importance: bool


def validate(cfg: ValidateConfig) -> Dict[str, Any]:
    # Paths
    spec_path = cfg.artifacts_dir / cfg.feature_spec_file
    feat_path = cfg.processed_dir / cfg.engineered_file

    if not spec_path.exists():
        fail(f"Missing feature spec: {spec_path}")
    if not feat_path.exists():
        fail(f"Missing engineered feature matrix: {feat_path}")

    info(f"Loading spec: {spec_path}")
    spec = read_json(spec_path)

    # 1) Spec integrity (hash)
    spec_hash_now = md5_file(spec_path)
    freeze = spec.get("freeze", {})
    spec_hash_expected = freeze.get("spec_hash_md5")
    if spec_hash_expected:
        if spec_hash_now != spec_hash_expected:
            warn(
                f"Spec hash mismatch: file md5={spec_hash_now} != freeze.spec_hash_md5={spec_hash_expected} (spec may have been edited)"
            )
        else:
            info("Spec hash matches freeze.spec_hash_md5")
    else:
        warn("Spec has no freeze.spec_hash_md5; skipping hash consistency check.")

    # 2) Optional interim leakage firewall
    if cfg.interim_file is not None:
        interim_path = cfg.interim_dir / cfg.interim_file
        if not interim_path.exists():
            fail(f"Missing interim dataset: {interim_path}")
        info(f"Loading interim (leakage firewall): {interim_path}")
        df_interim = pd.read_parquet(interim_path)
        forbidden_present = sorted(set(cfg.forbidden_cols).intersection(df_interim.columns))
        if forbidden_present:
            fail(
                f"Leakage firewall failed: forbidden columns present in interim data: {forbidden_present}"
            )
        info("Leakage firewall passed (interim data)")

    # 3) Feature matrix contract
    info(f"Loading engineered feature matrix: {feat_path}")
    df = pd.read_parquet(feat_path)

    target = spec["target"]["name"]
    allowed = set(spec["target"].get("allowed_values", [0, 1]))

    if target not in df.columns:
        fail(f"Target '{target}' missing from engineered feature matrix.")
    if df[target].isna().any():
        fail("Target contains missing values in engineered feature matrix.")
    unique_vals = set(int(v) for v in df[target].unique())
    if not unique_vals.issubset(allowed):
        fail(f"Target values {sorted(unique_vals)} not subset of allowed {sorted(allowed)}.")

    contracted = derive_contracted_feature_cols(df, spec)
    missing_features = [c for c in contracted if c not in df.columns]
    if missing_features:
        fail(
            f"Feature contract failed: missing {len(missing_features)} features. Example: {missing_features[:25]}"
        )

    # Extra: ensure forbidden leakage prefixes are not present in engineered matrix
    leakage_prefixes = ("loan_status_",)
    leakage_cols = [c for c in df.columns if c.startswith(leakage_prefixes)]
    if leakage_cols:
        fail(
            f"Leakage detected in engineered matrix: found columns with leakage prefix: {leakage_cols[:20]}"
        )

    info(f"Contract OK: {len(contracted)} feature columns + target '{target}'")

    # 4) Temporal framing
    anchor = make_anchor_timestamp(df)
    idx_train, idx_val, idx_test = time_based_split_indices(anchor)
    assert_no_temporal_overlap(anchor, idx_train, idx_val, idx_test)
    info("Temporal split guardrails passed (no overlap)")

    # Prepare X/y
    X = df[contracted].copy()
    y = df[target].astype(int).copy()

    # 5) Canary (optional)
    canary_metrics = None
    if cfg.run_canary:
        info("Running canary baseline LR checks...")
        canary_metrics = run_canary_lr(X, y, anchor)
        info(f"Canary passed: {canary_metrics}")

    # 6) Permutation importance dominance (optional)
    perm_metrics = None
    if cfg.run_perm_importance:
        if not cfg.run_canary:
            warn(
                "Permutation importance requested but canary model not run. Will fit a quick HGB for PI."
            )

        # Fit an HGB on train for PI checks on val
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        info("Fitting quick HGB for permutation importance checks...")
        Xtr, ytr = X.iloc[idx_train], y.iloc[idx_train]
        Xva, yva = X.iloc[idx_val], y.iloc[idx_val]

        hgb = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=300)),
            ]
        )
        hgb.fit(Xtr, ytr.to_numpy())
        info("Running permutation importance dominance checks...")
        perm_metrics = run_permutation_importance_checks(hgb, Xva, yva)
        info(f"Permutation-importance checks passed: {perm_metrics}")

    return {
        "spec_path": str(spec_path),
        "engineered_path": str(feat_path),
        "target": target,
        "n_rows": int(df.shape[0]),
        "n_cols_total": int(df.shape[1]),
        "n_features_contracted": int(len(contracted)),
        "spec_md5": spec_hash_now,
        "canary": canary_metrics,
        "permutation_importance": perm_metrics,
    }


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate pipeline guardrails end-to-end.")
    p.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    p.add_argument("--processed-dir", type=Path, default=Path(PROCESSED_DATA_DIR))
    p.add_argument("--interim-dir", type=Path, default=Path(INTERIM_DATA_DIR))

    p.add_argument("--feature-spec", type=str, default=DEFAULT_FEATURE_SPEC)
    p.add_argument("--engineered", type=str, default=DEFAULT_ENGINEERED)

    p.add_argument(
        "--interim",
        type=str,
        default=DEFAULT_INTERIM,
        help="Interim parquet to firewall-check (set to '' to skip).",
    )

    p.add_argument("--run-canary", action="store_true", help="Run baseline LR canary checks.")
    p.add_argument(
        "--run-permutation-importance",
        action="store_true",
        help="Run PI dominance checks (slower).",
    )

    p.add_argument(
        "--forbidden-col",
        action="append",
        default=[],
        help="Add extra forbidden columns (can repeat).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    interim_file = args.interim if args.interim.strip() else None
    forbidden = sorted(set(DEFAULT_FORBIDDEN_COLS).union(set(args.forbidden_col)))

    cfg = ValidateConfig(
        artifacts_dir=args.artifacts_dir,
        processed_dir=args.processed_dir,
        interim_dir=args.interim_dir,
        feature_spec_file=args.feature_spec,
        engineered_file=args.engineered,
        interim_file=interim_file,
        forbidden_cols=forbidden,
        run_canary=args.run_canary,
        run_perm_importance=args.run_permutation_importance,
    )

    try:
        result = validate(cfg)
        info("✅ VALIDATION PASSED")
        print(json.dumps(result, indent=2))
    except ValidationError as e:
        print(f"❌ VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
