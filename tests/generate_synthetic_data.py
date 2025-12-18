"""Generate synthetic data for CI testing and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_raw_data(output_path: Path, n_samples: int = 1000) -> None:
    """
    Generate synthetic raw loan CSV for CI testing (input to preprocess step).

    The output filename should match preprocess.py's discovery pattern:
      DEFAULT_PATTERN_CONTAINS = "_2007_to_2018Q4"
    """
    rng = np.random.default_rng(42)

    dates = pd.date_range("2015-01-01", periods=n_samples, freq="D")

    df = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(5_000, 35_000, n_samples),
            "annual_inc": rng.uniform(30_000, 150_000, n_samples),
            "int_rate": rng.uniform(5, 25, n_samples),
            "term": rng.choice([" 36 months", " 60 months"], n_samples),
            "dti": rng.uniform(0, 40, n_samples),
            "grade": rng.choice(list("ABCDEFG"), n_samples),
            "sub_grade": rng.choice(["A1", "A2", "B1", "B2", "C1", "C2", "D1"], n_samples),
            "emp_length": rng.choice(
                ["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"],
                n_samples,
            ),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n_samples),
            "issue_d": dates.strftime("%b-%Y"),  # Format like "Jan-2015"
            "loan_status": rng.choice(
                ["Fully Paid", "Charged Off", "Late (31-120 days)", "Late (16-30 days)", "Current"],
                n_samples,
            ),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} synthetic raw loans at {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    raw_file = raw_dir / "accepted_2007_to_2018Q4.csv"
    generate_synthetic_raw_data(raw_file)

    # Create expected dirs for downstream steps (harmless if already exist)
    (project_root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "bundles").mkdir(parents=True, exist_ok=True)
