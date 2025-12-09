"""Generate synthetic data for CI testing and evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_features(output_path: Path, n_samples: int = 1000):
    """
    Generate synthetic loans_features.parquet for CI testing.

    Args:
        output_path: Path to save the parquet file
        n_samples: Number of synthetic loan samples to generate
    """
    dates = pd.date_range("2015-01-01", periods=n_samples, freq="D")

    df = pd.DataFrame(
        {
            "loan_amnt": np.random.uniform(5000, 35000, n_samples),
            "annual_inc": np.random.uniform(30000, 150000, n_samples),
            "dti": np.random.uniform(0, 40, n_samples),
            "term": np.random.choice(["36 months", "60 months"], n_samples),
            "home_ownership": np.random.choice(["RENT", "MORTGAGE", "OWN"], n_samples),
            "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples),
            "sub_grade": np.random.choice(["A1", "A2", "B1", "B2", "C1", "C2", "D1"], n_samples),
            "loan_status": np.random.choice(["Fully Paid", "Charged Off"], n_samples),
            "default": np.random.choice([0, 1], n_samples),
            "issue_d": dates,
            # Engineered features
            "term_months": np.random.choice([36, 60], n_samples),
            "loan_to_income": np.random.uniform(0.1, 0.5, n_samples),
            "grade_numeric": np.random.choice([1, 2, 3, 4, 5, 6, 7], n_samples),
            "sub_grade_numeric": np.random.uniform(1.0, 7.5, n_samples),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Generated {n_samples} synthetic loans at {output_path}")


if __name__ == "__main__":
    from pathlib import Path

    # Generate synthetic data for CI
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    output_file = processed_dir / "loans_features.parquet"

    generate_synthetic_features(output_file)

    # Also create model directories
    (base_dir / "models" / "artifacts").mkdir(parents=True, exist_ok=True)
    (base_dir / "models" / "metadata").mkdir(parents=True, exist_ok=True)
