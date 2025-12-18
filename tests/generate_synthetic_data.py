"""Generate synthetic data for CI testing and evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_raw_data(output_path: Path, n_samples: int = 1000):
    """
    Generate synthetic raw loan CSV for CI testing (input to preprocess step).

    Args:
        output_path: Path to save the CSV file
        n_samples: Number of synthetic loan samples to generate
    """
    # Generate dates in the format expected by the data
    dates = pd.date_range("2015-01-01", periods=n_samples, freq="D")

    # Create raw data matching the expected schema from preprocess.py
    df = pd.DataFrame(
        {
            "loan_amnt": np.random.uniform(5000, 35000, n_samples),
            "annual_inc": np.random.uniform(30000, 150000, n_samples),
            "int_rate": np.random.uniform(5, 25, n_samples),
            "term": np.random.choice([" 36 months", " 60 months"], n_samples),
            "dti": np.random.uniform(0, 40, n_samples),
            "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples),
            "sub_grade": np.random.choice(["A1", "A2", "B1", "B2", "C1", "C2", "D1"], n_samples),
            "emp_length": np.random.choice(
                ["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"], n_samples
            ),
            "home_ownership": np.random.choice(["RENT", "MORTGAGE", "OWN"], n_samples),
            "issue_d": dates.strftime("%b-%Y"),  # Format like "Jan-2015"
            "loan_status": np.random.choice(["Fully Paid", "Charged Off", "Current"], n_samples),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} synthetic raw loans at {output_path}")


if __name__ == "__main__":
    from pathlib import Path

    # Generate synthetic raw data for CI (input to preprocessing pipeline)
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_file = raw_dir / "accepted_2007_to_2018Q4.csv"

    generate_synthetic_raw_data(raw_file)

    # Also create necessary directories
    (base_dir / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "artifacts").mkdir(parents=True, exist_ok=True)
    (base_dir / "models" / "artifacts").mkdir(parents=True, exist_ok=True)
    (base_dir / "models" / "metadata").mkdir(parents=True, exist_ok=True)
