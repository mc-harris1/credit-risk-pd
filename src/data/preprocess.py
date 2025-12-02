
from src.config import INTERIM_DATA_DIR
from src.data.load_data import load_raw_loans

DEFAULT_INPUT_FILE = "loans.csv"
DEFAULT_OUTPUT_FILE = "loans_preprocessed.parquet"


def preprocess_loans(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> None:
    """Basic preprocessing stub: load, lightly clean, and save interim data."""
    df = load_raw_loans(input_file)

    # TODO: add real cleaning logic
    df = df.dropna(axis=0, how="all")

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DATA_DIR / output_file
    df.to_parquet(output_path, index=False)
    print(f"Saved preprocessed loans to {output_path}")


if __name__ == "__main__":
    preprocess_loans()
