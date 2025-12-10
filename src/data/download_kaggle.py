# src/data/download_kaggle.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing kaggle
# Find the project root by going up from this file's directory
_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / ".env"
load_dotenv(_env_file)

from src.config import RAW_DATA_DIR  # noqa: E402

# Default Kaggle dataset slug (you can change this to your preferred dataset)
DEFAULT_DATASET_SLUG = "wordsforthewise/lending-club"

DEFAULT_OUTPUT_DIR = RAW_DATA_DIR


def download_dataset(
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR),
    unzip: bool = True,
) -> None:
    """
    Download a Kaggle dataset into RAW_DATA_DIR using the Kaggle API.

    Parameters
    ----------
    dataset_slug:
        The Kaggle dataset identifier in the form "owner/dataset".
        Example: "wordsforthewise/lending-club"

    output_dir:
        Directory where raw files will be placed.

    unzip:
        Whether to unzip the downloaded archive.
    """
    # Import KaggleApi only when needed to avoid authentication errors during import
    from kaggle.api.kaggle_api_extended import KaggleApi

    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        msg = (
            "Failed to authenticate with Kaggle API. "
            "Make sure you have Kaggle credentials configured:\n"
            "- KAGGLE_TOKEN environment variable (recommended), or\n"
            "- KAGGLE_USERNAME and KAGGLE_KEY environment variables, or\n"
            "- ~/.kaggle/kaggle.json file.\n"
            f"Original error: {exc}"
        )
        raise RuntimeError(msg) from exc

    print(f"Downloading dataset '{dataset_slug}' to {output_dir} (unzip={unzip})...")
    api.dataset_download_files(
        dataset_slug,
        path=str(output_dir),
        quiet=False,
        unzip=unzip,
    )
    print("Download complete.")


def cleanup_output_dir(output_dir: Path = Path(DEFAULT_OUTPUT_DIR)) -> None:
    # First, for CSV files in subdirectories (including nested ones), move them into the root directory
    for item in output_dir.iterdir():
        if item.is_dir():
            for csv_file in item.rglob("*.csv"):
                if csv_file.is_file():
                    target = output_dir / csv_file.name
                    # Handle naming conflicts by checking if target exists
                    if target.exists():
                        print(f"Warning: {target} already exists, skipping {csv_file.name}")
                    else:
                        csv_file.rename(target)
                        print(f"Moved {csv_file.name} to root directory")

    # Remove all non-CSV files in the root directory
    for item in output_dir.iterdir():
        if item.is_file() and item.suffix != ".csv":
            item.unlink()
            print(f"Removed non-CSV file: {item.name}")

    # Remove non-CSV files from subdirectories and delete empty subdirectories
    for item in output_dir.iterdir():
        if item.is_dir():
            # Remove only non-CSV files in subdirectories
            for subitem in item.rglob("*"):
                if subitem.is_file() and subitem.suffix != ".csv":
                    subitem.unlink()
                    print(f"Removed non-CSV file: {subitem.relative_to(output_dir)}")
            # Remove empty subdirectories (bottom-up)
            for subdir in sorted(item.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if subdir.is_dir():
                    try:
                        subdir.rmdir()
                    except OSError:
                        pass  # Directory not empty, skip
            # Remove the top-level subdirectory if empty
            try:
                item.rmdir()
                print(f"Removed empty directory: {item.name}")
            except OSError:
                # Directory not empty (may contain CSV files with naming conflicts)
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw loan data from Kaggle into RAW_DATA_DIR."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_SLUG,
        help=f"Kaggle dataset slug (default: {DEFAULT_DATASET_SLUG})",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not unzip the downloaded archive.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    unzip = not args.no_unzip

    download_dataset(
        dataset_slug=args.dataset,
        output_dir=Path(DEFAULT_OUTPUT_DIR),
        unzip=unzip,
    )

    cleanup_output_dir(
        output_dir=Path(DEFAULT_OUTPUT_DIR),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
