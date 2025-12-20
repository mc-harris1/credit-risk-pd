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

# Default Kaggle dataset slug
DEFAULT_DATASET_SLUG = "wordsforthewise/lending-club"

# DEFAULT_OUTPUT_DIR = RAW_DATA_DIR


def download_dataset(
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    output_dir: Path = Path(RAW_DATA_DIR),
    unzip: bool = True,
) -> None:
    """
    Download a Kaggle dataset into RAW_DATA_DIR using the Kaggle API.
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


def cleanup_output_dir(output_dir: Path = Path(RAW_DATA_DIR)) -> None:
    """
    Normalize Kaggle raw data layout.

    Steps:
      1) Delete all .gz files produced by Kaggle (transport artifacts).
      2) Remove non-CSV files from root and subdirectories.
      3) Move CSV files from subdirectories to root (handle naming conflicts).
      4) Remove now-empty nested directories.

    Fails fast if an overwrite would occur without user confirmation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Remove all .gz files (Kaggle transport artifacts)
    gz_files = list(output_dir.rglob("*.gz"))
    for gz in gz_files:
        try:
            gz.unlink()
            print(f"Removed transport artifact: {gz.relative_to(output_dir)}")
        except OSError as exc:
            raise RuntimeError(f"Failed to delete .gz file: {gz}") from exc

    # 2) Remove non-CSV files from root directory
    for item in output_dir.iterdir():
        if item.is_file() and item.suffix.lower() != ".csv":
            try:
                item.unlink()
                print(f"Removed non-CSV file: {item.relative_to(output_dir)}")
            except OSError as exc:
                raise RuntimeError(f"Failed to delete non-CSV file: {item}") from exc

    # 3) Collect CSV files from subdirectories and remove non-CSV files
    csv_moves: list[tuple[Path, Path]] = []
    subdirs_to_check: set[Path] = set()

    for subdir in output_dir.iterdir():
        if not subdir.is_dir():
            continue

        subdirs_to_check.add(subdir)

        # Remove non-CSV files from subdirectories
        for item in subdir.rglob("*"):
            if item.is_file() and item.suffix.lower() != ".csv":
                try:
                    item.unlink()
                    print(f"Removed non-CSV file: {item.relative_to(output_dir)}")
                except OSError as exc:
                    raise RuntimeError(f"Failed to delete non-CSV file: {item}") from exc

        # Collect CSV files for potential moving
        for csv_file in subdir.rglob("*.csv"):
            dst_path = output_dir / csv_file.name
            csv_moves.append((csv_file, dst_path))

    # 4) Move CSV files from subdirectories to root
    for src, dst in csv_moves:
        if dst.is_file():
            # Naming conflict: skip the file and keep it in subdirectory
            print(
                f"Warning: {dst.name} already exists at root. Keeping {src.relative_to(output_dir)}"
            )
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                print(f"Moved {src.relative_to(output_dir)} -> {dst.relative_to(output_dir)}")
            except OSError as exc:
                raise RuntimeError(f"Failed to move CSV file: {src}") from exc

    # 5) Remove now-empty nested directories
    # Sort in reverse to handle nested directories properly
    for d in sorted(subdirs_to_check, reverse=True):
        if d.exists():
            try:
                # Try to remove - will fail if directory not empty
                d.rmdir()
                print(f"Removed empty directory: {d.relative_to(output_dir)}")
            except OSError:
                # Directory not empty or other error - that's OK, just skip
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
        output_dir=Path(RAW_DATA_DIR),
        unzip=unzip,
    )

    cleanup_output_dir(
        output_dir=Path(RAW_DATA_DIR),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
