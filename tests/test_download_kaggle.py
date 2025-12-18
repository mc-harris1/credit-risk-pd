import pytest
from src.data.download_kaggle import cleanup_output_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


def test_cleanup_moves_csv_from_subdirectory_to_root(temp_output_dir):
    """Test that CSV files in subdirectories are moved to root directory."""
    subdir = temp_output_dir / "dataset_name"
    subdir.mkdir()
    csv_file = subdir / "data.csv"
    csv_file.write_text("test data")

    cleanup_output_dir(temp_output_dir)

    assert (temp_output_dir / "data.csv").exists()
    assert not csv_file.exists()


def test_cleanup_skips_csv_with_naming_conflict(temp_output_dir, capsys):
    """Test that CSV files with naming conflicts are skipped."""
    (temp_output_dir / "data.csv").write_text("existing data")
    subdir = temp_output_dir / "dataset_name"
    subdir.mkdir()
    (subdir / "data.csv").write_text("new data")

    cleanup_output_dir(temp_output_dir)

    assert (temp_output_dir / "data.csv").read_text() == "existing data"
    assert (subdir / "data.csv").exists()
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "already exists" in captured.out


def test_cleanup_removes_non_csv_files_from_root(temp_output_dir):
    """Test that non-CSV files are removed from root directory."""
    (temp_output_dir / "data.csv").write_text("csv data")
    (temp_output_dir / "readme.txt").write_text("readme")
    (temp_output_dir / "archive.zip").write_text("archive")

    cleanup_output_dir(temp_output_dir)

    assert (temp_output_dir / "data.csv").exists()
    assert not (temp_output_dir / "readme.txt").exists()
    assert not (temp_output_dir / "archive.zip").exists()


def test_cleanup_removes_non_csv_files_from_subdirectories(temp_output_dir):
    """Test that non-CSV files are removed from subdirectories."""
    subdir = temp_output_dir / "dataset_name"
    subdir.mkdir()
    (subdir / "data.csv").write_text("csv data")
    (subdir / "readme.txt").write_text("readme")

    cleanup_output_dir(temp_output_dir)

    assert (temp_output_dir / "data.csv").exists()
    assert not (subdir / "readme.txt").exists()


def test_cleanup_removes_empty_subdirectories(temp_output_dir):
    """Test that empty subdirectories are removed."""
    subdir = temp_output_dir / "dataset_name"
    subdir.mkdir()
    (subdir / "readme.txt").write_text("readme")

    cleanup_output_dir(temp_output_dir)

    assert not subdir.exists()


def test_cleanup_keeps_subdirectories_with_csv_files(temp_output_dir):
    """Test that subdirectories with CSV files that couldn't be moved are kept."""
    (temp_output_dir / "data.csv").write_text("existing")
    subdir = temp_output_dir / "dataset_name"
    subdir.mkdir()
    (subdir / "data.csv").write_text("conflicting")

    cleanup_output_dir(temp_output_dir)

    assert subdir.exists()
    assert (subdir / "data.csv").exists()


def test_cleanup_handles_nested_subdirectories(temp_output_dir):
    """Test cleanup with nested directory structures."""
    subdir = temp_output_dir / "dataset"
    nested = subdir / "nested"
    nested.mkdir(parents=True)
    (nested / "data.csv").write_text("csv data")
    (nested / "readme.txt").write_text("readme")

    cleanup_output_dir(temp_output_dir)

    assert (temp_output_dir / "data.csv").exists()
    assert not (nested / "readme.txt").exists()


def test_cleanup_with_empty_directory(temp_output_dir):
    """Test cleanup on an empty directory doesn't raise errors."""
    cleanup_output_dir(temp_output_dir)

    assert temp_output_dir.exists()


def test_cleanup_preserves_csv_files_in_root(temp_output_dir):
    """Test that existing CSV files in root are preserved."""
    (temp_output_dir / "existing.csv").write_text("existing data")
    (temp_output_dir / "another.csv").write_text("another data")

    cleanup_output_dir(temp_output_dir)

    assert (temp_output_dir / "existing.csv").exists()
    assert (temp_output_dir / "another.csv").exists()
