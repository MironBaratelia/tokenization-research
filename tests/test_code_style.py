"""Tests for publication-style repository hygiene."""

from pathlib import Path


def test_style_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "CODE_STYLE.md").is_file()
    assert (root / "pyproject.toml").is_file()
    assert (root / ".flake8").is_file()


def test_code_style_mentions_expected_tools():
    root = Path(__file__).resolve().parents[1]
    text = (root / "CODE_STYLE.md").read_text(encoding="utf-8")

    for keyword in ("black", "flake8", "mypy", "pytest"):
        assert keyword in text


def test_code_style_mentions_repo_layout():
    root = Path(__file__).resolve().parents[1]
    text = (root / "CODE_STYLE.md").read_text(encoding="utf-8")

    for keyword in ("src/", "tests/", "scripts/", "configs/"):
        assert keyword in text
