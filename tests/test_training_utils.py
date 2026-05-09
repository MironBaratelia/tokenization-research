"""Tests for training utilities."""

import os
import shutil
from pathlib import Path
from uuid import uuid4

from src.scripts.utils.training import resolve_data_paths


class TestResolveDataPaths:
    def _make_tmp_dir(self):
        root = Path(__file__).resolve().parents[1] / ".tmp_test_training_utils"
        root.mkdir(exist_ok=True)
        path = root / uuid4().hex
        path.mkdir()
        return path

    def _cleanup_tmp_dir(self, path):
        shutil.rmtree(path, ignore_errors=True)

    def test_shuffled_manifest_priority(self):
        config = {"data": {"train_file": "train.txt", "validation_file": "val.txt"}}
        tmpdir = self._make_tmp_dir()
        try:
            output_dir = tmpdir / "output"
            tokenized_dir = output_dir / "tokenized"
            tokenized_dir.mkdir(parents=True)

            shuffled = tokenized_dir / "train_packed_shuffled.manifest.json"
            shuffled.write_text("{}", encoding="utf-8")

            train_path, _ = resolve_data_paths(config, str(tmpdir), str(output_dir))

            assert "shuffled" in train_path
        finally:
            self._cleanup_tmp_dir(tmpdir)

    def test_packed_manifest_fallback(self):
        config = {"data": {"train_file": "train.txt", "validation_file": "val.txt"}}
        tmpdir = self._make_tmp_dir()
        try:
            output_dir = tmpdir / "output"
            tokenized_dir = output_dir / "tokenized"
            tokenized_dir.mkdir(parents=True)

            packed = tokenized_dir / "train_packed.arrow.manifest.json"
            packed.write_text("{}", encoding="utf-8")

            try:
                resolve_data_paths(config, str(tmpdir), str(output_dir))
                raise AssertionError("Expected FileNotFoundError for packed-only tokenized data")
            except FileNotFoundError as exc:
                assert "train_packed_shuffled.manifest.json" in str(exc)
        finally:
            self._cleanup_tmp_dir(tmpdir)

    def test_absolute_path_fallback(self):
        config = {"data": {"train_file": "/absolute/path/train.txt"}}
        tmpdir = self._make_tmp_dir()
        try:
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            train_path, _ = resolve_data_paths(config, str(tmpdir), str(output_dir))

            assert train_path == "/absolute/path/train.txt"
        finally:
            self._cleanup_tmp_dir(tmpdir)

    def test_relative_path_resolution(self):
        config = {"data": {"train_file": "data/train.txt"}}
        tmpdir = self._make_tmp_dir()
        try:
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            train_path, _ = resolve_data_paths(config, str(tmpdir), str(output_dir))
            expected = os.path.normpath(os.path.join(str(tmpdir), "data", "train.txt"))
            assert os.path.normpath(train_path) == expected
        finally:
            self._cleanup_tmp_dir(tmpdir)
