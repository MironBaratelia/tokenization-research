"""Tests for plotting utilities."""

import os
import tempfile
from src.scripts.utils.plotting import (
    load_metrics,
    normalize_metrics,
    extract_metric_series,
)


class TestLoadMetrics:
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        result = load_metrics(temp_path)
        assert result == []
        os.unlink(temp_path)

    def test_with_valid_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"step": 1, "loss": 0.5}\n')
            f.write('{"step": 2, "loss": 0.4}\n')
            temp_path = f.name

        result = load_metrics(temp_path)
        assert len(result) == 2
        os.unlink(temp_path)

    def test_with_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": true}\n')
            f.write("invalid json\n")
            f.write('{"also": true}\n')
            temp_path = f.name

        result = load_metrics(temp_path)
        assert len(result) == 2
        os.unlink(temp_path)


class TestNormalizeMetrics:
    def test_train_metrics_normalization(self):
        rows = [
            {"step": 1, "train_loss": 0.5, "Train/loss": 0.6},
            {"step": 2, "loss": 0.4},
        ]
        train_rows, val_rows = normalize_metrics(rows)

        assert len(train_rows) == 2
        assert any("train_loss" in r for r in train_rows)

    def test_val_metrics_normalization(self):
        rows = [
            {"step": 1, "val_loss": 0.5, "Val/val_loss": 0.6},
            {"step": 2, "Val/val_loss": 0.4},
        ]
        train_rows, val_rows = normalize_metrics(rows)

        assert len(val_rows) == 2

    def test_mixed_metrics(self):
        rows = [
            {"step": 1, "train_loss": 0.5, "val_loss": 0.6},
        ]
        train_rows, val_rows = normalize_metrics(rows)

        assert len(train_rows) == 1
        assert len(val_rows) == 1


class TestExtractMetricSeries:
    def test_basic_extraction(self):
        rows = [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4}]
        steps, values = extract_metric_series(rows, "loss")

        assert steps == [1, 2]
        assert values == [0.5, 0.4]

    def test_with_none_values(self):
        rows = [{"step": 1, "loss": 0.5}, {"step": 2}]
        steps, values = extract_metric_series(rows, "loss")

        assert steps == [1]
        assert values == [0.5]

    def test_missing_step_key(self):
        rows = [{"loss": 0.5}, {"step": 2, "loss": 0.4}]
        steps, values = extract_metric_series(rows, "loss")

        assert steps == [2]
        assert values == [0.4]
