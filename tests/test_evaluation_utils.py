"""Tests for evaluation utilities."""

import os
from src.scripts.utils.evaluation import (
    load_jsonl,
    safe_mean,
    safe_percentile,
    load_training_metadata,
    estimate_steps_per_epoch,
    build_training_summary,
)


class TestLoadJsonl:
    def _tmp_file(self, name: str):
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".tmp_test_evaluation_utils",
        )
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, name)

    def test_load_empty_file(self):
        temp_path = self._tmp_file("empty.jsonl")
        open(temp_path, "w", encoding="utf-8").close()
        result = load_jsonl(temp_path)
        assert result == []

    def test_load_valid_jsonl(self):
        temp_path = self._tmp_file("valid.jsonl")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write('{"step": 1, "loss": 0.5}\n')
            f.write('{"step": 2, "loss": 0.4}\n')
        result = load_jsonl(temp_path)
        assert len(result) == 2
        assert result[0]["step"] == 1
        assert result[1]["loss"] == 0.4

    def test_load_with_empty_lines(self):
        temp_path = self._tmp_file("empty_lines.jsonl")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write('{"step": 1}\n\n')
            f.write('{"step": 2}\n')
        result = load_jsonl(temp_path)
        assert len(result) == 2


class TestSafeMean:
    def test_empty_list(self):
        assert safe_mean([]) is None

    def test_valid_numbers(self):
        assert safe_mean([1, 2, 3, 4, 5]) == 3.0

    def test_with_none_values(self):
        assert safe_mean([1, 2, None, 3]) == 2.0

    def test_with_non_numeric(self):
        assert safe_mean([1, "two", 3]) == 2.0


class TestSafePercentile:
    def test_empty_list(self):
        assert safe_percentile([], 50) is None

    def test_percentile_median(self):
        assert safe_percentile([1, 2, 3, 4, 5], 50) == 3

    def test_percentile_0(self):
        assert safe_percentile([1, 2, 3, 4, 5], 0) == 1

    def test_percentile_100(self):
        assert safe_percentile([1, 2, 3, 4, 5], 100) == 5


class TestLoadTrainingMetadata:
    def test_missing_file(self):
        result = load_training_metadata("/nonexistent", "test")
        assert result == {}

    def test_invalid_json(self):
        temp_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".tmp_test_evaluation_utils",
            "invalid.json",
        )
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("not json")
        try:
            result = load_training_metadata(os.path.dirname(temp_path), "nonexistent")
            assert result == {}
        finally:
            pass


class TestEstimateStepsPerEpoch:
    def test_missing_paths(self):
        config = {"training": {"batch_size": 32, "gradient_accumulation_steps": 1}}
        result = estimate_steps_per_epoch("/nonexistent", "test", config)
        assert result is None

    def test_from_config(self):
        config = {
            "training": {
                "batch_size": 32,
                "gradient_accumulation_steps": 1,
                "max_steps": 1000,
            }
        }
        result = estimate_steps_per_epoch("/nonexistent", "test", config)
        assert result == 1000


class TestBuildTrainingSummary:
    def test_empty_metrics(self):
        result = build_training_summary("/nonexistent", "test", {})
        assert result is None
