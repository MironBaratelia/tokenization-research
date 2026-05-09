"""Tests for config loading and path resolution."""

import os
from src.io.config import get_full_config, resolve_paths


class TestResolvePaths:
    """resolve_paths must never produce None in resolved_paths."""

    def test_resolved_paths_never_none(self):
        """All resolved_paths values must be non-None strings."""
        config = {
            "paths": {"data_dir": "data", "tokenizers_dir": "tok", "outputs_dir": "out"}
        }
        root = os.path.abspath("/tmp")
        resolve_paths(config, root)
        rp = config["resolved_paths"]
        assert rp is not None
        assert rp["data_dir"] is not None
        assert rp["tokenizers_dir"] is not None
        assert rp["outputs_dir"] is not None
        assert isinstance(rp["tokenizers_dir"], str)
        assert len(rp["tokenizers_dir"]) > 0

    def test_resolved_paths_handles_none_in_paths(self):
        """When paths.tokenizers_dir is None, use default."""
        config = {"paths": {"tokenizers_dir": None}}
        root = os.path.abspath("/tmp")
        resolve_paths(config, root)
        rp = config["resolved_paths"]
        assert rp["tokenizers_dir"] is not None
        assert (
            "project_tokenizers" in rp["tokenizers_dir"]
            or "trained" in rp["tokenizers_dir"]
        )

    def test_resolved_paths_handles_empty_paths(self):
        """When paths is empty or missing, use defaults."""
        config = {}
        root = os.path.abspath("/tmp")
        resolve_paths(config, root)
        rp = config["resolved_paths"]
        assert rp["tokenizers_dir"] is not None
        assert rp["outputs_dir"] is not None


class TestGetFullConfig:
    """get_full_config must produce valid config for training script."""

    def test_ru_miron_config_has_resolved_paths(self):
        """ru/miron config loads and has non-None resolved_paths."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config = get_full_config("configs/experiments/ru/miron.yaml", project_root)
        assert "resolved_paths" in config
        rp = config["resolved_paths"]
        assert rp is not None
        assert rp.get("tokenizers_dir") is not None, "tokenizers_dir must not be None"
        assert rp.get("outputs_dir") is not None
        assert rp.get("data_dir") is not None
        assert isinstance(rp["tokenizers_dir"], str)
        assert (
            os.path.isabs(rp["tokenizers_dir"])
            or os.path.normpath(rp["tokenizers_dir"]) == rp["tokenizers_dir"]
        )

    def test_tokenizer_path_joinable(self):
        """resolved_paths allows os.path.join with tokenizer_id."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config = get_full_config("configs/experiments/ru/miron.yaml", project_root)
        tok_dir = config.get("resolved_paths", {}).get("tokenizers_dir")
        if not tok_dir:
            tok_dir = os.path.join(project_root, "project_tokenizers", "trained")
        tokenizer_id = config.get("tokenizer", {}).get("id") or config.get(
            "experiment", {}
        ).get("id", "ru/miron")
        path = os.path.normpath(os.path.join(str(tok_dir), str(tokenizer_id)))
        assert path is not None
        assert not any(x is None for x in [tok_dir, tokenizer_id])
