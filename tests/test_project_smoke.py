"""General repository smoke tests for publication readiness."""

import os


def test_repo_has_expected_root_layout():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("configs", "scripts", "src", "tests", "tex"):
        assert os.path.isdir(os.path.join(root, name))


def test_key_entrypoints_import():
    import src.io.config
    import src.reporting.run
    import src.training.base_trainer

    assert callable(src.io.config.get_full_config)
    assert callable(src.reporting.run.main)
    assert hasattr(src.training.base_trainer, "BaseTrainer")


def test_public_config_files_exist():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in (
        os.path.join("configs", "base.yaml"),
        os.path.join("configs", "base", "en.yaml"),
        os.path.join("configs", "base", "ru.yaml"),
    ):
        assert os.path.isfile(os.path.join(root, rel))
