"""Fast tests: no torch (see src.benchmark_config)."""

from src.benchmark_config import (
    merge_external_model_eval_config,
    resolve_benchmark_gen_do_sample,
    resolve_benchmark_gen_max_new_cap,
    resolve_benchmark_gen_total_tokens,
)


def test_resolve_caps():
    assert resolve_benchmark_gen_max_new_cap({}) == 32
    assert resolve_benchmark_gen_total_tokens({"benchmarks": {"gen_total_tokens": 128}}) == 128
    assert resolve_benchmark_gen_do_sample({"benchmarks": {"gen_do_sample": True}}) is True


def test_merge_root_overrides():
    cfg = {
        "external_model_eval": {"external_gen_temperature": 0.5},
        "external_gen_temperature": 2.0,
    }
    assert merge_external_model_eval_config(cfg)["external_gen_temperature"] == 2.0
