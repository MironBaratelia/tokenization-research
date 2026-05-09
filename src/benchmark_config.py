"""Benchmark / external_eval config helpers (no torch import).

Placed at src.benchmark_config (not under src.common) so imports do not pull
src.common.__init__ which eagerly imports torch.
"""

from __future__ import annotations

from typing import Optional

_EXTERNAL_GEN_KEYS = (
    "external_gen_batch_size",
    "external_gen_tune_warmup",
    "external_gen_torch_compile",
    "external_gen_reprobe",
    "external_gen_total_tokens",
    "external_gen_max_new_tokens_cap",
    "external_gen_plateau_max_batch",
    "external_gen_temperature",
    "external_gen_repetition_penalty",
    "external_gen_do_sample",
    "external_gen_suppress_eos_first_step",
    "trim_repeated_ngram",
    "trim_min_words",
)


def merge_external_model_eval_config(config: dict) -> dict:
    em = dict(config.get("external_model_eval") or {})
    for k in _EXTERNAL_GEN_KEYS:
        if k in config:
            em[k] = config[k]
    return em


def resolve_benchmark_gen_max_new_cap(config: dict) -> Optional[int]:
    b = config.get("benchmarks") or {}
    if "gen_max_new_tokens" not in b:
        return 32
    v = b["gen_max_new_tokens"]
    if v is None:
        return None
    return max(1, int(v))


def resolve_benchmark_gen_do_sample(config: dict) -> Optional[bool]:
    b = config.get("benchmarks") or {}
    if "gen_do_sample" not in b:
        return None
    v = b["gen_do_sample"]
    if v is None:
        return None
    return bool(v)


def resolve_benchmark_gen_total_tokens(config: dict) -> Optional[int]:
    b = config.get("benchmarks") or {}
    if "gen_total_tokens" not in b:
        return None
    v = b["gen_total_tokens"]
    if v is None:
        return None
    return max(2, int(v))
