"""Local batched generation aligned with external-model eval (context limits + batching)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.benchmark_config import merge_external_model_eval_config
from src.common.fast_generation import (
    MAX_GEN_BATCH_CAP,
    apply_inference_throughput_prefs,
    find_plateau_generation_batch_size,
    restore_inference_prefs,
)
from src.common.continuation_decode import decode_continuation
from src.common.inference import InferenceEngine, batched_generate
from src.common.tokenizer_utils import UniversalTokenizer

logger = logging.getLogger(__name__)

# Re-export for callers / tests that imported from here.
from src.benchmark_config import (  # noqa: E402
    resolve_benchmark_gen_do_sample,
    resolve_benchmark_gen_max_new_cap,
    resolve_benchmark_gen_total_tokens,
)


def gen_context_limits_like_external(
    tokenizer: Any,
    prompts: List[str],
    em: dict,
    *,
    gen_total_tokens: Optional[int] = None,
    min_new_tokens: Optional[int] = None,
) -> Tuple[int, int, int]:
    if gen_total_tokens is not None:
        total = max(2, int(gen_total_tokens))
    else:
        total = int(em.get("external_gen_total_tokens", 512))
        total = max(2, total)
    prefill_cap = total - 1
    if min_new_tokens is not None:
        r = max(1, int(min_new_tokens))
        prefill_cap = max(1, (total - 1) - r)
    ut = UniversalTokenizer(tokenizer)
    pad_side = "right" if ut.is_miron else "left"
    max_plen = 0
    for p in prompts:
        enc = ut.encode(
            [p],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
            add_bos_only=True,
            truncation=True,
            max_length=prefill_cap,
            padding_side=pad_side,
        )
        mask = enc.get("attention_mask")
        if mask is not None:
            plen = int(mask[0].sum().item())
        else:
            plen = int(enc["input_ids"].shape[1])
        max_plen = max(max_plen, plen)
    max_new = max(1, total - max_plen)
    return max_new, prefill_cap, max_plen


def _apply_miron_benchmark_max_new_words(max_new: int, tokenizer: Any, config: dict) -> int:
    """Cap MIRON word steps."""
    ut = UniversalTokenizer(tokenizer)
    if not ut.is_miron:
        return max_new
    b = config.get("benchmarks") or {}
    if "miron_max_new_words" not in b:
        return max_new
    raw = b["miron_max_new_words"]
    if raw is None:
        return max_new
    return min(max_new, max(1, int(raw)))


def _external_plateau_max_batch(em: dict) -> int:
    val = em.get("external_gen_plateau_max_batch")
    if val is not None:
        return min(MAX_GEN_BATCH_CAP, max(1, int(val)))
    return int(MAX_GEN_BATCH_CAP)


def _maybe_torch_compile_for_generate(model: Any, config: dict) -> Any:
    em = merge_external_model_eval_config(config)
    if not em.get("external_gen_torch_compile"):
        return model
    if getattr(model, "encode_batch", None) is not None:
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, dynamic=True)  # type: ignore[assignment]
    except Exception as e:
        logger.warning("local_model_generation: torch.compile skipped: %s", e)
        return model


def generate_continuations_external_style(
    model: Any,
    tokenizer: Any,
    config: dict,
    device: str,
    prompts: List[str],
    *,
    batch_size: int,
    seed: int,
    desc: str,
    gen_total_tokens: Optional[int] = None,
    max_new_tokens_cap: Optional[int] = None,
    allow_torch_compile: bool = True,
    use_throughput_prefs: bool = True,
) -> Tuple[List[str], int, Optional[Dict[str, Any]], Any]:
    em = merge_external_model_eval_config(config)
    miron_strict = bool(UniversalTokenizer(tokenizer).is_miron)
    reserve = max(1, int(max_new_tokens_cap)) if max_new_tokens_cap is not None else None
    max_new, prompt_budget, _ = gen_context_limits_like_external(
        tokenizer, prompts, em, gen_total_tokens=gen_total_tokens, min_new_tokens=reserve
    )
    if max_new_tokens_cap is not None:
        max_new = min(max_new, max(1, int(max_new_tokens_cap)))

    max_new = _apply_miron_benchmark_max_new_words(max_new, tokenizer, config)

    do_sample = bool(em.get("external_gen_do_sample", True))
    temperature = float(em.get("external_gen_temperature", 1.0))
    repetition_penalty = float(em.get("external_gen_repetition_penalty", 1.0))
    suppress_eos = bool(em.get("external_gen_suppress_eos_first_step", False))

    explicit_bs = em.get("external_gen_batch_size")
    reprobe = bool(em.get("external_gen_reprobe", False))
    max_batch_cap = _external_plateau_max_batch(em)

    prefs = apply_inference_throughput_prefs(device) if use_throughput_prefs else {}
    batch_search_meta: Optional[Dict[str, Any]] = None
    gen_bs = batch_size
    gen_model = model
    if allow_torch_compile:
        gen_model = _maybe_torch_compile_for_generate(model, config)
    try:
        if explicit_bs is not None:
            gen_bs = int(explicit_bs)
            batch_search_meta = {"source": "config", "external_gen_batch_size": gen_bs}
        elif reprobe:
            gen_bs, probe_meta = find_plateau_generation_batch_size(
                gen_model,
                tokenizer,
                device,
                prompts,
                max_new_tokens=max_new,
                max_batch_cap=max_batch_cap,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                seed=seed,
                suppress_eos_first_step=suppress_eos,
                warmup_rounds=int(em.get("external_gen_tune_warmup", 1)),
                max_context_tokens=prompt_budget,
            )
            batch_search_meta = {"source": "probed", **probe_meta}
        else:
            gen_bs = batch_size
            batch_search_meta = {"source": "eval_batch_size", "batch_size": gen_bs}

        texts = batched_generate(
            model=gen_model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=gen_bs,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            device=device,
            desc=desc,
            seed=seed,
            suppress_eos_first_step=suppress_eos,
            max_context_tokens=prompt_budget,
            miron_strict=miron_strict,
        )
        return texts, gen_bs, batch_search_meta, gen_model
    finally:
        if use_throughput_prefs:
            restore_inference_prefs(prefs)


def generate_continuations_external_style_long_and_short_prefix(
    model: Any,
    tokenizer: Any,
    config: dict,
    device: str,
    prompts: List[str],
    *,
    batch_size: int,
    seed: int,
    desc: str,
    gen_total_tokens: Optional[int] = None,
    short_continuation_tokens: int = 32,
    allow_torch_compile: bool = True,
    use_throughput_prefs: bool = True,
) -> Tuple[List[str], List[str], int, Optional[Dict[str, Any]], Any]:
    em = merge_external_model_eval_config(config)
    miron_strict = bool(UniversalTokenizer(tokenizer).is_miron)
    cap_dec = max(1, int(short_continuation_tokens))
    max_new, prompt_budget, _ = gen_context_limits_like_external(
        tokenizer, prompts, em, gen_total_tokens=gen_total_tokens, min_new_tokens=cap_dec
    )

    do_sample = bool(em.get("external_gen_do_sample", True))
    temperature = float(em.get("external_gen_temperature", 1.0))
    repetition_penalty = float(em.get("external_gen_repetition_penalty", 1.0))
    suppress_eos = bool(em.get("external_gen_suppress_eos_first_step", False))

    explicit_bs = em.get("external_gen_batch_size")
    reprobe = bool(em.get("external_gen_reprobe", False))
    max_batch_cap = _external_plateau_max_batch(em)

    cap = cap_dec

    max_new = _apply_miron_benchmark_max_new_words(max_new, tokenizer, config)

    prefs = apply_inference_throughput_prefs(device) if use_throughput_prefs else {}
    batch_search_meta: Optional[Dict[str, Any]] = None
    gen_bs = batch_size
    gen_model = model
    if allow_torch_compile:
        gen_model = _maybe_torch_compile_for_generate(model, config)
    try:
        if explicit_bs is not None:
            gen_bs = int(explicit_bs)
            batch_search_meta = {"source": "config", "external_gen_batch_size": gen_bs}
        elif reprobe:
            gen_bs, probe_meta = find_plateau_generation_batch_size(
                gen_model,
                tokenizer,
                device,
                prompts,
                max_new_tokens=max_new,
                max_batch_cap=max_batch_cap,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                seed=seed,
                suppress_eos_first_step=suppress_eos,
                warmup_rounds=int(em.get("external_gen_tune_warmup", 1)),
                max_context_tokens=prompt_budget,
            )
            batch_search_meta = {"source": "probed", **probe_meta}
        else:
            gen_bs = batch_size
            batch_search_meta = {"source": "eval_batch_size", "batch_size": gen_bs}

        engine = InferenceEngine(
            gen_model,
            tokenizer,
            device=device,
            miron_strict=miron_strict,
        )
        long_texts, pairs = engine.generate_with_prefix_gen_id_pairs(
            prompts,
            max_new_tokens=max_new,
            batch_size=gen_bs,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            seed=seed,
            desc=desc,
            suppress_eos_first_step=suppress_eos,
            max_context_tokens=prompt_budget,
        )
        short_texts = [
            decode_continuation(engine.tokenizer, pref, gen[:cap]) for pref, gen in pairs
        ]
        return long_texts, short_texts, gen_bs, batch_search_meta, gen_model
    finally:
        if use_throughput_prefs:
            restore_inference_prefs(prefs)
