"""Map YAML generation / human_eval layers to InferenceEngine.generate kwargs (incl. next-word mode)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.common.next_word_extract import max_new_tokens_for_tokenizer

if TYPE_CHECKING:
    from src.common.inference import InferenceEngine

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BIAS_FIRST_TOKENS: tuple[str, ...] = ("\n",)
DEFAULT_REPETITION_PENALTY_FULL = 1.15
DEFAULT_REPETITION_PENALTY_NEXT_WORD = 1.0
DEFAULT_REPETITION_PENALTY_CHAR_LEVEL_FULL = 1.0


def merge_generation_layers(config: dict) -> dict:
    out = dict(config.get("generation") or {})
    out.update(config.get("human_eval") or {})
    return out


def resolve_max_context_tokens(config: dict, gen_cfg: dict, model: Any) -> Optional[int]:
    if gen_cfg.get("max_context_tokens") is not None:
        return int(gen_cfg["max_context_tokens"])
    m = (config.get("model") or {}).get("max_position_embeddings")
    if m is not None:
        return int(m)
    mc = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if mc is not None:
        return int(mc)
    return None


def _default_repetition_penalty(config: dict, *, next_word: bool) -> float:
    if next_word:
        return DEFAULT_REPETITION_PENALTY_NEXT_WORD
    tok_type = (config.get("tokenizer") or {}).get("type") or ""
    if tok_type == "char_level":
        return DEFAULT_REPETITION_PENALTY_CHAR_LEVEL_FULL
    return DEFAULT_REPETITION_PENALTY_FULL


def resolved_max_new_tokens(gen_cfg: dict, tokenizer: Any, *, next_word: bool) -> int:
    base = int(gen_cfg.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
    if not next_word:
        return base
    return min(base, max_new_tokens_for_tokenizer(tokenizer))


def base_generate_kwargs(
    config: dict,
    model: Any,
    tokenizer: Any,
    *,
    next_word: bool,
) -> Dict[str, Any]:
    gen = merge_generation_layers(config)
    rep_def = _default_repetition_penalty(config, next_word=next_word)
    return {
        "max_new_tokens": resolved_max_new_tokens(gen, tokenizer, next_word=next_word),
        "max_context_tokens": resolve_max_context_tokens(config, gen, model),
        "do_sample": gen.get("do_sample", False),
        "temperature": float(gen.get("temperature", 0.7)),
        "repetition_penalty": float(gen.get("repetition_penalty", rep_def)),
    }


def generate_continuations_short(
    engine: InferenceEngine,
    config: dict,
    model: Any,
    prompts: List[str],
    batch_size: int,
    *,
    seed: Optional[int],
    desc: str,
    kw_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], int]:
    kw = base_generate_kwargs(config, model, engine.tokenizer, next_word=True)
    if kw_overrides:
        kw.update(kw_overrides)
    max_new = int(kw["max_new_tokens"])
    out = engine.generate(
        prompts=prompts,
        batch_size=batch_size,
        desc=desc,
        seed=seed,
        **kw,
    )
    return out, max_new
