"""Factual vocab / parameter counts from a saved checkpoint (not from YAML config)."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def find_checkpoint_path(project_root: str, experiment_id: str) -> Optional[str]:
    """Same resolution order as scripts/04_evaluate.find_checkpoint."""
    eid = experiment_id.replace("\\", "/")
    ckpt_dir = os.path.join(project_root, "outputs", eid, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    for name in ("last_checkpoint.pt", "best_model.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p
    step_files = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not step_files:
        return None

    def step_num(path: str) -> int:
        try:
            return int(os.path.basename(path).replace("step_", "").replace(".pt", ""))
        except ValueError:
            return 0

    return max(step_files, key=step_num)


def _safe_torch_load(path: str, map_location: str = "cpu") -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def _dedupe_unique_numel(state_dict: Dict[str, Any]) -> int:
    """Sum parameter elements counting each storage once (handles tied weights)."""
    import torch

    seen_ptrs: set[int] = set()
    total = 0
    for v in state_dict.values():
        if not isinstance(v, torch.Tensor):
            continue
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total += int(v.numel())
    return total


def _dedupe_unique_nbytes(state_dict: Dict[str, Any]) -> int:
    """Sum tensor payload bytes counting each storage once."""
    import torch

    seen_ptrs: set[int] = set()
    total = 0
    for value in state_dict.values():
        if not isinstance(value, torch.Tensor):
            continue
        ptr = value.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total += int(value.numel() * value.element_size())
    return total


def _unique_numel_for_prefixes(state_dict: Dict[str, Any], prefixes: Tuple[str, ...]) -> int:
    import torch

    seen_ptrs: set[int] = set()
    total = 0
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        ptr = value.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total += int(value.numel())
    return total


def _miron_component_params(state_dict: Dict[str, Any]) -> Dict[str, int]:
    encoder = _unique_numel_for_prefixes(state_dict, ("encoder.", "char_encoder."))
    decoder = _unique_numel_for_prefixes(state_dict, ("decoder.", "char_decoder."))
    enc_proj = _unique_numel_for_prefixes(state_dict, ("enc_proj.",))
    dec_proj = _unique_numel_for_prefixes(state_dict, ("dec_proj.",))
    core = _unique_numel_for_prefixes(state_dict, ("lm.", "core_lm."))
    out: Dict[str, int] = {}
    if encoder:
        out["miron_encoder_params_unique"] = encoder
    if decoder:
        out["miron_decoder_params_unique"] = decoder
    if enc_proj:
        out["miron_enc_proj_params_unique"] = enc_proj
    if dec_proj:
        out["miron_dec_proj_params_unique"] = dec_proj
    if core:
        out["miron_core_lm_params_unique"] = core
    if encoder or decoder or enc_proj or dec_proj:
        out["miron_char_stack_params_unique"] = encoder + decoder + enc_proj + dec_proj
    return out


def _pick_embed_key(candidates: list[str]) -> Optional[str]:
    if not candidates:
        return None
    for k in candidates:
        if k == "lm.embed_tokens.weight" or k.endswith(".lm.embed_tokens.weight"):
            return k
    return candidates[0]


def _pick_lm_head_key(candidates: list[str]) -> Optional[str]:
    if not candidates:
        return None
    for k in candidates:
        if k == "lm.lm_head.weight" or k.endswith(".lm.lm_head.weight"):
            return k
    return candidates[0]


def analyze_checkpoint_state_dict(state_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Infer token-level LM vocab / embedding+head parameter split (SmolLM2 inside root or under ``lm.``).

    Returns None if no ``embed_tokens.weight``-like key exists.
    """
    import torch

    keys = list(state_dict.keys())
    emb_k = _pick_embed_key([k for k in keys if k.endswith("embed_tokens.weight")])
    if not emb_k:
        return None
    emb = state_dict[emb_k]
    if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
        return None
    vocab_size, hidden_size = int(emb.shape[0]), int(emb.shape[1])

    head_k = _pick_lm_head_key([k for k in keys if k.endswith("lm_head.weight")])
    head = state_dict.get(head_k) if head_k else None
    tied = False
    if isinstance(head, torch.Tensor) and head.dim() == 2:
        tied = head.data_ptr() == emb.data_ptr()

    seen: set[int] = set()
    embed_head = 0
    for t in (emb, head if isinstance(head, torch.Tensor) else None):
        if t is None:
            continue
        p = t.data_ptr()
        if p in seen:
            continue
        seen.add(p)
        embed_head += int(t.numel())

    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "tied_word_embeddings": tied,
        "embed_head_params_unique": embed_head,
        "embed_tokens_key": emb_k,
        "lm_head_key": head_k,
    }


def load_checkpoint_param_stats(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from disk and return merged stats."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return None
    try:
        ckpt = _safe_torch_load(checkpoint_path, map_location="cpu")
    except Exception as e:
        logger.warning("Could not load checkpoint %s: %s", checkpoint_path, e)
        return None

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.endswith("embed_tokens.weight") for k in ckpt):
        sd = ckpt
    else:
        return None

    if not isinstance(sd, dict):
        return None

    total_unique = _dedupe_unique_numel(sd)
    total_bytes_unique = _dedupe_unique_nbytes(sd)
    lm_info = analyze_checkpoint_state_dict(sd)
    out: Dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "total_params_unique": total_unique,
        "param_bytes_unique": total_bytes_unique,
        "param_mib_unique": total_bytes_unique / (1024.0**2),
        "param_fp32_mib_unique": total_unique * 4 / (1024.0**2),
    }
    out.update(_miron_component_params(sd))
    if lm_info:
        out.update(
            {
                "vocab_size": lm_info["vocab_size"],
                "hidden_size": lm_info["hidden_size"],
                "tied_word_embeddings": lm_info["tied_word_embeddings"],
                "embed_head_params_unique": lm_info["embed_head_params_unique"],
            }
        )
        eb = lm_info["embed_head_params_unique"]
        rest = total_unique - eb
        out["rest_params_unique"] = rest if rest >= 0 else None
    return out


def checkpoint_stats_for_experiment(project_root: str, experiment_id: str) -> Optional[Dict[str, Any]]:
    path = find_checkpoint_path(project_root, experiment_id)
    if not path:
        return None
    return load_checkpoint_param_stats(path)
