"""Load outputs/<exp>/config.yaml; prefer factual vocab / params from checkpoint over YAML."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.reporting.checkpoint_param_analysis import checkpoint_stats_for_experiment


def load_output_yaml_config(
    project_root: str, outputs_root: str, experiment_id: str
) -> Optional[Dict[str, Any]]:
    if yaml is None:
        return None
    p = os.path.join(
        project_root, outputs_root, experiment_id.replace("\\", "/"), "config.yaml"
    )
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None


def vocab_hidden_tie(cfg: Dict[str, Any]) -> Tuple[int, int, bool]:
    tok = cfg.get("tokenizer") or {}
    mod = cfg.get("model") or {}
    vocab = mod.get("vocab_size")
    if vocab is None:
        vocab = tok.get("vocab_size")
    hidden = mod.get("hidden_size")
    if hidden is None:
        hidden = tok.get("d_model")
    tie = bool(mod.get("tie_word_embeddings", True))
    try:
        v = int(vocab) if vocab is not None else 0
        h = int(hidden) if hidden is not None else 0
    except (TypeError, ValueError):
        return 0, 0, True
    return v, h, tie


def embedding_lm_head_param_count(vocab: int, hidden: int, tie: bool) -> int:
    """Token embedding plus output head: shared weights if tie_word_embeddings else two matrices."""
    if vocab <= 0 or hidden <= 0:
        return 0
    return vocab * hidden if tie else 2 * vocab * hidden


def model_training_meta(
    project_root: str,
    outputs_root: str,
    experiment_id: str,
    training_summary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Returns dict with:
      pct_data, vocab_size, embed_head_params, rest_params, total_params, tied,
      param_stats_source, steps_per_epoch, last_step, epochs_completed, ...
    Prefers checkpoint-derived token LM stats; falls back to outputs/config.yaml.
    """
    cfg = load_output_yaml_config(project_root, outputs_root, experiment_id)
    v, h, tie = (0, 0, True)
    if cfg:
        v, h, tie = vocab_hidden_tie(cfg)

    ck = None
    try:
        ck = checkpoint_stats_for_experiment(project_root, experiment_id)
    except Exception:
        ck = None

    param_source = "config"
    eb: Optional[int] = None
    total: Optional[int] = None
    rest: Optional[int] = None

    if (
        ck
        and ck.get("vocab_size") is not None
        and int(ck["vocab_size"]) > 0
        and ck.get("embed_head_params_unique") is not None
    ):
        param_source = "checkpoint"
        v = int(ck["vocab_size"])
        if ck.get("hidden_size") is not None:
            h = int(ck["hidden_size"])
        tie = bool(ck.get("tied_word_embeddings", True))
        eb = int(ck["embed_head_params_unique"])
        if ck.get("total_params_unique") is not None:
            total = int(ck["total_params_unique"])
        if ck.get("rest_params_unique") is not None:
            rest = int(ck["rest_params_unique"])
        elif total is not None and eb is not None:
            r = total - eb
            rest = r if r >= 0 else None
    elif ck and ck.get("total_params_unique") is not None and v > 0 and h > 0:
        param_source = "checkpoint_total_yaml_split"
        total = int(ck["total_params_unique"])
        eb = embedding_lm_head_param_count(v, h, tie)
        r = total - eb
        rest = r if r >= 0 else None
    elif v > 0 and h > 0:
        eb = embedding_lm_head_param_count(v, h, tie)
        if training_summary:
            tp = training_summary.get("summary_num_parameters")
            if tp is not None:
                try:
                    total = int(tp)
                except (TypeError, ValueError):
                    total = None
        if total is not None and eb >= 0:
            r = total - eb
            rest = r if r >= 0 else None
    else:
        return None

    if eb is None:
        return None

    pct = None
    steps_per_epoch = None
    last_step = None
    epochs_completed = None
    max_steps_cfg = None
    frac_max = None
    if training_summary:
        p = training_summary.get("summary_pct_data_processed")
        if p is not None:
            try:
                pct = float(p)
            except (TypeError, ValueError):
                pct = None
        steps_per_epoch = training_summary.get("summary_steps_per_epoch")
        last_step = training_summary.get("summary_last_step")
        epochs_completed = training_summary.get("summary_epochs_completed")
        max_steps_cfg = training_summary.get("summary_max_steps_config")
        frac_max = training_summary.get("summary_fraction_of_config_max_steps")

    return {
        "pct_data": pct,
        "vocab_size": v,
        "embed_head_params": eb,
        "rest_params": rest,
        "total_params": total,
        "tied": tie,
        "param_stats_source": param_source,
        "hidden_size": h if h > 0 else None,
        "steps_per_epoch": steps_per_epoch,
        "last_step": last_step,
        "epochs_completed": epochs_completed,
        "max_steps_config": max_steps_cfg,
        "fraction_of_config_max_steps": frac_max,
        "miron_encoder_params": ck.get("miron_encoder_params_unique") if ck else None,
        "miron_decoder_params": ck.get("miron_decoder_params_unique") if ck else None,
        "miron_enc_proj_params": ck.get("miron_enc_proj_params_unique") if ck else None,
        "miron_dec_proj_params": ck.get("miron_dec_proj_params_unique") if ck else None,
        "miron_char_stack_params": ck.get("miron_char_stack_params_unique") if ck else None,
        "miron_core_lm_params": ck.get("miron_core_lm_params_unique") if ck else None,
        "param_mib": ck.get("param_mib_unique") if ck else None,
        "param_fp32_mib": ck.get("param_fp32_mib_unique") if ck else None,
    }
