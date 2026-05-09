"""Plots that read outputs/.../evaluation/*.json."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.visualization import ProfessionalPlotter

if TYPE_CHECKING:
    from src.reporting.context import ReportContext, ReportOutputDirs

logger = logging.getLogger(__name__)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Cannot read %s: %s", path, e)
        return None


def _primary_perplexity_block(ppl_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not ppl_results:
        return None
    for key in ("test_indomain", "test"):
        block = ppl_results.get(key)
        if isinstance(block, dict):
            return block
    first = next(iter(ppl_results.values()), None)
    return first if isinstance(first, dict) else None


def build_perplexity_comparison(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "perplexity_results.json")
    data = _load_json(path)
    if not data:
        return False
    ProfessionalPlotter(ctx.experiment_id, out.plots_dir).plot_perplexity_comparison(data)
    return True


def build_token_dist_vocab(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "perplexity_results.json")
    data = _load_json(path)
    if not data:
        return False
    block = _primary_perplexity_block(data)
    if not block:
        return False
    dist = block.get("token_length_dist_vocab") or {}
    if not dist:
        return False
    ProfessionalPlotter(ctx.experiment_id, out.plots_dir).plot_token_distribution(
        dist, "Vocabulary Token Length Distribution", "token_dist_vocab"
    )
    return True


def build_token_dist_text(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "perplexity_results.json")
    data = _load_json(path)
    if not data:
        return False
    block = _primary_perplexity_block(data)
    if not block:
        return False
    dist = block.get("token_length_dist_text") or {}
    if not dist:
        return False
    ProfessionalPlotter(ctx.experiment_id, out.plots_dir).plot_token_distribution(
        dist, "Text Tokenization Length Distribution", "token_dist_text"
    )
    return True


def build_token_length_vocab_vs_text(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "perplexity_results.json")
    data = _load_json(path)
    if not data:
        return False
    block = _primary_perplexity_block(data)
    if not block:
        return False
    dv = block.get("token_length_dist_vocab") or {}
    dt = block.get("token_length_dist_text") or {}
    if not dv and not dt:
        return False
    ProfessionalPlotter(ctx.experiment_id, out.plots_dir).plot_token_length_vocab_vs_text(dv, dt)
    return True


def build_blimp(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "blimp_results.json")
    data = _load_json(path)
    if not data:
        return False
    ProfessionalPlotter(ctx.experiment_id, out.plots_dir).plot_blimp_results(data)
    return True


def build_lambada(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "lambada_results.json")
    data = _load_json(path)
    if not data:
        return False
    ProfessionalPlotter(ctx.experiment_id, out.plots_dir).plot_lambada_results(data)
    return True


def build_canary(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "canary_results.json")
    data = _load_json(path)
    if not data:
        return False
    ProfessionalPlotter(ctx.experiment_id, out.plots_dir).plot_canary_results(data)
    return True


def build_training_summary(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "training_summary.json")
    data = _load_json(path)
    if not data:
        return False
    ProfessionalPlotter(ctx.experiment_id, out.plots_dir).plot_training_summary(data)
    return True
