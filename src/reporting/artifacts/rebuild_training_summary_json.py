"""Rewrite outputs/.../evaluation/training_summary.json from logs/metrics.jsonl (no model load)."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from src.io.config import get_full_config
from src.scripts.utils.evaluation import build_training_summary

if TYPE_CHECKING:
    from src.reporting.context import ReportContext, ReportOutputDirs

logger = logging.getLogger(__name__)


def rebuild_training_summary_json(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    if not os.path.isfile(ctx.metrics_jsonl):
        return False
    eid = ctx.experiment_id.replace("\\", "/")
    cfg_path = os.path.join(ctx.project_root, "configs", "experiments", *eid.split("/")) + ".yaml"
    config: dict = {}
    if os.path.isfile(cfg_path):
        try:
            config = get_full_config(cfg_path, ctx.project_root)
        except Exception:
            logger.warning("Could not load config %s; using empty config for summary", cfg_path)
    summary = build_training_summary(ctx.project_root, eid, config, model=None)
    if not summary:
        return False
    os.makedirs(ctx.eval_dir, exist_ok=True)
    path = os.path.join(ctx.eval_dir, "training_summary.json")
    prev: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                prev = json.load(f)
        except (json.JSONDecodeError, OSError):
            prev = {}
    p = prev.get("summary_num_parameters")
    if isinstance(p, int) and "summary_num_parameters" not in summary:
        summary["summary_num_parameters"] = p
    for k, v in prev.items():
        if k.startswith("summary_checkpoint_") and k not in summary:
            summary[k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return True
