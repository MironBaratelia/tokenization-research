"""Resolve training max_steps without importing training scripts (Hydra / logging side effects)."""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def resolve_training_max_steps(full_config: Dict[str, Any], train_loader) -> int:
    """
    Positive training.max_steps wins. Otherwise use one optimizer pass over train_loader
    (len(train_loader) // gradient_accumulation_steps).

    Structured configs often default max_steps to -1; that must not silently cap long experiments.
    """
    tc = full_config.get("training", {}) or {}
    raw_ms = tc.get("max_steps", 0)
    try:
        ms_i = int(raw_ms)
    except (TypeError, ValueError):
        ms_i = 0
    if ms_i > 0:
        full_config["training"]["max_steps"] = ms_i
        logger.info("Using configured max_steps=%s", ms_i)
        return ms_i

    grad_accum = int(tc.get("gradient_accumulation_steps", 1) or 1)
    max_steps = len(train_loader) // max(1, grad_accum)
    full_config["training"]["max_steps"] = max_steps
    logger.info(
        "training.max_steps is unset or non-positive (%r); targeting 1 epoch: %s optimizer steps (grad_accum=%s)",
        raw_ms,
        max_steps,
        grad_accum,
    )
    return max_steps
