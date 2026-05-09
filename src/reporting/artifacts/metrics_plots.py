"""Plots that read logs/metrics.jsonl."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt

from src.scripts.utils.plotting import extract_metric_series, load_metrics, normalize_metrics
from src.visualization import ProfessionalPlotter
from src.visualization.plotting import (
    CE_YLIM,
    FIG_SINGLE,
    PPL_YLIM,
    THEME,
    apply_axis_style,
    metric_ylim_minimum_band,
)

if TYPE_CHECKING:
    from src.reporting.context import ReportContext, ReportOutputDirs

logger = logging.getLogger(__name__)


def _load_training_summary_steps(ctx: "ReportContext") -> Tuple[Optional[float], Optional[float]]:
    p = os.path.join(ctx.eval_dir, "training_summary.json")
    if not os.path.isfile(p):
        return None, None
    try:
        with open(p, encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None
    spe = data.get("summary_steps_per_epoch")
    ls = data.get("summary_last_step")
    try:
        spe_f = float(spe) if spe is not None else None
    except (TypeError, ValueError):
        spe_f = None
    try:
        ls_f = float(ls) if ls is not None else None
    except (TypeError, ValueError):
        ls_f = None
    return spe_f, ls_f


def build_training_curves(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = ctx.metrics_jsonl
    if not os.path.isfile(path):
        return False
    rows = load_metrics(path)
    if not rows:
        logger.warning("Empty metrics: %s", path)
        return False
    train_rows, val_rows = normalize_metrics(rows)
    plotter = ProfessionalPlotter(ctx.experiment_id, out.plots_dir)

    t_steps, t_losses = extract_metric_series(train_rows, "train_loss")
    v_steps, v_losses = extract_metric_series(val_rows, "val_loss")
    plotted = False
    if t_losses and v_losses:
        plotter.plot_learning_curves(
            t_steps,
            t_losses,
            v_steps,
            v_losses,
            "Loss",
            "Cross Entropy",
            ylim_train=metric_ylim_minimum_band(t_losses, CE_YLIM[0], CE_YLIM[1]),
            ylim_val=metric_ylim_minimum_band(v_losses, CE_YLIM[0], CE_YLIM[1]),
        )
        plotted = True
    else:
        if t_losses:
            plotter.plot_curve(
                t_steps[: len(t_losses)],
                t_losses,
                "Train Loss",
                "Cross Entropy",
                "train_loss",
                color=THEME["train"],
                ylim=metric_ylim_minimum_band(t_losses, CE_YLIM[0], CE_YLIM[1]),
            )
            plotted = True
        if v_losses:
            plotter.plot_curve(
                v_steps[: len(v_losses)],
                v_losses,
                "Val Loss",
                "Cross Entropy",
                "val_loss",
                color=THEME["val"],
                ylim=metric_ylim_minimum_band(v_losses, CE_YLIM[0], CE_YLIM[1]),
            )
            plotted = True

    t_steps_p, t_ppls = extract_metric_series(train_rows, "train_ppl")
    v_steps_p, v_ppls = extract_metric_series(val_rows, "val_ppl")
    if t_ppls and v_ppls:
        plotter.plot_learning_curves(
            t_steps_p[: len(t_ppls)],
            t_ppls,
            v_steps_p[: len(v_ppls)],
            v_ppls,
            "Perplexity",
            "PPL",
            ylim_train=metric_ylim_minimum_band(t_ppls, PPL_YLIM[0], PPL_YLIM[1]),
            ylim_val=metric_ylim_minimum_band(v_ppls, PPL_YLIM[0], PPL_YLIM[1]),
        )
        plotted = True
    else:
        if t_ppls:
            plotter.plot_curve(
                t_steps_p[: len(t_ppls)],
                t_ppls,
                "Train Perplexity",
                "PPL",
                "train_perplexity",
                color=THEME["train"],
                ylim=metric_ylim_minimum_band(t_ppls, PPL_YLIM[0], PPL_YLIM[1]),
            )
            plotted = True
        if v_ppls:
            plotter.plot_curve(
                v_steps_p[: len(v_ppls)],
                v_ppls,
                "Val Perplexity",
                "PPL",
                "val_perplexity",
                color=THEME["val"],
                ylim=metric_ylim_minimum_band(v_ppls, PPL_YLIM[0], PPL_YLIM[1]),
            )
            plotted = True

    spe, last_s = _load_training_summary_steps(ctx)
    if v_losses and v_steps:
        plotter.plot_val_metric_with_training_progress_bands(
            v_steps[: len(v_losses)],
            v_losses,
            "Loss",
            "Cross Entropy",
            "val_loss_progress",
            steps_per_epoch=spe,
            last_step_hint=last_s,
            ylim=metric_ylim_minimum_band(v_losses, CE_YLIM[0], CE_YLIM[1]),
        )
        plotted = True
    if v_ppls and v_steps_p:
        plotter.plot_val_metric_with_training_progress_bands(
            v_steps_p[: len(v_ppls)],
            v_ppls,
            "Perplexity",
            "PPL",
            "val_perplexity_progress",
            steps_per_epoch=spe,
            last_step_hint=last_s,
            ylim=metric_ylim_minimum_band(v_ppls, PPL_YLIM[0], PPL_YLIM[1]),
        )
        plotted = True

    if _build_miron_diagnostics(ctx, out, rows):
        plotted = True

    return plotted


def _series_by_key(rows: list[Dict[str, Any]], *names: str) -> tuple[list[float], list[float]]:
    """Extract a metric series by case-insensitive exact key names."""
    wanted = {n.lower() for n in names}
    steps: list[float] = []
    values: list[float] = []
    for row in rows:
        step = row.get("step")
        if step is None:
            continue
        lower = {str(k).lower(): v for k, v in row.items()}
        value = None
        for name in wanted:
            if name in lower:
                value = lower[name]
                break
        if value is None:
            continue
        try:
            steps.append(float(step))
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return steps, values


def _build_miron_diagnostics(
    ctx: "ReportContext", out: "ReportOutputDirs", rows: list[Dict[str, Any]]
) -> bool:
    """Extra MIRON plots: char accuracy, gradient split, throughput, latent stability."""
    if "miron" not in ctx.experiment_id.lower():
        return False

    plotter = ProfessionalPlotter(ctx.experiment_id, out.plots_dir)
    plotted = False

    steps, acc = _series_by_key(rows, "Train/Lm_char_acc", "train/lm_char_acc")
    if acc:
        plotter.plot_curve(
            steps,
            acc,
            "MIRON char accuracy",
            "Character accuracy",
            "miron_char_accuracy",
            color=THEME["accent"],
            ylim=metric_ylim_minimum_band(acc, 0.0, 1.0),
        )
        plotted = True

    steps, tps = _series_by_key(rows, "Speed/TPS", "tokens_per_sec")
    if tps:
        plotter.plot_technical_metric(steps, tps, "MIRON training throughput", "Tokens/s", "miron_throughput")
        plotted = True

    grad_specs = [
        ("Grad/gn_encoder", "Encoder", THEME["train"]),
        ("Grad/gn_lm", "Core LM", THEME["accent"]),
        ("Grad/gn_decoder", "Decoder", THEME["val"]),
        ("Grad/gn_total", "Total", THEME["neutral"]),
    ]
    grad_series = []
    for key, label, color in grad_specs:
        s, v = _series_by_key(rows, key)
        if v:
            grad_series.append((s, v, label, color))
    if grad_series:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        for s, v, label, color in grad_series:
            n = min(len(s), len(v))
            ax.plot(s[:n], v[:n], label=label, color=color, linewidth=1.25)
        ax.set_title(f"Gradient norms — {ctx.experiment_id}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Global norm")
        ax.legend(loc="best", fontsize=8)
        apply_axis_style(ax)
        plotter._save_fig(fig, "miron_gradient_norms")
        plotted = True

    val_specs = [
        ("Val/Char_z_eff_rank", "Effective rank", THEME["accent"]),
        ("Val/Char_z_cov_k90_dim", "90% covariance rank", THEME["train"]),
        ("Val/Char_z_cov_k95_dim", "95% covariance rank", THEME["val"]),
    ]
    val_series = []
    for key, label, color in val_specs:
        s, v = _series_by_key(rows, key)
        if v:
            val_series.append((s, v, label, color))
    if val_series:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        for s, v, label, color in val_series:
            n = min(len(s), len(v))
            ax.plot(s[:n], v[:n], label=label, color=color, marker="o", linewidth=1.25)
        ax.set_title(f"Latent space diagnostics — {ctx.experiment_id}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Rank / dimension")
        ax.legend(loc="best", fontsize=8)
        apply_axis_style(ax)
        plotter._save_fig(fig, "miron_latent_space")
        plotted = True

    return plotted


def build_learning_rate(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    """Learning rate schedule only (no throughput)."""
    path = ctx.metrics_jsonl
    if not os.path.isfile(path):
        return False
    rows = load_metrics(path)
    if not rows:
        return False
    train_rows, _ = normalize_metrics(rows)
    plotter = ProfessionalPlotter(ctx.experiment_id, out.plots_dir)

    t_steps, lrs = extract_metric_series(train_rows, "lr")
    if not lrs:
        return False
    plotter.plot_technical_metric(t_steps[: len(lrs)], lrs, "Learning Rate", "LR", "lr")
    return True
