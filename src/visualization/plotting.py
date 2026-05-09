import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

CE_YLIM = (0.0, 3.0)
PPL_YLIM = (1.0, math.exp(3.0))


def metric_ylim_minimum_band(
    values: Sequence[float],
    base_lo: float,
    base_hi: float,
    coverage: float = 0.95,
    pad_frac: float = 0.04,
) -> Tuple[float, float]:
    """
    Use at least [base_lo, base_hi] on the y-axis. If fewer than ``coverage`` of the
    points fall inside that band, raise the upper limit (rarely the lower) so at least
    that fraction of points is visible — the band is a floor, not a hard cap.
    """
    if not values:
        return base_lo, base_hi
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    need = max(1, int(math.ceil(coverage * n)))
    in_band = int(np.sum((arr >= base_lo) & (arr <= base_hi)))
    if in_band >= need:
        return base_lo, base_hi

    lo = float(base_lo)

    def _count_upto(hi: float) -> int:
        return int(np.sum((arr >= lo) & (arr <= hi)))

    if _count_upto(base_hi) >= need:
        return base_lo, base_hi

    vmax = float(np.max(arr))
    lo_h, hi_h = float(base_hi), max(float(base_hi), vmax)
    guard = 0
    while _count_upto(hi_h) < need and guard < 60:
        hi_h = hi_h * 1.3 + 1e-6
        guard += 1
    for _ in range(90):
        mid = (lo_h + hi_h) / 2.0
        if _count_upto(mid) >= need:
            hi_h = mid
        else:
            lo_h = mid
    span = max(hi_h - lo, 1e-9)
    margin = max(span * pad_frac, abs(hi_h) * 1e-5, 1e-6)
    return lo, hi_h + margin


THEME: Dict[str, str] = {
    "train": "#0072B2",
    "val": "#D55E00",
    "accent": "#009E73",
    "accent2": "#CC79A7",
    "neutral": "#666666",
    "grid": "#B0B0B0",
    "spine": "#333333",
    "bar": "#56B4E9",
    "bar_alt": "#F0E442",
    "bar_edge": "#222222",
    "hist_fill": "#56B4E9",
    "hist_fill2": "#E69F00",
}

FIG_SINGLE = (7.0, 4.0)
FIG_LR = (7.0, 3.5)
FIG_WIDE = (14.0, 4.25)
FIG_BLIMP = (11.0, 7.5)


def set_scientific_style() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": THEME["spine"],
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "grid.alpha": 0.45,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": THEME["grid"],
        }
    )


def apply_axis_style(ax, grid_axis: str = "both") -> None:
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis=grid_axis,
        linestyle="--",
        linewidth=0.85,
        alpha=0.45,
        color=THEME["grid"],
    )
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=0.85,
        colors=THEME["spine"],
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(THEME["spine"])
        ax.spines[side].set_linewidth(0.85)


class ProfessionalPlotter:
    def __init__(self, exp_id: str, output_dir: str):
        self.exp_id = exp_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        set_scientific_style()

    def _title(self, main: str) -> str:
        return f"{main} — {self.exp_id}"

    def _get_y_limits(
        self, values: List[float], percentile: float = 95
    ) -> Optional[tuple]:
        if not values:
            return None
        if len(values) < 5:
            return min(values) * 0.9, max(values) * 1.1

        stable_values = values[max(1, len(values) // 20) :]
        if not stable_values:
            stable_values = values

        p_high = np.percentile(stable_values, percentile)
        p_low = np.percentile(stable_values, 100 - percentile)

        last_val = values[-1]

        y_max = max(p_high, last_val)
        y_min = min(p_low, last_val)

        if y_min > 0 and y_min < (y_max - y_min) * 0.5:
            y_min = 0

        y_range = y_max - y_min
        if y_range == 0:
            y_range = y_max * 0.1 or 0.1

        y_min_final = y_min - (y_range * 0.05) if y_min > 0 else 0
        y_max_final = y_max + (y_range * 0.1)

        return y_min_final, y_max_final

    def plot_curve(
        self,
        steps,
        values,
        label: str,
        ylabel: str,
        filename: str,
        color: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        color = color or THEME["train"]
        fig, ax = plt.subplots(figsize=FIG_SINGLE)

        limits = None if ylim is not None else self._get_y_limits(values)

        n = min(len(steps), len(values))
        ax.plot(
            steps[:n],
            values[:n],
            label=label,
            color=color,
            linewidth=1.35,
            alpha=1.0,
        )

        if ylim is not None:
            ax.set_ylim(ylim)
        elif limits:
            ax.set_ylim(limits)

        if len(steps) > 1:
            x_min, x_max = min(steps), max(steps)
            x_range = x_max - x_min
            ax.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)

        ax.set_title(self._title(label))
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        apply_axis_style(ax)

        self._save_fig(fig, filename)

    def plot_learning_curves(
        self,
        train_steps,
        train_values,
        val_steps,
        val_values,
        metric_name: str,
        ylabel: str,
        ylim: Optional[Tuple[float, float]] = None,
        ylim_train: Optional[Tuple[float, float]] = None,
        ylim_val: Optional[Tuple[float, float]] = None,
    ):
        stem = metric_name.lower().replace(" ", "_")
        yt = ylim_train if ylim_train is not None else ylim
        yv = ylim_val if ylim_val is not None else ylim
        self.plot_curve(
            train_steps,
            train_values,
            f"Train {metric_name}",
            ylabel,
            f"train_{stem}",
            color=THEME["train"],
            ylim=yt,
        )
        self.plot_curve(
            val_steps,
            val_values,
            f"Val {metric_name}",
            ylabel,
            f"val_{stem}",
            color=THEME["val"],
            ylim=yv,
        )

    def plot_val_metric_with_training_progress_bands(
        self,
        steps: Sequence[float],
        values: Sequence[float],
        metric_name: str,
        ylabel: str,
        filename: str,
        *,
        steps_per_epoch: Optional[float] = None,
        last_step_hint: Optional[float] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Plot validation metric over time with quartile bands."""
        if not steps or not values:
            return
        n = min(len(steps), len(values))
        if n < 2:
            return
        st = list(steps[:n])
        va = list(values[:n])
        color = THEME["val"]
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        last_step = (
            float(last_step_hint) if last_step_hint is not None else float(max(st))
        )
        band_colors = ("#E8EEF5", "#E5F0E8", "#F5E8EE", "#F0EBF5")
        for q in range(4):
            x0 = last_step * (q / 4.0)
            x1 = last_step * ((q + 1) / 4.0)
            ax.axvspan(
                x0, x1, facecolor=band_colors[q], edgecolor="none", alpha=1.0, zorder=0
            )
        ax.plot(
            st, va, label=f"Val {metric_name}", color=color, linewidth=1.35, zorder=2
        )
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            limits = self._get_y_limits(va)
            if limits:
                ax.set_ylim(limits)
        if steps_per_epoch is not None:
            try:
                spe = float(steps_per_epoch)
                if spe > 0:
                    k = 1
                    while k * spe <= last_step + 1e-9:
                        ax.axvline(
                            k * spe,
                            color=THEME["neutral"],
                            linestyle=":",
                            linewidth=0.95,
                            alpha=0.65,
                            zorder=1,
                        )
                        k += 1
            except (TypeError, ValueError):
                pass
        x_min, x_max = min(st), max(st)
        x_range = max(x_max - x_min, 1e-9)
        ax.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)
        ax.set_title(self._title(f"Val {metric_name} (training progress)"))
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        apply_axis_style(ax)
        note_parts = ["Shaded: run quartiles by step."]
        if steps_per_epoch is not None:
            try:
                spe = float(steps_per_epoch)
                if spe > 0 and last_step > 0:
                    note_parts.append(f"≈ {last_step / spe:.2f} epochs.")
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        fig.text(
            0.5,
            0.02,
            " ".join(note_parts),
            ha="center",
            fontsize=7.5,
            color=THEME["neutral"],
        )
        fig.subplots_adjust(bottom=0.18)
        self._save_fig(fig, filename)

    def plot_token_distribution(
        self, dist_dict: Dict[str, int], title: str, filename: str
    ):
        if not dist_dict:
            return

        pairs: List[Tuple[int, int]] = []
        for k, v in dist_dict.items():
            if not isinstance(v, int):
                continue
            if isinstance(k, str) and k.startswith("n_chars="):
                pairs.append((int(k.split("=", 1)[1]), v))
            else:
                try:
                    pairs.append((int(k), v))
                except (TypeError, ValueError):
                    continue
        if not pairs:
            return
        pairs.sort(key=lambda x: x[0])
        pairs = [p for p in pairs if p[0] > 0]
        if not pairs:
            return
        lengths = [p[0] for p in pairs]
        counts = [p[1] for p in pairs]

        max_count = max(counts) if counts else 1

        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        bars = ax.bar(
            lengths,
            counts,
            color=THEME["bar"],
            edgecolor=THEME["bar_edge"],
            linewidth=0.6,
            alpha=0.92,
        )

        ax.set_title(self._title(f"{title} (length $>$ 0)"))
        ax.set_xlabel("Token length (chars)")
        ax.set_ylabel("Count")
        apply_axis_style(ax, "y")

        if len(lengths) <= 30:
            for bar in bars:
                height = bar.get_height()
                if height > max_count * 0.01:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=THEME["spine"],
                    )

        if max_count > 0:
            ax.set_ylim(0, max_count * 1.1)

        self._save_fig(fig, filename)

    def plot_technical_metric(
        self, steps, values, title: str, ylabel: str, filename: str
    ):
        fig, ax = plt.subplots(figsize=FIG_LR)
        ax.plot(steps, values, color=THEME["accent"], linewidth=1.5)

        limits = self._get_y_limits(values)
        if limits:
            ax.set_ylim(limits)

        if len(steps) > 1:
            x_min, x_max = min(steps), max(steps)
            x_range = x_max - x_min
            ax.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)

        ax.set_title(self._title(title))
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        apply_axis_style(ax)

        self._save_fig(fig, filename)

    def plot_blimp_results(self, blimp_results: Dict):
        if "total_accuracy" not in blimp_results:
            return

        phenomena = [k for k in blimp_results.keys() if k != "total_accuracy"]
        accuracies = []
        for k in phenomena:
            v = blimp_results[k]
            if isinstance(v, dict):
                accuracies.append(v.get("accuracy", 0))
            else:
                accuracies.append(v)

        sorted_pairs = sorted(zip(phenomena, accuracies), key=lambda x: x[1])
        sorted_phenomena, sorted_accuracies = (
            zip(*sorted_pairs) if sorted_pairs else ([], [])
        )

        fig, ax = plt.subplots(figsize=FIG_BLIMP)
        y_pos = np.arange(len(sorted_phenomena))
        ax.barh(
            y_pos,
            list(sorted_accuracies),
            color=THEME["bar"],
            edgecolor=THEME["bar_edge"],
            linewidth=0.5,
            alpha=0.95,
            height=0.72,
        )

        ax.axvline(
            x=50,
            color=THEME["neutral"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
            label="Random baseline",
        )
        tot = float(blimp_results.get("total_accuracy", 0))
        ax.axvline(
            x=tot,
            color=THEME["val"],
            linestyle="-",
            linewidth=1.8,
            label=f"Mean: {tot:.2f}\\%",
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(sorted_phenomena), fontsize=8)
        ax.set_xlabel("Accuracy (\\%)")
        ax.set_title(self._title("BLiMP by phenomenon"))
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=9)
        apply_axis_style(ax, "x")

        self._save_fig(fig, "blimp_results")

    def plot_lambada_results(self, lambada_results: Dict):
        if "detailed_results" not in lambada_results:
            return

        detailed = lambada_results["detailed_results"]
        if not detailed:
            return

        accuracies = [r.get("accuracy", 0) for r in detailed if "accuracy" in r]
        confidences = [
            r.get("target_confidence", 0) for r in detailed if "target_confidence" in r
        ]

        if not accuracies:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_WIDE)

        ax1.hist(
            accuracies,
            bins=40,
            color=THEME["hist_fill"],
            edgecolor=THEME["bar_edge"],
            linewidth=0.5,
            alpha=0.88,
        )
        ax1.set_xlabel("Accuracy")
        ax1.set_ylabel("Count")
        ax1.set_title(self._title("LAMBADA: accuracy"))
        m_acc = float(lambada_results.get("accuracy", 0))
        ax1.axvline(
            x=m_acc,
            color=THEME["val"],
            linestyle="--",
            linewidth=1.6,
            label=f"Mean: {m_acc:.4f}",
        )
        ax1.legend(loc="best", fontsize=9)
        apply_axis_style(ax1)

        if confidences:
            ax2.hist(
                confidences,
                bins=40,
                color=THEME["hist_fill2"],
                edgecolor=THEME["bar_edge"],
                linewidth=0.5,
                alpha=0.88,
            )
            ax2.set_xlabel("Target confidence")
            ax2.set_ylabel("Count")
            ax2.set_title(self._title("LAMBADA: confidence"))
            m_cf = float(lambada_results.get("confidence", 0))
            ax2.axvline(
                x=m_cf,
                color=THEME["val"],
                linestyle="--",
                linewidth=1.6,
                label=f"Mean: {m_cf:.2f}",
            )
            ax2.legend(loc="best", fontsize=9)
            apply_axis_style(ax2)

        fig.tight_layout(pad=1.2)
        self._save_fig(fig, "lambada_results")

    def plot_training_summary(self, summary: Dict):
        quality_metrics: Dict[int, Dict[str, Any]] = {}
        for pct in (80, 90, 95, 99):
            step_key = f"summary_step_to_{pct}pct_quality"
            time_key = f"summary_time_to_{pct}pct_quality_sec"
            if step_key in summary:
                quality_metrics[pct] = {
                    "step": summary[step_key],
                    "time_sec": summary.get(time_key, 0),
                }

        if not quality_metrics:
            return

        percentages = sorted(quality_metrics.keys())
        steps = [quality_metrics[p]["step"] for p in percentages]
        times_hours = [quality_metrics[p]["time_sec"] / 3600.0 for p in percentages]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_WIDE)

        ax1.plot(
            percentages,
            steps,
            marker="o",
            linewidth=1.6,
            markersize=7,
            color=THEME["train"],
        )
        ax1.set_xlabel("Quality threshold (\\%)")
        ax1.set_ylabel("Training step")
        ax1.set_title(self._title("Steps to quality"))
        apply_axis_style(ax1)
        for p, s in zip(percentages, steps):
            ax1.text(
                p,
                s,
                f"{s:,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=THEME["spine"],
            )

        ax2.plot(
            percentages,
            times_hours,
            marker="s",
            linewidth=1.6,
            markersize=7,
            color=THEME["val"],
        )
        ax2.set_xlabel("Quality threshold (\\%)")
        ax2.set_ylabel("Time (h)")
        ax2.set_title(self._title("Time to quality"))
        apply_axis_style(ax2)
        for p, t in zip(percentages, times_hours):
            ax2.text(
                p,
                t,
                f"{t:.1f}h",
                ha="center",
                va="bottom",
                fontsize=8,
                color=THEME["spine"],
            )

        fig.tight_layout(pad=1.2)
        self._save_fig(fig, "training_quality_thresholds")

    def plot_canary_results(self, canary_results: Dict[str, Any]):
        rows: List[Tuple[str, float]] = []
        for key, value in sorted(canary_results.items()):
            if not key.startswith("accuracy_"):
                continue
            if isinstance(value, (int, float)):
                v = float(value)
                pct = v * 100.0 if v <= 1.0 else v
                label = key.replace("accuracy_", "", 1)
                rows.append((label, pct))

        if not rows:
            return

        labels = [r[0] for r in rows]
        accs = [r[1] for r in rows]

        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        x = np.arange(len(labels))
        colors = [
            THEME["bar"] if i % 2 == 0 else THEME["bar_alt"] for i in range(len(labels))
        ]
        bars = ax.bar(
            x,
            accs,
            color=colors,
            edgecolor=THEME["bar_edge"],
            linewidth=0.6,
            alpha=0.92,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy (\\%)")
        ax.set_title(self._title("Canary extraction"))
        ax.set_ylim(0, max(100.0, max(accs, default=0) * 1.12))
        apply_axis_style(ax, "y")

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=THEME["spine"],
            )

        self._save_fig(fig, "canary_results")

    def plot_perplexity_comparison(self, ppl_results: Dict):
        splits: List[str] = []
        ppls: List[float] = []

        for split_name, split_data in ppl_results.items():
            if isinstance(split_data, dict) and "ppl" in split_data:
                splits.append(split_name.replace("_", " ").title())
                ppls.append(float(split_data["ppl"]))

        if not splits:
            return

        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        x = np.arange(len(splits))
        bars = ax.bar(
            x,
            ppls,
            color=THEME["bar"],
            edgecolor=THEME["bar_edge"],
            linewidth=0.6,
            alpha=0.92,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(splits, rotation=20, ha="right")
        ax.set_ylabel("Perplexity")
        ax.set_title(self._title("Perplexity by split"))
        apply_axis_style(ax, "y")

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=THEME["spine"],
            )

        self._save_fig(fig, "perplexity_comparison")

    def plot_token_length_vocab_vs_text(
        self,
        dist_vocab: Dict[str, Any],
        dist_text: Dict[str, Any],
        filename: str = "token_length_vocab_vs_text",
    ):
        def _parse(dist: Dict[str, Any]) -> List[Tuple[int, int]]:
            pairs: List[Tuple[int, int]] = []
            for k, v in dist.items():
                if not isinstance(v, (int, float)) or int(v) <= 0:
                    continue
                if isinstance(k, str) and k.startswith("n_chars="):
                    try:
                        n = int(k.split("=", 1)[1])
                    except ValueError:
                        continue
                else:
                    try:
                        n = int(k)
                    except (TypeError, ValueError):
                        continue
                pairs.append((n, int(v)))
            return pairs

        v_pairs = [p for p in _parse(dist_vocab or {}) if p[0] > 0]
        t_pairs = [p for p in _parse(dist_text or {}) if p[0] > 0]
        if not v_pairs and not t_pairs:
            return

        def _normalize(pairs: List[Tuple[int, int]]) -> Tuple[List[int], List[float]]:
            if not pairs:
                return [], []
            total = sum(c for _, c in pairs)
            if total <= 0:
                return [], []
            pairs = sorted(pairs, key=lambda x: x[0])
            ks = [p[0] for p in pairs]
            ps = [p[1] / total for p in pairs]
            return ks, ps

        v_k, v_p = _normalize(v_pairs)
        t_k, t_p = _normalize(t_pairs)

        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        if v_k:
            ax.plot(
                v_k,
                v_p,
                marker="o",
                label="Vocabulary (types)",
                color=THEME["train"],
                linewidth=1.5,
            )
        if t_k:
            ax.plot(
                t_k,
                t_p,
                marker="s",
                label="Corpus (occurrences)",
                color=THEME["val"],
                linewidth=1.5,
            )

        ax.set_title(self._title("Token decode length (chars, excl. empty)"))
        ax.set_xlabel("Decoded length (Unicode code points)")
        ax.set_ylabel("Probability mass")
        ax.legend(loc="best", fontsize=9)
        apply_axis_style(ax)
        ax.set_ylim(
            0.0,
            max(max(v_p, default=0.0), max(t_p, default=0.0)) * 1.15 or 1.0,
        )

        self._save_fig(fig, filename)

    def _save_fig(self, fig, name: str) -> None:
        pdf_path = os.path.join(self.output_dir, f"{name}.pdf")
        png_path = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(pdf_path, facecolor="white", edgecolor="none")
        fig.savefig(png_path, facecolor="white", edgecolor="none", dpi=300)
        plt.close(fig)
