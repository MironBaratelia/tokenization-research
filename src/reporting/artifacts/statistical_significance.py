"""Bootstrap confidence intervals and paired model comparisons for report metrics."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MAIN_MODELS: Tuple[str, ...] = (
    "bpe_32k",
    "byte_bpe_32k",
    "byte_bpe_8k",
    "char",
    "miron",
    "morpheme",
    "unigram_32k",
    "wordpiece_32k",
)

N_BOOT = 5000
SEED = 42


@dataclass(frozen=True)
class MetricSamples:
    language: str
    model: str
    metric: str
    values: np.ndarray
    higher_is_better: bool


def _read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    out = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return np.asarray(out, dtype=np.float64)


def _ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> Dict[str, float]:
    n = int(values.size)
    mean = float(np.mean(values))
    if n <= 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": n}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high), "n": n}


def _paired_diff_ci(
    best: np.ndarray,
    other: np.ndarray,
    rng: np.random.Generator,
    higher_is_better: bool,
    n_boot: int = N_BOOT,
) -> Dict[str, float]:
    n = min(int(best.size), int(other.size))
    if n <= 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "n": 0}
    oriented = best[:n] - other[:n] if higher_is_better else other[:n] - best[:n]
    diff = float(np.mean(oriented))
    if n == 1:
        p = 0.0 if diff > 0 else 1.0
        return {"diff": diff, "ci_low": diff, "ci_high": diff, "p_value": p, "n": n}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = oriented[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    p_left = float(np.mean(boot <= 0.0))
    p_right = float(np.mean(boot >= 0.0))
    p_value = min(1.0, 2.0 * min(p_left, p_right))
    return {"diff": diff, "ci_low": float(low), "ci_high": float(high), "p_value": p_value, "n": n}


def _extract_miron(language: str, model: str, data: Dict[str, Any]) -> List[MetricSamples]:
    samples: List[MetricSamples] = []
    for section, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
        rows = section_data.get("detailed_results")
        if not isinstance(rows, list):
            continue
        for key in ("exact_match", "lev_score_thresholded", "target_confidence"):
            values = _finite_array(row.get(key) for row in rows)
            if values.size:
                samples.append(MetricSamples(language, model, f"miron_{section}_{key}", values, True))
    return samples


def _extract_lambada(language: str, model: str, data: Dict[str, Any]) -> List[MetricSamples]:
    rows = data.get("detailed_results")
    if not isinstance(rows, list):
        return []
    out = []
    for key in ("accuracy", "lev_score_thresholded", "target_confidence"):
        values = _finite_array(row.get(key) for row in rows)
        if values.size:
            out.append(MetricSamples(language, model, f"lambada_{key}", values, True))
    return out


def _extract_external(language: str, model: str, data: Dict[str, Any]) -> List[MetricSamples]:
    rows = data.get("examples")
    if not isinstance(rows, list):
        return []
    avg_scores = []
    for row in rows:
        scores = row.get("scores")
        if isinstance(scores, dict):
            vals = _finite_array(scores.values())
            if vals.size:
                avg_scores.append(float(np.mean(vals)))
    values = _finite_array(avg_scores)
    if not values.size:
        return []
    return [MetricSamples(language, model, "external_model_ppl_avg_per_example", values, False)]


def _extract_canary(language: str, model: str, data: Dict[str, Any]) -> List[MetricSamples]:
    out = []
    for key, rows in data.items():
        if not key.startswith("detailed_results_") or not isinstance(rows, list):
            continue
        suffix = key.removeprefix("detailed_results_")
        match_values = _finite_array(1.0 if row.get("match") else 0.0 for row in rows)
        if match_values.size:
            out.append(MetricSamples(language, model, f"canary_{suffix}_accuracy", match_values, True))
        prefix_values = _finite_array(row.get("prefix_match_len") for row in rows)
        if prefix_values.size:
            out.append(MetricSamples(language, model, f"canary_{suffix}_prefix_match_chars", prefix_values, True))
    return out


def _extract_human_speed(language: str, model: str, data: Dict[str, Any]) -> List[MetricSamples]:
    rows = data.get("generation_speed_runs")
    if not isinstance(rows, list):
        return []
    out = []
    for key in ("chars_per_sec", "steps_per_sec"):
        values = _finite_array(row.get(key) for row in rows)
        if values.size:
            out.append(MetricSamples(language, model, f"generation_{key}", values, True))
    return out


def _collect_language(outputs_root: str, language: str) -> List[MetricSamples]:
    out: List[MetricSamples] = []
    for model in MAIN_MODELS:
        eval_dir = os.path.join(outputs_root, language, model, "evaluation")
        for filename, extractor in (
            ("miron_results.json", _extract_miron),
            ("lambada_results.json", _extract_lambada),
            ("external_model_results.json", _extract_external),
            ("canary_results.json", _extract_canary),
            ("human_eval_speed_results.json", _extract_human_speed),
        ):
            data = _read_json(os.path.join(eval_dir, filename))
            if isinstance(data, dict):
                out.extend(extractor(language, model, data))
    return out


def _summarize_language(samples: Sequence[MetricSamples]) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    by_metric: Dict[str, List[MetricSamples]] = {}
    for sample in samples:
        by_metric.setdefault(sample.metric, []).append(sample)

    metrics: Dict[str, Any] = {}
    for metric, rows in sorted(by_metric.items()):
        if len(rows) < 2:
            continue
        higher_is_better = rows[0].higher_is_better
        estimates = {
            row.model: _ci(row.values, rng)
            for row in sorted(rows, key=lambda x: x.model)
        }
        best_model = max(estimates, key=lambda m: estimates[m]["mean"]) if higher_is_better else min(
            estimates, key=lambda m: estimates[m]["mean"]
        )
        best_values = next(row.values for row in rows if row.model == best_model)
        comparisons = {}
        for row in rows:
            if row.model == best_model:
                continue
            comparisons[row.model] = _paired_diff_ci(best_values, row.values, rng, higher_is_better)
        metrics[metric] = {
            "higher_is_better": higher_is_better,
            "best_model": best_model,
            "estimates": estimates,
            "paired_bootstrap_vs_best": comparisons,
        }
    return {
        "bootstrap": {"n_resamples": N_BOOT, "seed": SEED, "ci": "percentile_95"},
        "metrics": metrics,
    }


def _format_value(x: float, digits: int = 2) -> str:
    if not math.isfinite(float(x)):
        return "--"
    return f"{x:.{digits}f}"


def _write_latex(path: str, language: str, summary: Dict[str, Any]) -> None:
    rows = []
    for metric, info in summary["metrics"].items():
        best = info["best_model"]
        est = info["estimates"][best]
        significant = []
        for model, cmp_info in info["paired_bootstrap_vs_best"].items():
            if cmp_info["p_value"] < 0.05 and cmp_info["ci_low"] > 0:
                significant.append(model)
        sig_text = f"{len(significant)}/{len(info['paired_bootstrap_vs_best'])}"
        rows.append(
            (
                metric.replace("_", "\\_"),
                best.replace("_", "\\_"),
                f"{_format_value(est['mean'])} [{_format_value(est['ci_low'])}; {_format_value(est['ci_high'])}]",
                sig_text,
            )
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by statistical_significance.py\n")
        f.write("\\begin{tabular}{llrl}\n")
        f.write("\\toprule\n")
        f.write("Metric & Best model & Mean [95\\% CI] & Significant wins \\\\\n")
        f.write("\\midrule\n")
        for metric, best, ci, sig in rows:
            f.write(f"{metric} & {best} & {ci} & {sig} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def run_statistical_significance(project_root: str, outputs_root: str, languages: Sequence[str], tables_root: str) -> int:
    outputs_abs = outputs_root if os.path.isabs(outputs_root) else os.path.join(project_root, outputs_root)
    tables_abs = tables_root if os.path.isabs(tables_root) else os.path.join(project_root, tables_root)
    os.makedirs(tables_abs, exist_ok=True)

    written = 0
    combined: Dict[str, Any] = {}
    for language in languages:
        samples = _collect_language(outputs_abs, language)
        if not samples:
            continue
        summary = _summarize_language(samples)
        combined[language] = summary
        json_path = os.path.join(tables_abs, f"statistical_significance_{language}.json")
        tex_path = os.path.join(tables_abs, f"statistical_significance_{language}.tex")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _write_latex(tex_path, language, summary)
        written += 2

    if combined:
        with open(os.path.join(tables_abs, "statistical_significance_all.json"), "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        written += 1
    return written


def build_statistical_significance_unscoped(project_root: str, outputs_root: str, languages: Sequence[str], tables_root: str) -> int:
    return run_statistical_significance(project_root, outputs_root, languages, tables_root)
