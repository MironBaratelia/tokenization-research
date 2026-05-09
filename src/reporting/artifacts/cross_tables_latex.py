"""Per-language comparison tables across all tokenizers (one .tex per benchmark)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from src.reporting.cross_discovery import experiment_short_name, list_experiments_for_language
from src.reporting.tex_utils import (
    load_json,
    sort_split_names,
    tex_cell_scalar,
    tex_escape_label,
    tex_table_float,
    tex_write,
)

logger = logging.getLogger(__name__)

_EVAL = "evaluation"


def _load_significance(project_root: str, language: str) -> Dict[str, Any]:
    path = os.path.join(project_root, "reports", "tables", "cross", f"statistical_significance_{language}.json")
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def _statistically_best_models(sig: Dict[str, Any], metric: str) -> Set[str]:
    metrics = sig.get("metrics")
    if not isinstance(metrics, dict):
        return set()
    info = metrics.get(metric)
    if not isinstance(info, dict):
        return set()
    best = info.get("best_model")
    if not isinstance(best, str):
        return set()
    out = {best}
    comparisons = info.get("paired_bootstrap_vs_best")
    if isinstance(comparisons, dict):
        for model, cmp_info in comparisons.items():
            if not isinstance(cmp_info, dict):
                continue
            try:
                p_value = float(cmp_info.get("p_value", 1.0))
                ci_low = float(cmp_info.get("ci_low", 0.0))
            except (TypeError, ValueError):
                continue
            if p_value >= 0.05 or ci_low <= 0.0:
                out.add(str(model))
    return out


def _bold_if_best(cell: str, model: str, best_models: Set[str]) -> str:
    if model in best_models and cell != "---":
        return f"\\textbf{{{cell}}}"
    return cell


def _eval_path(project_root: str, outputs_root: str, exp_id: str, filename: str) -> str:
    return os.path.join(project_root, outputs_root, exp_id.replace("\\", "/"), _EVAL, filename)


def _int_cell(x: Optional[int]) -> str:
    if x is None:
        return "---"
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return "---"


def _summary_number_cell(summary: Dict[str, Any], key: str, digits: int = 4) -> str:
    value = summary.get(key)
    if value is None:
        return "---"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return tex_table_float(value, digits)
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return tex_cell_scalar(value)
    return tex_table_float(as_float, digits)


def _cross_perplexity(
    project_root: str, outputs_root: str, language: str, out_dir: str
) -> bool:
    exps = list_experiments_for_language(project_root, language, outputs_root)
    per_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    short_to_eid: Dict[str, str] = {}
    all_splits: Set[str] = set()
    for eid in exps:
        data = load_json(_eval_path(project_root, outputs_root, eid, "perplexity_results.json"))
        if not data:
            continue
        short = experiment_short_name(eid)
        short_to_eid[short] = eid
        per_model[short] = {}
        for split, block in data.items():
            if isinstance(block, dict) and "ppl" in block:
                all_splits.add(split)
                per_model[short][split] = block
    if not per_model:
        return False
    splits = sort_split_names(list(all_splits))

    header = [
        "Model",
        "Data \\%",
        "$|V|$",
        "$P_{\\mathrm{emb}}$",
        "$P_{\\mathrm{rest}}$",
    ]
    for sp in splits:
        sp_tex = tex_escape_label(sp)
        header.append(f"$\\mathrm{{PPL}}_{{{sp_tex}}}$")
        header.append(f"$\\widetilde{{\\mathrm{{PPL}}}}_{{{sp_tex}}}$")
        header.append(f"$\\rho_{{{sp_tex}}}$")
        header.append(f"OOV$_{{{sp_tex}}}$")
    col_spec = "l" + "rrrr" * (1 + len(splits))
    lines = [
        "% $P_{\\mathrm{emb}}$: token embedding + LM head (one $|V| \\times d$ if tied; else two). Prefer checkpoint counts.",
        "% Data \\% / $|V|$ / $P_{\\mathrm{emb}}$: training\\_summary + checkpoint when available.",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for short in sorted(per_model.keys()):
        cells = [tex_escape_label(short)]
        eid = short_to_eid.get(short, "")
        ts = load_json(_eval_path(project_root, outputs_root, eid, "training_summary.json"))
        if ts:
            cells.append(_summary_number_cell(ts, "summary_pct_data_processed"))
            cells.append(_summary_number_cell(ts, "summary_checkpoint_vocab_size", 0))
            cells.append(_summary_number_cell(ts, "summary_checkpoint_embed_head_params_unique", 0))
            cells.append(_summary_number_cell(ts, "summary_checkpoint_rest_params_unique", 0))
        else:
            cells.extend(["---", "---", "---", "---"])

        blocks = per_model[short]
        for sp in splits:
            b = blocks.get(sp)
            if not b:
                cells.extend(["---", "---", "---", "---"])
            else:
                cells.append(tex_table_float(b.get("ppl"), 4))
                cells.append(tex_table_float(b.get("char_ppl"), 4))
                cells.append(tex_table_float(b.get("compression_ratio"), 4))
                cells.append(tex_table_float(b.get("oov_rate"), 4))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "perplexity.tex"), "\n".join(lines))
    return True


def _cross_blimp(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    rows: List[Tuple[str, float]] = []
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "blimp_results.json"))
        if not data or "total_accuracy" not in data:
            continue
        rows.append((experiment_short_name(eid), float(data["total_accuracy"])))
    if not rows:
        return False
    rows.sort(key=lambda x: x[0])
    lines = [
        "\\begin{tabular}{lr}",
        "\\hline",
        "Model & BLiMP (\\% total) \\\\",
        "\\hline",
    ]
    for name, acc in rows:
        lines.append(f"{tex_escape_label(name)} & {tex_table_float(acc, 4)} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "blimp.tex"), "\n".join(lines))
    return True


def _cross_lambada(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    rows: List[Tuple[str, str, str, str]] = []
    sig = _load_significance(project_root, language)
    best_exact = _statistically_best_models(sig, "lambada_accuracy")
    best_score = _statistically_best_models(sig, "lambada_lev_score_thresholded")
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "lambada_results.json"))
        if not data or "accuracy" not in data:
            continue
        acc = float(data["accuracy"])
        acc_pct = acc * 100.0 if acc <= 1.0 else acc
        score = data.get("score", data.get("lev_score_thresholded"))
        conf = data.get("confidence")
        rows.append(
            (
                experiment_short_name(eid),
                tex_table_float(acc_pct, 4),
                tex_table_float(score, 4) if score is not None else "---",
                tex_table_float(conf, 4) if conf is not None else "---",
            )
        )
    if not rows:
        return False
    rows.sort(key=lambda x: x[0])
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Model & Accuracy (\\%) & Score & $P(\\mathrm{target})$ \\\\",
        "\\hline",
    ]
    for name, a, score, c in rows:
        lines.append(
            f"{tex_escape_label(name)} & "
            f"{_bold_if_best(a, name, best_exact)} & "
            f"{_bold_if_best(score, name, best_score)} & {c} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "lambada.tex"), "\n".join(lines))
    return True


def _scalar_keys_union(
    project_root: str, outputs_root: str, language: str, filename: str
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    data_by_model: Dict[str, Dict[str, Any]] = {}
    keys: Set[str] = set()
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, filename))
        if not data or not isinstance(data, dict):
            continue
        short = experiment_short_name(eid)
        scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        if not scalars:
            continue
        data_by_model[short] = scalars
        keys.update(scalars.keys())
    ordered = sorted(keys)
    return ordered, data_by_model


def _cross_oracle(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    keys, by_model = _scalar_keys_union(project_root, outputs_root, language, "oracle_results.json")
    if not by_model:
        return False
    col_spec = "l" + "r" * len(keys)
    header = ["Model"] + [tex_escape_label(k) for k in keys]
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for short in sorted(by_model.keys()):
        cells = [tex_escape_label(short)]
        row = by_model[short]
        for k in keys:
            v = row.get(k)
            cells.append(
                tex_table_float(v, 4) if isinstance(v, (int, float)) else tex_cell_scalar(v)
            )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "oracle.tex"), "\n".join(lines))
    return True


def _cross_canary(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    acc_keys: Set[str] = set()
    by_model: Dict[str, Dict[str, float]] = {}
    sig = _load_significance(project_root, language)
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "canary_results.json"))
        if not data:
            continue
        short = experiment_short_name(eid)
        row: Dict[str, float] = {}
        for k, v in data.items():
            if not k.startswith("accuracy_"):
                continue
            if isinstance(v, (int, float)):
                row[k] = float(v)
                acc_keys.add(k)
        if row:
            by_model[short] = row
    if not by_model:
        return False
    keys_sorted = sorted(acc_keys)
    col_spec = "l" + "r" * len(keys_sorted)
    header = ["Model"] + [tex_escape_label(k.replace("accuracy_", "", 1)) for k in keys_sorted]
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for short in sorted(by_model.keys()):
        cells = [tex_escape_label(short)]
        row = by_model[short]
        for k in keys_sorted:
            v = row.get(k)
            if v is None:
                cells.append("---")
            else:
                pct = v * 100.0 if v <= 1.0 else v
                metric = "canary_" + k.replace("accuracy_", "", 1) + "_accuracy"
                cells.append(_bold_if_best(tex_table_float(pct, 4), short, _statistically_best_models(sig, metric)))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "canary.tex"), "\n".join(lines))
    return True


def _cross_miron(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    categories: Set[str] = set()
    by_model: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    sig = _load_significance(project_root, language)
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "miron_results.json"))
        if not data:
            continue
        short = experiment_short_name(eid)
        row: Dict[str, Tuple[float, float, float]] = {}
        for cat, block in data.items():
            if not isinstance(block, dict):
                continue
            if "score" not in block and "lev_score_thresholded" not in block:
                continue
            exact = float(block.get("exact_match", block.get("accuracy", 0.0)))
            if exact <= 1.0:
                exact *= 100.0
            sc = float(block.get("lev_score_thresholded", block.get("score", 0.0)))
            cf = block.get("confidence")
            cf_f = float(cf) if cf is not None else float("nan")
            row[cat] = (exact, sc, cf_f)
            categories.add(cat)
        if row:
            by_model[short] = row
    if not by_model:
        return False
    cats = sorted(categories)
    header = ["Model"]
    for c in cats:
        ct = tex_escape_label(c)
        header.append(f"$\\mathrm{{accuracy}}_{{{ct}}}$")
        header.append(f"$\\mathrm{{score}}_{{{ct}}}$")
        header.append(f"$P(\\mathrm{{target}})_{{{ct}}}$")
    col_spec = "l" + "rrr" * len(cats)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for short in sorted(by_model.keys()):
        cells = [tex_escape_label(short)]
        row = by_model[short]
        for c in cats:
            pair = row.get(c)
            if not pair:
                cells.extend(["---", "---", "---"])
            else:
                exact, sc, cf = pair
                cells.append(
                    _bold_if_best(
                        tex_table_float(exact, 4),
                        short,
                        _statistically_best_models(sig, f"miron_{c}_exact_match"),
                    )
                )
                cells.append(
                    _bold_if_best(
                        tex_table_float(sc, 4),
                        short,
                        _statistically_best_models(sig, f"miron_{c}_lev_score_thresholded"),
                    )
                )
                conf_cell = tex_table_float(cf, 4) if cf == cf else "---"
                cells.append(conf_cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "miron.tex"), "\n".join(lines))
    return True


def _cross_training_summary(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    keys, by_model = _scalar_keys_union(project_root, outputs_root, language, "training_summary.json")
    if not by_model:
        return False
    wanted = [
        ("summary_checkpoint_vocab_size", "$|V|$"),
        ("summary_checkpoint_total_params_unique", "Params"),
        ("summary_checkpoint_param_mib_unique", "Weights MiB"),
        ("summary_checkpoint_param_fp32_mib_unique", "fp32 MiB"),
        ("summary_miron_encoder_params_unique", "Enc."),
        ("summary_miron_enc_proj_params_unique", "Enc. proj."),
        ("summary_miron_decoder_params_unique", "Dec."),
        ("summary_miron_dec_proj_params_unique", "Dec. proj."),
        ("summary_pct_data_processed", "Data \\%"),
        ("summary_best_val_ppl", "Best Val PPL"),
        ("summary_pct_data_to_95pct_quality", "95\\% data"),
        ("summary_step_to_95pct_quality", "95\\% step"),
        ("summary_p50_tokens_per_sec", "p50 tokens/s"),
        ("summary_steps_per_epoch", "Steps/epoch"),
    ]
    available = [(k, h) for k, h in wanted if k in keys]
    col_spec = "l" + "r" * len(available)
    header = ["Model"] + [h for _, h in available]
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for short in sorted(by_model.keys()):
        cells = [tex_escape_label(short)]
        row = by_model[short]
        for k, _ in available:
            cells.append(tex_cell_scalar(row.get(k)))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "training_summary.tex"), "\n".join(lines))
    return True


def _cross_human_eval(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    from src.reporting.artifacts.human_eval_latex import write_human_eval_cross_language

    return write_human_eval_cross_language(project_root, outputs_root, language, out_dir)


def _ext_model_col_label(full_name: str) -> str:
    short = full_name.rsplit("/", maxsplit=1)[-1]
    return tex_escape_label(short)


def _cross_external_model(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    """Cross-experiment table: mean Oracle-style PPL from each external scorer on our generations."""
    by_model: Dict[str, Dict[str, Any]] = {}
    ref_keys: Set[str] = set()
    sig = _load_significance(project_root, language)
    best_external = _statistically_best_models(sig, "external_model_ppl_avg_per_example")
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "external_model_results.json"))
        if not data or not isinstance(data, dict):
            continue
        models_block = data.get("models")
        if not isinstance(models_block, dict) or not models_block:
            continue
        short = experiment_short_name(eid)
        penalty = data.get("external_repetition_penalty") or {}
        by_model[short] = {
            "external_ppl_avg": data.get("external_ppl_avg"),
            "external_ppl_repetition_adjusted_avg": data.get(
                "external_ppl_repetition_adjusted_avg"
            ),
            "repeat_3": penalty.get("repeat_3"),
            "repeat_status": penalty.get("status"),
            "models": models_block,
        }
        ref_keys.update(models_block.keys())
    if not by_model:
        return False
    refs_sorted = sorted(ref_keys)
    header = [
        "Model",
        "$\\overline{\\mathrm{PPL}}_{\\mathrm{ext}}$",
        "$\\overline{\\mathrm{PPL}}_{\\mathrm{ext,adj}}$",
        "$r_3$",
        "Repeat status",
    ]
    header.extend(f"$\\mathrm{{PPL}}_{{{_ext_model_col_label(r)}}}$" for r in refs_sorted)
    col_spec = "lrrrl" + "r" * len(refs_sorted)
    lines = [
        "% External scorer PPL on continuations from our model (\\texttt{external\\_model\\_results.json}).",
        "% Adj. PPL combines tail-loop trimming diagnostics with a repetition penalty; r_3 is mean generated trigram repeat rate.",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]

    def _cell_ppl(v: Any) -> str:
        if v is None:
            return "---"
        try:
            xf = float(v)
        except (TypeError, ValueError):
            return "---"
        if xf != xf or xf == float("inf"):  # NaN or inf
            return "---"
        return tex_table_float(xf, 4)

    for short in sorted(by_model.keys()):
        row = by_model[short]
        cells = [
            tex_escape_label(short),
            _bold_if_best(_cell_ppl(row.get("external_ppl_avg")), short, best_external),
            _cell_ppl(row.get("external_ppl_repetition_adjusted_avg")),
            _cell_ppl(row.get("repeat_3")),
            tex_escape_label(str(row.get("repeat_status") or "---")),
        ]
        mb = row["models"]
        for r in refs_sorted:
            block = mb.get(r)
            if not isinstance(block, dict):
                cells.append("---")
            else:
                cells.append(_cell_ppl(block.get("external_ppl")))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "external_model.tex"), "\n".join(lines))
    return True


def _cross_repetition_diagnostics(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    """Cross-experiment loop/repetition diagnostics for generated continuations."""
    def _suffix_loop_start_units(units: List[str], *, min_prefix: int = 8, max_period: int = 64) -> Optional[int]:
        """Return start index when the tail is one unit sequence repeated >2 times."""
        n_units = len(units)
        if n_units < min_prefix + 3:
            return None
        best_start: Optional[int] = None
        for period in range(1, min(max_period, n_units // 3) + 1):
            max_repeats = n_units // period
            for repeats in range(max_repeats, 2, -1):
                tail_len = period * repeats
                start = n_units - tail_len
                if start < min_prefix:
                    continue
                pattern = units[start : start + period]
                ok = True
                for r in range(1, repeats):
                    if units[start + r * period : start + (r + 1) * period] != pattern:
                        ok = False
                        break
                if ok and (best_start is None or start < best_start):
                    best_start = start
        return best_start

    def _detect_loop(text: str) -> Tuple[bool, int, int]:
        """Loop = from some point to the end, any word/character n-gram repeats >2 times."""
        words = text.split()
        original_words = len(words)
        word_start = _suffix_loop_start_units(words, min_prefix=8, max_period=32)

        compact_chars = list(" ".join(text.split()))
        char_start = _suffix_loop_start_units(compact_chars, min_prefix=32, max_period=80)
        char_word_start: Optional[int] = None
        if char_start is not None:
            prefix = "".join(compact_chars[:char_start])
            char_word_start = len(prefix.split())

        candidates = [x for x in (word_start, char_word_start) if x is not None]
        if not candidates:
            return False, original_words, original_words
        start_words = min(candidates)
        return True, original_words, max(0, min(start_words, original_words))

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "external_model_results.json"))
        if not data or not isinstance(data, dict):
            continue
        penalty = data.get("external_repetition_penalty") or {}
        trim = data.get("loop_trim") or {}
        examples = data.get("examples") if isinstance(data.get("examples"), list) else []
        total = len(examples)
        trimmed = 0
        mild = 0
        removed_words = 0
        original_words = 0
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            source_text = str(ex.get("generated_raw") or ex.get("generated") or "")
            is_loop, ow, tw = _detect_loop(source_text)
            if not is_loop:
                original_words += ow
                continue
            removed = max(0, ow - tw)
            ratio = (removed / ow) if ow else 0.0
            trimmed += 1
            if ratio < 0.5:
                mild += 1
            original_words += ow
            removed_words += removed
        no_loop_pct = 100.0 * (1.0 - trimmed / total) if total else 0.0
        loop_pct = 100.0 * (trimmed / total) if total else 0.0
        mild_loop_pct = 100.0 * (mild / total) if total else 0.0
        loop_text_pct = 100.0 * (removed_words / original_words) if original_words else 0.0
        rows.append(
            (
                experiment_short_name(eid),
                {
                    "no_loop_pct": no_loop_pct,
                    "loop_pct": loop_pct,
                    "mild_loop_pct": mild_loop_pct,
                    "loop_text_pct": loop_text_pct,
                    "trimmed_ratio": penalty.get("trimmed_ratio", trim.get("trimmed_ratio")),
                    "trimmed_examples": trim.get("trimmed_examples"),
                    "multiplier": penalty.get("multiplier"),
                    "status": penalty.get("status"),
                },
            )
        )
    if not rows:
        return False
    rows.sort(key=lambda x: x[0])
    lines = [
        "% Repetition diagnostics for external-model generations.",
        "% loop: answers with detected repeated-ngram loop; trim: answers actually trimmed before external scoring.",
        "\\begin{tabular}{lrrrrrrl}",
        "\\hline",
        "Model & no-loop\\% & loop\\% & mild-loop\\% & loop-text\\% & trim\\% & penalty $\\times$ & status \\\\",
        "\\hline",
    ]
    for name, data in rows:
        trim_ratio = data.get("trimmed_ratio")
        trim_pct = None
        try:
            trim_pct = 100.0 * float(trim_ratio)
        except (TypeError, ValueError):
            trim_pct = None
        cells = [
            tex_escape_label(name),
            tex_table_float(data.get("no_loop_pct"), 1),
            tex_table_float(data.get("loop_pct"), 1),
            tex_table_float(data.get("mild_loop_pct"), 1),
            tex_table_float(data.get("loop_text_pct"), 1),
            tex_table_float(trim_pct, 1),
            tex_table_float(data.get("multiplier"), 2),
            tex_escape_label(str(data.get("status") or "---")),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "repetition_diagnostics.tex"), "\n".join(lines))
    return True


def _cross_human_eval_speed(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    """Cross-experiment generation speed table measured on human-eval prompts with batch_size=1."""
    rows: List[Tuple[str, Dict[str, Any]]] = []
    sig = _load_significance(project_root, language)
    best_chars = _statistically_best_models(sig, "generation_chars_per_sec")
    best_steps = _statistically_best_models(sig, "generation_steps_per_sec")
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "human_eval_speed_results.json"))
        if not data or not isinstance(data, dict):
            continue
        rows.append((experiment_short_name(eid), data))
    if not rows:
        return False
    rows.sort(key=lambda x: x[0])
    lines = [
        "% Generation speed from human_eval prompts, batch_size=1, median over repeated runs.",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Model & chars/s & tokens/s & rel. MAD & chars/token \\\\",
        "\\hline",
    ]
    for name, data in rows:
        cells = [
            tex_escape_label(name),
            _bold_if_best(tex_table_float(data.get("generation_chars_per_sec_median"), 2), name, best_chars),
            _bold_if_best(tex_table_float(data.get("generation_steps_per_sec_median"), 2), name, best_steps),
            tex_table_float(data.get("generation_chars_per_sec_rel_mad"), 4),
            tex_table_float(data.get("generation_chars_per_step_mean"), 2),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "human_eval_speed.tex"), "\n".join(lines))
    return True


def _cross_human_eval_repetition(project_root: str, outputs_root: str, language: str, out_dir: str) -> bool:
    """Loop diagnostics for the short qualitative human-eval generations."""

    def _suffix_loop_start_units(units: List[str], *, min_prefix: int = 8, max_period: int = 64) -> Optional[int]:
        n_units = len(units)
        if n_units < min_prefix + 3:
            return None
        best_start: Optional[int] = None
        for period in range(1, min(max_period, n_units // 3) + 1):
            max_repeats = n_units // period
            for repeats in range(max_repeats, 2, -1):
                tail_len = period * repeats
                start = n_units - tail_len
                if start < min_prefix:
                    continue
                pattern = units[start : start + period]
                if all(
                    units[start + r * period : start + (r + 1) * period] == pattern
                    for r in range(1, repeats)
                ):
                    if best_start is None or start < best_start:
                        best_start = start
        return best_start

    def _detect_loop(text: str) -> Tuple[bool, int, int]:
        words = text.split()
        original_words = len(words)
        word_start = _suffix_loop_start_units(words, min_prefix=8, max_period=32)
        compact_chars = list(" ".join(text.split()))
        char_start = _suffix_loop_start_units(compact_chars, min_prefix=32, max_period=80)
        char_word_start: Optional[int] = None
        if char_start is not None:
            prefix = "".join(compact_chars[:char_start])
            char_word_start = len(prefix.split())
        candidates = [x for x in (word_start, char_word_start) if x is not None]
        if not candidates:
            return False, original_words, original_words
        start_words = min(candidates)
        return True, original_words, max(0, min(start_words, original_words))

    rows: List[Tuple[str, Dict[str, float]]] = []
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "human_eval_results.json"))
        if not data or not isinstance(data, dict):
            continue
        examples = data.get("generations") or data.get("examples") or data.get("results")
        if not isinstance(examples, list) or not examples:
            continue
        total = 0
        loops = 0
        mild = 0
        removed_words = 0
        original_words = 0
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            text = str(ex.get("generated") or ex.get("generation") or "")
            is_loop, ow, tw = _detect_loop(text)
            total += 1
            original_words += ow
            if not is_loop:
                continue
            removed = max(0, ow - tw)
            loops += 1
            removed_words += removed
            if ow and removed / ow < 0.5:
                mild += 1
        if total:
            rows.append(
                (
                    experiment_short_name(eid),
                    {
                        "no_loop_pct": 100.0 * (1.0 - loops / total),
                        "loop_pct": 100.0 * loops / total,
                        "mild_loop_pct": 100.0 * mild / total,
                        "loop_text_pct": 100.0 * removed_words / original_words if original_words else 0.0,
                    },
                )
            )
    if not rows:
        return False
    rows.sort(key=lambda x: x[0])
    lines = [
        "% Repetition diagnostics for human_eval generations.",
        "% loop: answers with a repeated word/character n-gram from some point to the end.",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Model & no-loop\\% & loop\\% & mild-loop\\% & loop-text\\% \\\\",
        "\\hline",
    ]
    for name, data in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape_label(name),
                    tex_table_float(data.get("no_loop_pct"), 1),
                    tex_table_float(data.get("loop_pct"), 1),
                    tex_table_float(data.get("mild_loop_pct"), 1),
                    tex_table_float(data.get("loop_text_pct"), 1),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    tex_write(os.path.join(out_dir, "human_eval_repetition.tex"), "\n".join(lines))
    return True


_BENCHMARK_WRITERS = (
    ("perplexity", _cross_perplexity),
    ("blimp", _cross_blimp),
    ("lambada", _cross_lambada),
    ("oracle", _cross_oracle),
    ("external_model", _cross_external_model),
    ("repetition_diagnostics", _cross_repetition_diagnostics),
    ("human_eval_speed", _cross_human_eval_speed),
    ("human_eval_repetition", _cross_human_eval_repetition),
    ("canary", _cross_canary),
    ("miron", _cross_miron),
    ("training_summary", _cross_training_summary),
)


def run_cross_language_tables(
    project_root: str,
    outputs_root: str,
    languages: List[str],
    cross_tables_root: str,
) -> Tuple[int, int]:
    """
    For each language, write comparison tables under `<cross_tables_root>/<lang>/`.
    Returns (files_written, languages_touched).
    """
    written = 0
    touched = 0
    base = (
        cross_tables_root
        if os.path.isabs(cross_tables_root)
        else os.path.join(project_root, cross_tables_root)
    )
    for lang in languages:
        out_dir = os.path.join(base, lang)
        any_file = False
        for _name, fn in _BENCHMARK_WRITERS:
            try:
                if fn(project_root, outputs_root, lang, out_dir):
                    written += 1
                    any_file = True
            except Exception:
                logger.exception("Cross-table failed (%s, %s)", lang, fn.__name__)
        if any_file:
            touched += 1
            logger.info("Cross-language tables for %s -> %s", lang, out_dir)
    return written, touched


def build_latex_cross_tables_unscoped() -> bool:
    """Registry placeholder; real work is done in run_cli."""
    return False
