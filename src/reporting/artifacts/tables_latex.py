"""LaTeX tabular fragments from evaluation JSON (\\input{...} in thesis)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from src.reporting.experiment_meta import model_training_meta
from src.reporting.tex_utils import (
    load_json,
    sort_split_names,
    tex_cell_scalar,
    tex_escape_label,
    tex_table_float,
    tex_write,
)

if TYPE_CHECKING:
    from src.reporting.context import ReportContext, ReportOutputDirs


def build_latex_tables(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    """
    Writes:
    - perplexity_splits.tex — all splits from perplexity_results.json
    - model_training_meta.tex — data \\%, $|V|$, embedding+head vs rest (from config + training_summary)
    - training_summary.tex — scalar fields from training_summary.json
    - oracle.tex — oracle_results.json if present
    """
    written = 0
    tables_dir = out.tables_dir
    os.makedirs(tables_dir, exist_ok=True)

    ppl_path = os.path.join(ctx.eval_dir, "perplexity_results.json")
    ppl = load_json(ppl_path)
    if ppl:
        rows: List[str] = [
            "\\begin{tabular}{lrrrrrrr}",
            "\\hline",
            "Split & PPL & Char-PPL & Comp. & OOV & Voc.\\ util. & $\\bar{\\ell}_{\\mathrm{vocab}}$ & $\\bar{\\ell}_{\\mathrm{text}}$ \\\\",
            "\\hline",
        ]
        for split_name in sort_split_names(list(ppl.keys())):
            block = ppl[split_name]
            if not isinstance(block, dict) or "ppl" not in block:
                continue
            sn = tex_escape_label(str(split_name))
            rows.append(
                f"{sn} & {tex_table_float(block.get('ppl'), 4)} & {tex_table_float(block.get('char_ppl'), 4)} & "
                f"{tex_table_float(block.get('compression_ratio'), 4)} & {tex_table_float(block.get('oov_rate'), 4)} & "
                f"{tex_table_float(block.get('vocab_utilization'), 4)} & {tex_table_float(block.get('avg_token_length_vocab'), 4)} & "
                f"{tex_table_float(block.get('avg_token_length_text'), 4)} \\\\"
            )
        rows.append("\\hline")
        rows.append("\\end{tabular}")
        tex_write(os.path.join(tables_dir, "perplexity_splits.tex"), "\n".join(rows))
        written += 1

    ts_path = os.path.join(ctx.eval_dir, "training_summary.json")
    ts = load_json(ts_path)
    meta = model_training_meta(ctx.project_root, ctx.outputs_root, ctx.experiment_id, ts)
    if meta:
        tie_w = "yes" if meta["tied"] else "no"
        src = meta.get("param_stats_source") or "config"
        v_sz = int(meta["vocab_size"])
        h_sz = meta.get("hidden_size")
        eb = int(meta["embed_head_params"])
        eb_cell = str(eb)
        if h_sz is not None and int(h_sz) > 0:
            h_i = int(h_sz)
            if meta["tied"] and v_sz * h_i == eb:
                eb_cell = (
                    f"{eb} ($= \\lvert V\\rvert \\times d = {v_sz} \\times {h_i}$)"
                )
            elif (not meta["tied"]) and 2 * v_sz * h_i == eb:
                eb_cell = (
                    f"{eb} ($= 2\\lvert V\\rvert \\times d = 2 \\times {v_sz} \\times {h_i}$)"
                )

        mxs = meta.get("max_steps_config")
        mxs_cell = tex_cell_scalar(mxs)
        try:
            if mxs is not None and int(mxs) < 0:
                mxs_cell = "not set ($<0$; stop by epochs/data)"
        except (TypeError, ValueError):
            pass

        frac_cell = tex_table_float(meta.get("fraction_of_config_max_steps"), 4)
        try:
            if mxs is None or int(mxs) <= 0:
                frac_cell = "---"
        except (TypeError, ValueError):
            pass
        if meta.get("fraction_of_config_max_steps") is None:
            frac_cell = "---"

        mlines = [
            "% Vocab / emb: checkpoint embed_head_params_unique = unique scalars for token embed + lm_head",
            "% (one $|V|\\times d$ matrix if tie_word_embeddings; else two matrices).",
            "% Training steps: training\\_summary.json (metrics + training\\_metadata when present).",
            "\\begin{tabular}{lr}",
            "\\hline",
            "Field & Value \\\\",
            "\\hline",
            f"Param.\\ source & {tex_escape_label(str(src))} \\\\",
            f"Data processed (\\%) & {tex_table_float(meta.get('pct_data'), 4)} \\\\",
            f"Steps / epoch & {tex_cell_scalar(meta.get('steps_per_epoch'))} \\\\",
            f"Last training step & {tex_cell_scalar(meta.get('last_step'))} \\\\",
            f"Epochs completed (approx.) & {tex_table_float(meta.get('epochs_completed'), 4)} \\\\",
            f"Config max\\_steps & {mxs_cell} \\\\",
            f"Fraction of config max\\_steps & {frac_cell} \\\\",
        ]
        if h_sz is not None:
            mlines.append(
                f"$d_{{\\mathrm{{model}}}}$ (hidden size) & {int(h_sz)} \\\\",
            )
        if meta.get("param_mib") is not None:
            mlines.append(
                f"Weight memory, checkpoint dtype (MiB) & {tex_table_float(meta.get('param_mib'), 2)} \\\\"
            )
        if meta.get("param_fp32_mib") is not None:
            mlines.append(
                f"Weight memory, fp32 equivalent (MiB) & {tex_table_float(meta.get('param_fp32_mib'), 2)} \\\\"
            )
        if meta.get("miron_char_stack_params") is not None:
            mlines.extend(
                [
                    f"MIRON encoder params & {int(meta.get('miron_encoder_params') or 0)} \\\\",
                    f"MIRON encoder projection params & {int(meta.get('miron_enc_proj_params') or 0)} \\\\",
                    f"MIRON core LM params & {int(meta.get('miron_core_lm_params') or 0)} \\\\",
                    f"MIRON decoder projection params & {int(meta.get('miron_dec_proj_params') or 0)} \\\\",
                    f"MIRON decoder params & {int(meta.get('miron_decoder_params') or 0)} \\\\",
                    f"MIRON char stack params & {int(meta.get('miron_char_stack_params') or 0)} \\\\",
                ]
            )
        mlines.extend(
            [
                f"$|V|$ (token LM) & {v_sz} \\\\",
                "Unique params, token embed.\\ + LM head & "
                + eb_cell
                + " \\\\",
                f"$P_{{\\mathrm{{rest}}}}$ (transformer blocks) & "
                f"{int(meta['rest_params']) if meta.get('rest_params') is not None else '---'} \\\\",
                f"Total parameters & {int(meta['total_params']) if meta.get('total_params') is not None else '---'} \\\\",
                f"Tied word embeddings & {tie_w} \\\\",
                "\\hline",
                "\\end{tabular}",
            ]
        )
        tex_write(os.path.join(tables_dir, "model_training_meta.tex"), "\n".join(mlines))
        written += 1

    if ts:
        pairs: List[Tuple[str, str]] = []
        for k, v in sorted(ts.items()):
            if isinstance(v, (dict, list)):
                continue
            pairs.append((tex_escape_label(str(k)), tex_cell_scalar(v)))
        if pairs:
            lines = ["\\begin{tabular}{lr}", "\\hline", "Field & Value \\\\", "\\hline"]
            for key, val in pairs:
                lines.append(f"{key} & {val} \\\\")
            lines.extend(["\\hline", "\\end{tabular}"])
            tex_write(os.path.join(tables_dir, "training_summary.tex"), "\n".join(lines))
            written += 1

    oracle_path = os.path.join(ctx.eval_dir, "oracle_results.json")
    oracle = load_json(oracle_path)
    if oracle and isinstance(oracle, dict):
        lines = ["\\begin{tabular}{lr}", "\\hline", "Metric & Value \\\\", "\\hline"]
        for k, v in sorted(oracle.items()):
            if isinstance(v, (dict, list)):
                continue
            cell = (
                tex_table_float(v, 4)
                if isinstance(v, float)
                else (str(int(v)) if isinstance(v, int) else tex_escape_label(str(v)))
            )
            lines.append(f"{tex_escape_label(str(k))} & {cell} \\\\")
        lines.extend(["\\hline", "\\end{tabular}"])
        tex_write(os.path.join(tables_dir, "oracle.tex"), "\n".join(lines))
        written += 1

    return written > 0
