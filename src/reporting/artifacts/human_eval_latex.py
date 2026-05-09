"""LaTeX appendix: human-eval prompts + generations (generation capped at 150 chars + ...)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List

from src.reporting.tex_utils import load_json, tex_escape_body, tex_write

if TYPE_CHECKING:
    from src.reporting.context import ReportContext, ReportOutputDirs

GEN_CHAR_LIMIT = 150


def _eval_path(project_root: str, outputs_root: str, exp_id: str, filename: str) -> str:
    return os.path.join(
        project_root, outputs_root, exp_id.replace("\\", "/"), "evaluation", filename
    )


def _truncate_generation(text: str, max_len: int = GEN_CHAR_LIMIT) -> str:
    t = text if text is not None else ""
    if len(t) <= max_len:
        return t
    return t[:max_len] + "..."


def _examples_only(generations: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for g in generations:
        prompt = g.get("prompt") or ""
        gen_full = g.get("generated") or ""
        gen_show = _truncate_generation(gen_full)
        parts.append(f"\\noindent\\textbf{{Prompt:}} {tex_escape_body(prompt)}")
        parts.append("")
        parts.append(f"\\noindent\\textbf{{Generation:}} {tex_escape_body(gen_show)}")
        parts.append("")
        parts.append("\\vspace{0.5em}")
        parts.append("")
    return "\n".join(parts)


def build_human_eval_appendix(ctx: "ReportContext", out: "ReportOutputDirs") -> bool:
    path = os.path.join(ctx.eval_dir, "human_eval_results.json")
    data = load_json(path)
    if not data:
        return False
    gens = data.get("generations")
    if not isinstance(gens, list) or not gens:
        return False
    body = "\n".join(
        [
            "% Auto-generated: human evaluation sample generations",
            "\\begingroup",
            "\\sloppy",
            "\\small",
            "",
            _examples_only(gens),
            "\\endgroup",
            "",
        ]
    )
    tex_write(os.path.join(out.tables_dir, "human_eval_appendix.tex"), body)
    return True


def write_human_eval_cross_language(
    project_root: str,
    outputs_root: str,
    language: str,
    out_dir: str,
) -> bool:
    """One file per language; examples for all tokenizers with a separator."""
    from src.reporting.cross_discovery import list_experiments_for_language

    sections: List[str] = [
        f"% Human-eval: language `{language}'",
        "\\begingroup",
        "\\sloppy",
        "\\small",
        "",
    ]
    any_content = False
    for eid in list_experiments_for_language(project_root, language, outputs_root):
        data = load_json(_eval_path(project_root, outputs_root, eid, "human_eval_results.json"))
        if not data:
            continue
        gens = data.get("generations")
        if not isinstance(gens, list) or not gens:
            continue
        sections.append(_examples_only(gens))
        sections.append("\\vspace{1em}")
        sections.append("")
        any_content = True
    sections.append("\\endgroup")
    if not any_content:
        return False
    tex_write(os.path.join(out_dir, "human_eval_appendix.tex"), "\n".join(sections))
    return True
