"""List languages and experiments under outputs/<lang>/... for aggregate tables."""

from __future__ import annotations

import os
from typing import List


_ALIASES = {
    "miron_flatten_128": "miron",
}


def discover_languages(project_root: str, outputs_root: str = "outputs") -> List[str]:
    """Top-level dirs under outputs/ that contain at least one nested experiment with evaluation/."""
    root = os.path.join(project_root, outputs_root)
    if not os.path.isdir(root):
        return []
    langs: List[str] = []
    for name in sorted(os.listdir(root)):
        lang_dir = os.path.join(root, name)
        if not os.path.isdir(lang_dir) or name.startswith("."):
            continue
        has_eval = False
        for sub in os.listdir(lang_dir):
            ev = os.path.join(lang_dir, sub, "evaluation")
            if os.path.isdir(ev):
                has_eval = True
                break
        if has_eval:
            langs.append(name)
    return langs


def list_experiments_for_language(project_root: str, language: str, outputs_root: str = "outputs") -> List[str]:
    """Experiment ids `<lang>/<tok>` with an evaluation directory."""
    base = os.path.join(project_root, outputs_root, language)
    if not os.path.isdir(base):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(base)):
        exp_dir = os.path.join(base, name)
        if not os.path.isdir(exp_dir):
            continue
        if os.path.isdir(os.path.join(exp_dir, "evaluation")):
            out.append(f"{language}/{_ALIASES.get(name, name)}")
    return sorted(dict.fromkeys(out))


def experiment_short_name(experiment_id: str) -> str:
    parts = experiment_id.replace("\\", "/").split("/")
    return parts[-1] if parts else experiment_id
