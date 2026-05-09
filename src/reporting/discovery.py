"""Discover experiments under outputs/ that have metrics and/or evaluation JSON."""

from __future__ import annotations

import os
from typing import List, Set


_ALIASES = {
    "miron_flatten_128": "miron",
}


def _outputs_abs(project_root: str, outputs_root: str) -> str:
    return os.path.normpath(os.path.join(project_root, outputs_root))


def discover_experiments(project_root: str, outputs_root: str = "outputs") -> List[str]:
    """
    Union of:
    - dirs containing logs/metrics.jsonl
    - dirs containing evaluation/*.json
    Relative paths use '/' (e.g. ru/morpheme).
    """
    root = _outputs_abs(project_root, outputs_root)
    if not os.path.isdir(root):
        return []

    found: Set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel == ".":
            rel = ""

        if "logs" in dirnames:
            metrics = os.path.join(dirpath, "logs", "metrics.jsonl")
            if os.path.isfile(metrics):
                found.add(_ALIASES.get(rel, rel) if rel != "." else "")

        if "evaluation" in dirnames:
            eval_dir = os.path.join(dirpath, "evaluation")
            try:
                has_json = any(fn.endswith(".json") for fn in os.listdir(eval_dir))
            except OSError:
                has_json = False
            if has_json:
                found.add(_ALIASES.get(rel, rel) if rel != "." else "")

    # Drop empty string if any (experiment at outputs root — unusual)
    found.discard("")
    return sorted(found)
