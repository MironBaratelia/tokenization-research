"""Shared LaTeX escaping and JSON helpers for report tables."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def tex_escape_label(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def tex_escape_body(s: str) -> str:
    """Escape user/model text for typesetting in running text (prompts, generations)."""
    if not s:
        return ""
    t = (
        s.replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("$", "\\$")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("^", "\\textasciicircum{}")
        .replace("~", "\\textasciitilde{}")
    )
    return t.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " \\newline ")


def tex_num(x: Any, decimals: int = 4) -> str:
    if x is None:
        return "---"
    try:
        xf = float(x)
        if xf != xf:
            return "---"
        ax = abs(xf)
        if ax > 0 and ax < 1e-4:
            return f"{xf:.2e}"
        return f"{xf:.{decimals}f}"
    except (TypeError, ValueError):
        return tex_escape_label(str(x))


def tex_table_float(x: Any, decimals: int = 4) -> str:
    """Fixed decimal places for thesis tables (no scientific shorthand)."""
    if x is None:
        return "---"
    try:
        xf = float(x)
        if xf != xf:
            return "---"
        r = round(xf, decimals)
        return f"{r:.{decimals}f}"
    except (TypeError, ValueError):
        return "---"


def tex_cell_scalar(v: Any) -> str:
    if v is None:
        return "---"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return tex_table_float(v, 4)
    return tex_escape_label(str(v))


def tex_write(path: str, body: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")


def perplexity_split_sort_key(name: str) -> Tuple[int, str]:
    pref = ("test_indomain", "test", "ood_science", "ood_fiction")
    try:
        return (pref.index(name), name)
    except ValueError:
        return (len(pref), name)


def sort_split_names(names: List[str]) -> List[str]:
    return sorted(names, key=perplexity_split_sort_key)
