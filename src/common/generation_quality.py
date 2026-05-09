"""Lightweight continuation text sanity checks (for tests, not model quality)."""
from __future__ import annotations

import re
from typing import Optional


def assess_continuation_quality(
    generated: str,
    min_chars: int = 12,
    min_letters: int = 4,
) -> Optional[str]:
    t = generated.strip()
    if "\ufffd" in generated:
        return "replacement_char_U+FFFD"
    if len(t) < min_chars:
        return f"too_short(len={len(t)}<{min_chars})"
    letters = len(re.findall(r"[\u0400-\u04FFa-zA-Z]", t))
    if letters < min_letters:
        return f"too_few_letters({letters}<{min_letters})"
    return None
