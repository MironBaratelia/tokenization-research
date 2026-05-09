from __future__ import annotations

import math


def safe_exp(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if not math.isfinite(x):
        return float("inf")
    if x > 700:
        return float("inf")
    return math.exp(x)
