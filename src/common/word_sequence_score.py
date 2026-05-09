"""Normalized inverse Levenshtein similarity for short benchmark targets."""
from __future__ import annotations

import string
from typing import List

LEV_RANDOM_MATCH_CUTOFF = 50.0

_EDGE_STRIP = string.punctuation + "«»„\"'`´…—–−"


def _strip_word_edges(tok: str) -> str:
    return tok.strip(_EDGE_STRIP)


def tokenize_words(text: str, *, lowercase: bool = True) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for raw in text.split():
        t = _strip_word_edges(raw)
        if not t:
            continue
        out.append(t.lower() if lowercase else t)
    return out


def levenshtein_word_sequences(a: List[str], b: List[str]) -> int:
    na, nb = len(a), len(b)
    if na == 0:
        return nb
    if nb == 0:
        return na
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(na + 1):
        dp[i][0] = i
    for j in range(nb + 1):
        dp[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[na][nb]


def word_sequence_similarity_percent(generated: str, reference: str, *, lowercase: bool = True) -> float:
    """
    Benchmark methodology uses normalized inverse Levenshtein over generated
    suffix vs reference suffix. For next-word tasks we evaluate the cleaned
    surfaces directly at character level:

        (1 - Lev(gen, ref) / max(len(gen), len(ref))) * 100

    with light normalization (lowercasing + edge punctuation stripping).
    """
    gen = _strip_word_edges(generated or "")
    ref = _strip_word_edges(reference or "")
    if lowercase:
        gen = gen.lower()
        ref = ref.lower()

    if not gen and not ref:
        return 100.0
    denom = max(len(gen), len(ref))
    if denom == 0:
        return 100.0

    a = list(gen)
    b = list(ref)
    dist = levenshtein_word_sequences(a, b)
    return max(0.0, (1.0 - dist / denom) * 100.0)


def threshold_random_levenshtein(
    score: float, *, cutoff: float = LEV_RANDOM_MATCH_CUTOFF
) -> float:
    """Zero weak short-string overlap so random partial matches do not inflate benchmark scores."""
    return score if score >= cutoff else 0.0
