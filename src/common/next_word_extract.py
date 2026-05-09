"""First contentful word from raw model output; tokenizer-type caps for short generation."""
from __future__ import annotations

import re
import string
from typing import Any, Dict, FrozenSet, List, Optional

MAX_NEW_TOKENS_BY_TYPE: Dict[str, int] = {
    "char_level": 56,
    "bpe": 14,
    "unigram": 14,
    "wordpiece": 14,
    "byte_bpe": 40,
    "morpheme": 22,
}

_TOKENIZER_TYPE_ALIASES: Dict[str, str] = {
    "charlevel": "char_level",
    "bytebpe": "byte_bpe",
}

SKIP_LEADING_WORDS: FrozenSet[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "it",
        "as",
        "if",
        "и",
        "в",
        "на",
        "с",
        "к",
        "у",
        "из",
        "за",
        "по",
        "о",
        "а",
        "но",
        "не",
        "что",
        "как",
        "это",
        "я",
        "он",
        "мы",
        "вы",
    }
)

_EDGE_STRIP = string.punctuation + "«»„\"'`´…—–−"


def _strip_token_edges(tok: str) -> str:
    return tok.strip(_EDGE_STRIP)


def _token_has_alnum(tok: str) -> bool:
    return any(ch.isalnum() for ch in tok)


def _continuation_tokens(tail: str, max_lines: int = 3) -> List[str]:
    toks: List[str] = []
    for line in tail.split("\n")[:max_lines]:
        for raw in line.split():
            t = _strip_token_edges(raw)
            if t and _token_has_alnum(t):
                toks.append(t)
    return toks


def _surface_is_unk_token(tok: str, unk_literal: Optional[str]) -> bool:
    """True if surface form is an UNK (excluded from first-word scoring)."""
    cf = tok.casefold()
    if unk_literal and cf == unk_literal.strip().casefold():
        return True
    if cf in ("<unk>", "[unk]"):
        return True
    if unk_literal and cf == "unk":
        return True
    return False


def first_word_from_generation(gen_text: str, unk_token: Optional[str] = None) -> str:
    if not gen_text:
        return ""
    tail = gen_text.lstrip()
    if not tail:
        return ""
    cands = _continuation_tokens(tail)
    if not cands:
        m = re.search(r"\w+", gen_text, re.UNICODE)
        return m.group(0) if m else ""
    i = 0
    if len(cands[0]) == 1 and cands[0].isalpha():
        for j in range(1, len(cands)):
            if len(cands[j]) >= 3:
                i = j
                break
    fallback = cands[i]
    for k in range(i, len(cands)):
        if _surface_is_unk_token(cands[k], unk_token):
            continue
        if cands[k].lower() not in SKIP_LEADING_WORDS:
            return cands[k]
    if _surface_is_unk_token(fallback, unk_token):
        return ""
    return fallback


def first_word_for_scoring(gen_text: str, unk_token: Optional[str] = None) -> str:
    return first_word_from_generation(gen_text, unk_token=unk_token).strip(string.punctuation)


def max_new_tokens_for_tokenizer(tokenizer: Any) -> int:
    t = getattr(tokenizer, "type", None) or getattr(tokenizer, "tokenizer_type", None)
    if isinstance(t, str):
        t = t.lower()
        t = _TOKENIZER_TYPE_ALIASES.get(t, t)
    if t in (None, "", "standard"):
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None and inner is not tokenizer:
            return max_new_tokens_for_tokenizer(inner)
        return 16
    return MAX_NEW_TOKENS_BY_TYPE.get(t, 16)
