from __future__ import annotations

from typing import List, Protocol, Sequence, Union


class _TokenizerCont(Protocol):
    is_miron: bool

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        ...


def _is_all_pad_token(token: Union[int, Sequence[int]], pad_id: int) -> bool:
    if isinstance(token, int):
        return token == pad_id
    vals = list(token)
    return bool(vals) and all(int(v) == pad_id for v in vals)


def trim_generated_ids(token_ids: Sequence[Union[int, Sequence[int]]], pad_id: int) -> List[Union[int, List[int]]]:
    t = list(token_ids)
    n = len(t)
    i = 0
    while i < n and _is_all_pad_token(t[i], pad_id):
        i += 1
    t = t[i:]
    for j, tok in enumerate(t):
        if _is_all_pad_token(tok, pad_id):
            t = t[:j]
            break
    return t


def decode_continuation(tokenizer: _TokenizerCont, prefix_ids: List[int], gen_ids: List[int]) -> str:
    if not gen_ids:
        return ""
    if tokenizer.is_miron or not prefix_ids:
        return tokenizer.decode(gen_ids)
    merged = prefix_ids + gen_ids
    full = tokenizer.decode(merged)
    pref = tokenizer.decode(prefix_ids)
    if full.startswith(pref):
        return full[len(pref) :]
    return tokenizer.decode(gen_ids)
