from __future__ import annotations

import random
from typing import List, TypeVar

T = TypeVar("T")


def reproducible_subsample(items: List[T], max_count: int | None, seed: int) -> List[T]:
    if max_count is None or len(items) <= max_count:
        return list(items)
    rng = random.Random(int(seed))
    idx = rng.sample(range(len(items)), max_count)
    idx.sort()
    return [items[i] for i in idx]


def reproducible_index_sample(n: int, max_count: int | None, seed: int) -> List[int]:
    if max_count is None or n <= max_count:
        return list(range(n))
    rng = random.Random(int(seed))
    idx = rng.sample(range(n), max_count)
    idx.sort()
    return idx
