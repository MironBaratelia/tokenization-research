"""
Default RU/EN experiment ids for batch scripts (next-word, long-gen).
Matches outputs/{lang}/{name}; excludes miron, neural_segmenter, word_level.
"""
from __future__ import annotations

from typing import Dict, Tuple

DEFAULT_RU_NEXT_WORD_EXPERIMENTS: Tuple[str, ...] = (
    "ru/bpe_32k",
    "ru/byte_bpe_32k",
    "ru/byte_bpe_8k",
    "ru/char",
    "ru/morpheme",
    "ru/unigram_32k",
    "ru/wordpiece_32k",
)

DEFAULT_EN_NEXT_WORD_EXPERIMENTS: Tuple[str, ...] = (
    "en/bpe_32k",
    "en/byte_bpe_32k",
    "en/byte_bpe_8k",
    "en/char",
    "en/morpheme",
    "en/unigram_32k",
    "en/wordpiece_32k",
)

DEFAULT_EXPERIMENTS_BY_LANG: Dict[str, Tuple[str, ...]] = {
    "ru": DEFAULT_RU_NEXT_WORD_EXPERIMENTS,
    "en": DEFAULT_EN_NEXT_WORD_EXPERIMENTS,
}
