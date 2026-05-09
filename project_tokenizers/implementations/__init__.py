from .base import BaseTokenizer
from .word_level import WordLevelTokenizer
from .char_level import CharLevelTokenizer
from .bpe import BPETokenizer
from .unigram import UnigramTokenizer
from .wordpiece import WordPieceTokenizer
from .byte_bpe import ByteBPETokenizer
from .morpheme import MorphemeTokenizer
from .neural_segmenter import NeuralSegmenterTokenizer
from .miron import MIRONTokenizer

TOKENIZER_REGISTRY = {
    "word_level": WordLevelTokenizer,
    "char_level": CharLevelTokenizer,
    "bpe": BPETokenizer,
    "unigram": UnigramTokenizer,
    "wordpiece": WordPieceTokenizer,
    "byte_bpe": ByteBPETokenizer,
    "morpheme": MorphemeTokenizer,
    "neural_segmenter": NeuralSegmenterTokenizer,
    "miron": MIRONTokenizer
}