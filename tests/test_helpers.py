"""Tests for preprocessing helpers."""

from src.preprocessing.helpers import (
    build_sentence_end_ids,
    find_smart_split_point,
    get_pad_id,
    get_arrow_type,
)


class MockTokenizerWithIdToToken:
    def __init__(self, id_to_token=None):
        self.id_to_token = id_to_token or {}


class MockTokenizerWithVocab:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}


class TestBuildSentenceEndIds:
    def test_with_id_to_token(self):
        tokenizer = MockTokenizerWithIdToToken(
            id_to_token={0: "hello", 1: ".", 2: "!", 3: "?"}
        )
        result = build_sentence_end_ids(tokenizer)
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_with_vocab(self):
        tokenizer = MockTokenizerWithVocab(vocab={".": 0, "!": 1, "?": 2, "hello": 3})
        result = build_sentence_end_ids(tokenizer)
        assert 0 in result
        assert 1 in result
        assert 2 in result


class TestFindSmartSplitPoint:
    def test_short_tokens(self):
        tokenizer = MockTokenizerWithVocab()
        tokens = [1, 2, 3]
        result = find_smart_split_point(tokens, 10, tokenizer)
        assert result == 3

    def test_long_tokens_with_sentence_end(self):
        tokenizer = MockTokenizerWithIdToToken(id_to_token={0: "a", 1: ".", 2: "b"})
        sentence_end_ids = {1}
        tokens = [0] * 50 + [1] + [0] * 50
        result = find_smart_split_point(tokens, 100, tokenizer, sentence_end_ids)
        assert result == 51

    def test_no_sentence_end(self):
        tokenizer = MockTokenizerWithVocab()
        tokens = [1, 2, 3] * 50
        result = find_smart_split_point(tokens, 100, tokenizer)
        assert result == 100


class TestGetPadId:
    def test_standard_tokenizer(self):
        tokenizer = MockTokenizerWithVocab(vocab={"<pad>": 0, "hello": 1})
        result = get_pad_id(tokenizer, is_miron=False)
        assert result == 0

    def test_miron_tokenizer(self):
        class MironMock:
            char_to_id = {"<pad>": 5, "a": 1}

        result = get_pad_id(MironMock(), is_miron=True)
        assert result == 5

    def test_default_fallback(self):
        tokenizer = MockTokenizerWithVocab()
        result = get_pad_id(tokenizer, is_miron=False)
        assert result == 0


class TestGetArrowType:
    def test_small_vocab(self):
        result = get_arrow_type(100, is_miron=False)
        assert "uint8" in str(result)

    def test_medium_vocab(self):
        result = get_arrow_type(1000, is_miron=False)
        assert "uint16" in str(result)

    def test_large_vocab(self):
        result = get_arrow_type(100000, is_miron=False)
        assert "int32" in str(result)

    def test_miron_small(self):
        result = get_arrow_type(100, is_miron=True)
        assert "uint8" in str(result)
