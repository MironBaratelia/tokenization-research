"""Tests for src.common.word_sequence_score."""

from src.common.word_sequence_score import (
    levenshtein_word_sequences,
    threshold_random_levenshtein,
    tokenize_words,
    word_sequence_similarity_percent,
)


class TestLevenshteinWordSequences:
    def test_empty(self):
        assert levenshtein_word_sequences([], []) == 0
        assert levenshtein_word_sequences([], ["a"]) == 1
        assert levenshtein_word_sequences(["a"], []) == 1

    def test_equal(self):
        assert levenshtein_word_sequences(["a", "b"], ["a", "b"]) == 0

    def test_substitute(self):
        assert levenshtein_word_sequences(["x"], ["y"]) == 1


class TestWordSequenceSimilarityPercent:
    """Character-level normalized Levenshtein for already extracted benchmark answers."""

    def test_wrong_long_tail_is_penalized_if_not_extracted(self):
        gen = "sqrt8 " + "foo " * 10
        ref = "4"
        s = word_sequence_similarity_percent(gen, ref)
        assert s == 0.0

    def test_exact_extracted_answer(self):
        assert word_sequence_similarity_percent("4", "4") == 100.0

    def test_multiword_surface_similarity(self):
        gen = "hello world"
        ref = "hello there"
        wa = tokenize_words(gen)
        wb = tokenize_words(ref)
        assert len(wb) == 2
        d = levenshtein_word_sequences(wa, wb)
        assert d == 1  # world vs there
        s = word_sequence_similarity_percent(gen, ref)
        assert round(s, 2) == 54.55

    def test_empty_reference_no_gen(self):
        assert word_sequence_similarity_percent("", "") == 100.0

    def test_empty_reference_with_gen(self):
        assert word_sequence_similarity_percent("hello", "") == 0.0

    def test_case_insensitive(self):
        assert word_sequence_similarity_percent("Hello", "hello") == 100.0

    def test_threshold_random_levenshtein(self):
        assert threshold_random_levenshtein(49.99) == 0.0
        assert threshold_random_levenshtein(50.0) == 50.0
