"""Tests for tokenizer utilities."""
import os

import pytest
from src.scripts.utils.tokenizer import resolve_train_file, resolve_check_file
from src.common.tokenizer_utils import UniversalTokenizer


class _FakeMironInner:
    tokenizer_type = "miron"
    vocab_size = 8
    pad_token_id = 0
    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    max_word_length = 8
    char_to_id = {"<pad>": 0, "a": 1, "b": 2, "c": 3}
    id_to_char = {0: "<pad>", 1: "a", 2: "b", 3: "c"}

    def decode(self, ids):
        out = []
        for i in ids:
            ch = self.id_to_char.get(i, "")
            if ch and ch != self.pad_token:
                out.append(ch)
        return "".join(out)


class TestUniversalTokenizerMironSlotDecode:
    """Regression: InferenceEngine passes tail[r].tolist() as list of word-slot lists."""

    def test_decode_nested_word_slots(self):
        ut = UniversalTokenizer(_FakeMironInner())
        assert ut.decode([[1, 2, 0], [3, 0, 0]]) == "abc"

    def test_decode_flat_int_list_unchanged(self):
        ut = UniversalTokenizer(_FakeMironInner())
        assert ut.decode([1, 2, 3]) == "abc"


class TestResolveTrainFile:
    def test_missing_config_key(self):
        config = {}
        
        with pytest.raises(ValueError, match="train_tokenizer_file not specified"):
            resolve_train_file(config, "/tmp")
    
    def test_existing_file(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp_test_tokenizer_utils")
        os.makedirs(root, exist_ok=True)
        temp_path = os.path.join(root, "train.txt")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("test")
        config = {"data": {"train_tokenizer_file": temp_path}}

        result = resolve_train_file(config, "/nonexistent")

        assert result == temp_path
    
    def test_fallback_to_txt(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp_test_tokenizer_utils")
        os.makedirs(root, exist_ok=True)
        temp_path = os.path.join(root, "fallback.txt")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("test")
        arrow_path = temp_path.replace(".txt", ".arrow")

        config = {"data": {"train_tokenizer_file": arrow_path}}

        result = resolve_train_file(config, os.path.dirname(temp_path))

        assert result == temp_path


class TestResolveCheckFile:
    def test_existing_file(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp_test_tokenizer_utils")
        check_dir = os.path.join(root, "data", "base", "en")
        os.makedirs(check_dir, exist_ok=True)
        check_file = os.path.join(check_dir, "tokenizer_check.txt")

        with open(check_file, "w", encoding="utf-8") as f:
            f.write("test")

        result = resolve_check_file(root, "en")

        assert result == check_file
    
    def test_missing_file(self):
        result = resolve_check_file("/nonexistent", "en")
        
        assert result is None
