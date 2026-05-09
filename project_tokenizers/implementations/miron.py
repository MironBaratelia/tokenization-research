import json
import os
from collections import Counter
from typing import Any, Dict, List, Union

import pyarrow.ipc as ipc
import torch

from .base import BaseTokenizer


_RU_ALPHABET = (
    "абвгдежзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "ёЁ"
)
_MOJIBAKE_MARKERS = set("ÐÑРСЃЌўџ")


def _looks_like_mojibake(text: str) -> bool:
    if not text or text.startswith("[") or text.startswith("<"):
        return False
    if len(text) == 1 and ord(text) < 128:
        return False
    return any(ch in _MOJIBAKE_MARKERS for ch in text)


def _try_unmangle_utf8(text: str) -> str:
    for enc in ("cp1251", "latin1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except Exception:
            continue
        if fixed and fixed != text:
            return fixed
    return text


def _repair_loaded_vocab(vocab: Dict[str, int]) -> Dict[str, int]:
    repaired: Dict[str, int] = {}
    for token, idx in vocab.items():
        fixed = _try_unmangle_utf8(token) if _looks_like_mojibake(token) else token
        repaired[fixed] = idx
    return repaired


class MIRONTokenizer(BaseTokenizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.d_model = config.get("d_model", 768)
        self.max_word_length = config.get("max_word_length", 32)
        self.min_char_frequency = config.get("min_char_frequency", 100)

        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.eow_token = "<eow>"

        self.punctuation = set(".,!?/:;)]}")
        self.suffix_chars = set("\n\t")
        self.prefix_chars = set(" -_&@#+*([{")

    def _build_alphabet(self, file_path: str) -> None:
        chars = set()

        chars.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if self.language == "ru":
            chars.update(_RU_ALPHABET)
        chars.update("0123456789")
        chars.update(".,!?/:;()[]{}")
        chars.update([" ", "_", "-", "&", "@", "#", "+", "*", "\n", "\t"])

        for token in self.special_tokens:
            chars.update(token)
        chars.add(self.eow_token)

        freq = Counter()
        if file_path.endswith(".arrow"):
            reader = ipc.open_file(file_path)
            col_name = "text" if "text" in reader.schema.names else reader.schema.names[0]
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                for text in batch.column(col_name).to_pylist():
                    if isinstance(text, str):
                        freq.update(text)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for text in f:
                    if isinstance(text, str):
                        freq.update(text)

        chars = {c for c, count in freq.items() if count >= self.min_char_frequency}
        if self.language == "ru":
            chars.update(_RU_ALPHABET)
        for token in self.special_tokens:
            chars.update(token)
        chars.add(self.eow_token)
        chars.add(self.unk_token)
        chars.add(self.bos_token)
        chars.add(self.eos_token)

        sorted_chars = sorted(chars)
        if self.pad_token in sorted_chars:
            sorted_chars.remove(self.pad_token)

        self.char_to_id = {self.pad_token: 0}
        for idx, char in enumerate(sorted_chars, start=1):
            self.char_to_id[char] = idx

        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def train(self, file_path: str) -> None:
        self._build_alphabet(file_path)
        self.vocab_size = len(self.char_to_id)
        self.vocab = self.char_to_id.copy()
        self.id_to_token = self.id_to_char

    def _split_on_punct_before_letter(self, word: str) -> List[str]:
        if not word:
            return []
        parts = []
        current = []
        for i, c in enumerate(word):
            current.append(c)
            if c in self.punctuation and i + 1 < len(word) and word[i + 1].isalpha():
                parts.append("".join(current))
                current = []
        if current:
            parts.append("".join(current))
        return parts

    def _segment_text(self, text: str) -> List[str]:
        if not text:
            return []

        import re

        raw_tokens = re.findall(r"\s*\S+|\s+$", text)
        tokens = []
        for tok in raw_tokens:
            tokens.extend(self._split_on_punct_before_letter(tok))
        return tokens

    def _split_word_into_chunks(self, word: str) -> List[List[str]]:
        if not word:
            return []

        chunks = []
        for i in range(0, len(word), self.max_word_length):
            chunks.append(list(word[i : i + self.max_word_length]))
        return chunks

    def encode(self, text: str, add_special_tokens: bool = False) -> List[List[int]]:
        if not self.char_to_id:
            raise ValueError("Tokenizer not initialized. Call train() or load() first.")

        char_to_id_get = self.char_to_id.get
        pad_id = char_to_id_get(self.pad_token, 0)
        eow_id = char_to_id_get(self.eow_token, 0)
        space_id = char_to_id_get(" ", 0)
        unk_id = char_to_id_get(self.unk_token, space_id)

        encoded_words = []

        if add_special_tokens:
            bos_id = char_to_id_get(self.bos_token, pad_id)
            encoded_words.append([bos_id])

        words = self._segment_text(text)
        pad_token = self.pad_token
        eos_token = self.eos_token
        bos_token = self.bos_token

        for word in words:
            if word == pad_token or word == eos_token or word == bos_token:
                token_id = char_to_id_get(word, pad_id)
                encoded_words.append([token_id])
            elif word == self.eow_token:
                encoded_words.append([eow_id])
            else:
                if len(word) <= self.max_word_length:
                    char_ids = [char_to_id_get(char, unk_id) for char in word]
                    char_ids.append(eow_id)
                    encoded_words.append(char_ids)
                else:
                    chunks = self._split_word_into_chunks(word)
                    for i, chunk_chars in enumerate(chunks):
                        char_ids = [char_to_id_get(char, unk_id) for char in chunk_chars]
                        if i == len(chunks) - 1:
                            char_ids.append(eow_id)
                        encoded_words.append(char_ids)

        if add_special_tokens:
            eos_id = char_to_id_get(self.eos_token, pad_id)
            encoded_words.append([eos_id])

        return encoded_words

    def tokenize(self, text: str) -> List[List[str]]:
        if not self.char_to_id:
            raise ValueError("Tokenizer not initialized. Call train() or load() first.")

        words = self._segment_text(text)
        tokens = []

        for word in words:
            if word in (self.pad_token, self.eos_token, self.bos_token):
                tokens.append([word])
            elif word == self.eow_token:
                tokens.append([self.eow_token])
            else:
                chunks = self._split_word_into_chunks(word)
                for i, chunk in enumerate(chunks):
                    if i == len(chunks) - 1:
                        tokens.append(chunk + [self.eow_token])
                    else:
                        tokens.append(chunk)

        return tokens

    def decode(self, ids: Union[List[int], List[List[int]], torch.Tensor]) -> str:
        if not self.id_to_char:
            raise ValueError("Tokenizer not initialized. Call train() or load() first.")

        if ids is None:
            return ""

        if isinstance(ids, torch.Tensor):
            if ids.dim() == 0:
                ids = [ids.item()]
            else:
                ids = ids.tolist()

        if not ids:
            return ""

        if isinstance(ids, list) and isinstance(ids[0], list):
            return "".join(self.decode(word) for word in ids if word)

        id_to_char_get = self.id_to_char.get
        chars = []
        for id_val in ids:
            char = id_to_char_get(id_val)
            if char and char not in (self.eow_token, self.pad_token, self.bos_token, self.eos_token):
                chars.append(char)
        return "".join(chars)

    def decode_word(self, ids: Union[List[int], torch.Tensor]) -> str:
        return self.decode(ids)

    def save(self, path: str) -> None:
        if not self.char_to_id:
            raise ValueError("Tokenizer not initialized. Call train() first.")

        os.makedirs(path, exist_ok=True)

        params = {
            "d_model": self.d_model,
            "max_word_length": self.max_word_length,
            "char_to_id": self.char_to_id,
            "id_to_char": self.id_to_char,
            "vocab_size": self.vocab_size,
        }

        with open(os.path.join(path, "miron_config.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(os.path.join(path, "miron_config.json"), "r", encoding="utf-8") as f:
            params = json.load(f)

        self.d_model = params.get("d_model", 768)
        self.max_word_length = params.get("max_word_length", 32)
        self.char_to_id = _repair_loaded_vocab(params["char_to_id"])
        self.id_to_char = {idx: tok for tok, idx in self.char_to_id.items()}
        self.vocab_size = params["vocab_size"]

        self.vocab = self.char_to_id
        self.id_to_token = self.id_to_char
