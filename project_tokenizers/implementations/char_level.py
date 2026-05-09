import os
import json
import pyarrow.ipc
import unicodedata
from collections import Counter
from .base import BaseTokenizer


class CharLevelTokenizer(BaseTokenizer):
    FORCE_CHARS = ["й", "Й", "ё", "Ё"]

    def __init__(self, config):
        super().__init__(config)

    def _normalize(self, text):
        if not isinstance(text, str):
            text = str(text)
        return unicodedata.normalize("NFC", text)

    def train(self, file_path):
        char_counts = Counter()

        if file_path.endswith(".arrow"):
            table = pyarrow.ipc.open_file(file_path).read_all()
            text_data = table[0].to_pylist()
            for line in text_data:
                if line is None:
                    continue
                line = self._normalize(line)
                char_counts.update(line)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = self._normalize(line)
                    char_counts.update(line)

        valid_chars = [
            ch for ch, cnt in char_counts.items()
            if cnt >= self.min_frequency
        ]

        for ch in self.FORCE_CHARS:
            norm_ch = self._normalize(ch) 
            if norm_ch not in valid_chars:
                valid_chars.append(norm_ch)

        valid_chars.sort(key=lambda x: char_counts.get(x, 0), reverse=True)

        max_chars_slots = self.vocab_size - len(self.special_tokens)
        
        if len(valid_chars) > max_chars_slots:
            forced_set = set(self._normalize(c) for c in self.FORCE_CHARS)
            forced_list = [ch for ch in valid_chars if ch in forced_set]
            others_list = [ch for ch in valid_chars if ch not in forced_set]
            
            slots_for_others = max_chars_slots - len(forced_list)
            if slots_for_others < 0:
                valid_chars = forced_list[:max_chars_slots] 
            else:
                valid_chars = forced_list + others_list[:slots_for_others]

        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        start_idx = len(self.special_tokens)

        for idx, ch in enumerate(valid_chars):
            self.vocab[ch] = start_idx + idx

        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text, add_special_tokens=False):
        ids = []
        text = self._normalize(text)

        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])

        for ch in text:
            ids.append(self.vocab.get(ch, self.vocab[self.unk_token]))

        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])

        return ids

    def decode(self, ids):
        chars = []
        for idx in ids:
            idx = int(idx)
            tok = self.id_to_token.get(idx, self.unk_token)
            if tok not in self.special_tokens:
                chars.append(tok)
        return "".join(chars)

    def save(self, path):
        super().save(path)
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        project_path = f"project_tokenizers/trained/{self.language}/char"
        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, path):
        super().load(path)
        vocab_path = os.path.join(path, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.id_to_token = {int(idx): tok for tok, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
