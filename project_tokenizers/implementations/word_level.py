import os
import json
import pyarrow.ipc
import re
import unicodedata
from collections import Counter
from .base import BaseTokenizer


class WordLevelTokenizer(BaseTokenizer):
    def __init__(self, config):
        super().__init__(config)

    def _tokenize(self, text):
        text = unicodedata.normalize("NFC", text)
        text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]|\n', text)
        return tokens

    def train(self, file_path):
        if file_path.endswith('.arrow'):
            table = pyarrow.ipc.open_file(file_path).read_all()
            text_data = table[0].to_pylist()
            word_counts = Counter()
            for line in text_data:
                tokens = self._tokenize(line)
                word_counts.update(tokens)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = self._tokenize(text)
                word_counts = Counter(tokens)
        
        valid_words = [word for word, count in word_counts.most_common(self.vocab_size - len(self.special_tokens))]
        
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        start_idx = len(self.special_tokens)
        
        for idx, word in enumerate(valid_words):
            self.vocab[word] = start_idx + idx
            
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text, add_special_tokens=False):
        tokens = self._tokenize(text)
        ids = []
        
        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])
            
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
            
        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])
            
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if token not in self.special_tokens:
                tokens.append(token)
        
        decoded = []
        for token in tokens:
            if token == "\n":
                decoded.append(token)
            elif token in [".", ",", "!", "?", ":", ";"]:
                 decoded.append(token)
            else:
                 decoded.append(token)
        
        return " ".join(decoded).replace(" \n ", "\n").replace(" .", ".").replace(" ,", ",")

    def save(self, path):
        super().save(path)
        with open(os.path.join(path, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        project_path = f"project_tokenizers/trained/{self.language}/word"
        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, path):
        super().load(path)
        vocab_path = os.path.join(path, "vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.id_to_token = {int(idx): token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)