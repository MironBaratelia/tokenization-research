import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Tuple


class StraightThroughSigmoid(nn.Module):
    """Deterministic straight-through sigmoid (no Gumbel noise)."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.forward_with_temperature(logits, self.temperature)

    def forward_with_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        t = max(float(temperature), 1e-5)
        p_soft = torch.sigmoid(logits / t)
        y_hard = (p_soft > 0.5).to(p_soft.dtype)
        return y_hard - p_soft.detach() + p_soft


class TransformerSegmenter(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = 2048

        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(self.max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * max(1, int(ff_mult)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.boundary_proj = nn.Linear(d_model, 1)
        self.gumbel_sigmoid = StraightThroughSigmoid()

        self._pos_cache = {}
        self._init_weights()

    def _init_weights(self):
        import math

        std_emb = 0.02
        nn.init.normal_(self.char_embedding.weight, mean=0.0, std=std_emb)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=std_emb)

        fan_in = self.boundary_proj.weight.size(1)
        fan_out = self.boundary_proj.weight.size(0)
        std_proj = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.xavier_uniform_(self.boundary_proj.weight, gain=1.0)
        if self.boundary_proj.bias is not None:
            nn.init.constant_(self.boundary_proj.bias, 0)

    def forward(self, char_ids: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = char_ids.shape

        cache_key = (seq_len, char_ids.device)
        cached_positions = self._pos_cache.get(cache_key)
        use_cache = not torch.is_inference_mode_enabled()
        if cached_positions is None or getattr(cached_positions, "is_inference", lambda: False)():
            positions = torch.arange(seq_len, device=char_ids.device, dtype=torch.long, requires_grad=False)
            positions = torch.clamp(positions, max=self.max_seq_len - 1)
            if use_cache:
                self._pos_cache[cache_key] = positions
        else:
            positions = cached_positions
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        char_emb = self.char_embedding(char_ids)
        pos_emb = self.pos_embedding(positions)
        x = self.transformer(char_emb + pos_emb)

        logits = self.boundary_proj(x).squeeze(-1)
        boundaries = self.gumbel_sigmoid.forward_with_temperature(logits, temperature)
        return boundaries, logits


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.char_to_id[self.pad_token] = 0
        self.char_to_id[self.unk_token] = 1
        self.char_to_id[self.bos_token] = 2
        self.char_to_id[self.eos_token] = 3

        self.id_to_char[0] = self.pad_token
        self.id_to_char[1] = self.unk_token
        self.id_to_char[2] = self.bos_token
        self.id_to_char[3] = self.eos_token

        self.vocab_size = 4

    def build_vocab(self, texts: List[str]) -> None:
        seen_chars = set(self.char_to_id)
        for text in texts:
            for char in text:
                if char not in seen_chars:
                    char_id = self.vocab_size
                    self.char_to_id[char] = char_id
                    self.id_to_char[char_id] = char
                    self.vocab_size += 1
                    seen_chars.add(char)

    def encode(self, text: str) -> List[int]:
        if not hasattr(self, "_char_to_id_fast"):
            self._unk_id = self.char_to_id[self.unk_token]
            self._char_to_id_fast = [self.char_to_id.get(chr(i), self._unk_id) for i in range(65536)]

        try:
            return [self._char_to_id_fast[ord(c)] for c in text]
        except IndexError:
            char_to_id_get = self.char_to_id.get
            return [char_to_id_get(char, self._unk_id) for char in text]

    def decode(self, ids: List[int]) -> str:
        unk_token = self.unk_token
        id_to_char_get = self.id_to_char.get
        return "".join([id_to_char_get(idx, unk_token) for idx in ids])


class NeuralSegmenterDataset(Dataset):
    def __init__(self, texts: List[str], boundaries: List[List[int]], char_tokenizer: CharTokenizer):
        self.texts = texts
        self.boundaries = boundaries
        self.char_tokenizer = char_tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "boundaries": self.boundaries[idx],
        }


class NeuralSegmenterCollator:
    def __init__(self, char_tokenizer):
        self.char_tokenizer = char_tokenizer

    def __call__(self, batch):
        texts = [item["text"] for item in batch]
        boundaries = [item["boundaries"] for item in batch]

        max_len = max(len(text) for text in texts)
        char_to_id = self.char_tokenizer.char_to_id
        unk_id = char_to_id["<unk>"]

        char_ids = []
        target_boundaries = []
        attention_masks = []
        interior_masks = []
        char_to_id_get = char_to_id.get

        for text, boundary in zip(texts, boundaries):
            text_len = len(text)
            encoded_text = [char_to_id_get(char, unk_id) for char in text]
            if text_len < max_len:
                encoded_text.extend([0] * (max_len - text_len))

            encoded_boundary = list(boundary)
            if text_len < max_len:
                encoded_boundary.extend([0] * (max_len - text_len))

            encoded_interior = [0.0] * text_len
            for idx in range(1, text_len):
                encoded_interior[idx] = 1.0 - float(boundary[idx])
            if text_len < max_len:
                encoded_interior.extend([0.0] * (max_len - text_len))

            char_ids.append(encoded_text)
            target_boundaries.append(encoded_boundary)
            attention_masks.append([1] * text_len + [0] * (max_len - text_len))
            interior_masks.append(encoded_interior)

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "boundaries": torch.tensor(target_boundaries, dtype=torch.float),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.float),
            "interior": torch.tensor(interior_masks, dtype=torch.float),
        }
