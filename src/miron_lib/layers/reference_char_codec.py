"""
Reference MIRON char modules (miron/miron/model): CharEncoder + CharDecoder.
Pre-LN TransformerEncoder; decoder injects h_t as position-0 context then causal chars.
Our word slots from the project tokenizer are [chars..., eow, pad...] without leading BOW;
the encoder prepends BOW (bos id) inside forward(), matching the reference layout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ReferenceCharEncoder(nn.Module):
    """Non-causal encoder over one word slot; BOW at position 0; z = output at BOW."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        word_slot_len: int,
        pad_id: int,
        bow_id: int,
        max_sequence_positions: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bow_id = bow_id
        self.d_model = d_model
        self.word_slot_len = word_slot_len
        self.max_sequence_positions = max_sequence_positions

        self.char_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_sequence_positions, d_model)
        self.dropout = nn.Dropout(dropout)
        self._pos_cache = {}

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.char_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, word_slot_len) or (N, word_slot_len), slots without leading BOW.
        Returns z: (B, T, d_model) or (N, d_model).
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        B, T, L = x.shape
        inner_len = L + 1
        if inner_len > self.max_sequence_positions:
            raise ValueError(
                f"ReferenceCharEncoder: slot length {L} (+BOW) exceeds max_sequence_positions="
                f"{self.max_sequence_positions} (raise tokenizer max_word_length or encoder_max_pos)"
            )

        bow = torch.full(
            (B * T, 1), self.bow_id, device=x.device, dtype=x.dtype
        )
        x_flat = x.reshape(B * T, L)
        x_flat = torch.cat([bow, x_flat], dim=1)

        pad_mask = x_flat == self.pad_id
        positions = torch.arange(inner_len, device=x.device).unsqueeze(0)
        h = self.char_emb(x_flat) + self.pos_emb(positions)
        h = self.dropout(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        z_flat = h[:, 0, :]
        z = z_flat.view(B, T, self.d_model)
        return z.squeeze(0) if squeeze else z

    def forward_sequence(self, x: Tensor) -> Tensor:
        """
        x: (B, T, word_slot_len) or (N, word_slot_len) without leading BOW.
        Returns all non-BOW states: (B, T, word_slot_len, d_model) or (N, word_slot_len, d_model).
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        B, T, L = x.shape
        inner_len = L + 1
        if inner_len > self.max_sequence_positions:
            raise ValueError(
                f"ReferenceCharEncoder: slot length {L} (+BOW) exceeds max_sequence_positions="
                f"{self.max_sequence_positions} (raise tokenizer max_word_length or encoder_max_pos)"
            )

        bow = torch.full((B * T, 1), self.bow_id, device=x.device, dtype=x.dtype)
        x_flat = x.reshape(B * T, L)
        x_flat = torch.cat([bow, x_flat], dim=1)

        pad_mask = x_flat == self.pad_id
        positions = torch.arange(inner_len, device=x.device).unsqueeze(0)
        h = self.char_emb(x_flat) + self.pos_emb(positions)
        h = self.dropout(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        states = h[:, 1:, :].view(B, T, L, self.d_model)
        return states.squeeze(0) if squeeze else states


class ReferenceCharDecoder(nn.Module):
    """Causal char generation conditioned on CoreLM state h_t (context token at 0)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        word_slot_len: int,
        pad_id: int,
        max_sequence_positions: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.word_slot_len = word_slot_len
        self.pad_id = pad_id
        self.max_sequence_positions = max_sequence_positions

        self.char_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_sequence_positions, d_model)
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self._pos_cache = {}
        self._causal_mask_cache = {}
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.char_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.context_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _get_positions(self, total_len: int, device: torch.device) -> Tensor:
        if self.training:
            return torch.arange(total_len, device=device).unsqueeze(0)
        key = (total_len, device)
        positions = self._pos_cache.get(key)
        if positions is None:
            positions = torch.arange(total_len, device=device).unsqueeze(0)
            self._pos_cache[key] = positions
        return positions

    def _get_causal_mask(self, total_len: int, device: torch.device) -> Tensor:
        if self.training:
            return self._causal_mask(total_len, device)
        key = (total_len, device)
        causal_mask = self._causal_mask_cache.get(key)
        if causal_mask is None:
            causal_mask = self._causal_mask(total_len, device)
            self._causal_mask_cache[key] = causal_mask
        return causal_mask

    def forward(self, h_t: Tensor, char_input: Tensor) -> Tensor:
        """
        h_t: (B, T, d_model) or (B, T, context_len, d_model)
        char_input: (B, T, word_slot_len), teacher input [<bow>, c0, ...] aligned with reference.
        Returns logits (B, T, word_slot_len, vocab_size).
        """
        B, T, s_in = char_input.shape
        if h_t.dim() == 3:
            context_len = 1
            h_ctx = h_t.reshape(B * T, 1, self.d_model)
        elif h_t.dim() == 4:
            context_len = h_t.size(2)
            h_ctx = h_t.reshape(B * T, context_len, self.d_model)
        else:
            raise ValueError("ReferenceCharDecoder expects h_t with shape (B,T,D) or (B,T,C,D)")

        total_len = context_len + s_in
        if total_len > self.max_sequence_positions:
            raise ValueError(
                f"ReferenceCharDecoder: sequence len {total_len} > max_sequence_positions="
                f"{self.max_sequence_positions}"
            )

        chars_flat = char_input.reshape(B * T, s_in)

        ctx = self.context_proj(h_ctx)
        char_embs = self.char_emb(chars_flat)
        seq = torch.cat([ctx, char_embs], dim=1)
        positions = self._get_positions(total_len, h_t.device)
        seq = self.dropout(seq + self.pos_emb(positions))

        causal_mask = self._get_causal_mask(total_len, h_t.device)
        try:
            out = self.transformer(seq, mask=causal_mask, is_causal=True)
        except TypeError:
            out = self.transformer(seq, mask=causal_mask)
        out = self.norm(out)
        out_chars = out[:, context_len:, :]
        logits_flat = self.out_proj(out_chars)
        return logits_flat.view(B, T, s_in, -1)
