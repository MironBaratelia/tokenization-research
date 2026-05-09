"""
Supervised boundary training for the neural segmenter (no joint LM).

Loss = BCEWithLogits between encoder boundary logits and gold BPE start positions,
plus an optional penalty for high boundary probability inside a multi-character BPE token
(where gold boundary is always 0).

Optional OOV segment penalty (matches inference in NeuralSegmenterTokenizer.encode):
hard boundaries from detached sigmoid → character segments; if a segment string is not
in tokenizer.vocab (so encode would use byte fallback), penalize logits at those cuts:
high sigmoid at a bad segment start (s>0), and low sigmoid on missing interior cuts
inside multi-character OOV spans. Gradients flow through sigmoid(logits); discrete
structure is STE-style (constant from the hard pass).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _char_ids_to_text(char_ids: List[int], id_to_char: Dict[int, str], pad_id: int) -> str:
    parts: List[str] = []
    for cid in char_ids:
        if cid == pad_id:
            break
        parts.append(id_to_char.get(int(cid), ""))
    return "".join(parts)


def compute_gold_interior_lists(
    char_ids: List[int],
    tokenizer_or_base: Any,
    id_to_char: Dict[int, str],
    pad_id: int,
) -> Tuple[List[float], List[float]]:
    """CPU-side gold boundary + interior masks for one char-id sequence (length = len(char_ids))."""
    L = len(char_ids)
    gold = [0.0] * L
    interior = [0.0] * L

    text = _char_ids_to_text(char_ids, id_to_char, pad_id)
    if not text.strip():
        return gold, interior

    offsets: List[Tuple[int, int]] = []
    if hasattr(tokenizer_or_base, "surface_tokenize_with_offsets"):
        offsets = [
            (int(start), int(end))
            for start, end, _ in tokenizer_or_base.surface_tokenize_with_offsets(text)
        ]
    else:
        enc_model = getattr(tokenizer_or_base, "tokenizer", None)
        if enc_model is None:
            raise ValueError("tokenizer must expose .tokenizer or surface_tokenize_with_offsets")
        encoding = enc_model.encode(text)
        offsets = list(encoding.offsets or [])

    char_len = len(text)
    for start, end in offsets:
        if 0 <= start < L:
            gold[start] = 1.0
            if end > start + 1:
                hi = min(int(end), char_len, L)
                lo = min(start + 1, L)
                if lo < hi:
                    for j in range(lo, hi):
                        interior[j] = 1.0
    return gold, interior


def _gold_boundaries_and_interior(
    input_ids: torch.Tensor,
    tokenizer_or_base: Any,
    id_to_char: Dict[int, str],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per (batch, char_pos): gold boundary at BPE token starts; interior=1 inside multi-char BPE spans."""
    device = input_ids.device
    bsz, seqlen = input_ids.shape
    gold = torch.zeros((bsz, seqlen), device=device, dtype=torch.float32)
    interior = torch.zeros((bsz, seqlen), device=device, dtype=torch.float32)
    rows = input_ids.detach().cpu().tolist()
    for b in range(bsz):
        g, inter = compute_gold_interior_lists(rows[b], tokenizer_or_base, id_to_char, pad_id)
        for j in range(min(len(g), seqlen)):
            gold[b, j] = g[j]
            interior[b, j] = inter[j]
    return gold, interior


def _nonpad_lengths_np(ids_np: np.ndarray, pad_id: int) -> np.ndarray:
    """Per-row length of valid prefix (right-padded rows)."""
    return (ids_np != int(pad_id)).sum(axis=1).astype(np.int64, copy=False)


def _segment_ranges_from_hard(hard_row: torch.Tensor, length: int) -> List[Tuple[int, int]]:
    """Inclusive (start, end) from hard flags (1 = char index starts a new segment). Used by tests."""
    if length <= 0:
        return []
    ranges: List[Tuple[int, int]] = []
    t = 0
    hr = hard_row[:length]
    for i in range(1, length):
        if float(hr[i].item()) > 0.5:
            ranges.append((t, i - 1))
            t = i
    ranges.append((t, length - 1))
    return ranges


def _build_vocab_segment_bytes(tokenizer: Any) -> Set[bytes]:
    """Byte keys of segments that are truly encodable as one base token."""
    vocab = getattr(tokenizer, "vocab", None) or {}
    ct = getattr(tokenizer, "char_tokenizer", None)
    if ct is None:
        return set()
    out: Set[bytes] = set()
    for tok in vocab.keys():
        single_id_fn = getattr(tokenizer, "_segment_to_single_base_id", None)
        if callable(single_id_fn) and single_id_fn(tok) is None:
            continue
        try:
            ids = ct.encode(tok)
        except Exception:
            continue
        if ids:
            out.add(np.asarray(ids, dtype=np.int32).tobytes())
    return out


def _oov_append_penalty_indices(
    row: np.ndarray,
    b: int,
    s: int,
    e: int,
    vocab_seg_bytes: Set[bytes],
    idx_b_s: List[int],
    idx_s: List[int],
    idx_b_k: List[int],
    idx_k: List[int],
) -> None:
    key = np.asarray(row[s : e + 1], dtype=np.int32).tobytes()
    if key in vocab_seg_bytes:
        return
    if s > 0:
        idx_b_s.append(b)
        idx_s.append(s)
    for k in range(s + 1, e + 1):
        idx_b_k.append(b)
        idx_k.append(k)


def _oov_segment_penalty_batch(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_id: int,
    vocab_seg_bytes: Set[bytes],
    mask: torch.Tensor,
) -> torch.Tensor:
    """One CPU sync per batch; numpy segment walk; sum penalty terms on GPU via index tensors."""
    if not vocab_seg_bytes:
        return logits.new_zeros(())

    bsz, seqlen = input_ids.shape
    p = torch.sigmoid(logits.float())
    with torch.no_grad():
        hard_np = (p.detach().cpu().numpy() > 0.5)
        ids_np = input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    lengths = _nonpad_lengths_np(ids_np, pad_id)

    idx_b_s: List[int] = []
    idx_s: List[int] = []
    idx_b_k: List[int] = []
    idx_k: List[int] = []

    for b in range(bsz):
        L = int(lengths[b])
        if L <= 0:
            continue
        row = ids_np[b]
        hrow = hard_np[b]
        hb = np.zeros(L, dtype=np.bool_)
        hb[1:L] = hrow[1:L]
        t = 0
        for i in range(1, L):
            if hb[i]:
                _oov_append_penalty_indices(row, b, t, i - 1, vocab_seg_bytes, idx_b_s, idx_s, idx_b_k, idx_k)
                t = i
        _oov_append_penalty_indices(row, b, t, L - 1, vocab_seg_bytes, idx_b_s, idx_s, idx_b_k, idx_k)

    device = logits.device
    total = torch.zeros((), device=device, dtype=torch.float32)
    if idx_s:
        bi = torch.tensor(idx_s, device=device, dtype=torch.long)
        bb = torch.tensor(idx_b_s, device=device, dtype=torch.long)
        total = total + p[bb, bi].sum()
    if idx_k:
        bk = torch.tensor(idx_k, device=device, dtype=torch.long)
        bbk = torch.tensor(idx_b_k, device=device, dtype=torch.long)
        total = total + (1.0 - p[bbk, bk]).sum()

    denom = mask.sum().clamp(min=1.0)
    return total / denom


class NeuralSegmenterSupervised(nn.Module):
    """Train only TransformerSegmenter to match BPE boundaries (+ optional penalties)."""

    ignore_index = -100

    def __init__(self, segmenter: nn.Module, tokenizer: Any, config: Dict[str, Any]):
        super().__init__()
        self.segmenter = segmenter
        self.tokenizer = tokenizer
        training_cfg = config.get("training", {})
        self.lambda_interior = float(training_cfg.get("lambda_interior", 0.05))
        self.lambda_oov_segment = float(training_cfg.get("lambda_oov_segment", 0.0))
        self.pad_id = int(getattr(tokenizer, "pad_token_id", 0))
        self._vocab_seg_bytes = _build_vocab_segment_bytes(tokenizer)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del labels  # gold built from BPE, not dataloader labels
        kwargs.pop("attention_mask", None)
        boundary_targets = kwargs.pop("boundary_targets", None)
        interior_targets = kwargs.pop("interior_targets", None)

        if getattr(self.tokenizer, "base_tokenizer", None) is None:
            raise ValueError("Tokenizer must have base_tokenizer (BPE) for supervised boundaries.")
        if getattr(self.tokenizer, "char_tokenizer", None) is None:
            raise ValueError("Tokenizer must have char_tokenizer.")

        id_to_char = self.tokenizer.char_tokenizer.id_to_char
        _, logits = self.segmenter(input_ids, temperature=1.0)
        if boundary_targets is not None:
            gold = boundary_targets.to(device=logits.device, dtype=torch.float32)
            if interior_targets is not None:
                interior = interior_targets.to(device=logits.device, dtype=torch.float32)
            else:
                interior = torch.zeros_like(gold)
        else:
            gold, interior = _gold_boundaries_and_interior(input_ids, self.tokenizer, id_to_char, self.pad_id)

        mask = (input_ids != self.pad_id).to(logits.dtype)
        denom = mask.sum().clamp(min=1.0)

        bce = F.binary_cross_entropy_with_logits(logits, gold, reduction="none")
        loss = (bce * mask).sum() / denom

        interior_pen = logits.new_zeros(())
        if self.training and self.lambda_interior > 0.0:
            p = torch.sigmoid(logits.float())
            interior_pen = (p * interior * mask).sum() / denom
            loss = loss + self.lambda_interior * interior_pen

        oov_pen = logits.new_zeros(())
        if self.training and self.lambda_oov_segment > 0.0 and self._vocab_seg_bytes:
            oov_pen = _oov_segment_penalty_batch(logits, input_ids, self.pad_id, self._vocab_seg_bytes, mask)
            loss = loss + self.lambda_oov_segment * oov_pen

        return {
            "loss": loss,
            "bce": (bce * mask).sum() / denom,
            "interior_pen": interior_pen,
            "oov_pen": oov_pen,
            "logits": logits,
        }
