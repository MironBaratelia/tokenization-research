import json
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import pyarrow.ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from project_tokenizers.implementations.base import BaseTokenizer
from project_tokenizers.implementations.bpe import BPETokenizer
from project_tokenizers.implementations.wordpiece import WordPieceTokenizer

from .core import (
    CharTokenizer,
    NeuralSegmenterCollator,
    NeuralSegmenterDataset,
    TransformerSegmenter,
)

logger = logging.getLogger(__name__)


class NeuralSegmenterTokenizer(BaseTokenizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.segmenter_model = None
        self.base_tokenizer = None
        self.char_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_tokenizer_type = config.get("base_tokenizer_type", "bpe")
        self.max_segment_length = config.get("max_segment_length", 512)
        self.char_max_length = int(config.get("char_max_length", self.max_segment_length))
        self.d_model = config.get("d_model", 128)
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 2)
        self.ff_mult = int(config.get("ff_mult", 4))

        self.temperature = config.get("temperature", 1.0)
        self.min_temperature = config.get("min_temperature", 0.1)
        self.temperature_decay = config.get("temperature_decay", 0.95)
        self.segmenter_pretrain_epochs = int(config.get("segmenter_pretrain_epochs", 3))
        self.segmenter_pretrain_batch_size = int(config.get("segmenter_pretrain_batch_size", 128))
        self.segmenter_pretrain_lr = float(config.get("segmenter_pretrain_lr", 1e-4))
        self.segmenter_pretrain_pos_weight = config.get("segmenter_pretrain_pos_weight")
        self.segmenter_pretrain_lambda_mean = float(config.get("segmenter_pretrain_lambda_mean", 2.0))
        self.segmenter_pretrain_lambda_interior = float(config.get("segmenter_pretrain_lambda_interior", 0.05))
        self.segmenter_pretrain_lambda_oov = float(config.get("segmenter_pretrain_lambda_oov", 0.1))
        boundary_threshold = config.get("boundary_threshold", 0.5)
        self.boundary_threshold = 0.5 if boundary_threshold is None else float(boundary_threshold)
        boundary_target_rate = config.get("boundary_target_rate")
        self.boundary_target_rate = None if boundary_target_rate is None else float(boundary_target_rate)
        boundary_selection_mode = config.get("boundary_selection_mode", "threshold")
        self.boundary_selection_mode = "threshold" if boundary_selection_mode is None else str(boundary_selection_mode)
        calib_batches = config.get("segmenter_pretrain_calibration_batches", 256)
        self.segmenter_pretrain_calibration_batches = 256 if calib_batches is None else int(calib_batches)

        locked_length_reward = config.get("locked_length_reward", 0.35)
        locked_end_boundary_weight = config.get("locked_end_boundary_weight", 0.25)
        locked_runtime_decode_mode = config.get("locked_runtime_decode_mode", "greedy_boundaries")
        self.locked_length_reward = 0.35 if locked_length_reward is None else float(locked_length_reward)
        self.locked_end_boundary_weight = (
            0.25 if locked_end_boundary_weight is None else float(locked_end_boundary_weight)
        )
        self.locked_runtime_decode_mode = (
            "greedy_boundaries" if locked_runtime_decode_mode is None else str(locked_runtime_decode_mode)
        )
        self.training_stage = config.get("training_stage", "init")
        self._base_vocab_tokens = None
        self._base_vocab_token_set = None
        self._single_token_cache: Dict[str, int | None] = {}
        self._surface_to_id: Dict[str, int] = {}
        self._surface_by_first_char: Dict[str, List[str]] = {}
        self._id_to_surface: Dict[int, str] = {}
        self._surface_trie: Dict[str, Any] = {}
        self._char_surface_trie: Dict[Any, Any] = {}
        self._single_char_token_ids: Dict[int, int] = {}
        self._max_surface_len = 0

    def train(self, file_path):
        self._train_init_stage(file_path)

    def _train_init_stage(self, file_path):
        if self.base_tokenizer_type not in ["bpe", "wordpiece"]:
            raise ValueError(f"Unsupported base tokenizer type: {self.base_tokenizer_type}")

        base_config = self.config.copy()
        base_config["vocab_size"] = int(self.vocab_size or 32000)

        if self.base_tokenizer_type == "bpe":
            base_tokenizer = BPETokenizer(base_config)
        else:
            base_tokenizer = WordPieceTokenizer(base_config)

        base_tokenizer.train(file_path)
        self.base_tokenizer = base_tokenizer
        self._refresh_base_vocab_cache()

        self.char_tokenizer = CharTokenizer()

        train_texts = []
        train_boundaries = []

        if file_path.endswith(".arrow"):
            table = pyarrow.ipc.open_file(file_path).read_all()
            col_name = "text" if "text" in table.column_names else table.column_names[0]
            raw_data = table[col_name].to_pylist()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = f.readlines()

        for line in tqdm(raw_data, desc="Preparing training data"):
            line_stripped = line.strip() if isinstance(line, str) else str(line).strip()
            if not line_stripped:
                continue

            char_len = len(line_stripped)
            boundaries = [0] * char_len
            for start, _, _ in self.surface_tokenize_with_offsets(line_stripped):
                if 0 <= start < char_len:
                    boundaries[start] = 1

            chunk_size = min(max(1, self.char_max_length), 2048)
            for i in range(0, char_len, chunk_size):
                train_texts.append(line_stripped[i : i + chunk_size])
                train_boundaries.append(boundaries[i : i + chunk_size])

        self.char_tokenizer.build_vocab(train_texts)

        segmenter_dataset = NeuralSegmenterDataset(train_texts, train_boundaries, self.char_tokenizer)
        vocab_size = len(self.char_tokenizer.char_to_id)
        self.segmenter_model = TransformerSegmenter(
            vocab_size=vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            ff_mult=self.ff_mult,
        ).to(self.device)

        self._train_segmenter_bce(segmenter_dataset)
        self._calibrate_boundary_threshold(segmenter_dataset)
        self._build_vocab(file_path)
        self.segmenter_model.eval()

    def _train_segmenter_bce(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.segmenter_pretrain_batch_size,
            shuffle=True,
            collate_fn=NeuralSegmenterCollator(self.char_tokenizer),
        )

        optimizer = torch.optim.Adam(self.segmenter_model.parameters(), lr=self.segmenter_pretrain_lr)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        pos_weight = self._compute_pretrain_pos_weight(dataset)
        vocab_seg_bytes = None
        try:
            from src.models.neural_segmenter_supervised import _build_vocab_segment_bytes

            vocab_seg_bytes = _build_vocab_segment_bytes(self)
        except Exception:
            vocab_seg_bytes = None

        self.segmenter_model.train()
        for epoch_idx in range(self.segmenter_pretrain_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            epoch_bce = 0.0
            epoch_mean = 0.0
            epoch_interior = 0.0
            epoch_oov = 0.0
            for batch in tqdm(dataloader, desc=f"Training segmenter (epoch {epoch_idx + 1}/{self.segmenter_pretrain_epochs})"):
                char_ids = batch["char_ids"].to(self.device)
                target_boundaries = batch["boundaries"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                interior_targets = batch["interior"].to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device_type):
                    _, logits = self.segmenter_model(char_ids, temperature=1.0)
                    mask = attention_mask.to(logits.dtype)
                    denom = mask.sum().clamp(min=1.0)
                    bce = F.binary_cross_entropy_with_logits(
                        logits,
                        target_boundaries,
                        reduction="none",
                        pos_weight=pos_weight,
                    )
                    bce = (bce * mask).sum() / denom

                    p = torch.sigmoid(logits.float())
                    pred_mean = ((p > 0.5).float() * mask).sum() / denom
                    gold_mean = (target_boundaries * mask).sum() / denom
                    mean_pen = (pred_mean - gold_mean) ** 2

                    interior_pen = (p * interior_targets * mask).sum() / denom

                    oov_pen = logits.new_zeros(())
                    if self.segmenter_pretrain_lambda_oov > 0.0 and vocab_seg_bytes:
                        from src.models.neural_segmenter_supervised import _oov_segment_penalty_batch

                        oov_pen = _oov_segment_penalty_batch(
                            logits,
                            char_ids,
                            int(self.char_tokenizer.char_to_id.get("<pad>", 0)),
                            vocab_seg_bytes,
                            mask,
                        )

                    loss = (
                        bce
                        + self.segmenter_pretrain_lambda_mean * mean_pen
                        + self.segmenter_pretrain_lambda_interior * interior_pen
                        + self.segmenter_pretrain_lambda_oov * oov_pen
                    )

                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().item())
                epoch_bce += float(bce.detach().item())
                epoch_mean += float(mean_pen.detach().item())
                epoch_interior += float(interior_pen.detach().item())
                epoch_oov += float(oov_pen.detach().item())
                epoch_steps += 1
            if epoch_steps:
                logger.info(
                    "[NeuralSegmenterTokenizer] segmenter_pretrain epoch=%s loss=%.4f "
                    "bce=%.4f mean=%.4f interior=%.4f oov=%.4f",
                    epoch_idx + 1,
                    epoch_loss / epoch_steps,
                    epoch_bce / epoch_steps,
                    epoch_mean / epoch_steps,
                    epoch_interior / epoch_steps,
                    epoch_oov / epoch_steps,
                )

    def _compute_pretrain_pos_weight(self, dataset) -> torch.Tensor:
        if self.segmenter_pretrain_pos_weight is not None:
            value = max(1.0, float(self.segmenter_pretrain_pos_weight))
            return torch.tensor(value, device=self.device, dtype=torch.float32)

        pos = 0
        total = 0
        for boundary in dataset.boundaries:
            pos += int(sum(boundary))
            total += int(len(boundary))
        neg = max(0, total - pos)
        if pos <= 0:
            value = 1.0
        else:
            value = max(1.0, min(8.0, float(neg) / float(pos)))
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def _calibrate_boundary_threshold(self, dataset) -> None:
        if self.segmenter_model is None:
            return
        if len(dataset) == 0:
            self.boundary_threshold = 0.5
            return

        dataloader = DataLoader(
            dataset,
            batch_size=self.segmenter_pretrain_batch_size,
            shuffle=False,
            collate_fn=NeuralSegmenterCollator(self.char_tokenizer),
        )
        max_batches = max(1, self.segmenter_pretrain_calibration_batches)
        probs_cpu = []
        gold_cpu = []
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.segmenter_model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                char_ids = batch["char_ids"].to(self.device)
                target_boundaries = batch["boundaries"]
                attention_mask = batch["attention_mask"]
                with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    _, logits = self.segmenter_model(char_ids, temperature=1.0)
                probs = torch.sigmoid(logits.float()).cpu()
                valid = attention_mask > 0.5
                probs_cpu.append(probs[valid])
                gold_cpu.append(target_boundaries[valid])

        if not probs_cpu:
            self.boundary_threshold = 0.5
            return

        probs_flat = torch.cat(probs_cpu)
        gold_flat = torch.cat(gold_cpu).float()
        target_rate = float(gold_flat.mean().item())
        self.boundary_target_rate = target_rate
        if target_rate <= 0.0:
            self.boundary_threshold = 0.5
            return

        q = max(0.0, min(1.0, 1.0 - target_rate))
        max_points = 1_000_000
        if probs_flat.numel() > max_points:
            idx = torch.randperm(probs_flat.numel())[:max_points]
            probs_flat = probs_flat[idx]
        threshold = float(torch.quantile(probs_flat, q).item())
        self.boundary_threshold = max(0.02, min(0.5, threshold))
        logger.info(
            "[NeuralSegmenterTokenizer] calibrated boundary_threshold=%.4f target_rate=%.4f",
            self.boundary_threshold,
            target_rate,
        )

    def hard_boundary_mask_from_logits(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        if self.boundary_selection_mode != "target_rate":
            hard = (probs >= float(self.boundary_threshold)) & valid_mask
            if hard.numel() > 0:
                hard[:, 0] = valid_mask[:, 0]
            return hard
        hard = torch.zeros_like(valid_mask, dtype=torch.bool)
        batch_size = int(logits.shape[0])
        for b in range(batch_size):
            row_valid = valid_mask[b]
            valid_len = int(row_valid.sum().item())
            if valid_len <= 0:
                continue
            hard[b, 0] = True
            if self.boundary_target_rate is not None and self.boundary_target_rate > 0.0:
                target_total = max(1, int(round(self.boundary_target_rate * valid_len)))
                extra = min(valid_len - 1, max(0, target_total - 1))
                if extra > 0:
                    scores = probs[b, 1:valid_len]
                    top_idx = torch.topk(scores, k=extra, dim=0).indices + 1
                    hard[b, top_idx] = True
            else:
                hard[b, :valid_len] = probs[b, :valid_len] >= float(self.boundary_threshold)
                hard[b, 0] = True
        return hard

    def _build_vocab(self, file_path):
        if self.segmenter_model is None or self.base_tokenizer is None:
            raise ValueError("Segmenter model or base tokenizer not trained")
        self.vocab = dict(self.base_tokenizer.vocab)
        self.id_to_token = dict(self.base_tokenizer.id_to_token)
        self.vocab_size = len(self.vocab)
        self._refresh_base_vocab_cache()
        self._single_token_cache.clear()

    def _refresh_base_vocab_cache(self):
        if self.base_tokenizer is None:
            self._base_vocab_tokens = []
            self._base_vocab_token_set = set()
            self._surface_to_id = {}
            self._surface_by_first_char = {}
            self._id_to_surface = {}
            self._surface_trie = {}
            self._char_surface_trie = {}
            self._single_char_token_ids = {}
            self._max_surface_len = 0
            return
        if hasattr(self.base_tokenizer, "tokenizer") and hasattr(self.base_tokenizer.tokenizer, "get_vocab"):
            base_vocab = self.base_tokenizer.tokenizer.get_vocab()
            self._base_vocab_tokens = list(base_vocab.keys())
            self._base_vocab_token_set = set(base_vocab.keys())
        else:
            self._base_vocab_tokens = list(getattr(self.base_tokenizer, "vocab", {}).keys())
            self._base_vocab_token_set = set(self._base_vocab_tokens)
        self._rebuild_surface_vocab()

    def _rebuild_surface_vocab(self) -> None:
        self._surface_to_id = {}
        self._surface_by_first_char = {}
        self._id_to_surface = {}
        self._surface_trie = {}
        self._char_surface_trie = {}
        self._single_char_token_ids = {}
        self._max_surface_len = 0
        if self.base_tokenizer is None:
            return

        special_ids = {
            tid
            for tid in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
            )
            if tid is not None
        }
        for token_str, token_id in getattr(self.base_tokenizer, "vocab", {}).items():
            token_id = int(token_id)
            if token_id in special_ids:
                continue
            surface = str(token_str)
            if not surface:
                continue
            prev = self._surface_to_id.get(surface)
            if prev is None or token_id < prev:
                self._surface_to_id[surface] = token_id
            self._id_to_surface[token_id] = surface

        for surface in self._surface_to_id.keys():
            self._max_surface_len = max(self._max_surface_len, len(surface))
            first = surface[0]
            bucket = self._surface_by_first_char.setdefault(first, [])
            bucket.append(surface)
            node = self._surface_trie
            for ch in surface:
                node = node.setdefault(ch, {})
            node["_id"] = int(self._surface_to_id[surface])

        for bucket in self._surface_by_first_char.values():
            bucket.sort(key=len, reverse=True)
        self._rebuild_char_surface_trie()

    def _rebuild_char_surface_trie(self) -> None:
        self._char_surface_trie = {}
        self._single_char_token_ids = {}
        if self.char_tokenizer is None:
            return
        unk_id = int(self.char_tokenizer.char_to_id.get("<unk>", 1))
        for surface, token_id in self._surface_to_id.items():
            char_ids = self.char_tokenizer.encode(surface)
            if not char_ids:
                continue
            if any(int(cid) == unk_id for cid in char_ids):
                continue
            node = self._char_surface_trie
            for cid in char_ids:
                node = node.setdefault(int(cid), {})
            node["_id"] = int(token_id)
            if len(char_ids) == 1:
                self._single_char_token_ids[int(char_ids[0])] = int(token_id)

    def _longest_surface_prefix(self, text: str, start: int) -> Tuple[int | None, int]:
        node = self._surface_trie
        best_id = None
        best_end = start
        limit = min(len(text), start + self._max_surface_len)
        pos = start
        while pos < limit:
            ch = text[pos]
            nxt = node.get(ch)
            if nxt is None:
                break
            node = nxt
            pos += 1
            token_id = node.get("_id")
            if token_id is not None:
                best_id = int(token_id)
                best_end = pos
        return best_id, best_end

    def _enumerate_surface_prefixes(self, text: str, start: int) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        node = self._surface_trie
        limit = min(len(text), start + self._max_surface_len)
        pos = start
        while pos < limit:
            ch = text[pos]
            nxt = node.get(ch)
            if nxt is None:
                break
            node = nxt
            pos += 1
            token_id = node.get("_id")
            if token_id is not None:
                matches.append((int(token_id), pos))
        return matches

    def _enumerate_char_id_prefixes(
        self,
        char_ids: List[int],
        start: int,
        valid_len: int,
    ) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        node = self._char_surface_trie
        pos = start
        while pos < valid_len:
            nxt = node.get(int(char_ids[pos]))
            if nxt is None:
                break
            node = nxt
            pos += 1
            token_id = node.get("_id")
            if token_id is not None:
                matches.append((int(token_id), pos))
        return matches

    def _exact_char_id_span_token_id(
        self,
        char_ids: List[int],
        start: int,
        end: int,
    ) -> int | None:
        if start >= end:
            return None
        node = self._char_surface_trie
        pos = start
        while pos < end:
            nxt = node.get(int(char_ids[pos]))
            if nxt is None:
                return None
            node = nxt
            pos += 1
        token_id = node.get("_id")
        return None if token_id is None else int(token_id)

    def _longest_char_id_prefix(
        self,
        char_ids: List[int],
        start: int,
        end: int,
    ) -> Tuple[int | None, int]:
        node = self._char_surface_trie
        pos = start
        best_id = None
        best_end = start
        while pos < end:
            nxt = node.get(int(char_ids[pos]))
            if nxt is None:
                break
            node = nxt
            pos += 1
            token_id = node.get("_id")
            if token_id is not None:
                best_id = int(token_id)
                best_end = pos
        return best_id, best_end

    def _greedy_tokenize_char_ids_with_boundaries(
        self,
        char_ids: List[int],
        boundary_starts: List[bool],
        valid_len: int | None = None,
    ) -> List[Tuple[int, int, int]]:
        if not char_ids:
            return []
        if valid_len is None:
            valid_len = len(char_ids)
        valid_len = max(0, min(int(valid_len), len(char_ids)))
        if valid_len <= 0:
            return []

        if len(boundary_starts) < valid_len:
            boundary_starts = list(boundary_starts) + [False] * (valid_len - len(boundary_starts))
        elif len(boundary_starts) > valid_len:
            boundary_starts = list(boundary_starts[:valid_len])

        start_positions = [0]
        for pos in range(1, valid_len):
            if boundary_starts[pos]:
                start_positions.append(pos)
        start_positions.append(valid_len)

        out: List[Tuple[int, int, int]] = []
        unk_id = int(self.vocab.get(self.unk_token, 0))
        for seg_idx in range(len(start_positions) - 1):
            seg_start = int(start_positions[seg_idx])
            seg_end = int(start_positions[seg_idx + 1])
            if seg_end <= seg_start:
                continue

            exact_id = self._exact_char_id_span_token_id(char_ids, seg_start, seg_end)
            if exact_id is not None:
                out.append((exact_id, seg_start, seg_end))
                continue

            pos = seg_start
            while pos < seg_end:
                token_id, end = self._longest_char_id_prefix(char_ids, pos, seg_end)
                if token_id is None or end <= pos:
                    token_id = self._single_char_token_ids.get(int(char_ids[pos]), unk_id)
                    end = pos + 1
                out.append((int(token_id), int(pos), int(end)))
                pos = int(end)
        return out

    def greedy_tokenize_char_ids_batch_with_boundaries(
        self,
        batch_char_ids: List[List[int]],
        batch_boundary_starts: List[List[bool]],
        valid_lens: List[int],
    ) -> List[List[Tuple[int, int, int]]]:
        out: List[List[Tuple[int, int, int]]] = []
        for char_ids, boundary_starts, valid_len in zip(batch_char_ids, batch_boundary_starts, valid_lens):
            out.append(
                self._greedy_tokenize_char_ids_with_boundaries(
                    [int(cid) for cid in char_ids[: int(valid_len)]],
                    boundary_starts[: int(valid_len)],
                    int(valid_len),
                )
            )
        return out

    def _locked_tokenize_char_ids_with_offsets_from_probs(
        self,
        char_ids: List[int],
        boundary_probs: List[float],
    ) -> List[Tuple[int, int, int]]:
        if not char_ids:
            return []
        valid_len = len(char_ids)
        if len(boundary_probs) < valid_len:
            boundary_probs = list(boundary_probs) + [0.5] * (valid_len - len(boundary_probs))
        elif len(boundary_probs) > valid_len:
            boundary_probs = list(boundary_probs[:valid_len])

        if not self._char_surface_trie:
            text = self.char_tokenizer.decode(char_ids) if self.char_tokenizer is not None else ""
            return [(tok_id, start, end) for tok_id, start, end in self._locked_tokenize_with_offsets_from_probs(text, boundary_probs)]

        eps = 1e-6
        log_p = [0.0] * valid_len
        log_not_p = [0.0] * valid_len
        for i, prob in enumerate(boundary_probs):
            p = min(1.0 - eps, max(eps, float(prob)))
            log_p[i] = math.log(p)
            log_not_p[i] = math.log1p(-p)

        prefix_non_boundary = [0.0] * (valid_len + 1)
        for i in range(valid_len):
            prefix_non_boundary[i + 1] = prefix_non_boundary[i] + log_not_p[i]

        dp = [-1e30] * (valid_len + 1)
        choice: List[Tuple[int, int] | None] = [None] * valid_len
        dp[valid_len] = 0.0

        for pos in range(valid_len - 1, -1, -1):
            matches = self._enumerate_char_id_prefixes(char_ids, pos, valid_len)
            if not matches:
                fallback_id = self._single_char_token_ids.get(int(char_ids[pos]), self.vocab.get(self.unk_token, 0))
                matches = [(int(fallback_id), pos + 1)]

            best_score = -1e30
            best_choice: Tuple[int, int] | None = None
            start_bonus = 0.0 if pos == 0 else log_p[pos]
            for token_id, end in matches:
                interior_pen = prefix_non_boundary[end] - prefix_non_boundary[pos + 1]
                span = end - pos
                score = start_bonus + interior_pen + self.locked_length_reward * float(max(0, span - 1)) + dp[end]
                if score > best_score:
                    best_score = score
                    best_choice = (int(token_id), int(end))
            dp[pos] = best_score
            choice[pos] = best_choice

        out: List[Tuple[int, int, int]] = []
        pos = 0
        while pos < valid_len:
            selected = choice[pos]
            if selected is None:
                fallback_id = self._single_char_token_ids.get(int(char_ids[pos]), self.vocab.get(self.unk_token, 0))
                end = pos + 1
                out.append((int(fallback_id), pos, end))
                pos = end
                continue
            token_id, end = selected
            if end <= pos:
                end = pos + 1
            out.append((int(token_id), pos, end))
            pos = end
        return out

    def locked_tokenize_char_ids_with_offsets_from_probs(
        self,
        char_ids: List[int],
        boundary_probs: List[float],
    ) -> List[Tuple[int, int, int]]:
        return self._locked_tokenize_char_ids_with_offsets_from_probs(char_ids, boundary_probs)

    def _locked_tokenize_with_offsets_from_probs(
        self,
        text: str,
        boundary_probs: List[float],
    ) -> List[Tuple[int, int, int]]:
        if not text:
            return []
        if len(boundary_probs) < len(text):
            boundary_probs = list(boundary_probs) + [0.5] * (len(text) - len(boundary_probs))
        elif len(boundary_probs) > len(text):
            boundary_probs = list(boundary_probs[: len(text)])
        char_ids = self.char_tokenizer.encode(text) if self.char_tokenizer is not None else []
        if len(char_ids) == len(text):
            return self._locked_tokenize_char_ids_with_offsets_from_probs(char_ids, boundary_probs)

        token_ids = self._locked_tokenize_from_probs(text, boundary_probs)
        out: List[Tuple[int, int, int]] = []
        pos = 0
        for token_id in token_ids:
            surface = self._id_to_surface.get(int(token_id), "")
            step = max(1, len(surface))
            end = min(len(text), pos + step)
            out.append((int(token_id), pos, end))
            pos = end
        return out

    def _locked_tokenize_from_probs(
        self,
        text: str,
        boundary_probs: List[float],
    ) -> List[int]:
        if not text:
            return []

        n = len(text)
        if n == 0:
            return []
        if len(boundary_probs) < n:
            boundary_probs = list(boundary_probs) + [0.5] * (n - len(boundary_probs))
        elif len(boundary_probs) > n:
            boundary_probs = list(boundary_probs[:n])

        if not self._surface_trie:
            return self._fallback_segment_to_base_ids(text)

        eps = 1e-6
        log_p = [0.0] * n
        log_not_p = [0.0] * n
        for i, prob in enumerate(boundary_probs):
            p = min(1.0 - eps, max(eps, float(prob)))
            log_p[i] = math.log(p)
            log_not_p[i] = math.log1p(-p)

        prefix_non_boundary = [0.0] * (n + 1)
        for i in range(n):
            prefix_non_boundary[i + 1] = prefix_non_boundary[i] + log_not_p[i]

        dp = [-1e30] * (n + 1)
        choice: List[Tuple[int, int] | None] = [None] * n
        dp[n] = 0.0

        for pos in range(n - 1, -1, -1):
            matches = self._enumerate_surface_prefixes(text, pos)
            if not matches:
                fallback_id = self._surface_to_id.get(text[pos], self.vocab.get(self.unk_token, 0))
                matches = [(int(fallback_id), pos + 1)]
            best_score = -1e30
            best_choice: Tuple[int, int] | None = None
            start_bonus = 0.0 if pos == 0 else log_p[pos]
            for token_id, end in matches:
                span = end - pos
                interior_pen = prefix_non_boundary[end] - prefix_non_boundary[pos + 1]
                score = start_bonus + interior_pen + self.locked_length_reward * float(max(0, span - 1)) + dp[end]
                if score > best_score:
                    best_score = score
                    best_choice = (int(token_id), int(end))
            dp[pos] = best_score
            choice[pos] = best_choice

        token_ids: List[int] = []
        pos = 0
        while pos < n:
            selected = choice[pos]
            if selected is None:
                token_ids.extend(self._fallback_segment_to_base_ids(text[pos : pos + 1]))
                pos += 1
                continue
            token_id, end = selected
            token_ids.append(int(token_id))
            pos = max(pos + 1, end)
        return token_ids

    def _encode_locked_char_ids_batch(
        self,
        batch_char_ids: List[List[int]],
    ) -> List[List[int]]:
        if not batch_char_ids:
            return []

        max_len = max(len(c) for c in batch_char_ids)
        pad_id = self.char_tokenizer.char_to_id.get("<pad>", 0)
        char_ids_tensor = torch.full(
            (len(batch_char_ids), max_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for row_idx, c in enumerate(batch_char_ids):
            if c:
                char_ids_tensor[row_idx, : len(c)] = torch.as_tensor(c, dtype=torch.long, device=self.device)

        temp = self.temperature
        with torch.no_grad():
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            if device_type == "cuda":
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    _, logits = self.segmenter_model(char_ids_tensor, temperature=temp)
            else:
                _, logits = self.segmenter_model(char_ids_tensor, temperature=temp)
            valid_mask = char_ids_tensor != pad_id
            if self.locked_runtime_decode_mode == "greedy_boundaries":
                hard = self.hard_boundary_mask_from_logits(logits, valid_mask).cpu().tolist()
                valid_lens = [len(char_ids) for char_ids in batch_char_ids]
                triples_batch = self.greedy_tokenize_char_ids_batch_with_boundaries(
                    batch_char_ids,
                    hard,
                    valid_lens,
                )
                return [[int(token_id) for token_id, _, _ in triples] for triples in triples_batch]

            probs = torch.sigmoid(logits.float()).cpu().tolist()

        batch_ids: List[List[int]] = []
        for row_idx, char_ids in enumerate(batch_char_ids):
            triples = self._locked_tokenize_char_ids_with_offsets_from_probs(
                [int(cid) for cid in char_ids],
                probs[row_idx][: len(char_ids)],
            )
            token_ids = [token_id for token_id, _, _ in triples]
            batch_ids.append(token_ids)
        return batch_ids

    def _segment_text(self, text):
        if not text:
            return []

        if self.segmenter_model is None:
            encoded = self.base_tokenizer.encode(text)
            decoded = self.base_tokenizer.decode(encoded)
            return decoded.split()

        char_ids = self.char_tokenizer.encode(text)
        char_len = len(char_ids)

        if char_len > self.max_segment_length:
            max_len = self.max_segment_length
            parts = []
            for i in range(0, char_len, max_len):
                parts.append(char_ids[i : i + max_len])

            all_segments = []
            batch_size = 32
            for i in range(0, len(parts), batch_size):
                batch_parts = parts[i : i + batch_size]
                part_segments_list = self._segment_char_ids_batch(batch_parts)
                for p_segs in part_segments_list:
                    all_segments.extend(p_segs)
            return all_segments

        return self._segment_char_ids_batch([char_ids])[0]

    def _segments_to_base_ids(self, segments: List[str], add_special_tokens: bool = False) -> List[int]:
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])
        for segment in segments:
            segment_id = self._segment_to_single_base_id(segment)
            if segment_id is not None:
                ids.append(int(segment_id))
            else:
                ids.extend(self._fallback_segment_to_base_ids(segment))
        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])
        return ids

    def encode_locked_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        max_token_length: int | None = None,
        batch_size: int = 64,
    ) -> List[List[int]]:
        if not texts:
            return []
        if self.segmenter_model is None or self.base_tokenizer is None:
            out: List[List[int]] = []
            for text in texts:
                out.append(self.base_tokenizer.encode(text, add_special_tokens))
            return out

        flat_parts: List[List[int]] = []
        owners: List[int] = []
        per_text_ids: List[List[int]] = [[] for _ in texts]
        max_len = max(1, int(self.max_segment_length))
        for text_idx, text in enumerate(texts):
            char_ids = self.char_tokenizer.encode(text)
            if not char_ids:
                continue
            for i in range(0, len(char_ids), max_len):
                flat_parts.append(char_ids[i : i + max_len])
                owners.append(text_idx)

        if flat_parts:
            for i in range(0, len(flat_parts), max(1, int(batch_size))):
                batch_parts = flat_parts[i : i + batch_size]
                batch_token_ids = self._encode_locked_char_ids_batch(batch_parts)
                for owner_idx, token_ids in zip(owners[i : i + batch_size], batch_token_ids):
                    per_text_ids[owner_idx].extend(token_ids)

        out_ids: List[List[int]] = []
        for token_ids in per_text_ids:
            ids = list(token_ids)
            if add_special_tokens:
                ids = [self.vocab[self.bos_token]] + ids + [self.vocab[self.eos_token]]
            if max_token_length is not None and len(ids) > int(max_token_length):
                ids = ids[: int(max_token_length)]
            out_ids.append(ids)
        return out_ids

    def debug_locked_metrics(self, text: str) -> Dict[str, float | int | bool]:
        metrics: Dict[str, float | int | bool] = {
            "exact_roundtrip": False,
            "pred_tokens": 0,
            "gold_tokens": 0,
            "span_exact_ratio": 0.0,
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
        }
        if not text or self.segmenter_model is None or self.base_tokenizer is None or self.char_tokenizer is None:
            return metrics

        gold_offsets = self.surface_tokenize_with_offsets(text)
        gold_spans = [(int(start), int(end), int(tok_id)) for start, end, tok_id in gold_offsets if end > start]
        gold_starts = {int(start) for start, _, _ in gold_spans}
        pred_ids = self.encode_locked_batch([text], add_special_tokens=False, batch_size=1)[0]
        pred_text = self.decode(pred_ids)
        metrics["exact_roundtrip"] = pred_text == text

        char_ids = self.char_tokenizer.encode(text)
        max_len = len(char_ids)
        if max_len == 0:
            return metrics
        pad_id = self.char_tokenizer.char_to_id.get("<pad>", 0)
        char_ids_tensor = torch.full((1, max_len), pad_id, dtype=torch.long, device=self.device)
        char_ids_tensor[0, :max_len] = torch.as_tensor(char_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, logits = self.segmenter_model(char_ids_tensor, temperature=self.temperature)
            probs = torch.sigmoid(logits.float())[0, :max_len].cpu().tolist()
        pred_spans = self._locked_tokenize_char_ids_with_offsets_from_probs(char_ids, probs)
        pred_starts = {int(start) for _, start, _ in pred_spans}

        gold_span_set = {(int(start), int(end), int(tok_id)) for start, end, tok_id in gold_spans}
        pred_span_set = {(int(start), int(end), int(tok_id)) for tok_id, start, end in pred_spans}
        exact_span_matches = len(gold_span_set & pred_span_set)
        tp = len(pred_starts & gold_starts)
        fp = len(pred_starts - gold_starts)
        fn = len(gold_starts - pred_starts)
        precision = float(tp) / float(max(1, tp + fp))
        recall = float(tp) / float(max(1, tp + fn))
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

        metrics["pred_tokens"] = len(pred_spans)
        metrics["gold_tokens"] = len(gold_spans)
        metrics["span_exact_ratio"] = float(exact_span_matches) / float(max(1, len(gold_spans)))
        metrics["boundary_precision"] = precision
        metrics["boundary_recall"] = recall
        metrics["boundary_f1"] = f1
        return metrics

    def _fallback_segment_to_base_ids(self, segment: str) -> List[int]:
        if not segment:
            return []
        if self.base_tokenizer is None:
            return [self.vocab.get(self.unk_token, 0)]

        pieces = self._fallback_segment_to_base_pieces(segment)
        if pieces:
            return [tok_id for tok_id, _ in pieces]
        return [self.vocab.get(self.unk_token, 0)]

    def _segment_to_single_base_id(self, segment: str) -> int | None:
        if not segment or self.base_tokenizer is None:
            return None
        cached = self._single_token_cache.get(segment, None)
        if segment in self._single_token_cache:
            return cached

        token_id = self._surface_to_id.get(segment)
        self._single_token_cache[segment] = token_id
        return token_id

    def _fallback_segment_to_base_pieces(self, segment: str) -> List[Tuple[int, int]]:
        if not segment:
            return []
        if self.base_tokenizer is None:
            return [(self.vocab.get(self.unk_token, 0), 0)]
        pieces: List[Tuple[int, int]] = []
        i = 0
        while i < len(segment):
            token_id, end = self._longest_surface_prefix(segment, i)
            if token_id is None or end <= i:
                fallback_id = self._surface_to_id.get(segment[i])
                if fallback_id is None:
                    fallback_id = self.vocab.get(self.unk_token, 0)
                pieces.append((int(fallback_id), i))
                i += 1
                continue
            pieces.append((int(token_id), i))
            i = end
        return pieces

    def surface_tokenize_with_offsets(self, text: str) -> List[Tuple[int, int, int]]:
        if not text:
            return []
        if (
            self.base_tokenizer is not None
            and hasattr(self.base_tokenizer, "tokenizer")
            and self.base_tokenizer.tokenizer is not None
        ):
            encoding = self.base_tokenizer.tokenizer.encode(text)
            offsets = list(getattr(encoding, "offsets", []) or [])
            ids = list(getattr(encoding, "ids", []) or [])
            out: List[Tuple[int, int, int]] = []
            for (start, end), token_id in zip(offsets, ids):
                start = int(start)
                end = int(end)
                token_id = int(token_id)
                if end <= start:
                    continue
                out.append((start, end, token_id))
            if out:
                return out
        pieces = self._fallback_segment_to_base_pieces(text)
        out: List[Tuple[int, int, int]] = []
        for token_id, start in pieces:
            token_text = self._id_to_surface.get(int(token_id), "")
            if not token_text:
                token_text = text[start : start + 1]
            end = min(len(text), start + len(token_text))
            if end <= start:
                end = min(len(text), start + 1)
            out.append((int(start), int(end), int(token_id)))
        return out

    def _segment_char_ids_batch(self, batch_char_ids):
        if not batch_char_ids:
            return []

        max_len = max(len(c) for c in batch_char_ids)
        pad_id = self.char_tokenizer.char_to_id.get("<pad>", 0)
        char_ids_tensor = torch.full(
            (len(batch_char_ids), max_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for row_idx, c in enumerate(batch_char_ids):
            if c:
                char_ids_tensor[row_idx, : len(c)] = torch.as_tensor(c, dtype=torch.long, device=self.device)
        temp = self.temperature

        with torch.no_grad():
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            if device_type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    _, logits = self.segmenter_model(char_ids_tensor, temperature=temp)
            else:
                _, logits = self.segmenter_model(char_ids_tensor, temperature=temp)
            valid_mask = char_ids_tensor != pad_id
            boundaries = self.hard_boundary_mask_from_logits(logits, valid_mask)

        boundaries_list = boundaries.cpu().tolist()
        all_segments = []
        id_to_char_get = self.char_tokenizer.id_to_char.get

        for b_idx, char_ids in enumerate(batch_char_ids):
            segments = []
            current_segment = []
            b_bounds = boundaries_list[b_idx]

            for char_id, is_boundary in zip(char_ids, b_bounds):
                char = id_to_char_get(char_id, "")
                if is_boundary and current_segment:
                    segments.append("".join(current_segment))
                    current_segment = [char]
                else:
                    current_segment.append(char)

            if current_segment:
                segments.append("".join(current_segment))

            all_segments.append(segments)

        return all_segments

    def encode(self, text, add_special_tokens=False):
        if not text:
            if add_special_tokens:
                return [self.vocab[self.bos_token], self.vocab[self.eos_token]]
            return []

        if self.segmenter_model is None or self.base_tokenizer is None:
            if self.base_tokenizer_type == "bpe":
                base_tokenizer = BPETokenizer(self.config)
            elif self.base_tokenizer_type == "wordpiece":
                base_tokenizer = WordPieceTokenizer(self.config)
            else:
                raise ValueError(f"Unsupported base tokenizer type: {self.base_tokenizer_type}")
            return base_tokenizer.encode(text, add_special_tokens)

        if self.training_stage in ["joint", "adaptation"]:
            char_ids = self.char_tokenizer.encode(text)
            if add_special_tokens:
                bos_id = self.char_tokenizer.char_to_id.get("<bos>", 0)
                eos_id = self.char_tokenizer.char_to_id.get("<eos>", 0)
                return [bos_id] + char_ids + [eos_id]
            return char_ids

        return self.encode_locked_batch([text], add_special_tokens=add_special_tokens, batch_size=1)[0]

    def decode(self, ids):
        if not ids:
            return ""

        if self.base_tokenizer is not None:
            special_ids = {
                tid
                for tid in (
                    self.pad_token_id,
                    self.bos_token_id,
                    self.eos_token_id,
                    self.unk_token_id,
                )
                if tid is not None
            }
            parts = []
            for token_id in ids:
                token_id = int(token_id)
                if token_id in special_ids:
                    continue
                surface = self._id_to_surface.get(token_id)
                if surface is not None:
                    parts.append(surface)
                else:
                    parts.append(self.id_to_token.get(token_id, self.unk_token))
            return "".join(parts)

        id_to_token_get = self.id_to_token.get
        unk_token = self.unk_token
        special_tokens_set = set(self.special_tokens)
        result_parts = []

        for token_id in ids:
            token = id_to_token_get(token_id, unk_token)
            if token in special_tokens_set:
                continue
            if len(token) > 7 and token[:6] == "<byte_" and token[-1] == ">":
                try:
                    byte_val = int(token[6:-1])
                    if 0 <= byte_val <= 255:
                        result_parts.append(bytes([byte_val]).decode("utf-8", errors="replace"))
                    else:
                        result_parts.append(token)
                except ValueError:
                    result_parts.append(token)
            else:
                result_parts.append(token)

        return "".join(result_parts)

    def update_temperature(self):
        if self.training_stage == "adaptation":
            self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
            return self.temperature
        return self.temperature

    def save(self, path):
        super().save(path)

        if self.segmenter_model is not None:
            torch.save(self.segmenter_model.state_dict(), os.path.join(path, "segmenter_model.pt"))

        if self.char_tokenizer is not None:
            with open(os.path.join(path, "char_tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "char_to_id": self.char_tokenizer.char_to_id,
                        "id_to_char": self.char_tokenizer.id_to_char,
                        "vocab_size": self.char_tokenizer.vocab_size,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        if self.base_tokenizer is not None:
            base_path = os.path.join(path, "base_tokenizer")
            os.makedirs(base_path, exist_ok=True)
            self.base_tokenizer.save(base_path)

        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(os.path.join(path, "neural_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_tokenizer_type": self.base_tokenizer_type,
                    "max_segment_length": self.max_segment_length,
                    "char_max_length": self.char_max_length,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "ff_mult": self.ff_mult,
                    "temperature": self.temperature,
                    "min_temperature": self.min_temperature,
                    "temperature_decay": self.temperature_decay,
                    "segmenter_pretrain_epochs": self.segmenter_pretrain_epochs,
                    "segmenter_pretrain_batch_size": self.segmenter_pretrain_batch_size,
                    "segmenter_pretrain_lr": self.segmenter_pretrain_lr,
                    "segmenter_pretrain_pos_weight": self.segmenter_pretrain_pos_weight,
                    "segmenter_pretrain_lambda_mean": self.segmenter_pretrain_lambda_mean,
                    "segmenter_pretrain_lambda_interior": self.segmenter_pretrain_lambda_interior,
                    "segmenter_pretrain_lambda_oov": self.segmenter_pretrain_lambda_oov,
                    "boundary_threshold": self.boundary_threshold,
                    "boundary_target_rate": self.boundary_target_rate,
                    "boundary_selection_mode": self.boundary_selection_mode,
                    "segmenter_pretrain_calibration_batches": self.segmenter_pretrain_calibration_batches,
                    "locked_length_reward": self.locked_length_reward,
                    "locked_end_boundary_weight": self.locked_end_boundary_weight,
                    "training_stage": self.training_stage,
                },
                f,
                indent=2,
            )

    def load(self, path):
        super().load(path)

        config_path = os.path.join(path, "neural_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                neural_config = json.load(f)
            self.base_tokenizer_type = neural_config.get("base_tokenizer_type", "bpe")
            self.max_segment_length = neural_config.get("max_segment_length", 512)
            self.char_max_length = int(neural_config.get("char_max_length", self.max_segment_length))
            self.d_model = neural_config.get("d_model", 128)
            self.nhead = neural_config.get("nhead", 8)
            self.num_layers = neural_config.get("num_layers", 2)
            self.ff_mult = int(self.config.get("ff_mult", neural_config.get("ff_mult", 4)))
            self.temperature = neural_config.get("temperature", 1.0)
            self.min_temperature = neural_config.get("min_temperature", 0.1)
            self.temperature_decay = neural_config.get("temperature_decay", 0.95)
            self.segmenter_pretrain_epochs = int(
                self.config.get("segmenter_pretrain_epochs", neural_config.get("segmenter_pretrain_epochs", 3))
            )
            self.segmenter_pretrain_batch_size = int(
                self.config.get(
                    "segmenter_pretrain_batch_size",
                    neural_config.get("segmenter_pretrain_batch_size", 128),
                )
            )
            self.segmenter_pretrain_lr = float(
                self.config.get("segmenter_pretrain_lr", neural_config.get("segmenter_pretrain_lr", 1e-4))
            )
            self.segmenter_pretrain_pos_weight = self.config.get(
                "segmenter_pretrain_pos_weight",
                neural_config.get("segmenter_pretrain_pos_weight"),
            )
            self.segmenter_pretrain_lambda_mean = float(
                self.config.get(
                    "segmenter_pretrain_lambda_mean",
                    neural_config.get("segmenter_pretrain_lambda_mean", 2.0),
                )
            )
            self.segmenter_pretrain_lambda_interior = float(
                self.config.get(
                    "segmenter_pretrain_lambda_interior",
                    neural_config.get("segmenter_pretrain_lambda_interior", 0.05),
                )
            )
            self.segmenter_pretrain_lambda_oov = float(
                self.config.get(
                    "segmenter_pretrain_lambda_oov",
                    neural_config.get("segmenter_pretrain_lambda_oov", 0.1),
                )
            )
            boundary_threshold = self.config.get("boundary_threshold", neural_config.get("boundary_threshold", 0.5))
            self.boundary_threshold = 0.5 if boundary_threshold is None else float(boundary_threshold)
            boundary_target_rate = self.config.get(
                "boundary_target_rate",
                neural_config.get("boundary_target_rate"),
            )
            self.boundary_target_rate = None if boundary_target_rate is None else float(boundary_target_rate)
            boundary_selection_mode = self.config.get(
                "boundary_selection_mode",
                neural_config.get("boundary_selection_mode", "threshold"),
            )
            self.boundary_selection_mode = "threshold" if boundary_selection_mode is None else str(boundary_selection_mode)
            calib_batches = self.config.get(
                "segmenter_pretrain_calibration_batches",
                neural_config.get("segmenter_pretrain_calibration_batches", 256),
            )
            self.segmenter_pretrain_calibration_batches = 256 if calib_batches is None else int(calib_batches)
            locked_length_reward = self.config.get(
                "locked_length_reward",
                neural_config.get("locked_length_reward", 0.35),
            )
            locked_end_boundary_weight = self.config.get(
                "locked_end_boundary_weight",
                neural_config.get("locked_end_boundary_weight", 0.25),
            )
            self.locked_length_reward = 0.35 if locked_length_reward is None else float(locked_length_reward)
            self.locked_end_boundary_weight = (
                0.25 if locked_end_boundary_weight is None else float(locked_end_boundary_weight)
            )

            if "max_segment_length" in self.config:
                self.max_segment_length = int(self.config["max_segment_length"])
            if "char_max_length" in self.config:
                self.char_max_length = int(self.config["char_max_length"])

            if "training_stage" in self.config:
                self.training_stage = self.config["training_stage"]
            else:
                self.training_stage = neural_config.get("training_stage", "init")

        char_tokenizer_path = os.path.join(path, "char_tokenizer.json")
        if os.path.exists(char_tokenizer_path):
            with open(char_tokenizer_path, "r", encoding="utf-8") as f:
                char_tokenizer_data = json.load(f)
            self.char_tokenizer = CharTokenizer()
            self.char_tokenizer.char_to_id = char_tokenizer_data["char_to_id"]
            self.char_tokenizer.id_to_char = {int(k): v for k, v in char_tokenizer_data["id_to_char"].items()}
            self.char_tokenizer.vocab_size = char_tokenizer_data["vocab_size"]
            if self._surface_to_id:
                self._rebuild_char_surface_trie()

        segmenter_path = os.path.join(path, "segmenter_model.pt")
        if os.path.exists(segmenter_path) and self.char_tokenizer is not None:
            vocab_size = len(self.char_tokenizer.char_to_id)
            self.segmenter_model = TransformerSegmenter(
                vocab_size=vocab_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                ff_mult=self.ff_mult,
            ).to(self.device)
            try:
                state = torch.load(segmenter_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(segmenter_path, map_location=self.device)
            self.segmenter_model.load_state_dict(state)
            self.segmenter_model.eval()

        base_path = os.path.join(path, "base_tokenizer")
        if os.path.exists(base_path):
            if self.base_tokenizer_type == "bpe":
                self.base_tokenizer = BPETokenizer(self.config)
            elif self.base_tokenizer_type == "wordpiece":
                self.base_tokenizer = WordPieceTokenizer(self.config)
            else:
                raise ValueError(f"Unsupported base tokenizer type: {self.base_tokenizer_type}")

            self.base_tokenizer.load(base_path)
            self._refresh_base_vocab_cache()
        elif self._surface_to_id:
            self._rebuild_char_surface_trie()

        vocab_path = os.path.join(path, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            self.id_to_token = dict(zip(self.vocab.values(), self.vocab.keys()))
        self._single_token_cache.clear()
