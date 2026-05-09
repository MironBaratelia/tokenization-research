import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_right
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pyarrow.ipc
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from src.common.tqdm_utils import tqdm
from src.common.tokenizer_utils import get_pad_id

class DataCollator:
    """Collates variable-length sequences into batches, supporting 3D MIRON tensors."""
    def __init__(self, pad_id: int = 0, max_seq_len: int = 512, max_word_len: int = 32):
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len

    def _has_segment_supervision(self, batch: List[Dict[str, Any]]) -> bool:
        return bool(batch) and isinstance(batch[0], dict) and "boundary_targets" in batch[0]
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [item['input_ids'] for item in batch]
        has_seg_sup = self._has_segment_supervision(batch)
        has_frozen_ids = bool(batch) and isinstance(batch[0], dict) and "frozen_input_ids" in batch[0]
        
        # Detect if we have a MIRON-style 3D dataset [batch, seq, word]
        if input_ids and isinstance(input_ids[0], list) and input_ids[0] and isinstance(input_ids[0][0], list):
            max_seq_len = max(len(seq) for seq in input_ids)
            max_word_len = max(len(word) for seq in input_ids for word in seq)
            
            padded_ids = torch.full((len(input_ids), max_seq_len, max_word_len), self.pad_id, dtype=torch.long)
            padded_labels = torch.full((len(input_ids), max_seq_len, max_word_len), -100, dtype=torch.long)
            
            for i, seq in enumerate(input_ids):
                for j, word in enumerate(seq):
                    if word:
                        word_tensor = torch.tensor(word, dtype=torch.long)
                        padded_ids[i, j, :len(word)] = word_tensor
                        padded_labels[i, j, :len(word)] = word_tensor
            
            mask = (padded_ids != self.pad_id).any(dim=-1)
            if mask.all():
                return {"input_ids": padded_ids, "labels": padded_labels}
            return {
                "input_ids": padded_ids, 
                "labels": padded_labels, 
                "attention_mask": mask
            }
        
        # Standard 2D token padding
        input_ids_padded = pad_sequence([torch.as_tensor(x) for x in input_ids], batch_first=True, padding_value=self.pad_id)
        labels_padded = pad_sequence([torch.as_tensor(x) for x in input_ids], batch_first=True, padding_value=-100)
        
        mask = (input_ids_padded != self.pad_id)
        # OPTIMIZATION: Check on CPU if mask is fully packed. If so, drop it to enable Flash Attention in model.
        if mask.all():
            out: Dict[str, torch.Tensor] = {"input_ids": input_ids_padded, "labels": labels_padded}
            if has_frozen_ids:
                frozen_padded = pad_sequence(
                    [torch.as_tensor(item["frozen_input_ids"]) for item in batch],
                    batch_first=True,
                    padding_value=self.pad_id,
                )
                frozen_labels = pad_sequence(
                    [torch.as_tensor(item["frozen_input_ids"]) for item in batch],
                    batch_first=True,
                    padding_value=-100,
                )
                out["frozen_input_ids"] = frozen_padded
                out["frozen_labels"] = frozen_labels
                out["frozen_attention_mask"] = frozen_padded != self.pad_id
            if has_seg_sup:
                out["boundary_targets"] = pad_sequence(
                    [torch.as_tensor(item["boundary_targets"], dtype=torch.float32) for item in batch],
                    batch_first=True,
                    padding_value=0.0,
                )
                out["interior_targets"] = pad_sequence(
                    [torch.as_tensor(item["interior_targets"], dtype=torch.float32) for item in batch],
                    batch_first=True,
                    padding_value=0.0,
                )
            return out
            
        out2: Dict[str, torch.Tensor] = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": mask,
        }
        if has_frozen_ids:
            frozen_padded = pad_sequence(
                [torch.as_tensor(item["frozen_input_ids"]) for item in batch],
                batch_first=True,
                padding_value=self.pad_id,
            )
            frozen_labels = pad_sequence(
                [torch.as_tensor(item["frozen_input_ids"]) for item in batch],
                batch_first=True,
                padding_value=-100,
            )
            out2["frozen_input_ids"] = frozen_padded
            out2["frozen_labels"] = frozen_labels
            out2["frozen_attention_mask"] = frozen_padded != self.pad_id
        if has_seg_sup:
            out2["boundary_targets"] = pad_sequence(
                [torch.as_tensor(item["boundary_targets"], dtype=torch.float32) for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            out2["interior_targets"] = pad_sequence(
                [torch.as_tensor(item["interior_targets"], dtype=torch.float32) for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
        return out2

def _worker_init_fn(worker_id: int) -> None:
    """Set per-worker seed for reproducibility when num_workers > 0."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ChunkedSortedSampler(Sampler):
    """Sampler that groups similar-length sequences together to minimize padding noise."""
    def __init__(self, dataset: Dataset, batch_size: int, chunk_size: int = 32000, shuffle: bool = True, generator: Optional[torch.Generator] = None, seed: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.generator = generator
        self.seed = seed
        
    def __iter__(self):
        dataset_len = len(self.dataset)
        indices = list(range(dataset_len))
        
        if self.shuffle:
            if self.generator is not None:
                perm = torch.randperm(dataset_len, generator=self.generator).tolist()
                indices = [indices[i] for i in perm]
            elif self.seed is not None:
                gen = torch.Generator().manual_seed(self.seed)
                perm = torch.randperm(dataset_len, generator=gen).tolist()
                indices = [indices[i] for i in perm]
            else:
                random.shuffle(indices)

        for start in range(0, dataset_len, self.chunk_size):
            end = min(start + self.chunk_size, dataset_len)
            chunk = indices[start:end]
            if hasattr(self.dataset, "get_length"):
                chunk.sort(key=lambda idx: self.dataset.get_length(idx))
            yield from chunk
            
    def __len__(self):
        return len(self.dataset)

class LanguageModelingDataset(Dataset):
    """Packed Arrow LM data with LRU shard cache and optional pre-tokenization."""
    def __init__(
        self,
        file_path: str,
        tokenizer: Any,
        max_length: int = 512,
        max_shards_in_memory: int = 2,
        pre_tokenize_on_load: bool = True,
        context_length: Optional[int] = None,
        precompute_segment_boundaries: bool = False,
        segment_supervision_enabled: Optional[bool] = None,
        locked_runtime_enabled: bool = False,
        locked_max_length: Optional[int] = None,
    ):
        self.file_path = file_path
        self.max_length = context_length if context_length is not None else max_length
        self.tokenizer = tokenizer
        self.max_shards_in_memory = max_shards_in_memory
        self.pre_tokenize_on_load = pre_tokenize_on_load
        self.precompute_segment_boundaries = bool(precompute_segment_boundaries)
        self.locked_runtime_enabled = bool(locked_runtime_enabled)
        self.locked_max_length = int(locked_max_length) if locked_max_length is not None else None
        self.locked_runtime_batch_size = 256
        if segment_supervision_enabled is None:
            self.segment_supervision_enabled = bool(precompute_segment_boundaries)
        else:
            self.segment_supervision_enabled = bool(segment_supervision_enabled)
        self.tokenized_block_size = 4096
        self.max_tokenized_blocks_per_shard = 8
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lm-ds-prefetch")

        manifest_path = file_path if file_path.endswith(".json") else file_path + ".manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.shard_paths = manifest.get("shards", [])
        self._shard_sizes = manifest.get("shard_sizes", [])
        self._total_rows = manifest.get("total_samples", sum(self._shard_sizes))
        self._shard_offsets = np.cumsum([0] + self._shard_sizes[:-1])
        self._shard_readers = {}
        self._batch_map = {}

        for i, shard_path in enumerate(self.shard_paths):
            batch_sizes = manifest.get("shard_batch_sizes", [])[i] if "shard_batch_sizes" in manifest else None
            if batch_sizes:
                self._batch_map[shard_path] = np.cumsum([0] + batch_sizes[:-1])

    def __len__(self) -> int:
        return self._total_rows

    def get_length(self, idx: int) -> int:
        return self.max_length

    def enable_segment_supervision(self) -> None:
        changed = False
        if not self.segment_supervision_enabled:
            self.segment_supervision_enabled = True
            changed = True
        if self.locked_runtime_enabled:
            self.locked_runtime_enabled = False
            changed = True
        if not changed:
            return
        for shard_state in self._shard_readers.values():
            if isinstance(shard_state, dict):
                futures = shard_state.get("prefetch_futures", {})
                for future in futures.values():
                    try:
                        future.cancel()
                    except Exception:
                        pass
        self._shard_readers.clear()

    def _evict_shard_if_needed(self, current_shard_idx: int):
        """Evict shards when over limit. For sequential access, keep current and next."""
        if len(self._shard_readers) <= self.max_shards_in_memory:
            return
        def _idx(p):
            try:
                return self.shard_paths.index(p)
            except ValueError:
                return 9999
        to_evict = min(self._shard_readers.keys(), key=_idx)
        del self._shard_readers[to_evict]

    def _tokenize_rows(
        self,
        rows: List[Any],
        shard_path: Optional[str] = None,
    ) -> List[Any]:
        """Tokenize a small contiguous block of rows on demand."""
        from src.models.neural_segmenter_supervised import compute_gold_interior_lists

        result = []
        pad_id = int(get_pad_id(self.tokenizer, 0))
        base = getattr(self.tokenizer, "base_tokenizer", None)
        char_tok = getattr(self.tokenizer, "char_tokenizer", None)
        id_to_char = getattr(char_tok, "id_to_char", None) if char_tok is not None else None
        is_neural_segmenter = type(self.tokenizer).__name__ == "NeuralSegmenterTokenizer"
        raw_texts: List[str] = []
        locked_rows: List[List[int]] = []
        if self.locked_runtime_enabled and is_neural_segmenter and hasattr(self.tokenizer, "encode_locked_batch"):
            raw_texts = [
                value.replace("\\n", "\n") if isinstance(value, str) and "\\n" in value else value
                for value in rows
                if isinstance(value, str)
            ]
            if raw_texts:
                locked_rows = self.tokenizer.encode_locked_batch(
                    raw_texts,
                    add_special_tokens=True,
                    max_token_length=self.locked_max_length,
                    batch_size=self.locked_runtime_batch_size,
                )
        raw_text_idx = 0
        can_precompute = (
            self.segment_supervision_enabled
            and base is not None
            and id_to_char is not None
            and getattr(base, "tokenizer", None) is not None
        )
        for i, s in enumerate(rows):
            if isinstance(s, str):
                normalized_text = s.replace("\\n", "\n") if "\\n" in s else s
                if self.locked_runtime_enabled and is_neural_segmenter:
                    locked_ids = locked_rows[raw_text_idx] if raw_text_idx < len(locked_rows) else []
                    raw_text_idx += 1
                    if self.locked_max_length is not None and len(locked_ids) > self.locked_max_length:
                        locked_ids = locked_ids[: self.locked_max_length]
                    result.append({"input_ids": [pad_id], "frozen_input_ids": locked_ids})
                    continue
                if is_neural_segmenter and char_tok is not None:
                    char_ids = char_tok.encode(normalized_text)
                    bos_id = int(char_tok.char_to_id.get("<bos>", 0))
                    eos_id = int(char_tok.char_to_id.get("<eos>", 0))
                    s = [bos_id] + char_ids + [eos_id]
                else:
                    s = self.tokenizer.encode(normalized_text, add_special_tokens=True)
                if len(s) > self.max_length:
                    s = s[: self.max_length]
            else:
                if isinstance(s, list) and len(s) > self.max_length:
                    s = s[: self.max_length]
            if can_precompute and isinstance(s, list):
                g, inter = compute_gold_interior_lists(s, base, id_to_char, pad_id)
                result.append({"input_ids": s, "boundary_targets": g, "interior_targets": inter})
            else:
                result.append(s)
        return result

    def _open_shard_column(self, shard_path: str) -> Any:
        reader = pyarrow.ipc.open_file(shard_path)
        if reader.num_record_batches <= 1:
            batch = reader.get_batch(0)
            return batch.column(0)
        return reader.read_all().column(0)

    def _load_tokenized_block(self, shard_state: Dict[str, Any], shard_row: int, shard_path: str) -> List[Any]:
        block_size = self.tokenized_block_size
        block_idx = shard_row // block_size
        cached = shard_state["blocks"].get(block_idx)
        if cached is not None:
            self._schedule_next_block_prefetch(shard_state, block_idx, shard_path)
            return cached

        future = shard_state["prefetch_futures"].pop(block_idx, None)
        if future is not None:
            tokenized_rows = future.result()
        else:
            start = block_idx * block_size
            stop = min(start + block_size, shard_state["num_rows"])
            raw_rows = shard_state["column"].slice(start, stop - start).to_pylist()
            tokenized_rows = self._tokenize_rows(raw_rows, shard_path=shard_path)
        shard_state["blocks"][block_idx] = tokenized_rows
        shard_state["block_order"].append(block_idx)
        if len(shard_state["block_order"]) > self.max_tokenized_blocks_per_shard:
            evict_idx = shard_state["block_order"].pop(0)
            shard_state["blocks"].pop(evict_idx, None)
            old_future = shard_state["prefetch_futures"].pop(evict_idx, None)
            if old_future is not None:
                try:
                    old_future.cancel()
                except Exception:
                    pass
        self._schedule_next_block_prefetch(shard_state, block_idx, shard_path)
        return tokenized_rows

    def _schedule_next_block_prefetch(self, shard_state: Dict[str, Any], block_idx: int, shard_path: str) -> None:
        next_idx = int(block_idx) + 1
        num_blocks = (int(shard_state["num_rows"]) + self.tokenized_block_size - 1) // self.tokenized_block_size
        if next_idx >= num_blocks:
            return
        if next_idx in shard_state["blocks"] or next_idx in shard_state["prefetch_futures"]:
            return

        start = next_idx * self.tokenized_block_size
        stop = min(start + self.tokenized_block_size, shard_state["num_rows"])
        column = shard_state["column"]

        def _job() -> List[Any]:
            raw_rows = column.slice(start, stop - start).to_pylist()
            return self._tokenize_rows(raw_rows, shard_path=shard_path)

        shard_state["prefetch_futures"][next_idx] = self._prefetch_executor.submit(_job)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx = bisect_right(self._shard_offsets, idx) - 1
        shard_row = idx - self._shard_offsets[shard_idx]
        shard_path = self.shard_paths[shard_idx]

        if shard_path not in self._shard_readers:
            _ds_logger = logging.getLogger("Dataset")
            _ds_logger.info("Loading shard %s/%s: %s", shard_idx + 1, len(self.shard_paths), os.path.basename(shard_path))
            self._evict_shard_if_needed(shard_idx)
            if self.pre_tokenize_on_load:
                column = self._open_shard_column(shard_path)
                self._shard_readers[shard_path] = {
                    "column": column,
                    "num_rows": len(column),
                    "blocks": {},
                    "block_order": [],
                    "prefetch_futures": {},
                }
            else:
                self._shard_readers[shard_path] = self._open_shard_column(shard_path)

        data = self._shard_readers[shard_path]
        if self.pre_tokenize_on_load:
            block_rows = self._load_tokenized_block(data, shard_row, shard_path)
            block_start = (shard_row // self.tokenized_block_size) * self.tokenized_block_size
            sample = block_rows[shard_row - block_start]
        else:
            sample = data[shard_row].as_py()

        if not self.pre_tokenize_on_load and isinstance(sample, str):
            if "\\n" in sample:
                sample = sample.replace("\\n", "\n")
            sample = self.tokenizer.encode(sample, add_special_tokens=True)
            if len(sample) > self.max_length:
                sample = sample[: self.max_length]

        if (
            self.precompute_segment_boundaries
            and isinstance(sample, list)
            and getattr(self.tokenizer, "base_tokenizer", None) is not None
            and getattr(self.tokenizer.char_tokenizer, "id_to_char", None) is not None
            and getattr(self.tokenizer.base_tokenizer, "tokenizer", None) is not None
        ):
            from src.models.neural_segmenter_supervised import compute_gold_interior_lists

            pad_id = int(get_pad_id(self.tokenizer, 0))
            g, inter = compute_gold_interior_lists(
                sample, self.tokenizer.base_tokenizer, self.tokenizer.char_tokenizer.id_to_char, pad_id
            )
            return {"input_ids": sample, "boundary_targets": g, "interior_targets": inter}

        if isinstance(sample, dict):
            return sample
        return {"input_ids": sample}


class PlainTextLMDataset(Dataset):
    """Simple dataset for perplexity eval: reads .txt line-by-line, no manifest required."""
    def __init__(self, file_path: str, tokenizer: Any, max_length: int = 512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.lines[idx]
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = [tid for word in ids for tid in word]
        ids = ids[: self.max_length] if len(ids) > self.max_length else ids
        return {"input_ids": ids}


def load_data_splits(config: Dict[str, Any], tokenizer: Any) -> Tuple[DataLoader, DataLoader]:
    """Prepares Training and Validation DataLoaders with optimized settings."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    ctx_len = config['model']["max_position_embeddings"]
    bs = train_cfg["batch_size"]
    max_shards = int(data_cfg.get("max_shards_in_memory", 2))
    pre_tok = data_cfg.get("pre_tokenize_on_load", True)

    if type(tokenizer).__name__ == "NeuralSegmenterTokenizer":
        # Char-level encoder input may be longer than the LM token window.
        # The LM itself is still capped later by model.config.max_position_embeddings.
        ds_ctx_len = int(getattr(tokenizer, "char_max_length", ctx_len) or ctx_len)
    else:
        ds_ctx_len = ctx_len

    neural_supervised = (
        type(tokenizer).__name__ == "NeuralSegmenterTokenizer"
        and str(train_cfg.get("neural_train_mode", "")).lower() == "supervised_boundaries"
    )
    neural_joint_aux = (
        type(tokenizer).__name__ == "NeuralSegmenterTokenizer"
        and str(train_cfg.get("neural_train_mode", "")).lower() == "joint"
        and (
            float(train_cfg.get("lambda_boundary_bce", 0.0)) > 0.0
            or float(train_cfg.get("lambda_interior", 0.0)) > 0.0
        )
    )
    segmenter_online_training = bool(train_cfg.get("segmenter_online_training", True))
    segmenter_unfreeze_step = int(train_cfg.get("segmenter_unfreeze_step", 0))
    precompute_segment_boundaries = neural_supervised or (
        neural_joint_aux and segmenter_online_training
    )
    initial_segment_supervision_enabled = precompute_segment_boundaries and (
        neural_supervised or segmenter_unfreeze_step <= 0
    )
    initial_locked_runtime_enabled = False

    train_ds = LanguageModelingDataset(
        data_cfg['train_file'],
        tokenizer,
        max_length=ds_ctx_len,
        max_shards_in_memory=max_shards,
        pre_tokenize_on_load=pre_tok,
        precompute_segment_boundaries=precompute_segment_boundaries,
        segment_supervision_enabled=initial_segment_supervision_enabled,
        locked_runtime_enabled=initial_locked_runtime_enabled,
        locked_max_length=ctx_len,
    )
    val_ds = LanguageModelingDataset(
        data_cfg['validation_file'],
        tokenizer,
        max_length=ds_ctx_len,
        max_shards_in_memory=max_shards,
        pre_tokenize_on_load=pre_tok,
        precompute_segment_boundaries=precompute_segment_boundaries,
        segment_supervision_enabled=initial_segment_supervision_enabled,
        locked_runtime_enabled=initial_locked_runtime_enabled,
        locked_max_length=ctx_len,
    )
    
    requested_workers = train_cfg.get("num_workers", None)
    if requested_workers is None:
        num_workers = 0 if sys.platform.startswith("win") else 4
    else:
        num_workers = max(0, int(requested_workers))
    pin_memory = torch.cuda.is_available()
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    seed = train_cfg.get("seed", 42)
    
    collate = DataCollator(
        pad_id=get_pad_id(tokenizer, 0),
        max_seq_len=ds_ctx_len,
        max_word_len=getattr(tokenizer, 'max_word_length', 32)
    )
    
    # If the data is already pre-shuffled on disk, reading it sequentially
    # is perfectly random AND incredibly fast due to Arrow's linear reading.
    # If it's not shuffled, we use ChunkedSortedSampler for efficiency.
    is_shuffled = "shuffled" in data_cfg['train_file']
    
    worker_init = _worker_init_fn if num_workers > 0 else None
    
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate,
        "worker_init_fn": worker_init,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)

    if is_shuffled:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=False,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=False,
            sampler=ChunkedSortedSampler(train_ds, bs, seed=seed),
            **loader_kwargs,
        )
    
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        **loader_kwargs,
    )
    
    return train_loader, val_loader
