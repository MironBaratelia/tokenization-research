"""Unified dataset packing utilities."""
import os
import json
from array import array
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List, Any, Protocol, Tuple
import pyarrow as pa
import pyarrow.ipc as ipc

from src.common.tqdm_utils import tqdm
from src.preprocessing.helpers import (
    build_sentence_end_ids,
    find_smart_split_point,
    count_total_lines,
    iter_lines,
    get_pad_id,
    get_arrow_type,
)

class PackingStrategy(Protocol):
    def process_line(self, line: str) -> List[Any]: ...
    def get_arrow_schema(self) -> pa.Schema: ...

class TokenizedPackingStrategy:
    def __init__(self, tokenizer, context_length: int, is_miron: bool, vocab_size: int):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.is_miron = is_miron
        self.pad_id = get_pad_id(tokenizer, is_miron)
        self.sentence_end_ids = build_sentence_end_ids(tokenizer)
        self.arrow_type = get_arrow_type(vocab_size, is_miron)
        self.current_sample = []

    def feed_encoded_line(self, line_tokens: List[int]) -> List[List[int]]:
        samples: List[List[int]] = []
        lt = line_tokens
        while lt:
            space_left = self.context_length - len(self.current_sample)
            if len(lt) <= space_left:
                self.current_sample.extend(lt)
                lt = []
            else:
                if not self.current_sample:
                    split_point = find_smart_split_point(
                        lt, self.context_length, self.tokenizer, self.sentence_end_ids
                    )
                    self.current_sample.extend(lt[:split_point])
                    lt = lt[split_point:]

                if not self.is_miron and len(self.current_sample) < self.context_length:
                    self.current_sample.extend(
                        [self.pad_id] * (self.context_length - len(self.current_sample))
                    )

                samples.append(list(self.current_sample))
                self.current_sample = []
        return samples

    def process_line(self, line: str) -> List[List[int]]:
        line_tokens = self.tokenizer.encode(line.strip(), add_special_tokens=True)
        return self.feed_encoded_line(line_tokens)

    def get_arrow_schema(self) -> pa.Schema:
        return pa.schema([pa.field("input_ids", self.arrow_type)])

class LazyPackingStrategy:
    def __init__(self, tokenizer, context_length: int, is_miron: bool):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.is_miron = is_miron
        self.budget = context_length - 2
        self.current_text_buffer = ""
        self.current_token_count = 0
        self.sentence_end_chars = {'.', '!', '?', '\n', '。', '！', '？'}

    def _flush_lazy_buffer_if_nonempty(self, samples: List[str]) -> None:
        if self.current_text_buffer:
            samples.append(self.current_text_buffer.replace("\n", "\\n"))
            self.current_text_buffer = ""
            self.current_token_count = 0

    def _append_lazy_pieces(self, samples: List[str], pieces: List[Tuple[str, int]]) -> None:
        for w, w_len in pieces:
            if self.current_token_count + w_len > self.budget:
                self._flush_lazy_buffer_if_nonempty(samples)
            self.current_text_buffer += w
            self.current_token_count += w_len

    def _get_token_len(self, text: str) -> int:
        if self.is_miron:
            words = self.tokenizer._segment_text(text)
            count = 0
            for w in words:
                if w in self.tokenizer.special_tokens or w == self.tokenizer.eow_token:
                    count += 1
                else:
                    if len(w) <= self.tokenizer.max_word_length:
                        count += 1
                    else:
                        chunks = self.tokenizer._split_word_into_chunks(w)
                        count += len(chunks)
            return count
        return len(text)

    def process_line(self, line: str) -> List[str]:
        line_str = line.strip() + "\n"
        t_len = self._get_token_len(line_str)
        samples = []
        
        if self.current_token_count + t_len <= self.budget:
            self.current_text_buffer += line_str
            self.current_token_count += t_len
        else:
            self._flush_lazy_buffer_if_nonempty(samples)

            if t_len > self.budget:
                if self.is_miron:
                    pieces = [
                        (w, self._get_token_len(w))
                        for w in self.tokenizer._segment_text(line_str)
                    ]
                else:
                    pieces = [
                        (line_str[j : j + self.budget], min(self.budget, len(line_str) - j))
                        for j in range(0, len(line_str), self.budget)
                    ]
                self._append_lazy_pieces(samples, pieces)
            else:
                self.current_text_buffer = line_str
                self.current_token_count = t_len
        return samples

    def get_arrow_schema(self) -> pa.Schema:
        return pa.schema([pa.field("input_ids", pa.string())])

def _tokenize_chunk_worker(args: Tuple) -> List[List[int]]:
    lines, tokenizer_type, tokenizer_config, tokenizer_path = args
    from project_tokenizers.implementations import TOKENIZER_REGISTRY
    tok = TOKENIZER_REGISTRY[tokenizer_type](tokenizer_config)
    tok.load(tokenizer_path)
    return [tok.encode(line.strip(), add_special_tokens=True) for line in lines]


def pack_dataset(
    file_path: str,
    tokenizer,
    context_length: int,
    output_path: str,
    strategy_type: str = "tokenized",
    flush_every: int = 1000,
    max_shard_bytes: int = 500_000_000,
    max_input_rows: Optional[int] = None,
    num_workers: int = 0,
    tokenizer_type: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> int:
    """Unified packer using Strategy pattern."""
    is_miron = type(tokenizer).__name__ == "MIRONTokenizer"
    vocab_size = getattr(tokenizer, 'vocab_size', 50000)
    
    if strategy_type == "lazy":
        strategy = LazyPackingStrategy(tokenizer, context_length, is_miron)
    else:
        strategy = TokenizedPackingStrategy(tokenizer, context_length, is_miron, vocab_size)

    sink = None
    writer = None
    length_sink = None
    shard_idx = 0
    shard_paths, length_paths, shard_rows, shard_batch_sizes = [], [], [], []
    current_shard_rows = 0
    current_shard_batch_sizes = []
    total_samples = 0
    samples_buffer = []

    def open_shard():
        nonlocal sink, writer, length_sink, shard_idx
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base = os.path.splitext(os.path.basename(output_path))[0]
        shard_path = os.path.join(os.path.dirname(output_path), f"{base}.part{shard_idx:03d}.arrow")
        length_path = os.path.join(os.path.dirname(output_path), f"{base}.part{shard_idx:03d}.len")
        shard_idx += 1
        sink = pa.OSFile(shard_path, "wb")
        length_sink = open(length_path, "wb")
        shard_paths.append(shard_path)
        length_paths.append(length_path)
        writer = ipc.new_file(sink, strategy.get_arrow_schema())

    def flush():
        nonlocal samples_buffer, current_shard_rows, total_samples
        if not samples_buffer: return
        if sink is None: open_shard()
        
        batch = pa.RecordBatch.from_arrays([pa.array(samples_buffer)], schema=strategy.get_arrow_schema())
        writer.write_batch(batch)
        
        n_rows = len(samples_buffer)
        total_samples += n_rows
        current_shard_rows += n_rows
        current_shard_batch_sizes.append(n_rows)
        
        # Lengths are not strictly used by all loaders but kept for manifest
        array("I", [len(s) for s in samples_buffer]).tofile(length_sink)
        samples_buffer.clear()

        if sink.tell() >= max_shard_bytes:
            writer.close(); sink.close(); length_sink.close()
            shard_rows.append(current_shard_rows)
            shard_batch_sizes.append(list(current_shard_batch_sizes))
            current_shard_rows = 0
            current_shard_batch_sizes.clear()
            open_shard()

    total_lines = count_total_lines(file_path)
    if max_input_rows:
        total_lines = min(total_lines, max_input_rows)
    pbar = tqdm(total=total_lines, desc=f"Packing ({strategy_type})")

    use_parallel = (
        num_workers > 0
        and strategy_type == "tokenized"
        and not is_miron
        and tokenizer_type is not None
        and tokenizer_path is not None
    )

    if use_parallel:
        chunk_size = max(500, (total_lines + num_workers * 4 - 1) // (num_workers * 4))
        cfg = getattr(tokenizer, "config", None) or {}

        def make_chunks():
            current = []
            for line in iter_lines(file_path, max_input_rows):
                current.append(line)
                if len(current) >= chunk_size:
                    yield (current, tokenizer_type, dict(cfg), tokenizer_path)
                    current = []
            if current:
                yield (current, tokenizer_type, dict(cfg), tokenizer_path)

        def line_tokens_iter():
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for chunk_result in ex.map(_tokenize_chunk_worker, make_chunks()):
                    for line_tokens in chunk_result:
                        yield line_tokens

        for line_tokens in line_tokens_iter():
            new_samples = strategy.feed_encoded_line(line_tokens)
            samples_buffer.extend(new_samples)
            if len(samples_buffer) >= flush_every:
                flush()
            pbar.update(1)
    else:
        for line in iter_lines(file_path, max_input_rows):
            new_samples = strategy.process_line(line)
            samples_buffer.extend(new_samples)
            if len(samples_buffer) >= flush_every:
                flush()
            pbar.update(1)

    # Flush tail so the last partial window is not dropped.
    if strategy_type == "tokenized":
        if strategy.current_sample:
            tail = list(strategy.current_sample)
            if not is_miron and len(tail) < context_length:
                tail.extend([strategy.pad_id] * (context_length - len(tail)))
            samples_buffer.append(tail)
            strategy.current_sample.clear()
    elif strategy_type == "lazy" and strategy.current_text_buffer.strip():
        samples_buffer.append(strategy.current_text_buffer.replace("\n", "\\n"))
        strategy.current_text_buffer = ""
        strategy.current_token_count = 0

    flush()
    if writer: writer.close(); sink.close(); length_sink.close()
    if current_shard_rows > 0:
        shard_rows.append(current_shard_rows)
        shard_batch_sizes.append(current_shard_batch_sizes)

    manifest = {
        "shards": shard_paths,
        "length_files": length_paths,
        "shard_sizes": shard_rows,
        "total_samples": total_samples,
        "shard_batch_sizes": shard_batch_sizes
    }
    with open(output_path + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    pbar.close()
    return total_samples

# Legacy aliases for compatibility
def smart_pack_dataset(*args, **kwargs):
    return pack_dataset(*args, strategy_type="tokenized", **kwargs)

def smart_pack_dataset_lazy(*args, **kwargs):
    return pack_dataset(*args, strategy_type="lazy", **kwargs)
