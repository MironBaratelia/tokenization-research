"""Helper functions for data preprocessing."""
import os
from typing import Optional, Set

import pyarrow as pa
import pyarrow.ipc as ipc
import datasets


def build_sentence_end_ids(tokenizer) -> Set[int]:
    """Build set of token IDs that represent sentence endings."""
    sentence_end_tokens = {'.', '!', '?', '\n', '。', '！', '？'}
    sentence_end_ids = set()
    
    if hasattr(tokenizer, 'id_to_token'):
        for token_id, token_text in tokenizer.id_to_token.items():
            if token_text in sentence_end_tokens:
                sentence_end_ids.add(token_id)
    elif hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab
        for token_text, token_id in vocab.items():
            if token_text in sentence_end_tokens:
                sentence_end_ids.add(token_id)
    
    return sentence_end_ids


def find_smart_split_point(
    tokens, 
    max_length: int, 
    tokenizer, 
    sentence_end_ids: Optional[Set[int]] = None
) -> int:
    """Find optimal split point at sentence boundary within max_length."""
    if len(tokens) <= max_length:
        return len(tokens)
    
    if sentence_end_ids is None:
        sentence_end_ids = build_sentence_end_ids(tokenizer)
    
    if sentence_end_ids:
        for i in range(max_length - 1, max(0, max_length - 200), -1):
            if i < len(tokens):
                token = tokens[i]
                if isinstance(token, list):
                    if any(char_id in sentence_end_ids for char_id in token):
                        return i + 1
                elif token in sentence_end_ids:
                    return i + 1
    
    return max_length


def count_total_lines(file_path: str) -> int:
    """Count total lines in file (supports .arrow and text)."""
    if file_path.endswith(".arrow"):
        if os.path.isdir(file_path):
            dataset = datasets.load_from_disk(file_path)
            return len(dataset)
        reader = ipc.open_file(file_path)
        total = 0
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            total += batch.num_rows
        return total
    
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)


def iter_texts(file_path: str):
    """Iterate over text lines from file (supports .arrow and text)."""
    if file_path.endswith(".arrow"):
        if os.path.isdir(file_path):
            dataset = datasets.load_from_disk(file_path)
            col_name = "text" if "text" in dataset.column_names else dataset.column_names[0]
            for item in dataset[col_name]:
                yield item
        else:
            reader = ipc.open_file(file_path)
            col_name = "text" if "text" in reader.schema.names else reader.schema.names[0]
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                column_data = batch.column(col_name)
                for j in range(len(column_data)):
                    yield column_data[j].as_py()
    else:
        with open(file_path, "r", encoding="utf-8", buffering=1024*1024) as f:
            for line in f:
                yield line


def iter_lines(file_path: str, max_rows: Optional[int] = None):
    """Iterate lines with optional row limit."""
    count = 0
    for line in iter_texts(file_path):
        if max_rows is not None and count >= max_rows:
            break
        yield line
        count += 1


def get_pad_id(tokenizer, is_miron: bool = False) -> int:
    """Get padding token ID from tokenizer."""
    tokenizer_vocab = getattr(tokenizer, "vocab", None)
    pad_token = getattr(tokenizer, "pad_token", None)
    if tokenizer_vocab and pad_token:
        return tokenizer_vocab.get(pad_token, 0)
    elif is_miron:
        return tokenizer.char_to_id.get("<pad>", 0)
    return 0


def get_arrow_type(vocab_size: Optional[int], is_miron: bool = False):
    """Get appropriate PyArrow type based on vocab size."""
    if is_miron:
        if vocab_size < 256:
            return pa.list_(pa.list_(pa.uint8()))
        elif vocab_size < 65536:
            return pa.list_(pa.list_(pa.uint16()))
        else:
            return pa.list_(pa.list_(pa.int32()))
    else:
        if vocab_size < 256:
            return pa.list_(pa.uint8())
        elif vocab_size < 65536:
            return pa.list_(pa.uint16())
        else:
            return pa.list_(pa.int32())
