import os
import json
from array import array
import pyarrow.ipc
import pyarrow as pa
from src.common.tqdm_utils import tqdm
from typing import Dict, Any, List, Optional

import datasets
HAS_DATASETS = True


def materialize_locked_token_cache(
    manifest_path: str,
    tokenizer,
    frozen_max_length: int,
    batch_size: int = 256,
) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    shard_paths = manifest.get("shards", [])
    if not shard_paths:
        return

    for shard_path in tqdm(shard_paths, desc="Frozen token cache"):
        root, ext = os.path.splitext(shard_path)
        cache_path = f"{root}.frozen{ext}"
        if os.path.exists(cache_path):
            continue

        table = pyarrow.ipc.open_file(shard_path).read_all()
        col = table.column("input_ids")
        texts = []
        for i in range(len(col)):
            value = col[i].as_py()
            if isinstance(value, str) and "\\n" in value:
                value = value.replace("\\n", "\n")
            texts.append(value)

        prev_stage = getattr(tokenizer, "training_stage", "locked")
        try:
            tokenizer.training_stage = "locked"
            frozen_rows: List[List[int]] = []
            for i in range(0, len(texts), max(1, int(batch_size))):
                frozen_rows.extend(
                    tokenizer.encode_locked_batch(
                        texts[i : i + batch_size],
                        add_special_tokens=True,
                        max_token_length=frozen_max_length,
                    )
                )
        finally:
            tokenizer.training_stage = prev_stage

        batch = pa.RecordBatch.from_arrays(
            [pa.array(frozen_rows)],
            names=["frozen_input_ids"],
        )
        with pa.OSFile(cache_path, "wb") as sink:
            with pyarrow.ipc.new_file(sink, batch.schema) as writer:
                writer.write_batch(batch)


def _iter_texts_neural(file_path):
    if file_path.endswith(".arrow"):
        if os.path.isdir(file_path):
            if not HAS_DATASETS:
                raise ImportError("datasets library is required for loading directory-based arrow files")
            dataset = datasets.load_from_disk(file_path)
            col_name = "text" if "text" in dataset.column_names else dataset.column_names[0]
            for item in dataset[col_name]:
                yield item
        else:
            reader = pyarrow.ipc.open_file(file_path)
            col_name = "text" if "text" in reader.schema.names else reader.schema.names[0]
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                for item in batch.column(col_name).to_pylist():
                    yield item
    else:
        with open(file_path, "r", encoding="utf-8", buffering=1024*1024) as f:
            for line in f:
                yield line


def smart_pack_dataset_neural(
    file_path,
    tokenizer,
    context_length,
    output_path,
    max_chars_per_line: int = 1000,
    flush_every: int = 5000,
    max_shard_bytes: int = 1_000_000_000,
    max_input_rows: Optional[int] = None
):
    samples = []
    tokenizer_vocab = getattr(tokenizer, 'vocab', None)
    pad_token = getattr(tokenizer, 'pad_token', None)
    pad_id = tokenizer_vocab.get(pad_token, 0) if tokenizer_vocab and pad_token else 0
    total_samples = 0
    sink = None
    writer = None
    length_sink = None
    shard_idx = 0
    shard_paths = []
    shard_rows = []
    shard_batch_sizes = []
    current_shard_batch_sizes = []
    length_paths = []
    current_shard_rows = 0

    def _open_shard():
        nonlocal sink, writer, length_sink, shard_idx
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base = os.path.splitext(os.path.basename(output_path))[0]
        shard_name = f"{base}.part{shard_idx:03d}.arrow"
        shard_path = os.path.join(os.path.dirname(output_path), shard_name)
        length_name = f"{base}.part{shard_idx:03d}.len"
        length_path = os.path.join(os.path.dirname(output_path), length_name)
        shard_idx += 1
        sink = pa.OSFile(shard_path, "wb")
        writer = None
        length_sink = open(length_path, "wb")
        shard_paths.append(shard_path)
        length_paths.append(length_path)

    def flush():
        nonlocal samples, writer, sink, length_sink, total_samples, current_shard_rows, current_shard_batch_sizes
        if not samples:
            return
        n_rows = len(samples)
        arrays = [pa.array(samples)]
        batch = pa.RecordBatch.from_arrays(arrays, names=["input_ids"])
        if sink is None:
            _open_shard()
        if writer is None:
            writer = pa.ipc.new_file(sink, batch.schema)
        writer.write_batch(batch)
        total_samples += n_rows
        current_shard_rows += n_rows
        current_shard_batch_sizes.append(n_rows)
        if length_sink is not None:
            lens = array("I", lengths)
            lens.tofile(length_sink)
        samples = []
        lengths.clear()

        if max_shard_bytes and sink.tell() >= max_shard_bytes:
            writer.close()
            sink.close()
            sink = None
            writer = None
            if length_sink is not None:
                length_sink.close()
                length_sink = None
            shard_rows.append(current_shard_rows)
            shard_batch_sizes.append(current_shard_batch_sizes)
            current_shard_rows = 0
            current_shard_batch_sizes = []
    
    lengths = []
    processed_rows = 0
    for line in tqdm(_iter_texts_neural(file_path), desc="Neural packing"):
        if max_input_rows is not None and processed_rows >= max_input_rows:
            break
        line = line.strip() if isinstance(line, str) else str(line).strip()
        if not line:
            continue
        processed_rows += 1
            
        if len(line) > max_chars_per_line:
            line = line[:max_chars_per_line]
        
        line_tokens = tokenizer.encode(line, add_special_tokens=True)
        
        if len(line_tokens) > context_length:
            remaining_tokens = line_tokens
            while remaining_tokens:
                split_point = find_smart_split_point_neural(remaining_tokens, context_length, tokenizer)
                chunk_tokens = remaining_tokens[:split_point]
                
                if len(chunk_tokens) < context_length:
                    chunk_tokens.extend([pad_id] * (context_length - len(chunk_tokens)))
                
                samples.append(chunk_tokens)
                lengths.append(len(chunk_tokens))
                remaining_tokens = remaining_tokens[split_point:]
        else:
            if len(line_tokens) < context_length:
                line_tokens.extend([pad_id] * (context_length - len(line_tokens)))
            
            samples.append(line_tokens)
            lengths.append(len(line_tokens))

        if len(samples) >= flush_every:
            flush()
        
    flush()

    if writer is not None:
        writer.close()
    if sink is not None:
        sink.close()
    if length_sink is not None:
        length_sink.close()
    if shard_paths and current_shard_rows > 0:
        shard_rows.append(current_shard_rows)
        shard_batch_sizes.append(current_shard_batch_sizes)

    if shard_paths:
        manifest_path = output_path + ".manifest.json"
        manifest = {
            "shards": shard_paths,
            "length_files": length_paths,
            "length_dtype": "uint32",
            "shard_sizes": shard_rows,
            "total_samples": total_samples,
        }
        if shard_batch_sizes:
            manifest["shard_batch_sizes"] = shard_batch_sizes
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            
    return total_samples


def find_smart_split_point_neural(tokens, max_length, tokenizer):
    if len(tokens) <= max_length:
        return len(tokens)
    
    for i in range(max_length - 1, max(0, max_length - 200), -1):
        if i < len(tokens):
            token_id = tokens[i]
            if hasattr(tokenizer, 'id_to_token'):
                token_text = tokenizer.id_to_token.get(token_id, '')
            elif hasattr(tokenizer, 'decode'):
                token_text = tokenizer.decode([token_id])
            else:
                continue
            if token_text in ['.', '!', '?', '\n']:
                return i + 1
    
    return max_length


def analyze_tokenization_distribution(tokenizer, file_path, num_samples: int = 1000):
    raw_data = []
    if file_path.endswith('.arrow'):
        table = pyarrow.ipc.open_file(file_path).read_all()
        col_name = 'text' if 'text' in table.column_names else table.column_names[0]
        raw_data = table[col_name].to_pylist()
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = f.readlines()
    
    if len(raw_data) > num_samples:
        raw_data = raw_data[:num_samples]
    
    token_lengths = []
    segment_lengths = []
    
    for line in tqdm(raw_data, desc="Analyzing"):
        if isinstance(line, str):
            line = line.strip()
        else:
            line = str(line).strip()
            
        if not line:
            continue
            
        tokens = tokenizer.encode(line, add_special_tokens=True)
        token_lengths.append(len(tokens))
        
        segments = tokenizer._segment_text(line)
        segment_lengths.extend([len(seg) for seg in segments])
    
    sorted_token_lengths = sorted(token_lengths)
    sorted_segment_lengths = sorted(segment_lengths)
    
    token_len = len(token_lengths)
    segment_len = len(segment_lengths)
    
    stats = {
        'token_lengths': {
            'mean': sum(token_lengths) / token_len if token_len > 0 else 0,
            'min': min(token_lengths) if token_lengths else 0,
            'max': max(token_lengths) if token_lengths else 0,
            'median': sorted_token_lengths[token_len // 2] if sorted_token_lengths else 0,
            'p95': sorted_token_lengths[int(token_len * 0.95)] if sorted_token_lengths else 0,
        },
        'segment_lengths': {
            'mean': sum(segment_lengths) / segment_len if segment_len > 0 else 0,
            'min': min(segment_lengths) if segment_lengths else 0,
            'max': max(segment_lengths) if segment_lengths else 0,
            'median': sorted_segment_lengths[segment_len // 2] if sorted_segment_lengths else 0,
            'p95': sorted_segment_lengths[int(segment_len * 0.95)] if sorted_segment_lengths else 0,
        },
        'samples_analyzed': len(raw_data),
        'total_tokens': sum(token_lengths),
        'total_segments': len(segment_lengths),
    }
    
    return stats


def create_neural_dataset_config(tokenizer, file_path, output_path):
    stats = analyze_tokenization_distribution(tokenizer, file_path)
    
    config = {
        'tokenizer_type': 'neural_segmenter',
        'vocab_size': tokenizer.vocab_size,
        'base_tokenizer_type': tokenizer.base_tokenizer_type,
        'max_segment_length': tokenizer.max_segment_length,
        'd_model': tokenizer.d_model,
        'nhead': tokenizer.nhead,
        'num_layers': tokenizer.num_layers,
        'temperature': tokenizer.temperature,
        'min_temperature': tokenizer.min_temperature,
        'temperature_decay': tokenizer.temperature_decay,
        'training_stage': tokenizer.training_stage,
        'tokenization_stats': stats,
        'recommended_context_length': min(512, int(stats['token_lengths']['p95'] * 1.2)),
        'data_source': file_path,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config
