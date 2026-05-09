"""Dataset shuffling utilities."""
import json
import logging
import os
import glob
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from src.common.tqdm_utils import tqdm
from typing import Optional


logger = logging.getLogger(__name__)


def shuffle_dataset(
    manifest_path: str,
    output_dir: str,
    seed: int = 42,
    temp_dir: Optional[str] = None,
    max_ram_per_shard_gb: float = 0.5
) -> str:
    """
    Performs global shuffle of Arrow dataset using partition/bucket method.
    
    Args:
        manifest_path: Path to input dataset manifest.json
        output_dir: Directory to save shuffled dataset
        seed: Random seed
        temp_dir: Directory for temporary files
        max_ram_per_shard_gb: Maximum partition size in GB
        
    Returns:
        Path to new manifest file.
    """
    np.random.seed(seed)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    input_shards = manifest['shards']
    if not input_shards:
        logger.warning("No shards found in manifest.")
        return manifest_path
    missing_shards = [shard for shard in input_shards if not os.path.exists(shard)]
    if missing_shards:
        preview = ", ".join(missing_shards[:3])
        if len(missing_shards) > 3:
            preview += f", ... ({len(missing_shards)} total)"
        raise FileNotFoundError(
            "Packed manifest references missing shard(s). "
            "This usually means a previous preprocessing run was interrupted or two step-2 runs "
            f"wrote the same output directory. Missing: {preview}. "
            "Rerun preprocessing from a clean tokenized directory."
        )

    total_size_bytes = sum(os.path.getsize(shard) for shard in input_shards if os.path.exists(shard))
    total_size_gb = total_size_bytes / (1024**3)
    
    num_output_shards = max(1, int(np.ceil(total_size_gb / max_ram_per_shard_gb)))
    
    MAX_OPEN_FILES = 128
    if num_output_shards > MAX_OPEN_FILES:
        logger.warning(f"Calculated {num_output_shards} shards, capping at {MAX_OPEN_FILES}")
        num_output_shards = MAX_OPEN_FILES

    os.makedirs(output_dir, exist_ok=True)
    if temp_dir is None:
        temp_dir = os.path.join(output_dir, "temp_shuffle")
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    logger.info(f"Shuffle: {total_size_gb:.2f} GB -> {num_output_shards} partitions")

    length_files = manifest.get('length_files', [])
    base_name = os.path.basename(manifest_path).replace(".manifest.json", "").replace(".arrow", "")
    for old_path in glob.glob(os.path.join(output_dir, f"{base_name}_shuffled_*.arrow")):
        os.remove(old_path)
    old_manifest = os.path.join(output_dir, f"{base_name}_shuffled.manifest.json")
    if os.path.exists(old_manifest):
        os.remove(old_manifest)

    with ipc.open_file(input_shards[0]) as reader:
        schema = reader.schema

    temp_writers = []
    temp_files = []
    
    try:
        for i in range(num_output_shards):
            path = os.path.join(temp_dir, f"temp_{i:03d}.arrow")
            f = pa.OSFile(path, 'wb')
            writer = ipc.new_file(f, schema)
            temp_files.append(f)
            temp_writers.append(writer)

        for shard_idx, input_shard in enumerate(tqdm(input_shards, desc="Distributing")):
            with ipc.open_file(input_shard) as reader:
                for i in range(reader.num_record_batches):
                    batch = reader.get_batch(i)
                    num_rows = batch.num_rows
                    assignments = np.random.randint(0, num_output_shards, size=num_rows)
                    
                    for shard_id in range(num_output_shards):
                        mask = (assignments == shard_id)
                        if np.any(mask):
                            indices = np.where(mask)[0]
                            sub_batch = batch.take(indices)
                            temp_writers[shard_id].write_batch(sub_batch)

            # Free space before finalization; data is already copied into temp partitions.
            if os.path.exists(input_shard):
                try:
                    os.remove(input_shard)
                except Exception:
                    pass
            if shard_idx < len(length_files):
                len_file = length_files[shard_idx]
                if len_file and os.path.exists(len_file):
                    try:
                        os.remove(len_file)
                    except Exception:
                        pass

    finally:
        for w in temp_writers:
            w.close()
        for f in temp_files:
            f.close()

    final_shards = []
    final_shard_sizes = []
    total_samples = 0
    
    for i in tqdm(range(num_output_shards), desc="Finalizing"):
        temp_path = os.path.join(temp_dir, f"temp_{i:03d}.arrow")
        if not os.path.exists(temp_path):
            continue
            
        with ipc.open_file(temp_path) as reader:
            table = reader.read_all()
            
        if table.num_rows == 0:
            continue
            
        indices = np.arange(table.num_rows)
        np.random.shuffle(indices)
        shuffled_table = table.take(indices)
        
        final_shard_name = f"{base_name}_shuffled_{i:03d}.arrow"
        final_shard_path = os.path.join(output_dir, final_shard_name)
        
        with pa.OSFile(final_shard_path, 'wb') as sink:
            with ipc.new_file(sink, schema) as writer:
                writer.write_table(shuffled_table)
        
        final_shards.append(final_shard_path)
        final_shard_sizes.append(table.num_rows)
        total_samples += table.num_rows
        
        for _ in range(3):
            try:
                os.remove(temp_path)
                break
            except PermissionError:
                time.sleep(0.5)

    for _ in range(3):
        try:
            shutil.rmtree(temp_dir)
            break
        except OSError:
            time.sleep(0.5)

    new_manifest_path = os.path.join(output_dir, f"{base_name}_shuffled.manifest.json")
    new_manifest = {
        "shards": final_shards,
        "shard_sizes": final_shard_sizes,
        "total_samples": total_samples,
        "shuffled": True
    }
    
    with open(new_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(new_manifest, f, indent=2)

    for len_file in length_files:
        if os.path.exists(len_file):
            try:
                os.remove(len_file)
            except Exception:
                pass

    for shard in input_shards:
        if os.path.exists(shard) and shard not in final_shards:
            try:
                os.remove(shard)
            except Exception:
                pass

    if os.path.exists(manifest_path) and os.path.abspath(manifest_path) != os.path.abspath(new_manifest_path):
        try:
            os.remove(manifest_path)
        except Exception:
            pass

    logger.info(f"Shuffle complete: {new_manifest_path}")
    return new_manifest_path
