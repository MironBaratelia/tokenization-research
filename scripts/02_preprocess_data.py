import argparse
import atexit
import glob
import os
import shutil
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from typing import Optional

from src.common.system import setup_project_env, configure_logging
setup_project_env(project_root)
configure_logging()

from src.io.config import get_full_config
from src.preprocessing.packing import smart_pack_dataset, smart_pack_dataset_lazy
from src.preprocessing.helpers import count_total_lines
from src.preprocessing.shuffling import shuffle_dataset
from src.common.reproducibility import seed_everything
from project_tokenizers.implementations import TOKENIZER_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def acquire_preprocess_lock(tokenized_dir: str) -> str:
    """Prevent two preprocessing runs from writing the same shard names."""
    lock_path = os.path.join(tokenized_dir, ".preprocess.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        details = ""
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                details = f.read().strip()
        except OSError:
            pass
        message = (
            f"Preprocessing output is locked: {lock_path}. "
            "Another step-2 run may still be writing this experiment. "
            "If no such process exists, remove the stale lock and rerun."
        )
        if details:
            message += f" Lock details: {details}"
        raise RuntimeError(message) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\n")
        f.write(f"created={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return lock_path

def release_preprocess_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass

def _remove_path(path: str) -> None:
    for attempt in range(3):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.5)

def clean_generated_outputs(tokenized_dir: str) -> None:
    """Remove generated packing/shuffle artifacts so manifests cannot mix runs."""
    patterns = [
        "train_packed.part*.arrow",
        "train_packed.part*.frozen.arrow",
        "train_packed.part*.len",
        "train_packed.arrow.manifest.json",
        "train_packed_shuffled_*.arrow",
        "train_packed_shuffled_*.frozen.arrow",
        "train_packed_shuffled.manifest.json",
        "val_packed.part*.arrow",
        "val_packed.part*.frozen.arrow",
        "val_packed.part*.len",
        "val_packed.arrow.manifest.json",
    ]
    removed = 0
    temp_shuffle = os.path.join(tokenized_dir, "temp_shuffle")
    if os.path.exists(temp_shuffle):
        _remove_path(temp_shuffle)
        removed += 1

    for pattern in patterns:
        for path in glob.glob(os.path.join(tokenized_dir, pattern)):
            _remove_path(path)
            removed += 1

    if removed:
        logger.info("Removed %d stale generated preprocessing artifact(s).", removed)

def estimate_dataset_size(file_path: str) -> Optional[int]:
    """Provides a rough estimate of rows in the raw dataset. Supports .arrow and .txt."""
    if not os.path.exists(file_path):
        return None
    try:
        if file_path.endswith(".arrow"):
            return count_total_lines(file_path)
        if os.path.getsize(file_path) < 100 * 1024 * 1024:
            with open(file_path, "rb") as f:
                return sum(1 for _ in f)
        with open(file_path, "r", encoding="utf-8") as f:
            sample = [len(f.readline()) for _ in range(1000)]
            avg_len = sum(sample) / len(sample)
            return int(os.path.getsize(file_path) / avg_len) if avg_len > 0 else 0
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="MIRON Data Preprocessing Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Experiment config name")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction of data to use (0.0-1.0)")
    parser.add_argument("--max-gb", type=float, default=None, help="Capped data size in GB")
    parser.add_argument("--skip-shuffle", action="store_true", help="Disable dataset shuffling")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clean existing packed/shuffled outputs before packing")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers for tokenization (morpheme etc.)")
    args = parser.parse_args()

    config_path = args.config if args.config.endswith(".yaml") else f"configs/experiments/{args.config}.yaml"
    full_config = get_full_config(config_path, project_root)
    
    experiment_id = full_config["experiment"]["id"]
    seed = full_config.get("training", {}).get("seed", 42)
    seed_everything(seed)
    
    output_dir = os.path.normpath(os.path.join(full_config['resolved_paths']['outputs_dir'], experiment_id))
    tokenized_dir = os.path.join(output_dir, "tokenized")
    os.makedirs(tokenized_dir, exist_ok=True)
    
    logger.info(f"Preprocessing Experiment: {experiment_id}")
    lock_path = acquire_preprocess_lock(tokenized_dir)
    atexit.register(release_preprocess_lock, lock_path)
    if not args.keep_existing:
        clean_generated_outputs(tokenized_dir)
    
    tokenizer_config = full_config.get("tokenizer", {})
    tokenizer_type = tokenizer_config.get("type")
    tokenizer_id = tokenizer_config.get("id") or experiment_id
    tokenizer_path = os.path.normpath(
        os.path.join(full_config["resolved_paths"]["tokenizers_dir"], str(tokenizer_id))
    )
    
    tokenizer = TOKENIZER_REGISTRY[tokenizer_type](tokenizer_config)
    tokenizer.load(tokenizer_path)
    
    context_length = full_config['model']['max_position_embeddings']
    pack_context_length = context_length
    if tokenizer_type == "neural_segmenter":
        pack_context_length = int(getattr(tokenizer, "char_max_length", context_length) or context_length)
        logger.info(
            "Neural segmenter packing uses char context_length=%d with LM token budget=%d.",
            pack_context_length,
            context_length,
        )

    train_raw_path = os.path.join(project_root, full_config['data']['train_file'])
    max_rows = None
    
    if args.max_gb:
        total_size = os.path.getsize(train_raw_path)
        limit_bytes = args.max_gb * 1024**3
        if limit_bytes < total_size:
            est_total = estimate_dataset_size(train_raw_path)
            if est_total:
                max_rows = int(est_total * (limit_bytes / total_size))
                logger.info(f"Capping at {args.max_gb}GB (~{max_rows} rows)")
                if not args.skip_shuffle:
                    logger.warning("Shuffle needs ~2x disk (packed + shuffled copy). Free ~{:.0f}GB+ recommended.".format(args.max_gb * 2))
    elif args.train_fraction < 1.0:
        est_total = estimate_dataset_size(train_raw_path)
        if est_total:
            max_rows = int(est_total * args.train_fraction)
            logger.info(f"Using {args.train_fraction*100}% of data (~{max_rows} rows)")

    logger.info(f"Packing Training Data...")
    if tokenizer_type not in ("miron", "char_level", "neural_segmenter") and args.workers == 0:
        logger.info(
            "Tip: add --workers 8 (or min(CPU-1,8)) to parallelize line tokenization; "
            "each process loads the tokenizer (more RAM)."
        )
    train_out = os.path.join(tokenized_dir, "train_packed.arrow")
    
    if tokenizer_type == "neural_segmenter":
        logger.info(
            "Using Lazy Tokenization for Neural Segmenter (char budget=%d, LM token budget=%d).",
            pack_context_length,
            context_length,
        )
        smart_pack_dataset_lazy(train_raw_path, tokenizer, pack_context_length, train_out, max_input_rows=max_rows)
    elif tokenizer_type == "miron":
        logger.info("Using Lazy Tokenization (string storage) for MIRON.")
        smart_pack_dataset_lazy(train_raw_path, tokenizer, context_length, train_out, max_input_rows=max_rows)
    elif tokenizer_type == "char_level":
        logger.info(
            "Lazy packing for char_level: len(text) ≈ token count, budget=context_length-2 (BOS+EOS); "
            "encode only at train time."
        )
        smart_pack_dataset_lazy(train_raw_path, tokenizer, context_length, train_out, max_input_rows=max_rows)
    else:
        smart_pack_dataset(
            train_raw_path,
            tokenizer,
            context_length,
            train_out,
            max_input_rows=max_rows,
            num_workers=args.workers,
            tokenizer_type=tokenizer_type,
            tokenizer_path=tokenizer_path,
        )
        
    val_raw = os.path.join(project_root, full_config['data'].get('validation_file', ''))
    if os.path.exists(val_raw) and not val_raw.endswith('.arrow'):
        logger.info(f"Packing Validation Data...")
        val_out = os.path.join(tokenized_dir, "val_packed.arrow")
        if tokenizer_type in ("neural_segmenter", "miron", "char_level"):
            val_context_length = pack_context_length if tokenizer_type == "neural_segmenter" else context_length
            smart_pack_dataset_lazy(val_raw, tokenizer, val_context_length, val_out)
        else:
            smart_pack_dataset(
                val_raw,
                tokenizer,
                context_length,
                val_out,
                num_workers=args.workers,
                tokenizer_type=tokenizer_type,
                tokenizer_path=tokenizer_path,
            )

    if not args.skip_shuffle:
        manifest = train_out + ".manifest.json"
        if os.path.exists(manifest):
            logger.info("Shuffling dataset...")
            shuffle_dataset(manifest, tokenized_dir, seed=seed)

    logger.info("Preprocessing complete.")
    release_preprocess_lock(lock_path)

if __name__ == "__main__":
    main()
