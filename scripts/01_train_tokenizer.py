#!/usr/bin/env python3
import argparse
import os
import sys
import unicodedata

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import time
from statistics import median

from src.common.system import setup_project_env, configure_logging
setup_project_env(project_root)

from src.io.config import get_full_config
from src.common.reproducibility import seed_everything
from project_tokenizers.implementations import TOKENIZER_REGISTRY
from src.scripts.utils.tokenizer import resolve_train_file, resolve_check_file

logger = logging.getLogger(__name__)

def setup_logging(log_dir: str, verbose: bool = False):
    os.makedirs(log_dir, exist_ok=True)
    configure_logging(verbose=verbose)
    log_file = os.path.join(log_dir, 'tokenizer_training.log')
    logging.getLogger().addHandler(logging.FileHandler(log_file, encoding='utf-8'))


def load_tokenizer(full_config: dict):
    """Initialize and return tokenizer."""
    tokenizer_config = dict(full_config.get("tokenizer", {}))
    tokenizer_type = tokenizer_config.get("type")
    if not tokenizer_config.get("language") and full_config.get("language"):
        tokenizer_config["language"] = full_config["language"]
    if tokenizer_type not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Available: {list(TOKENIZER_REGISTRY.keys())}")
    logger.info(f"Initializing {tokenizer_type} tokenizer...")
    return TOKENIZER_REGISTRY[tokenizer_type](tokenizer_config)


def train_tokenizer(tokenizer, train_file: str):
    """Train tokenizer on data."""
    logger.info(f"Training tokenizer on: {train_file}")
    start_time = time.time()
    tokenizer.train(train_file)
    return time.time() - start_time


def save_tokenizer(tokenizer, save_dir: str):
    """Save tokenizer to directory."""
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(save_dir)
    logger.info(f"Tokenizer saved to: {save_dir}")


def run_neural_segmenter_training(config_path: str, train_file: str, output_dir: str, project_root: str):
    """Run neural segmenter training via subprocess."""
    import subprocess
    
    cmd = [
        sys.executable,
        os.path.join(project_root, "src/training/run_neural_segmenter_training.py"),
        "--config", config_path,
        "--data", train_file,
        "--output", output_dir,
        "--stage", "init"
    ]
    
    logger.info(f"Running neural segmenter: {' '.join(cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        raise RuntimeError(f"Neural segmenter training failed with exit code {result.returncode}")
    
    return time.time() - start_time


def verify_tokenizer(tokenizer, tokenizer_type: str, check_file_path: str):
    """Verify tokenizer by encoding/decoding sample text."""
    with open(check_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    logger.info("Tokenizer verification on sample text:")
    logger.info("Original:\n%s", text)
    
    prev_stage = None
    if tokenizer_type == "neural_segmenter":
        prev_stage = getattr(tokenizer, "training_stage", None)
        tokenizer.training_stage = "locked"

    try:
        if tokenizer_type == "miron":
            tokens = tokenizer.tokenize(text)
            logger.info("Tokens: %s", tokens)

            encoded_ids = tokenizer.encode(text)
            flat_ids = [id for word_ids in encoded_ids for id in word_ids]
            decoded = tokenizer.decode(flat_ids)
            logger.info("Decoded:\n%s", decoded)
        else:
            ids = tokenizer.encode(text)
            tokens = [tokenizer.id_to_token.get(idx, tokenizer.unk_token) for idx in ids]
            logger.info("Tokens: %s", tokens)

            decoded = tokenizer.decode(ids)
            logger.info("Decoded:\n%s", decoded)
    finally:
        if prev_stage is not None:
            tokenizer.training_stage = prev_stage

    exact_match = decoded == text
    nfc_match = unicodedata.normalize("NFC", decoded) == unicodedata.normalize("NFC", text)
    logger.info("Round-trip exact: %s | NFC-normalized: %s", exact_match, nfc_match)
    if not exact_match:
        limit = min(len(text), len(decoded))
        mismatch = next((i for i in range(limit) if text[i] != decoded[i]), limit)
        logger.info(
            "First mismatch at char %d | original=%r | decoded=%r",
            mismatch,
            text[mismatch : mismatch + 40],
            decoded[mismatch : mismatch + 40],
        )
    if tokenizer_type == "neural_segmenter" and hasattr(tokenizer, "debug_locked_metrics"):
        metrics = tokenizer.debug_locked_metrics(text)
        logger.info(
            "Locked metrics | exact=%s | gold_tokens=%s | pred_tokens=%s | span_exact=%.3f | "
            "boundary_P=%.3f | boundary_R=%.3f | boundary_F1=%.3f",
            metrics.get("exact_roundtrip", False),
            metrics.get("gold_tokens", 0),
            metrics.get("pred_tokens", 0),
            metrics.get("span_exact_ratio", 0.0),
            metrics.get("boundary_precision", 0.0),
            metrics.get("boundary_recall", 0.0),
            metrics.get("boundary_f1", 0.0),
        )
        timed_texts = [text] * 8
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            for sample_text in timed_texts:
                tokenizer.encode(sample_text)
            elapsed = time.perf_counter() - t0
            chars = sum(len(sample_text) for sample_text in timed_texts)
            if elapsed > 0:
                samples.append(chars / elapsed)
        if samples:
            logger.info("Locked encode speed | median_chars_per_sec=%.1f", median(samples))


def main():
    parser = argparse.ArgumentParser(description="Step 1: Train Tokenizer")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    start_time = time.time()
    
    config_path = args.config
    if not config_path.startswith("configs/experiments/"):
        config_path = f"configs/experiments/{config_path}"
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"
    config_path = os.path.join(project_root, config_path)
    
    logger.info(f"Loading config: {config_path}")
    full_config = get_full_config(config_path, project_root)
    
    experiment_id = full_config.get('experiment', {}).get('id', 'unknown')
    outputs_dir = full_config.get('paths', {}).get('outputs_dir', 'outputs')
    log_dir = os.path.join(project_root, outputs_dir, experiment_id)
    
    setup_logging(log_dir, args.verbose)
    
    seed = full_config.get('training', {}).get('seed', 42)
    seed_everything(seed)
    logger.info(f"Random seed: {seed}")
    
    tokenizer_config = full_config.get("tokenizer", {})
    tokenizer_type = tokenizer_config.get("type")
    
    logger.info(f"Experiment: {experiment_id}")
    logger.info(f"Tokenizer type: {tokenizer_type}")
    
    tokenizer = load_tokenizer(full_config)
    
    train_file = resolve_train_file(full_config, project_root)
    logger.info(f"Training data: {train_file}")
    
    save_dir = os.path.join(full_config['resolved_paths']['tokenizers_dir'], experiment_id)
    
    if tokenizer_type == "neural_segmenter":
        output_dir = os.path.join(full_config['resolved_paths']['tokenizers_dir'], experiment_id)
        train_time = run_neural_segmenter_training(config_path, train_file, output_dir, project_root)
        tokenizer.load(output_dir)
    else:
        train_time = train_tokenizer(tokenizer, train_file)
        save_tokenizer(tokenizer, save_dir)
    
    logger.info(f"Training completed in {train_time:.2f}s")
    
    check_file = resolve_check_file(project_root, full_config.get('language', 'en'))
    if check_file:
        verify_tokenizer(tokenizer, tokenizer_type, check_file)
    else:
        logger.warning(f"Check file not found")
    
    logger.info(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
