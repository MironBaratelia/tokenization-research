#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from project_tokenizers.implementations.neural_segmenter import NeuralSegmenterTokenizer
from src.io.config import load_config, merge_configs
from src.common.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train neural segmenter tokenizer")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--stage", type=str, default="init", choices=["init", "locked", "adaptation"])
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output, 'neural_segmenter_training.log'))
        ]
    )
    start_time = time.time()
    logger.info(f"Neural segmenter | stage: {args.stage}")

    base_config_path = os.path.join(project_root, "configs/base.yaml")
    base_config = load_config(base_config_path)
    
    exp_config = load_config(args.config)
    
    config = base_config.copy()
    if 'defaults' in exp_config:
        defaults = exp_config.pop('defaults')
        for default in defaults:
            if '@' in default:
                parts = default.split('@')
                path = parts[0]
                if path.startswith('/'):
                    path = path[1:]
                default_path = os.path.join(project_root, "configs", path + ".yaml")
                if os.path.exists(default_path):
                    default_config = load_config(default_path)
                    config = merge_configs(config, default_config)
    
    config = merge_configs(config, exp_config)
    
    training_config = config['training']
    seed = training_config['seed']
    seed_everything(seed)
    tokenizer_config = {
        'vocab_size': config['tokenizer']['vocab_size'],
        'special_tokens': config['tokenizer'].get('special_tokens', ['<pad>', '<unk>', '<bos>', '<eos>']),
        'language': config.get('language', 'en'),
        'base_tokenizer_type': config['tokenizer'].get('base_tokenizer_type', 'bpe'),
        'max_segment_length': config['tokenizer'].get('max_segment_length', 512),
        'char_max_length': config['tokenizer'].get('char_max_length', config['tokenizer'].get('max_segment_length', 512)),
        'd_model': config['tokenizer'].get('d_model', 128),
        'nhead': config['tokenizer'].get('nhead', 8),
        'num_layers': config['tokenizer'].get('num_layers', 2),
        'ff_mult': config['tokenizer'].get('ff_mult', 4),
        'temperature': config['tokenizer'].get('temperature', 1.0),
        'min_temperature': config['tokenizer'].get('min_temperature', 0.1),
        'temperature_decay': config['tokenizer'].get('temperature_decay', 0.95),
        'segmenter_pretrain_epochs': config['tokenizer'].get('segmenter_pretrain_epochs', 3),
        'segmenter_pretrain_batch_size': config['tokenizer'].get('segmenter_pretrain_batch_size', 128),
        'segmenter_pretrain_lr': config['tokenizer'].get('segmenter_pretrain_lr', 1e-4),
        'segmenter_pretrain_pos_weight': config['tokenizer'].get('segmenter_pretrain_pos_weight'),
        'segmenter_pretrain_lambda_mean': config['tokenizer'].get('segmenter_pretrain_lambda_mean', 2.0),
        'segmenter_pretrain_lambda_interior': config['tokenizer'].get('segmenter_pretrain_lambda_interior', 0.05),
        'segmenter_pretrain_lambda_oov': config['tokenizer'].get('segmenter_pretrain_lambda_oov', 0.1),
        'boundary_threshold': config['tokenizer'].get('boundary_threshold', 0.5),
        'boundary_target_rate': config['tokenizer'].get('boundary_target_rate'),
        'boundary_selection_mode': config['tokenizer'].get('boundary_selection_mode', 'threshold'),
        'segmenter_pretrain_calibration_batches': config['tokenizer'].get('segmenter_pretrain_calibration_batches', 256),
        'locked_length_reward': config['tokenizer'].get('locked_length_reward', 0.35),
        'locked_end_boundary_weight': config['tokenizer'].get('locked_end_boundary_weight', 0.25),
        'locked_runtime_decode_mode': config['tokenizer'].get('locked_runtime_decode_mode', 'greedy_boundaries'),
        'training_stage': args.stage,
        'vocab_logit_mask_enabled': config['tokenizer'].get('vocab_logit_mask_enabled', True),
        'vocab_mask_max_token_chars': config['tokenizer'].get('vocab_mask_max_token_chars', 64),
        'vocab_mask_neg_value': config['tokenizer'].get('vocab_mask_neg_value', -1e4),
    }
    
    neural_tokenizer = NeuralSegmenterTokenizer(tokenizer_config)
    if os.path.exists(args.output) and args.stage != "init":
        neural_tokenizer.load(args.output)
        neural_tokenizer.training_stage = args.stage
    else:
        if args.stage == "init":
            if not os.path.exists(args.data):
                raise FileNotFoundError(f"Training data not found: {args.data}")
            logger.info("Training neural segmenter...")
            train_start_time = time.time()
            neural_tokenizer.train(args.data)
            train_time = time.time() - train_start_time
            os.makedirs(args.output, exist_ok=True)
            neural_tokenizer.save(args.output)
            logger.info(f"Saved to {args.output} in {train_time:.2f}s")
        else:
            logger.warning(f"Stage {args.stage} requires existing tokenizer at {args.output}")
            return
    total_time = time.time() - start_time
    logger.info(f"Completed in {total_time:.2f}s")


if __name__ == "__main__":
    main()
