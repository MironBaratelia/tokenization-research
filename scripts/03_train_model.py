import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import subprocess
import time
import logging
import yaml
import json

from src.common.system import setup_project_env, configure_logging
setup_project_env(project_root)

from src.io.config import get_full_config
from src.common.device_utils import get_device
from src.common.performance import TrainingOptimizer
from src.common.reproducibility import seed_everything
from src.common.tokenizer_utils import get_pad_id
from src.models.smollm2 import SmolLM2Model, SmolLM2Config
from src.training.base_trainer import BaseTrainer as Trainer
from src.training.optimizer import build_optimizer, build_scheduler
from src.datasets.lm_dataset import load_data_splits
from project_tokenizers.implementations import TOKENIZER_REGISTRY
from src.scripts.utils.training import resolve_data_paths


logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False):
    from src.common.system import configure_logging
    configure_logging(verbose=verbose)


def build_model_config(full_config: dict, tokenizer):
    """Build model configuration."""
    model_cfg = full_config['model']
    pad_id = get_pad_id(tokenizer, 0)
    
    config = SmolLM2Config(
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        intermediate_size=model_cfg['intermediate_size'],
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=model_cfg['max_position_embeddings'],
        rope_theta=model_cfg['rope_theta'],
        pad_token_id=pad_id,
        label_smoothing=model_cfg.get('label_smoothing', 0.0),
    )
    return config


def build_model(tokenizer_type: str, tokenizer, config, full_config: dict):
    """Build model and get trainer class."""
    if tokenizer_type == "miron":
        from src.miron_lib import MironConfig, MironForCausalLM
        from src.training.miron_trainer import MIRONTrainer

        from transformers import AutoConfig, AutoModel

        if not hasattr(SmolLM2Model, '_from_config'):
            @classmethod
            def _from_config(cls, config, **kwargs):
                return cls(config, **kwargs)
            SmolLM2Model._from_config = _from_config
            
        AutoConfig.register("smollm2", SmolLM2Config)
        AutoModel.register(SmolLM2Config, SmolLM2Model)
        
        m = full_config.get('miron', {})
        base_cfg = config.__dict__ if hasattr(config, '__dict__') else config
        _cti = tokenizer.char_to_id
        _pad = get_pad_id(tokenizer, 0)
        _eow = _cti.get(getattr(tokenizer, "eow_token", "<eow>"))
        if _eow is None:
            _eow = _cti.get("<eow>")
        if _eow is None:
            raise KeyError("eow token not in char_to_id (expected tokenizer.eow_token or '<eow>')")
        _bow = _cti.get(getattr(tokenizer, "bos_token", "[CLS]"))
        if _bow is None:
            _bow = _cti.get("<bow>")
        if _bow is None:
            _bow = _pad
        miron_config = MironConfig(
            base_model_config=base_cfg,
            base_model_type="smollm2",
            vocab_size=tokenizer.vocab_size,
            pad_token_id=int(_pad),
            eow_token_id=int(_eow),
            bow_token_id=int(_bow),
            max_word_length=tokenizer.max_word_length if hasattr(tokenizer, 'max_word_length') else 32,
            encoder_char_dim=m.get('encoder_char_dim') or m.get('d_char_emb', 128),
            decoder_char_dim=m.get('decoder_char_dim') or m.get('d_char_emb', 128),
            encoder_pooling=m.get('encoder_pooling', 'flatten'),
            decoder_conditioning=m.get('decoder_conditioning', 'flatten'),
            encoder_nhead=m.get('nhead', 8),
            encoder_num_layers=m.get('nlayers', 2),
            encoder_d_ff=m.get('d_ff', 1024),
            encoder_dropout=m.get('encoder_dropout', 0.1),
            encoder_max_pos=m.get('encoder_max_pos', 64),
            encoder_chunk_size=m.get('encoder_chunk_size', 2048),
            decoder_chunk_size=m.get('decoder_chunk_size', 2048),
            init_gain=m.get('init_gain', 1.1),
        )
        smol_config = SmolLM2Config(
            hidden_size=miron_config.d_model,
            num_layers=miron_config.core_num_layers,
            num_heads=miron_config.core_num_heads,
            intermediate_size=miron_config.core_intermediate_size,
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=miron_config.max_position_embeddings,
            rope_theta=miron_config.rope_theta,
            pad_token_id=miron_config.pad_token_id,
        )
        lm_model = SmolLM2Model(smol_config)
        model = MironForCausalLM(miron_config, lm_model=lm_model)

        return model, MIRONTrainer
    elif tokenizer_type == "neural_segmenter" and getattr(tokenizer, "training_stage", "init") in [
        "joint",
        "adaptation",
    ]:
        from src.models.neural_segmenter_lm import NeuralSegmenterLM
        from src.training.neural_segmenter_trainer import NeuralSegmenterTrainer

        smol_config = SmolLM2Config(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            pad_token_id=config.pad_token_id,
        )
        lm_model = SmolLM2Model(smol_config)

        model = NeuralSegmenterLM(tokenizer.segmenter_model, lm_model, tokenizer, full_config)
        return model, NeuralSegmenterTrainer
    elif tokenizer_type == "neural_segmenter" and str(
        (full_config.get("training") or {}).get("neural_train_mode", "joint")
    ).lower() == "supervised_boundaries":
        from src.models.neural_segmenter_supervised import NeuralSegmenterSupervised

        if getattr(tokenizer, "segmenter_model", None) is None:
            from project_tokenizers.implementations.neural_segmenter import TransformerSegmenter

            char_tok = getattr(tokenizer, "char_tokenizer", None)
            if char_tok is None or not getattr(char_tok, "char_to_id", None):
                raise ValueError(
                    "neural_train_mode=supervised_boundaries needs tokenizer.segmenter_model or "
                    "a loaded char_tokenizer with char_to_id (load tokenizer from disk after init)."
                )
            tcfg = full_config.get("tokenizer") or {}
            vs = len(char_tok.char_to_id)
            tokenizer.segmenter_model = TransformerSegmenter(
                vocab_size=vs,
                d_model=int(tcfg.get("d_model", 128)),
                nhead=int(tcfg.get("nhead", 8)),
                num_layers=int(tcfg.get("num_layers", 2)),
            )
        model = NeuralSegmenterSupervised(tokenizer.segmenter_model, tokenizer, full_config)
        return model, Trainer
    else:
        model = SmolLM2Model(config)
        return model, Trainer


def save_training_metadata(output_dir: str, full_config: dict, train_loader, train_packed_path: str):
    """Save training metadata for reproducibility."""
    total_samples = len(train_loader.dataset)
    grad_accum = full_config.get('training', {}).get('gradient_accumulation_steps', 1)
    batch_size = full_config.get('training', {}).get('batch_size', 1)
    steps_per_epoch = total_samples // (batch_size * grad_accum)
    
    metadata = {
        "config": full_config, 
        "timestamp": time.time(), 
        "train_path": train_packed_path,
        "total_samples": total_samples,
        "steps_per_epoch": max(1, steps_per_epoch),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum
    }
    
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def start_tensorboard(output_dir: str):
    """Start TensorBoard process."""
    log_dir = os.path.join(output_dir, "logs", "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    
    tb_proc = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    logger.info("TensorBoard: http://localhost:6006")
    return tb_proc


def calculate_max_steps(full_config: dict, train_loader):
    from src.training.max_steps_resolution import resolve_training_max_steps

    return resolve_training_max_steps(full_config, train_loader)


def main():
    parser = argparse.ArgumentParser(description="MIRON Model Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Experiment configuration name (e.g. ru/miron)")
    parser.add_argument("--verbose", action="store_true", help="Detailed debug logging")
    parser.add_argument("--memory-profile", action="store_true", help="Print CUDA memory at each forward stage (for OOM debug)")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load checkpoint; start from init (e.g. after architecture change)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If > 0, override training max_steps (smoke tests / short runs)",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="If > 0, stop after this many optimizer steps without changing scheduler max_steps.",
    )
    args, _ = parser.parse_known_args()

    if args.memory_profile:
        os.environ["MIRON_MEMORY_PROFILE"] = "1"
        logger.info("Memory profiling enabled (MIRON_MEMORY_PROFILE=1)")
    else:
        os.environ.pop("MIRON_MEMORY_PROFILE", None)

    configure_logging(args.verbose)

    config_path = args.config if args.config.startswith("configs/experiments/") else f"configs/experiments/{args.config}"
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"
    
    full_config = get_full_config(config_path, project_root)

    if getattr(args, "max_steps", 0) and int(args.max_steps) > 0:
        full_config.setdefault("training", {})["max_steps"] = int(args.max_steps)
    if getattr(args, "stop_after", 0) and int(args.stop_after) > 0:
        full_config.setdefault("training", {})["stop_after_steps"] = int(args.stop_after)

    experiment_id = full_config.get('experiment', {}).get('id', 'unknown')
    seed = full_config.get('training', {}).get('seed', 42)
    seed_everything(seed)
    
    logger.info(f"Starting Experiment: {experiment_id} | Seed: {seed}")
    
    out_dir = full_config.get("resolved_paths", {}).get("outputs_dir") or os.path.join(project_root, "outputs")
    output_dir = os.path.normpath(os.path.join(out_dir, experiment_id))
    os.makedirs(output_dir, exist_ok=True)

    if getattr(args, "no_resume", False):
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            import shutil
            shutil.rmtree(ckpt_dir)
            logger.info("Removed checkpoint dir (--no-resume), starting fresh")

    train_packed_path, val_packed_path = resolve_data_paths(full_config, project_root, output_dir)
    
    if not train_packed_path or not os.path.exists(train_packed_path):
        raise FileNotFoundError(f"Missing training data: {train_packed_path}")
        
    full_config["data"]["train_file"] = train_packed_path
    full_config["data"]["validation_file"] = val_packed_path

    bs = full_config.get("training", {}).get("batch_size")
    ctx = full_config.get("model", {}).get("max_position_embeddings")
    mwl = full_config.get("tokenizer", {}).get("max_word_length")
    logger.info("Config: batch_size=%s max_position_embeddings=%s max_word_length=%s", bs, ctx, mwl)
    
    tokenizer_config = full_config.get("tokenizer", {})
    tokenizer_type = tokenizer_config.get("type")
    tokenizer_cls = TOKENIZER_REGISTRY[tokenizer_type]
    
    tokenizer_id = tokenizer_config.get("id") or experiment_id
    resolved = full_config.get("resolved_paths") or {}
    tok_dir = resolved.get("tokenizers_dir")
    if not tok_dir:
        tok_dir = os.path.join(project_root, "project_tokenizers", "trained")
    tokenizer_path = os.path.normpath(os.path.join(str(tok_dir), str(tokenizer_id)))
    
    tokenizer = tokenizer_cls(tokenizer_config)
    tokenizer.load(tokenizer_path)

    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    train_loader, val_loader = load_data_splits(full_config, tokenizer)
    
    model_cfg = build_model_config(full_config, tokenizer)
    model, TrainerClass = build_model(tokenizer_type, tokenizer, model_cfg, full_config)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    if full_config.get('training', {}).get('gradient_checkpointing', False):
        if hasattr(model, 'lm') and hasattr(model.lm, 'gradient_checkpointing_enable'):
            model.lm.gradient_checkpointing_enable()

    model = model.to(get_device())
    enable_compile = full_config.get("training", {}).get("enable_compile")
    if enable_compile is None:
        enable_compile = not sys.platform.startswith("win")
    model = TrainingOptimizer.optimize_for_training(model, enable_compile=bool(enable_compile))

    calculate_max_steps(full_config, train_loader)

    optimizer = build_optimizer(model, full_config)
    use_scheduler = full_config.get("training", {}).get("use_scheduler", True)
    scheduler = build_scheduler(optimizer, full_config) if use_scheduler else None
    
    tb_proc = start_tensorboard(output_dir)

    save_training_metadata(output_dir, full_config, train_loader, train_packed_path)

    trainer = TrainerClass(
        model=model, optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        config=full_config, experiment_id=experiment_id, tokenizer=tokenizer
    )

    try:
        logger.info("Starting Training...")
        trainer.train()
        es = getattr(trainer, "early_stopping", None)
        if es is not None and getattr(es, "early_stop", False):
            logger.info(
                "Training finished (early stopping at global_step=%s).",
                getattr(trainer, "global_step", None),
            )
        else:
            logger.info(
                "Training finished (global_step=%s / max_steps=%s).",
                getattr(trainer, "global_step", None),
                getattr(trainer, "max_steps", None),
            )
    except KeyboardInterrupt:
        step = getattr(trainer, "global_step", None)
        logger.warning(
            "Training stopped by user (KeyboardInterrupt / Ctrl+C). "
            "This is not a model error. Last global_step=%s",
            step,
        )
        raise
    except Exception:
        step = getattr(trainer, "global_step", None)
        logger.exception("Training aborted with an exception (global_step=%s)", step)
        raise
    finally:
        if tb_proc:
            tb_proc.terminate()


if __name__ == "__main__":
    main()
