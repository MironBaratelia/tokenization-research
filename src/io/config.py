import os
import yaml
import hydra
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "silu"
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    label_smoothing: float = 0.0

@dataclass
class MironConfig:
    """Single source of truth for MIRON model parameters. All defaults live here."""
    nhead: int = 8
    nlayers: int = 2
    d_ff: int = 1024
    d_char_emb: int = 128
    encoder_char_dim: Optional[int] = None  # None = use d_char_emb
    decoder_char_dim: Optional[int] = None  # None = use d_char_emb
    encoder_dropout: float = 0.1
    encoder_max_pos: int = 64  # char pos emb span; keep >= max_word_length + 2 (BOW + slot)
    max_char_words: int = 512
    encoder_chunk_size: int = 2048
    decoder_chunk_size: int = 2048
    encoder_pooling: str = "flatten"
    decoder_conditioning: str = "flatten"
    init_gain: float = 1.1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    warmup_ratio: float = 0.05
    warmup_min_steps: int = 10000
    lr_scheduler_type: str = "cosine"
    save_steps: int = 1000
    eval_steps: int = 2000
    logging_steps: int = 250
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    grad_clip_norm: float = 1.0
    use_scheduler: bool = True
    min_lr: float = 1e-6
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    early_stopping_min_steps: int = 100000
    save_every: int = 5
    num_workers: int = 4
    prefetch_factor: int = 2
    encoder_lr_ratio: float = 0.1
    decoder_lr_ratio: float = 0.1  # decoder LR = base_lr * this (0.1 = slower than encoder)
    freeze_except_modules_steps: int = 0
    freeze_except_modules: Optional[List[str]] = None
    lm_lr_ratio: float = 0.2
    enc_dec_lr_ratio: float = 0.2
    lm_grad_scale: float = 1.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    curriculum_warmup: int = 1000
    max_steps: int = -1
    stop_after_steps: int = 0
    max_wall_clock_minutes: float = 0.0
    speed_gate_min_tps: float = 0.0
    speed_gate_max_step: int = 0
    enable_compile: bool = False
    val_generation_enabled: bool = False
    val_max_new_tokens: int = 16
    neural_train_mode: str = "joint"
    lambda_interior: float = 0.0
    lambda_oov_segment: float = 0.0
    segmenter_lr_ratio: float = 0.2
    segmenter_lr_ratio_start: float = 1.0
    segmenter_sparse_update_interval: int = 1
    segmenter_joint_decode_mode: str = "locked_dp"
    segmenter_force_fp32: bool = False
    segmenter_lr_decay_start: int = 0
    segmenter_lr_decay_steps: int = 0
    segmenter_online_training: bool = True
    segmenter_unfreeze_step: int = 0
    segmenter_unfreeze_from_end_steps: int = 0
    segmenter_freeze_step: int = 0
    lambda_len: float = 0.001
    lambda_entropy: float = 0.001
    lambda_boundary_bce: float = 0.0
    lambda_boundary_ceiling: float = 0.0
    boundary_ceiling_slack: float = 0.0
    min_soft_boundary_mean: float = 0.02
    lambda_boundary_floor: float = 0.05
    boundary_warmup_steps: int = 0
    joint_lm_ramp_steps: int = 0
    ramp_lm_update_interval: int = 1
    warmup_boundary_bce_weight: float = 1.0
    warmup_boundary_floor_weight: float = 0.0
    warmup_boundary_mean_weight: float = 0.0
    warmup_boundary_pos_weight: float = 1.0
    warmup_interior_weight: float = 0.0
    warmup_oov_weight: float = 0.0

@dataclass
class OptimizerConfig:
    type: str = "adamw"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.9998])
    eps: float = 1e-8

@dataclass
class TokenizerConfig:
    type: str = "word"
    id: Optional[str] = None
    vocab_size: int = 50000
    d_model: int = 768
    max_word_length: int = 32
    min_char_frequency: Optional[int] = None
    min_frequency: Optional[int] = None
    language: Optional[str] = None
    limit_alphabet: Optional[int] = None
    base_tokenizer_type: Optional[str] = None
    max_segment_length: Optional[int] = None
    char_max_length: Optional[int] = None
    nhead: Optional[int] = None
    num_layers: Optional[int] = None
    ff_mult: Optional[int] = None
    temperature: Optional[float] = None
    min_temperature: Optional[float] = None
    temperature_decay: Optional[float] = None
    training_stage: Optional[str] = None
    vocab_logit_mask_enabled: Optional[bool] = None
    vocab_mask_max_token_chars: Optional[int] = None
    vocab_mask_neg_value: Optional[float] = None
    segmenter_pretrain_epochs: Optional[int] = None
    segmenter_pretrain_batch_size: Optional[int] = None
    segmenter_pretrain_lr: Optional[float] = None
    segmenter_pretrain_pos_weight: Optional[float] = None
    segmenter_pretrain_lambda_mean: Optional[float] = None
    segmenter_pretrain_lambda_interior: Optional[float] = None
    segmenter_pretrain_lambda_oov: Optional[float] = None
    boundary_threshold: Optional[float] = None
    boundary_target_rate: Optional[float] = None
    boundary_selection_mode: Optional[str] = None
    segmenter_pretrain_calibration_batches: Optional[int] = None
    locked_length_reward: Optional[float] = None
    locked_end_boundary_weight: Optional[float] = None
    locked_runtime_decode_mode: Optional[str] = None

@dataclass
class PathsConfig:
    data_dir: str = "data/base"
    tokenizers_dir: str = "project_tokenizers/trained"
    outputs_dir: str = "outputs"

@dataclass
class ExperimentConfig:
    id: str = ""
    description: str = ""

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    miron: MironConfig = field(default_factory=MironConfig)
    data: Dict[str, str] = field(default_factory=dict)
    language: str = "en"

def load_config_with_inheritance(config_path: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Loads config using Hydra's composition system to handle inheritance."""
    if project_root is None:
        project_root = os.getcwd()
    
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        config_dir = os.path.join(project_root, "configs")
        hydra.initialize_config_dir(config_dir=config_dir, version_base="1.2")
    
    config_dir = os.path.join(project_root, "configs")
    if os.path.isabs(config_path):
        rel_path = os.path.relpath(config_path, config_dir)
    else:
        rel_path = config_path
        if rel_path.startswith("configs/"):
            rel_path = rel_path[8:]
            
    if rel_path.endswith(".yaml"):
        rel_path = rel_path[:-5]
    
    cfg = hydra.compose(config_name=rel_path.replace("\\", "/"))

    result = OmegaConf.to_container(cfg, resolve=True)

    if isinstance(result, dict) and len(result) == 1 and "experiments" in result:
        experiments = result.get("experiments")
        if isinstance(experiments, dict) and len(experiments) == 1:
            result = next(iter(experiments.values()))

    return result

def resolve_paths(config: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    """Resolves relative paths to absolute paths based on project_root."""
    paths = config.get("paths") or {}
    data_dir = paths.get("data_dir") or "data/base"
    tokenizers_dir = paths.get("tokenizers_dir") or "project_tokenizers/trained"
    outputs_dir = paths.get("outputs_dir") or "outputs"

    resolved = {
        "data_dir": os.path.join(project_root, str(data_dir)),
        "tokenizers_dir": os.path.join(project_root, str(tokenizers_dir)),
        "outputs_dir": os.path.join(project_root, str(outputs_dir)),
    }

    if "data" in config:
        for key, value in config["data"].items():
            if (key.endswith("_file") or key.endswith("_path")) and isinstance(value, str):
                if not os.path.isabs(value):
                    config["data"][key] = os.path.normpath(os.path.join(resolved["data_dir"], value))

    config["resolved_paths"] = resolved
    return config

def build_miron_config(miron: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build complete MIRON config from user overrides. No hardcoded defaults in model code."""
    default = OmegaConf.to_container(OmegaConf.structured(MironConfig), resolve=True)
    user = dict(miron or {})
    merged = OmegaConf.merge(OmegaConf.create(default), OmegaConf.create(user))
    return OmegaConf.to_container(merged, resolve=True)

def get_full_config(config_path: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Unified entry point to get fully loaded and resolved config with defaults."""
    if project_root is None:
        project_root = os.getcwd()
    
    config_dict = load_config_with_inheritance(config_path, project_root)

    default_cfg = OmegaConf.create(OmegaConf.structured(Config))
    loaded_cfg = OmegaConf.create(config_dict)
    
    OmegaConf.set_struct(default_cfg, False)
    OmegaConf.set_struct(loaded_cfg, False)
    merged_cfg = OmegaConf.merge(default_cfg, loaded_cfg)
    OmegaConf.set_struct(merged_cfg, False)
    
    full_config = OmegaConf.to_container(merged_cfg, resolve=True)
    full_config = resolve_paths(full_config, project_root)
    
    return full_config

def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def merge_configs(base, override):
    base_conf = OmegaConf.create(base)
    override_conf = OmegaConf.create(override)
    merged = OmegaConf.merge(base_conf, override_conf)
    return OmegaConf.to_container(merged, resolve=True)
