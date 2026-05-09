"""Optimizer and scheduler builders."""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer with differential learning rates for MIRON model."""
    training_config = config['training']
    optimizer_config = config.get('optimizer', {'betas': [0.9, 0.999], 'eps': 1e-8})
    lr = float(training_config['learning_rate'])
    weight_decay = float(training_config['weight_decay'])
    
    # NeuralSegmenterLM: lower LR on the segmenter than on the LM (same LR on both destabilizes joint training).
    if getattr(model.__class__, "__name__", "") == "NeuralSegmenterLM" and hasattr(model, "segmenter") and hasattr(model, "lm"):
        seg_ratio = float(training_config.get("segmenter_lr_ratio_start", training_config.get("segmenter_lr_ratio", 0.2)))
        seg_params = list(model.segmenter.parameters())
        seg_ids = {id(p) for p in seg_params}
        other_params = [p for p in model.parameters() if id(p) not in seg_ids]
        return AdamW(
            [
                {"params": seg_params, "lr": lr * seg_ratio, "weight_decay": weight_decay},
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            ],
            betas=tuple(optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
        )

    if hasattr(model, 'encoder') and hasattr(model, 'decoder') and (hasattr(model, 'core_lm') or hasattr(model, 'lm')):
        encoder_lr_ratio = float(training_config.get('encoder_lr_ratio', 0.1))
        decoder_lr_ratio = float(training_config.get('decoder_lr_ratio', encoder_lr_ratio))
        all_params = set(model.parameters())
        enc_params = set(model.encoder.parameters())
        dec_params = set(model.decoder.parameters()) - enc_params
        loss_params = set(model.loss_fn.parameters()) if hasattr(model, 'loss_fn') else set()
        other_params = all_params - enc_params - dec_params - loss_params
        
        return AdamW([
            {'params': list(enc_params), 'lr': lr * encoder_lr_ratio, 'weight_decay': weight_decay},
            {'params': list(dec_params), 'lr': lr * decoder_lr_ratio, 'weight_decay': weight_decay},
            {'params': list(other_params), 'lr': lr, 'weight_decay': weight_decay},
            {'params': list(loss_params), 'lr': lr, 'weight_decay': 0.0}
        ], betas=tuple(optimizer_config['betas']), eps=float(optimizer_config['eps']))
    
    return AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(optimizer_config['betas']),
        eps=float(optimizer_config['eps']),
        weight_decay=weight_decay
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """Build learning rate scheduler with warmup."""
    training_config = config['training']
    max_steps = int(training_config['max_steps'])
    lr = float(training_config['learning_rate'])
    
    warmup_ratio = float(training_config.get('warmup_ratio', 0.05))
    warmup_min_steps = int(training_config.get('warmup_min_steps', 10000))
    warmup_from_ratio = int(max_steps * warmup_ratio)
    
    warmup_steps = min(warmup_from_ratio, warmup_min_steps)
    warmup_steps = min(warmup_steps, max_steps - 1)
    warmup_steps = max(1, warmup_steps)
    
    cosine_steps = max_steps - warmup_steps
    if cosine_steps <= 0:
        cosine_steps = 1
    
    warmup = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=lr * 0.02)
    
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
