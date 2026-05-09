"""MIRON char bottleneck stats: z_c covariance after char_encoder and SVD of enc_proj."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def participation_ratio(eigenvalues: torch.Tensor) -> float:
    ev = eigenvalues[eigenvalues > 1e-12]
    if ev.numel() == 0:
        return 0.0
    s = ev.sum()
    return float((s * s) / (ev * ev).sum().clamp_min(1e-12))


def variance_explained_dim_count(eigenvalues: torch.Tensor, frac: float) -> int:
    ev = torch.sort(eigenvalues, descending=True).values
    ev = ev[ev > 1e-12]
    if ev.numel() == 0:
        return 0
    t = ev.sum() * frac
    c = 0.0
    for k in range(ev.numel()):
        c += ev[k].item()
        if c >= t - 1e-9:
            return k + 1
    return ev.numel()


def enc_proj_svd_metrics(enc_proj: nn.Module) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(enc_proj, nn.Linear):
        out["val_encproj_is_identity"] = 1.0
        return out
    out["val_encproj_is_identity"] = 0.0
    w = enc_proj.weight.detach().float().cpu()
    s = torch.linalg.svdvals(w)
    energy = s * s
    te = float(energy.sum().clamp_min(1e-12))
    cume = torch.cumsum(energy / te, dim=0)
    out["val_encproj_sigma_1"] = float(s[0].item()) if s.numel() else 0.0
    out["val_encproj_energy50_min_rank"] = float(
        int((cume < 0.5).sum().item()) + 1 if s.numel() else 0
    )
    out["val_encproj_energy90_min_rank"] = float(
        int((cume < 0.9).sum().item()) + 1 if s.numel() else 0
    )
    return out


def collect_zc_matrix(
    model: nn.Module,
    data_loader,
    device: torch.device,
    max_batches: int,
    pad_id: int,
) -> torch.Tensor:
    """Rows are word positions (non-pad); float32 on CPU."""
    rows: List[torch.Tensor] = []
    n = 0
    with torch.inference_mode():
        for batch in data_loader:
            if n >= max_batches:
                break
            n += 1
            inp = batch["input_ids"].to(device, non_blocking=True)
            if inp.dim() != 3 or inp.size(1) < 2:
                continue
            enc_in = inp[:, :-1, :].contiguous()
            z_c = model.char_encoder(enc_in)
            z_c = z_c.float()
            z_flat = z_c.reshape(-1, z_c.size(-1))
            mask = enc_in.reshape(-1, enc_in.size(-1))
            mask = (mask != pad_id).any(dim=-1)
            rows.append(z_flat[mask].cpu())
    if not rows:
        return torch.empty(0, 0)
    return torch.cat(rows, dim=0)


def z_covariance_metrics(Z: torch.Tensor) -> Dict[str, float]:
    if Z.numel() == 0 or Z.size(0) < 2:
        return {
            "val_char_z_eff_rank": 0.0,
            "val_char_z_cov_k90_dim": 0.0,
            "val_char_z_cov_k95_dim": 0.0,
        }
    Zm = Z - Z.mean(dim=0, keepdim=True)
    n = Zm.size(0)
    cov = (Zm.T @ Zm) / max(n - 1, 1)
    ev = torch.linalg.eigvalsh(cov)
    return {
        "val_char_z_eff_rank": float(participation_ratio(ev)),
        "val_char_z_cov_k90_dim": float(variance_explained_dim_count(ev, 0.9)),
        "val_char_z_cov_k95_dim": float(variance_explained_dim_count(ev, 0.95)),
    }


def char_bottleneck_metrics_on_val(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_batches: int,
    pad_id: int,
) -> Dict[str, float]:
    model.eval()
    Z = collect_zc_matrix(model, val_loader, device, max_batches, pad_id)
    metrics = z_covariance_metrics(Z)
    metrics.update(enc_proj_svd_metrics(getattr(model, "enc_proj", nn.Identity())))
    return metrics
