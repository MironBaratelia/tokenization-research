import torch

from src.common.math_utils import safe_exp


def tensor_to_float_for_logging(v: torch.Tensor) -> float:
    """Scalar for JSON/TensorBoard. Never use O(N) ``.tolist()`` on large tensors."""
    if v.numel() == 1:
        return float(v.detach().item())
    return float(v.detach().float().mean().cpu())


def perplexity(loss):
    return safe_exp(loss)


def tokens_per_second(num_tokens, time_elapsed):
    if time_elapsed <= 0:
        return 0.0
    return num_tokens / time_elapsed


def gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
