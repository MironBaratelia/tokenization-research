"""Internal utilities for miron_lib. No external project dependencies."""
import logging
import os

logger = logging.getLogger(__name__)


def memory_profile_enabled() -> bool:
    """Check if memory profiling is enabled via env var."""
    return os.environ.get("MIRON_MEMORY_PROFILE", "").lower() in ("1", "true", "yes")


def _mem(label: str, prev_alloc: float = 0.0) -> float:
    """Log CUDA memory at checkpoint. Returns current allocated GiB."""
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    delta = alloc - prev_alloc
    delta_str = f" (+{delta:.2f})" if prev_alloc > 0 else ""
    logger.info(
        "[MEM] %s: %.2f GiB allocated, %.2f GiB reserved%s",
        label,
        alloc,
        reserved,
        delta_str,
    )
    return alloc
