import logging
import os
import sys
from typing import Optional

import torch

def setup_project_env(project_root: Optional[str] = None):
    """Sets up project root in sys.path and configures system-specific settings."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    if sys.platform == 'win32':
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    return project_root

def configure_logging(level=logging.INFO, verbose: bool = False):
    if verbose:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def get_torch_accelerator():
    """Returns the best available torch device and sets optimization flags."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
