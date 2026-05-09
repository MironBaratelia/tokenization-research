"""Utilities for tokenizer operations."""
import os
from typing import Optional


def resolve_train_file(full_config: dict, project_root: str) -> str:
    """Resolve training file path with fallback to .txt extension."""
    data = full_config.get("data", {})
    train_file = data.get("train_tokenizer_file") if isinstance(data, dict) else None

    if not train_file:
        raise ValueError("train_tokenizer_file not specified in config")
    
    if not os.path.exists(train_file):
        resolved = os.path.join(project_root, train_file)
        if os.path.exists(resolved):
            return resolved
        
        txt_file = train_file.replace(".arrow", ".txt")
        if os.path.exists(txt_file):
            return txt_file
        resolved_txt = os.path.join(project_root, txt_file)
        if os.path.exists(resolved_txt):
            return resolved_txt
    
    return train_file


def resolve_check_file(project_root: str, language: str) -> Optional[str]:
    """Resolve path to tokenizer check file."""
    check_file = os.path.join(project_root, "data", "base", language, "tokenizer_check.txt")
    if os.path.exists(check_file):
        return check_file
    return None
