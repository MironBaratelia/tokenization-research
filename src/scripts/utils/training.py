"""Utilities for training pipeline."""
import os
from typing import Any, Dict, Optional, Tuple


def resolve_data_paths(full_config: Dict[str, Any], project_root: str, output_dir: str) -> Tuple[str, str]:
    """Determine absolute paths for training and validation data, prioritizing tokenized versions."""
    data_config = full_config.get("data", {})
    tokenized_id = data_config.get("tokenized_id")
    if tokenized_id:
        outputs_dir = full_config.get("resolved_paths", {}).get("outputs_dir") or os.path.join(project_root, "outputs")
        tokenized_dir = os.path.join(outputs_dir, str(tokenized_id), "tokenized")
    else:
        tokenized_dir = os.path.join(output_dir, "tokenized")
    
    shuffled_manifest = os.path.join(tokenized_dir, "train_packed_shuffled.manifest.json")
    packed_manifest = os.path.join(tokenized_dir, "train_packed.arrow.manifest.json")
    val_manifest = os.path.join(tokenized_dir, "val_packed.arrow.manifest.json")
    
    if os.path.exists(shuffled_manifest):
        train_path = shuffled_manifest
    elif os.path.exists(packed_manifest):
        raise FileNotFoundError(
            f"Expected shuffled training manifest at {shuffled_manifest}, but only packed data exists at "
            f"{packed_manifest}. Run preprocessing step 2 with shuffling enabled so training uses the prepared "
            "tokenized pipeline instead of silently changing the data order."
        )
    else:
        train_path = data_config.get("train_file")
        if train_path and not os.path.isabs(train_path):
            train_path = os.path.normpath(os.path.join(project_root, train_path))

    if os.path.exists(val_manifest):
        val_path = val_manifest
    else:
        val_path = data_config.get("validation_file")
        if val_path and not os.path.isabs(val_path):
            val_path = os.path.normpath(os.path.join(project_root, val_path))

    return train_path, val_path


def resolve_check_file(project_root: str, language: str) -> Optional[str]:
    """Resolve path to tokenizer check file."""
    check_file = os.path.join(project_root, f"data/base/{language}/tokenizer_check.txt")
    if os.path.exists(check_file):
        return check_file
    return None
