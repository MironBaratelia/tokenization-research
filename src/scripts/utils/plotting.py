"""Utilities for plotting and visualization."""
import json
import os
from typing import Any, Dict, List


def load_metrics(path: str) -> List[Dict]:
    """Load metrics from JSONL file."""
    rows = []
    if not os.path.exists(path):
        return rows
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_metrics(rows: List[Dict]) -> tuple:
    """Normalize metric keys (case-insensitive)."""
    train_rows = []
    val_rows = []
    
    for r in rows:
        r_lower = {k.lower(): v for k, v in r.items()}
        
        if "train/loss" in r_lower or "train_loss" in r_lower or "loss" in r_lower:
            normalized = dict(r)
            for key, val in r.items():
                low = key.lower()
                if low in ("train/loss", "train_loss", "loss"):
                    normalized["train_loss"] = val
                elif low in ("train/lr", "lr", "train_lr"):
                    normalized["lr"] = val
                elif low in ("speed/tps", "tokens_per_sec", "tps"):
                    normalized["tokens_per_sec"] = val
                elif low in ("train/ppl", "train_ppl"):
                    normalized["train_ppl"] = val
            train_rows.append(normalized)

        # Normalize common validation metric keys.
        if any(
            k in r_lower
            for k in ("val/loss", "val/val_loss", "val_loss", "val/ppl", "val/val_ppl", "val_ppl")
        ):
            normalized = dict(r)
            for key, val in r.items():
                low = key.lower()
                if low in ("val/loss", "val/val_loss", "val_loss"):
                    normalized["val_loss"] = val
                elif low in ("val/ppl", "val/val_ppl", "val_ppl"):
                    normalized["val_ppl"] = val
            val_rows.append(normalized)
    
    return train_rows, val_rows


def extract_metric_series(rows: List[Dict], metric: str, step_key: str = "step") -> tuple:
    """Extract aligned (step, value) pairs; skip rows missing either field."""
    steps: List[Any] = []
    values: List[Any] = []
    for r in rows:
        s = r.get(step_key)
        if s is None:
            continue
        v = r.get(metric)
        if v is None:
            continue
        steps.append(s)
        values.append(v)
    return steps, values
