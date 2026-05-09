import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    """Load metrics from JSONL file."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_mean(values: List[Any]) -> Optional[float]:
    """Calculate mean of numeric values."""
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def safe_percentile(values: List[Any], pct: int) -> Optional[float]:
    """Calculate percentile of numeric values."""
    values = sorted([v for v in values if isinstance(v, (int, float))])
    if not values:
        return None
    k = int(round((pct / 100.0) * (len(values) - 1)))
    return values[k]


def estimate_steps_per_epoch(project_root: str, experiment_id: str, config: dict) -> Optional[int]:
    """Estimate steps per epoch from training metadata or config."""
    outputs_dir = os.path.join(project_root, "outputs", experiment_id)
    
    metadata_path = os.path.join(outputs_dir, "training_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                m = json.load(f)
                if m.get("steps_per_epoch"):
                    return m["steps_per_epoch"]
        except Exception:
            pass

    manifest_path = os.path.join(outputs_dir, "tokenized", "train_packed.arrow.manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        total_samples = manifest.get("total_samples")
        training_config = config.get("training", {})
        batch_size = training_config.get("batch_size", 1) if isinstance(training_config, dict) else 1
        grad_accum = training_config.get("gradient_accumulation_steps", 1) if isinstance(training_config, dict) else 1
        if total_samples and batch_size and grad_accum:
            steps = total_samples // (batch_size * grad_accum)
            return max(steps, 1)
    
    log_path = os.path.join(outputs_dir, "model_training.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Steps per epoch:" in line:
                    return int(line.strip().split("Steps per epoch:")[-1].strip())
    
    training_config = config.get("training", {})
    return training_config.get("max_steps") if isinstance(training_config, dict) else None


def _lower_key_map(r: Dict) -> Dict[str, Any]:
    return {k.lower(): v for k, v in r.items()}


def _is_train_metric_row(r: Dict) -> bool:
    lk = _lower_key_map(r)
    return "train/loss" in lk or "train_loss" in lk


def _is_val_metric_row(r: Dict) -> bool:
    lk = _lower_key_map(r)
    return any(
        k in lk for k in ("val/loss", "val/val_loss", "val_loss")
    )


def _backfill_val_steps(rows: List[Dict]) -> None:
    for i, r in enumerate(rows):
        if not _is_val_metric_row(r):
            continue
        if r.get("step") is not None:
            continue
        prev = None
        for j in range(i):
            s = rows[j].get("step")
            if s is not None:
                prev = s
        if prev is not None:
            r["step"] = prev


def _normalize_val_row(r: Dict) -> Dict:
    out = dict(r)
    for k, v in r.items():
        low = k.lower()
        if low in ("val/loss", "val/val_loss", "val_loss"):
            out["val_loss"] = v
        elif low in ("val/ppl", "val/val_ppl", "val_ppl"):
            out["val_ppl"] = v
    return out


def _train_row_tps(r: Dict) -> Optional[Any]:
    for k, v in r.items():
        if k.lower() in ("speed/tps", "tokens_per_sec", "tps"):
            return v
    return None


def load_training_metadata(project_root: str, experiment_id: str) -> Dict[str, Any]:
    """Load training metadata from file."""
    outputs_dir = os.path.join(project_root, "outputs", experiment_id)
    metadata_path = os.path.join(outputs_dir, "training_metadata.json")
    
    if not os.path.exists(metadata_path):
        return {}
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _enrich_summary_from_checkpoint(project_root: str, experiment_id: str, summary: Dict[str, Any]) -> None:
    """Add factual parameter / vocab counts from the saved checkpoint (tied weights deduped)."""
    try:
        from src.reporting.checkpoint_param_analysis import checkpoint_stats_for_experiment
    except ImportError:
        return
    try:
        ck = checkpoint_stats_for_experiment(project_root, experiment_id)
    except Exception as e:
        logger.debug("checkpoint stats skipped for %s: %s", experiment_id, e)
        return
    if not ck:
        return
    summary["summary_param_stats_source"] = "checkpoint"
    tu = ck.get("total_params_unique")
    if tu is not None:
        summary["summary_checkpoint_total_params_unique"] = int(tu)
        summary["summary_num_parameters"] = int(tu)
        summary.pop("summary_num_parameters_trainable_only", None)
    for key in ("param_bytes_unique", "param_mib_unique", "param_fp32_mib_unique"):
        if ck.get(key) is not None:
            summary[f"summary_checkpoint_{key}"] = ck[key]
    if ck.get("vocab_size") is not None:
        summary["summary_checkpoint_vocab_size"] = int(ck["vocab_size"])
    if ck.get("hidden_size") is not None:
        summary["summary_checkpoint_hidden_size"] = int(ck["hidden_size"])
    if ck.get("embed_head_params_unique") is not None:
        summary["summary_checkpoint_embed_head_params_unique"] = int(ck["embed_head_params_unique"])
    if ck.get("rest_params_unique") is not None:
        summary["summary_checkpoint_rest_params_unique"] = int(ck["rest_params_unique"])
    if "tied_word_embeddings" in ck:
        summary["summary_checkpoint_tied_word_embeddings"] = bool(ck["tied_word_embeddings"])
    for key in (
        "miron_encoder_params_unique",
        "miron_decoder_params_unique",
        "miron_enc_proj_params_unique",
        "miron_dec_proj_params_unique",
        "miron_char_stack_params_unique",
        "miron_core_lm_params_unique",
    ):
        if ck.get(key) is not None:
            summary[f"summary_{key}"] = int(ck[key])


def build_training_summary(project_root: str, experiment_id: str, config: dict, model=None) -> Optional[Dict[str, Any]]:
    """Build training summary from metrics log."""
    outputs_dir = os.path.join(project_root, "outputs", experiment_id)
    metrics_path = os.path.join(outputs_dir, "logs", "metrics.jsonl")
    rows = load_jsonl(metrics_path)
    
    if not rows:
        return None

    _backfill_val_steps(rows)
    train_rows = [r for r in rows if _is_train_metric_row(r)]
    val_rows = [_normalize_val_row(r) for r in rows if _is_val_metric_row(r)]

    steps = [r.get("step") for r in rows if isinstance(r.get("step"), int)]
    last_step = max(steps) if steps else None
    
    metadata = load_training_metadata(project_root, experiment_id)
    steps_per_epoch = metadata.get("steps_per_epoch") or estimate_steps_per_epoch(project_root, experiment_id, config)
    total_samples = metadata.get("total_samples")
    batch_size = metadata.get("batch_size") or config.get("training", {}).get("batch_size", 1)
    grad_accum = metadata.get("gradient_accumulation_steps") or config.get("training", {}).get("gradient_accumulation_steps", 1)
    context_length = metadata.get("context_length") or config.get("training", {}).get("context_length", 512)

    num_parameters = None
    if model is not None:
        from src.models.model_utils import count_parameters
        num_parameters = count_parameters(model)

    tokens_per_sec = [v for r in train_rows for v in (_train_row_tps(r),) if v is not None]
    avg_tps = safe_mean(tokens_per_sec)
    p50_tps = safe_percentile(tokens_per_sec, 50)

    summary = {
        "summary_last_step": last_step,
        "summary_steps_per_epoch": steps_per_epoch,
        "summary_avg_tokens_per_sec": avg_tps,
        "summary_p50_tokens_per_sec": p50_tps,
    }

    _tr = config.get("training") if isinstance(config.get("training"), dict) else {}
    if _tr.get("max_steps") is not None:
        try:
            summary["summary_max_steps_config"] = int(_tr["max_steps"])
        except (TypeError, ValueError):
            pass

    if num_parameters is not None:
        summary["summary_num_parameters"] = num_parameters
        summary["summary_num_parameters_trainable_only"] = True

    if last_step and steps_per_epoch:
        try:
            spe = float(steps_per_epoch)
            if spe > 0:
                summary["summary_epochs_completed"] = float(last_step) / spe
        except (TypeError, ValueError):
            pass
        summary["summary_pct_data_processed"] = (last_step / steps_per_epoch) * 100.0

    mxs = summary.get("summary_max_steps_config")
    if last_step is not None and mxs is not None:
        try:
            mxf = float(mxs)
            if mxf > 0:
                summary["summary_fraction_of_config_max_steps"] = float(last_step) / mxf
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    _enrich_summary_from_checkpoint(project_root, experiment_id, summary)
        
    if total_samples:
        summary["summary_total_samples"] = total_samples
    elif steps_per_epoch:
        summary["summary_total_samples"] = steps_per_epoch * batch_size * grad_accum

    if val_rows:
        losses = [float(r["val_loss"]) for r in val_rows if r.get("val_loss") is not None]
        if not losses:
            pass
        else:
            first_val_loss = float(val_rows[0]["val_loss"])
            best_val_loss = min(losses)

            ppls = [float(r["val_ppl"]) for r in val_rows if r.get("val_ppl") is not None]
            best_val_ppl = min(ppls) if ppls else None

            quality_metrics: Dict[str, Any] = {}
            for pct in (80, 90, 95, 99):
                total_improvement = first_val_loss - best_val_loss
                target_loss = first_val_loss - (pct / 100.0) * total_improvement
                reached = next((r for r in val_rows if r.get("val_loss") is not None and float(r["val_loss"]) <= target_loss), None)

                if reached:
                    step = reached.get("step")
                    if step is not None:
                        quality_metrics[f"summary_step_to_{pct}pct_quality"] = step
                        if steps_per_epoch:
                            quality_metrics[f"summary_pct_data_to_{pct}pct_quality"] = (step / steps_per_epoch) * 100.0
                        if avg_tps:
                            est_time = (step * batch_size * grad_accum * context_length) / avg_tps
                            quality_metrics[f"summary_time_to_{pct}pct_quality_sec"] = est_time

            val_block: Dict[str, Any] = {
                "summary_first_val_loss": first_val_loss,
                "summary_best_val_loss": best_val_loss,
                **quality_metrics,
            }
            if best_val_ppl is not None:
                val_block["summary_best_val_ppl"] = best_val_ppl
            summary.update(val_block)

    return summary
