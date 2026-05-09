import json
import math
import os
import time
import logging
import warnings
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from src.common.tqdm_utils import tqdm

from .callbacks import TensorBoardCallback, MetricsLogger, EarlyStoppingCallback
from .metrics import perplexity, tokens_per_second

def _grad_scaler_cuda():
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return torch.cuda.amp.GradScaler()

class _NoOpScaler:
    def is_enabled(self) -> bool: return False
    def scale(self, x: torch.Tensor) -> torch.Tensor: return x
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: pass
    def step(self, optimizer: torch.optim.Optimizer) -> None: optimizer.step()
    def update(self, new_scale: Optional[float] = None) -> None: pass

def _get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        experiment_id: str,
        tokenizer: Any = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_id = experiment_id
        self.tokenizer = tokenizer

        self.device = next(model.parameters()).device
        train_cfg = _get(config, "training", config)
        if isinstance(train_cfg, dict):
            train_cfg = type("Cfg", (), train_cfg)()

        self.gradient_accumulation_steps = int(_get(train_cfg, "gradient_accumulation_steps", 1))
        self.max_steps = int(_get(train_cfg, "max_steps", 0))
        self.stop_after_steps = int(_get(train_cfg, "stop_after_steps", 0))
        self.max_wall_clock_minutes = float(_get(train_cfg, "max_wall_clock_minutes", 0.0))
        self.speed_gate_min_tps = float(_get(train_cfg, "speed_gate_min_tps", 0.0))
        self.speed_gate_max_step = int(_get(train_cfg, "speed_gate_max_step", 0))
        self.max_grad_norm = float(_get(train_cfg, "max_grad_norm", 1.0))
        use_fp16 = bool(_get(train_cfg, "fp16", False))
        use_bf16 = bool(_get(train_cfg, "bf16", False))
        self._use_amp = (use_fp16 or use_bf16) and self.device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        self.scaler = _grad_scaler_cuda() if use_fp16 and self.device.type == "cuda" else _NoOpScaler()

        self.eval_steps = int(_get(train_cfg, "eval_steps", 500))
        self.logging_steps = int(_get(train_cfg, "logging_steps", 100))
        self.save_steps = int(_get(train_cfg, "save_steps", 2000))
        self.metrics_log_steps = int(_get(train_cfg, "metrics_log_steps", 0))

        paths = _get(config, "resolved_paths", _get(config, "paths", {}))
        outputs_dir = _get(paths, "outputs_dir", "outputs")
        if isinstance(outputs_dir, str) and not os.path.isabs(outputs_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            outputs_dir = os.path.join(project_root, outputs_dir)
        
        self._save_dir = os.path.join(outputs_dir, experiment_id)
        self._checkpoint_dir = os.path.join(self._save_dir, "checkpoints")
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        log_dir = os.path.join(self._save_dir, "logs", "tensorboard")
        metrics_log_path = os.path.join(self._save_dir, "logs", "metrics.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
        
        self.tensorboard_callback = TensorBoardCallback(log_dir)
        self.metrics_logger = MetricsLogger(metrics_log_path)
        self.early_stopping = EarlyStoppingCallback(
            patience=int(_get(train_cfg, "early_stopping_patience", 10)),
            min_delta=float(_get(train_cfg, "early_stopping_min_delta", 0.0001)),
            min_steps=int(_get(train_cfg, "early_stopping_min_steps", 100000)),
        )

        self.global_step = 0
        self.best_val_loss: Optional[float] = None
        self.best_checkpoint_metric: Optional[float] = None
        self.logger = logging.getLogger(f"Trainer")
        self._inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        self._generation_seed: Optional[str] = None

    def _get_generation_seed(self) -> str:
        """Resolve generation seed text: human_eval.txt, else generation_seed.txt or tokenizer_check.txt."""
        if self._generation_seed is not None:
            return self._generation_seed
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        paths = _get(self.config, "resolved_paths", _get(self.config, "paths", {}))
        data_dir = _get(paths, "data_dir", None)
        if not data_dir or not os.path.isabs(data_dir):
            data_dir = data_dir or os.path.join(project_root, "data", "base")
            if not os.path.isabs(data_dir):
                data_dir = os.path.join(project_root, data_dir)
        lang = _get(self.config, "language", "en")
        benchmarks_dir = os.path.join(project_root, "data", "benchmarks")
        for path in (
            os.path.join(benchmarks_dir, lang, "human_eval.txt"),
            os.path.join(data_dir, lang, "generation_seed.txt"),
            os.path.join(data_dir, lang, "tokenizer_check.txt"),
        ):
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                self._generation_seed = line
                                return self._generation_seed
                except Exception:
                    pass
        self._generation_seed = " "
        return self._generation_seed

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dev = self.device
        inp = {k: v.to(dev, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        if self._use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=self._amp_dtype):
                out = self.model(**inp)
        else:
            out = self.model(**inp)
            
        return out.get("loss", out) if isinstance(out, dict) else out

    def _loss_metric_weight(self, batch: Dict[str, Any]) -> Optional[int]:
        """Override in subclasses. If not None, replaces char-token weighting for logging averages."""
        return None

    def _token_count_for_speed(self, batch: Dict[str, Any]) -> int:
        input_ids = batch.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            return int(input_ids.numel())
        return 0

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        loss = self.compute_loss(batch)

        if not isinstance(loss, torch.Tensor) or not loss.requires_grad or not torch.isfinite(loss).all():
            return {"loss": 0.0, "_skip": True}

        scaled = loss / self.gradient_accumulation_steps

        if self.scaler.is_enabled():
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        return {"loss": loss.detach().item()}

    def validation_step(self) -> Dict[str, float]:
        self.model.eval()
        total_loss_sum = 0.0
        total_valid_tokens = 0
        
        with self._inference_ctx():
            for batch in self.val_loader:
                labels = batch.get("labels", batch.get("input_ids"))
                if labels is not None and isinstance(labels, torch.Tensor):
                    ignore_idx = getattr(self.model, "ignore_index", -100)
                    num_valid = (labels[:, 1:] != ignore_idx).sum().item() if labels.dim() == 2 else (labels != ignore_idx).sum().item()
                else:
                    num_valid = 1
                
                loss = self.compute_loss(batch)
                total_loss_sum += (loss.item() if isinstance(loss, torch.Tensor) else float(loss)) * num_valid
                total_valid_tokens += num_valid
                
        if total_valid_tokens == 0: return {"val_loss": 0.0, "val_ppl": 1.0}
        vloss = total_loss_sum / total_valid_tokens
        return {"val_loss": vloss, "val_ppl": perplexity(vloss)}

    def _generate_sample(self) -> Optional[str]:
        """Return decoded continuation, or None if this model is not wired to InferenceEngine.generate."""
        from src.common.inference import InferenceEngine
        if self.tokenizer is None:
            return None
        if not hasattr(self.model, "generate") and not hasattr(self.model, "lm"):
            return None
        if hasattr(self.model, "lm") and not hasattr(self.model, "generate"):
            return None

        self.model.eval()
        engine = InferenceEngine(self.model, self.tokenizer, device=self.device)
        prompt = self._get_generation_seed()
        
        results = engine.generate([prompt], max_new_tokens=128, temperature=0.8, do_sample=True, repetition_penalty=1.1)
        return prompt + (results[0] if results else "")

    def _optimizer_step(self) -> bool:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
            
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, foreach=False)

        if not torch.isfinite(grad_norm).item() and self.scaler.is_enabled():
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            return False
            
        self.scaler.step(self.optimizer) if self.scaler.is_enabled() else self.optimizer.step()
        self.scaler.update()
        if self.scheduler:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def _log_step_console(self, step: int, formatted: Dict[str, Any], raw_metrics: Dict[str, Any]):
        """Override to customize step logging. Default: Loss, PPL, TPS, LR."""
        ppl = formatted.get("Train/PPL", 0)
        ppl_str = f"{ppl:.2e}" if ppl > 1e6 else f"{ppl:.2f}"
        self.logger.info(
            "Step %s | Loss: %.4f | PPL: %s | TPS: %.0f | LR: %.2e",
            step,
            formatted.get("Train/Loss", 0),
            ppl_str,
            raw_metrics.get("Speed/TPS", 0),
            raw_metrics.get("lr", 0),
        )

    def _log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = "Train"):
        """Centralized logging for TensorBoard and JSONL."""
        formatted_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item() if v.numel() == 1 else v.tolist()
            
            if k.startswith("gn_"):
                tag = f"Grad/{k}"
            elif k.startswith("Speed/") or k.startswith("Val/"):
                tag = k
            else:
                tag = f"{prefix}/{k.capitalize()}" if not k.isupper() else f"{prefix}/{k}"
            
            formatted_metrics[tag] = v

        if "Train/Loss" in formatted_metrics:
            formatted_metrics["Train/PPL"] = perplexity(formatted_metrics["Train/Loss"])

        self.tensorboard_callback.log(formatted_metrics, step)
        if self.metrics_log_steps <= 0 or step % self.metrics_log_steps == 0 or step >= self.max_steps:
            self.metrics_logger.log({"step": step, **formatted_metrics})
        return formatted_metrics

    def _save_checkpoint(self, step: int):
        """Save training checkpoint and cleanup old ones."""
        path = os.path.join(self._checkpoint_dir, f"step_{step}.pt")
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        
        for f in os.listdir(self._checkpoint_dir):
            full_path = os.path.join(self._checkpoint_dir, f)
            if f.startswith("step_") and f.endswith(".pt") and full_path != path:
                try:
                    os.remove(full_path)
                except Exception as e:
                    self.logger.warning(f"Could not remove old checkpoint {f}: {e}")

    def _save_best_checkpoint(self, step: int, metric_name: str, metric_value: float, val_metrics: Dict[str, Any]):
        path = os.path.join(self._checkpoint_dir, "best_model.pt")
        torch.save(
            {
                "step": step,
                "monitor_name": metric_name,
                "monitor_value": metric_value,
                "val_metrics": val_metrics,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            },
            path,
        )

    def _checkpoint_metric_info(self, val_metrics: Dict[str, Any]) -> Dict[str, Any]:
        value = val_metrics.get("val_loss", float("inf"))
        return {"name": "val_loss", "mode": "min", "value": value}

    def _is_better_checkpoint(self, metric_info: Dict[str, Any]) -> bool:
        value = metric_info.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
        mode = str(metric_info.get("mode", "min")).lower()
        if self.best_checkpoint_metric is None:
            return True
        if mode == "max":
            return float(value) > float(self.best_checkpoint_metric)
        return float(value) < float(self.best_checkpoint_metric)

    def _validate_optimizer_state_shapes(self) -> None:
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if not state:
                    continue
                for name, value in state.items():
                    if not torch.is_tensor(value) or value.ndim == 0:
                        continue
                    if tuple(value.shape) != tuple(param.shape):
                        raise ValueError(
                            f"optimizer state {name} has shape {tuple(value.shape)} "
                            f"for parameter shape {tuple(param.shape)}"
                        )

    def _load_checkpoint_if_exists(self):
        """Loads the latest checkpoint if it exists in the checkpoint directory."""
        if not os.path.exists(self._checkpoint_dir):
            return
            
        checkpoints = [f for f in os.listdir(self._checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")]
        if not checkpoints:
            return
            
        latest_ckpt = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
        ckpt_path = os.path.join(self._checkpoint_dir, latest_ckpt)
        
        self.logger.info(f"Loading checkpoint from {ckpt_path}...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            load_result = self.model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            if load_result.missing_keys or load_result.unexpected_keys:
                self.logger.warning(
                    "Checkpoint architecture mismatch (missing=%d, unexpected=%d). "
                    "Not loading optimizer, starting from step 0.",
                    len(load_result.missing_keys), len(load_result.unexpected_keys),
                )
                self.global_step = 0
            else:
                try:
                    opt_state = checkpoint.get("optimizer_state_dict")
                    if opt_state:
                        self.optimizer.load_state_dict(opt_state)
                        self._validate_optimizer_state_shapes()
                    else:
                        self.logger.warning(
                            "Checkpoint has no optimizer state; resuming with fresh optimizer."
                        )
                    sched_state = checkpoint.get("scheduler_state_dict")
                    if self.scheduler and sched_state:
                        self.scheduler.load_state_dict(sched_state)
                except (KeyError, RuntimeError, ValueError) as opt_err:
                    self.logger.warning(
                        "Optimizer/scheduler state incompatible (config changed?): %s. Resuming with fresh optimizer.",
                        opt_err,
                    )
                    self.optimizer.state.clear()
                self.global_step = checkpoint["step"]
            self.logger.info(f"Successfully resumed from step {self.global_step}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")

    def train(self) -> None:
        if self.max_steps <= 0: return
        
        self._load_checkpoint_if_exists()
        effective_max_steps = self.max_steps
        if self.stop_after_steps > 0:
            effective_max_steps = min(effective_max_steps, self.stop_after_steps)
        
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        step_metrics_sum = {}
        step_valid_tokens, accum, tokens_processed = 0, 0, 0
        log_start_time = time.perf_counter()
        train_start_time = log_start_time
        stop_reason: Optional[str] = None
        
        all_tps = []
        best_val_loss = float('inf')

        pbar = tqdm(total=effective_max_steps, desc="Training", unit="step", dynamic_ncols=True, initial=self.global_step)

        should_stop = False
        while not should_stop:
            for batch_idx, batch in enumerate(self.train_loader):
                if self.global_step >= effective_max_steps:
                    should_stop = True
                    break
                if self.max_wall_clock_minutes > 0.0:
                    elapsed_minutes = (time.perf_counter() - train_start_time) / 60.0
                    if elapsed_minutes >= self.max_wall_clock_minutes:
                        stop_reason = (
                            f"wall_clock_limit ({elapsed_minutes:.1f} min >= {self.max_wall_clock_minutes:.1f} min)"
                        )
                        should_stop = True
                        break

                try:
                    metrics = self.training_step(batch)
                except KeyboardInterrupt:
                    self.logger.warning(
                        "KeyboardInterrupt during training_step (train batch_idx=%s, global_step=%s).",
                        batch_idx,
                        self.global_step,
                    )
                    raise
                except Exception:
                    self.logger.exception(
                        "training_step failed (train batch_idx=%s, global_step=%s, batch_keys=%s)",
                        batch_idx,
                        self.global_step,
                        list(batch.keys()) if isinstance(batch, dict) else None,
                    )
                    raise

                if "_skip" in metrics: continue
                
                labels = batch.get("labels", batch.get("input_ids"))
                ignore_idx = getattr(self.model, "ignore_index", -100)
                if labels is not None and isinstance(labels, torch.Tensor):
                    current_num_valid = (labels[:, 1:] != ignore_idx).sum().item() if labels.dim() == 2 else (labels != ignore_idx).sum().item()
                else:
                    current_num_valid = 1
                if current_num_valid == 0 and isinstance(batch.get("input_ids"), torch.Tensor):
                    current_num_valid = max(1, int(batch["input_ids"].numel()))

                metric_weight = self._loss_metric_weight(batch)
                if metric_weight is None:
                    metric_weight = current_num_valid

                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        step_metrics_sum[k] = step_metrics_sum.get(k, 0.0) + v * metric_weight
                    else:
                        step_metrics_sum[k] = v  # keep last (weights, log_vars, lists, etc.)

                step_valid_tokens += metric_weight
                accum += 1
                if accum < self.gradient_accumulation_steps: continue

                accum = 0
                try:
                    opt_ok = self._optimizer_step()
                except KeyboardInterrupt:
                    self.logger.warning(
                        "KeyboardInterrupt during optimizer step (global_step=%s).",
                        self.global_step,
                    )
                    raise
                except Exception:
                    self.logger.exception(
                        "optimizer step failed (global_step=%s)",
                        self.global_step,
                    )
                    raise
                if not opt_ok:
                    step_metrics_sum, step_valid_tokens = {}, 0
                    continue

                tokens_processed += self._token_count_for_speed(batch)

                _last_mw = metric_weight
                _aux_from_model = (
                    "lm_loss",
                    "length_penalty",
                    "num_lm_ce_targets",
                    "merged_lm_len",
                    "num_merged_segments",
                    "num_in_vocab_segments",
                    "num_oov_segments",
                    "oov_segment_ratio",
                    "in_vocab_segment_ratio",
                    "fallback_bpe_tokens",
                    "avg_segment_chars",
                    "num_lm_input_tokens",
                    "mean_soft_boundary",
                    "had_segment_truncation",
                    "used_lm_loss_fallback",
                    "boundary_bce",
                    "interior_penalty",
                    "boundary_floor_penalty",
                    "pred_boundary_mean",
                    "gold_boundary_mean",
                    "oov_segment_penalty",
                    "lm_loss_weight",
                    "segmenter_lr_ratio",
                )
                if hasattr(self, '_last_batch_metrics'):
                    for k, v in self._last_batch_metrics.items():
                        if k.startswith('gn_') and k not in step_metrics_sum:
                            step_metrics_sum[k] = v.item() if isinstance(v, torch.Tensor) else v
                        elif k in _aux_from_model and isinstance(v, torch.Tensor) and v.numel() == 1:
                            fv = float(v.detach().item())
                            step_metrics_sum[k] = step_metrics_sum.get(k, 0.0) + fv * _last_mw
                        elif k == "train_phase" and k not in step_metrics_sum:
                            step_metrics_sum[k] = v

                self.global_step += 1
                gs = self.global_step
                pbar.update(1)
                
                den = max(1, step_valid_tokens)
                avg_loss = step_metrics_sum.get("loss", 0.0) / den
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"}, refresh=False)

                if gs % self.logging_steps == 0:
                    elapsed = time.perf_counter() - log_start_time
                    tps = tokens_per_second(tokens_processed, elapsed)
                    all_tps.append(tps)
                    
                    raw_metrics = {
                        "Speed/TPS": tps,
                        "lr": self.optimizer.param_groups[-1]['lr']  # main/default LR
                    }
                    if len(self.optimizer.param_groups) > 1:
                        raw_metrics["lr_encoder"] = self.optimizer.param_groups[0]['lr']
                    for k, v in step_metrics_sum.items():
                        if k.startswith('gn_'):
                            raw_metrics[k] = v  # grad norms are per-step, don't average
                        elif isinstance(v, (int, float)):
                            raw_metrics[k] = v / max(1, step_valid_tokens)
                        else:
                            raw_metrics[k] = v

                    formatted = self._log_metrics(raw_metrics, gs)
                    self._log_step_console(gs, formatted, raw_metrics)

                    if (
                        self.speed_gate_min_tps > 0.0
                        and self.speed_gate_max_step > 0
                        and gs <= self.speed_gate_max_step
                        and tps < self.speed_gate_min_tps
                    ):
                        stop_reason = (
                            f"speed_gate (step={gs}, TPS={tps:.0f} < {self.speed_gate_min_tps:.0f})"
                        )
                        self.logger.warning("Aborting run by %s", stop_reason)
                        should_stop = True

                    step_metrics_sum, step_valid_tokens, tokens_processed = {}, 0, 0
                    log_start_time = time.perf_counter()
                    if should_stop:
                        break

                if gs % self.eval_steps == 0:
                    pause_start = time.perf_counter()
                    val_m = self.validation_step()
                    val_metrics = {f"Val/{k.replace('val_', '').capitalize()}": v for k, v in val_m.items()}
                    self._log_metrics(val_metrics, gs, prefix="Val")
                    
                    vl = val_m.get("val_loss", float("inf"))
                    if isinstance(vl, (int, float)) and math.isfinite(vl) and vl < best_val_loss:
                        best_val_loss = vl

                    metric_info = self._checkpoint_metric_info(val_m)
                    if self._is_better_checkpoint(metric_info):
                        self.best_checkpoint_metric = float(metric_info["value"])
                        self._save_best_checkpoint(
                            gs,
                            str(metric_info.get("name", "metric")),
                            float(metric_info["value"]),
                            val_m,
                        )
                        self.logger.info(
                            "Step %s | New best checkpoint by %s=%s",
                            gs,
                            metric_info.get("name", "metric"),
                            f"{float(metric_info['value']):.6f}",
                        )

                    sample = self._generate_sample()
                    if sample:
                        sample_preview = (sample[:120] + "...") if len(sample) > 120 else sample
                    else:
                        if self.__class__.__name__ == "NeuralSegmenterTrainer" and hasattr(
                            self.model, "segmenter"
                        ):
                            sample_preview = (
                                "Neural joint: val text preview unavailable "
                                "(see Trainer WARNING if generation failed; else 04_evaluate.py / human_eval)."
                            )
                        elif (
                            hasattr(self.model, "segmenter")
                            and hasattr(self.model, "lm")
                            and not hasattr(self.model, "generate")
                        ):
                            sample_preview = (
                                "NeuralSegmenterLM: no training-time text sample in this trainer "
                                "(Val loss is valid; use scripts/04_evaluate.py or human_eval for generation)."
                            )
                        else:
                            sample_preview = (
                                "generation skipped (model has no .generate in this training stack)"
                            )
                    vl = val_m["val_loss"]
                    vl_str = f"{vl:.4f}" if isinstance(vl, (int, float)) and math.isfinite(float(vl)) else str(vl)
                    self.logger.info("Step %s | Val Loss: %s | Sample: %s", gs, vl_str, sample_preview)
                    
                    torch.cuda.empty_cache()
                    self.early_stopping(val_m["val_loss"], gs)
                    if self.early_stopping.early_stop:
                        should_stop = True
                        break
                    self.model.train()
                    pause_elapsed = time.perf_counter() - pause_start
                    log_start_time += pause_elapsed

                if gs % self.save_steps == 0:
                    pause_start = time.perf_counter()
                    self._save_checkpoint(gs)
                    pause_elapsed = time.perf_counter() - pause_start
                    log_start_time += pause_elapsed

        summary_path = os.path.join(self._save_dir, "training_summary_raw.json")
        try:
            summary = {
                "total_steps": self.global_step,
                "avg_tps": sum(all_tps) / len(all_tps) if all_tps else 0,
                "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
                "timestamp": time.time()
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save training summary: {e}")

        if getattr(self, "early_stopping", None) and self.early_stopping.early_stop:
            self.logger.info("Training loop exit: early stopping (validation patience).")
        elif stop_reason is not None:
            self.logger.info("Training loop exit: %s.", stop_reason)
        elif self.global_step >= effective_max_steps:
            self.logger.info(
                "Training loop exit: reached target_steps=%s (configured max_steps=%s).",
                effective_max_steps,
                self.max_steps,
            )
        else:
            self.logger.info(
                "Training loop exit: global_step=%s (target_steps=%s, max_steps=%s, early_stop=%s).",
                self.global_step,
                effective_max_steps,
                self.max_steps,
                getattr(self.early_stopping, "early_stop", False),
            )

        pbar.close()
        self.tensorboard_callback.close()
