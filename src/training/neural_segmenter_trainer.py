import math
from typing import Any, Dict, List, Optional

import torch

from src.training.base_trainer import BaseTrainer
from src.training.metrics import perplexity


class NeuralSegmenterTrainer(BaseTrainer):
    """
    Trainer for Neural Segmenter End-to-End.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_cfg = self.config.get("training", {}) if isinstance(self.config, dict) else getattr(self.config, "training", {})
        segmenter_online_training = bool(train_cfg.get("segmenter_online_training", True))
        explicit_unfreeze = int(train_cfg.get("segmenter_unfreeze_step", 0))
        from_end = int(train_cfg.get("segmenter_unfreeze_from_end_steps", 0))
        if not segmenter_online_training:
            self._resolved_segmenter_unfreeze_step = int(self.max_steps) + 1
        elif from_end > 0:
            self._resolved_segmenter_unfreeze_step = max(0, int(self.max_steps) - from_end)
        else:
            self._resolved_segmenter_unfreeze_step = explicit_unfreeze
        if hasattr(self.model, "segmenter_unfreeze_step"):
            self.model.segmenter_unfreeze_step = int(self._resolved_segmenter_unfreeze_step)
        if not segmenter_online_training:
            self.logger.info(
                "Segmenter online training disabled; using frozen fast-token path for all %s steps.",
                self.max_steps,
            )
        elif from_end > 0:
            self.logger.info(
                "Resolved segmenter unfreeze step: %s (max_steps=%s, from_end=%s)",
                self._resolved_segmenter_unfreeze_step,
                self.max_steps,
                from_end,
            )

    def _generation_prompt_candidates(self) -> List[str]:
        prompts: List[str] = []
        seed = self._get_generation_seed()
        if isinstance(seed, str) and seed.strip():
            prompts.append(seed.strip())

        if isinstance(self.config, dict):
            language = str(self.config.get("language", "en"))
        else:
            language = str(getattr(self.config, "language", "en"))

        defaults = {
            "ru": [
                "Александр Пушкин родился",
                "Москва — столица",
                "Машинное обучение — это",
            ],
            "en": [
                "The capital of France is",
                "Machine learning is",
            ],
        }
        for prompt in defaults.get(language, defaults["en"]):
            if prompt not in prompts:
                prompts.append(prompt)
        return prompts

    def _generate_sample(self) -> Optional[str]:
        """
        Val-time preview: LM.generate on segment ids from the trained segmenter (tokenizer in 'locked'
        path), while training stays on char-level joint. Not identical to joint forward but useful sanity.
        """
        if self.tokenizer is None or not hasattr(self.model, "lm"):
            return None
        stage = getattr(self.tokenizer, "training_stage", "")
        if stage not in ("joint", "adaptation"):
            return super()._generate_sample()

        prev_stage = self.tokenizer.training_stage
        try:
            self.tokenizer.training_stage = "locked"
            mpe = int(getattr(self.model.lm.config, "max_position_embeddings", 512))
            max_new = min(64, max(8, mpe // 8))
            max_prompt = max(1, mpe - max_new - 2)
            veos = getattr(self.tokenizer, "vocab", {}).get(getattr(self.tokenizer, "eos_token", "<eos>"))
            vpad = getattr(self.tokenizer, "pad_token_id", 0)
            self.model.lm.eval()

            last_error: Optional[str] = None
            for prompt in self._generation_prompt_candidates():
                try:
                    ids: List[int] = self.tokenizer.encode(prompt, add_special_tokens=True)
                    if not ids:
                        last_error = f"prompt produced zero locked ids: {prompt!r}"
                        continue
                    if len(ids) > max_prompt:
                        ids = ids[:max_prompt]
                    t = torch.tensor([ids], device=self.device, dtype=torch.long)
                    out = self.model.lm.generate(
                        t,
                        max_new_tokens=max_new,
                        eos_token_id=veos,
                        pad_token_id=vpad,
                        temperature=0.85,
                        do_sample=True,
                        repetition_penalty=1.1,
                    )
                    text = self.tokenizer.decode(out[0].tolist())
                    if text and text.strip():
                        return text[:280] + ("..." if len(text) > 280 else "")
                    last_error = f"empty decode for prompt: {prompt!r}"
                except Exception as e:
                    last_error = f"{prompt!r}: {e}"
                    continue

            if last_error:
                self.logger.warning("Neural val preview unavailable after prompt fallbacks: %s", last_error)
            return None
        finally:
            self.tokenizer.training_stage = prev_stage

    def _maybe_enable_joint_char_inputs(self) -> None:
        unfreeze_step = int(getattr(self, "_resolved_segmenter_unfreeze_step", 0))
        if unfreeze_step <= 0 or self.global_step < unfreeze_step:
            return
        for loader in (self.train_loader, self.val_loader):
            ds = getattr(loader, "dataset", None)
            if ds is not None and hasattr(ds, "enable_segment_supervision"):
                ds.enable_segment_supervision()

    def _segmenter_lr_ratio_for_step(self, step: int) -> float:
        train_cfg = self.config.get("training", {}) if isinstance(self.config, dict) else getattr(self.config, "training", {})
        unfreeze_step = int(getattr(self, "_resolved_segmenter_unfreeze_step", 0))
        if unfreeze_step > 0 and step < unfreeze_step:
            return 0.0
        freeze_step = int(train_cfg.get("segmenter_freeze_step", 0))
        if freeze_step > 0 and step >= freeze_step:
            return 0.0
        start_ratio = float(train_cfg.get("segmenter_lr_ratio_start", train_cfg.get("segmenter_lr_ratio", 0.1)))
        end_ratio = float(train_cfg.get("segmenter_lr_ratio", 0.1))
        freeze_step = int(train_cfg.get("segmenter_freeze_step", 0))
        decay_start = int(
            train_cfg.get(
                "segmenter_lr_decay_start",
                int(train_cfg.get("boundary_warmup_steps", 0)) + int(train_cfg.get("joint_lm_ramp_steps", 0)),
            )
        )
        decay_steps = int(train_cfg.get("segmenter_lr_decay_steps", 12000))

        if freeze_step > 0 and step >= freeze_step:
            return 0.0

        if decay_steps <= 0 or step <= decay_start:
            return start_ratio

        progress = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_ratio + (start_ratio - end_ratio) * cosine

    def _apply_segmenter_lr_schedule(self, step: int) -> None:
        if len(self.optimizer.param_groups) < 2:
            return
        if getattr(self.model.__class__, "__name__", "") != "NeuralSegmenterLM":
            return
        main_lr = float(self.optimizer.param_groups[-1]["lr"])
        seg_ratio = self._segmenter_lr_ratio_for_step(step)
        self.optimizer.param_groups[0]["lr"] = main_lr * seg_ratio
        if seg_ratio <= 0.0:
            for param in self.model.segmenter.parameters():
                param.requires_grad = False
        else:
            for param in self.model.segmenter.parameters():
                param.requires_grad = True

    def _use_sparse_frozen_joint_step(self, step: int) -> bool:
        train_cfg = self.config.get("training", {}) if isinstance(self.config, dict) else getattr(self.config, "training", {})
        interval = int(train_cfg.get("segmenter_sparse_update_interval", 1))
        if interval <= 1:
            return False
        unfreeze_step = int(getattr(self, "_resolved_segmenter_unfreeze_step", 0))
        if unfreeze_step <= 0 or step < unfreeze_step:
            return False
        if getattr(self.model.__class__, "__name__", "") != "NeuralSegmenterLM":
            return False
        return (int(step) - unfreeze_step) % interval != 0

    def _loss_metric_weight(self, batch: Dict) -> Optional[int]:
        # Char-level labels in the batch do not match LM CE positions; weighting by them corrupts the running average.
        return 1

    def _token_count_for_speed(self, batch: Dict[str, Any]) -> int:
        frozen = batch.get("frozen_attention_mask")
        if isinstance(frozen, torch.Tensor):
            return int(frozen.sum().item())
        frozen_ids = batch.get("frozen_input_ids")
        if isinstance(frozen_ids, torch.Tensor):
            return int((frozen_ids != self.tokenizer.pad_token_id).sum().item())
        input_ids = batch.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            return int(input_ids.numel())
        return 0

    def validation_step(self) -> Dict[str, float]:
        """Mean loss per validation batch (char-token weighting is meaningless for NeuralSegmenterLM)."""
        self._maybe_enable_joint_char_inputs()
        self.model.eval()
        total, n_batches, n_skip_nonfinite, n_seen = 0.0, 0, 0, 0
        first_bad_batch: Optional[Dict[str, Any]] = None
        agg: Dict[str, float] = {
            "val_boundary_bce": 0.0,
            "val_oov_segment_ratio": 0.0,
            "val_in_vocab_segment_ratio": 0.0,
            "val_pred_boundary_mean": 0.0,
            "val_gold_boundary_mean": 0.0,
            "val_oov_segment_penalty": 0.0,
            "val_truncation_rate": 0.0,
            "val_avg_segment_chars": 0.0,
        }
        with self._inference_ctx():
            for batch in self.val_loader:
                n_seen += 1
                loss = self.compute_loss(batch)
                if isinstance(loss, torch.Tensor) and torch.isfinite(loss).all():
                    total += float(loss.detach().item())
                    n_batches += 1
                    for key, out_key in (
                        ("boundary_bce", "val_boundary_bce"),
                        ("oov_segment_ratio", "val_oov_segment_ratio"),
                        ("in_vocab_segment_ratio", "val_in_vocab_segment_ratio"),
                        ("pred_boundary_mean", "val_pred_boundary_mean"),
                        ("gold_boundary_mean", "val_gold_boundary_mean"),
                        ("oov_segment_penalty", "val_oov_segment_penalty"),
                        ("had_segment_truncation", "val_truncation_rate"),
                        ("avg_segment_chars", "val_avg_segment_chars"),
                    ):
                        value = getattr(self, "_last_batch_metrics", {}).get(key)
                        if isinstance(value, torch.Tensor) and value.numel() == 1 and torch.isfinite(value).all():
                            agg[out_key] += float(value.detach().item())
                else:
                    n_skip_nonfinite += 1
                    if first_bad_batch is None:
                        first_bad_batch = {
                            "keys": list(batch.keys()) if isinstance(batch, dict) else None,
                            "input_ids_shape": tuple(batch["input_ids"].shape)
                            if isinstance(batch, dict) and isinstance(batch.get("input_ids"), torch.Tensor)
                            else None,
                            "diag": {
                                k: float(v.detach().item())
                                for k, v in getattr(self, "_last_batch_metrics", {}).items()
                                if isinstance(v, torch.Tensor) and v.numel() == 1
                            },
                        }
        if n_batches == 0:
            self.logger.error(
                "Neural joint validation: zero finite-loss batches (seen=%d, nonfinite_skipped=%d). "
                "Old code reported val_loss=0.0 here — bogus; fix NaN/Inf in joint forward or val batches. "
                "first_bad_batch=%s",
                n_seen,
                n_skip_nonfinite,
                first_bad_batch,
            )
            return {"val_loss": float("nan"), "val_ppl": float("nan")}
        if n_skip_nonfinite:
            self.logger.warning(
                "Neural joint validation: skipped %d batches with non-finite loss (used %d batches).",
                n_skip_nonfinite,
                n_batches,
            )
        vloss = total / n_batches
        metrics = {"val_loss": vloss, "val_ppl": perplexity(vloss)}
        for key, value in agg.items():
            metrics[key] = value / max(1, n_batches)
        return metrics

    def _checkpoint_metric_info(self, val_metrics: Dict[str, Any]) -> Dict[str, Any]:
        val_loss = float(val_metrics.get("val_loss", float("inf")))
        oov_ratio = float(val_metrics.get("val_oov_segment_ratio", 1.0))
        pred_mean = float(val_metrics.get("val_pred_boundary_mean", 0.0))
        gold_mean = float(val_metrics.get("val_gold_boundary_mean", 0.0))
        trunc_rate = float(val_metrics.get("val_truncation_rate", 1.0))
        boundary_bce = float(val_metrics.get("val_boundary_bce", 1.0))

        # Composite objective for checkpoint selection:
        # keep LM quality important, but strongly punish segmentation collapse / OOV drift.
        score = (
            val_loss
            + 2.0 * oov_ratio
            + 2.0 * trunc_rate
            + 4.0 * abs(pred_mean - gold_mean)
            + 0.25 * boundary_bce
        )
        return {"name": "val_composite_score", "mode": "min", "value": score}

    def _log_step_console(self, step: int, formatted: dict, raw_metrics: dict):
        ppl = formatted.get("Train/PPL", 0)
        ppl_str = f"{ppl:.2e}" if ppl > 1e6 else f"{ppl:.2f}"
        msg = "Step %s | Loss: %.4f | PPL: %s | TPS: %.0f | LR: %.2e" % (
            step,
            formatted.get("Train/Loss", 0),
            ppl_str,
            raw_metrics.get("Speed/TPS", 0),
            raw_metrics.get("lr", 0),
        )
        
        phase = raw_metrics.get("train_phase")
        if phase:
            msg += f" | phase:{phase}"
        lmw = raw_metrics.get("lm_loss_weight")
        if lmw is not None and lmw < 0.999:
            msg += f" | lmw:{lmw:.2f}"
        seg_ratio = raw_metrics.get("segmenter_lr_ratio")
        if seg_ratio is not None:
            msg += f" | seglr:{seg_ratio:.2f}"
            if seg_ratio <= 0.0:
                msg += " | seg:freeze"
        if "lm_loss" in raw_metrics:
            msg += f" | LM: {raw_metrics['lm_loss']:.4f} | Pen: {raw_metrics['length_penalty']:.4f}"
        nce = raw_metrics.get("num_lm_ce_targets")
        if nce is not None:
            msg += f" | CE_tok: {nce:.1f}"
            if nce < 1.0:
                msg += " (WARN: ~no LM CE targets — loss near 0 is the L2 fallback, not real perplexity)"
        if formatted.get("Train/Loss", 0) < 1e-4 and (nce is None or nce < 8.0):
            msg += " | WARN: degenerate loss logging window"
        mlen = raw_metrics.get("merged_lm_len")
        nseg = raw_metrics.get("num_merged_segments")
        niv = raw_metrics.get("num_in_vocab_segments")
        noov = raw_metrics.get("num_oov_segments")
        oovr = raw_metrics.get("oov_segment_ratio")
        ivr = raw_metrics.get("in_vocab_segment_ratio")
        avg_chars = raw_metrics.get("avg_segment_chars")
        fbpe = raw_metrics.get("fallback_bpe_tokens")
        mb = raw_metrics.get("mean_soft_boundary")
        tr = raw_metrics.get("had_segment_truncation")
        fb = raw_metrics.get("used_lm_loss_fallback")
        if mlen is not None and nseg is not None and mb is not None:
            msg += f" | LMlen:{mlen:.0f} segs:{nseg:.0f} bmean:{mb:.3f}"
        if niv is not None and noov is not None and oovr is not None:
            msg += f" | iv:{niv:.0f} oov:{noov:.0f} oovr:{oovr:.2f}"
            if ivr is not None:
                msg += f" ivr:{ivr:.2f}"
        if avg_chars is not None:
            msg += f" | avgC:{avg_chars:.1f}"
        if fbpe is not None and fbpe > 0:
            msg += f" fbpe:{fbpe:.0f}"
        if tr is not None and tr > 0.01:
            msg += f" | trunc:{tr:.2f}"
        if fb is not None and fb > 0.01:
            msg += f" | LMfb:{fb:.2f}"
        bfp = raw_metrics.get("boundary_floor_penalty")
        if bfp is not None and bfp > 1e-5:
            msg += f" | Bfloor:{bfp:.4f}"
        pbm = raw_metrics.get("pred_boundary_mean")
        gbm = raw_metrics.get("gold_boundary_mean")
        if pbm is not None and gbm is not None and gbm > 0:
            msg += f" | pb:{pbm:.3f} gb:{gbm:.3f}"
        bbce = raw_metrics.get("boundary_bce")
        if bbce is not None and bbce > 1e-5:
            msg += f" | Bbce:{bbce:.4f}"
        ipin = raw_metrics.get("interior_penalty")
        if ipin is not None and ipin > 1e-5:
            msg += f" | Int:{ipin:.4f}"
        bcp = raw_metrics.get("boundary_ceiling_penalty")
        if bcp is not None and bcp > 1e-5:
            msg += f" | Bceil:{bcp:.4f}"
        oovp = raw_metrics.get("oov_segment_penalty")
        if oovp is not None and oovp > 1e-5:
            msg += f" | OOVseg:{oovp:.4f}"

        self.logger.info(msg)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self._maybe_enable_joint_char_inputs()
        metrics = super().training_step(batch)
        if "_skip" in metrics:
            self.logger.error(
                "training_step skipped (non-finite loss before backward); batch_keys=%s input_ids.shape=%s "
                "last_diag=%s",
                list(batch.keys()) if isinstance(batch, dict) else None,
                tuple(batch["input_ids"].shape)
                if isinstance(batch, dict) and isinstance(batch.get("input_ids"), torch.Tensor)
                else None,
                {
                    k: float(v.detach().item())
                    for k, v in getattr(self, "_last_batch_metrics", {}).items()
                    if isinstance(v, torch.Tensor) and v.numel() == 1
                },
            )
        return metrics

    def _optimizer_step(self) -> bool:
        ok = super()._optimizer_step()
        if ok:
            self._apply_segmenter_lr_schedule(self.global_step + 1)
            if hasattr(self, "_last_batch_metrics"):
                self._last_batch_metrics["segmenter_lr_ratio"] = torch.tensor(
                    self._segmenter_lr_ratio_for_step(self.global_step + 1),
                    device=self.device,
                    dtype=torch.float32,
                )
        return ok

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dev = self.device
        inputs = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        inputs["step"] = self.global_step
        if self.model.training and self._use_sparse_frozen_joint_step(self.global_step):
            inputs["force_frozen_path"] = True

        if self._use_amp:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                res = self.model(**inputs)
        else:
            res = self.model(**inputs)

        if isinstance(res, dict):
            self._last_batch_metrics = {}
            for k, v in res.items():
                if k == "loss":
                    continue
                if isinstance(v, torch.Tensor):
                    self._last_batch_metrics[k] = v.detach()
                else:
                    self._last_batch_metrics[k] = v
            loss_t = res["loss"]
            if isinstance(loss_t, torch.Tensor) and not torch.isfinite(loss_t).all():
                self.logger.error(
                    "Joint forward non-finite loss before train skip; batch_keys=%s input_ids.shape=%s diag=%s",
                    list(batch.keys()) if isinstance(batch, dict) else None,
                    tuple(batch["input_ids"].shape)
                    if isinstance(batch, dict) and isinstance(batch.get("input_ids"), torch.Tensor)
                    else None,
                    {
                        k: float(v.detach().item())
                        for k, v in self._last_batch_metrics.items()
                        if isinstance(v, torch.Tensor) and v.numel() == 1
                    },
                )
            return loss_t
        return res
