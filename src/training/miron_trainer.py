import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .base_trainer import BaseTrainer
from .metrics import tensor_to_float_for_logging


class MIRONTrainer(BaseTrainer):
    """
    Trainer for reference-aligned MIRON (CE on next-word chars; SmolLM2 core).
    Differential LR via build_optimizer (encoder / decoder / rest).
    """

    def _apply_freeze_schedule(self) -> None:
        train_cfg = self.config.get("training", {}) if isinstance(self.config, dict) else {}
        freeze_steps = int(train_cfg.get("freeze_except_modules_steps", 0) or 0)
        modules = train_cfg.get("freeze_except_modules") or []
        if isinstance(modules, str):
            modules = [m.strip() for m in modules.split(",") if m.strip()]
        modules = tuple(str(m) for m in modules)
        phase = "adapter" if freeze_steps > 0 and self.global_step < freeze_steps else "full"
        if getattr(self, "_miron_freeze_phase", None) == phase:
            return

        if phase == "adapter" and modules:
            for _, p in self.model.named_parameters():
                p.requires_grad_(False)
            trainable = 0
            for module_name in modules:
                module = getattr(self.model, module_name, None)
                if isinstance(module, torch.nn.Module):
                    for p in module.parameters():
                        p.requires_grad_(True)
                        trainable += p.numel()
            self.logger.info(
                "MIRON freeze schedule: first %s steps train only %s (%s params).",
                freeze_steps,
                ", ".join(modules),
                trainable,
            )
        else:
            for _, p in self.model.named_parameters():
                p.requires_grad_(True)
            self.logger.info("MIRON freeze schedule: full model trainable at step %s.", self.global_step)
        self._miron_freeze_phase = phase

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
        if raw_metrics.get("lm_char_loss") is not None:
            acc = raw_metrics.get("lm_char_acc", 0)
            acc_v = acc if isinstance(acc, float) else getattr(acc, "item", lambda: acc)()
            msg += " | char CE: %.4f acc: %.4f" % (
                raw_metrics.get("lm_char_loss", 0),
                float(acc_v),
            )
        self.logger.info(msg)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self._apply_freeze_schedule()
        loss = self.compute_loss(batch)
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad or not torch.isfinite(loss).all():
            return {"loss": 0.0, "_skip": True}

        scaled_loss = loss / self.gradient_accumulation_steps
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        metrics = {"loss": loss.detach().item()}
        if hasattr(self, "_last_batch_metrics"):
            for k, v in self._last_batch_metrics.items():
                if k != "loss":
                    if isinstance(v, torch.Tensor):
                        metrics[k] = tensor_to_float_for_logging(v)
                    else:
                        metrics[k] = v

        return metrics

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dev = self.device
        inputs = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        inputs["step"] = self.global_step

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
            return res["loss"]
        return res

    def _optimizer_step(self) -> bool:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)

        gn_lm_val = 0.0
        with torch.no_grad():
            def _param_norm(params):
                total = 0.0
                for p in params:
                    if p.grad is not None:
                        total += p.grad.data.norm(2).item() ** 2
                return total**0.5 if total > 0 else 0.0

            m = self.model
            gn_encoder = _param_norm(m.encoder.parameters()) if hasattr(m, "encoder") else 0.0
            gn_decoder = _param_norm(m.decoder.parameters()) if hasattr(m, "decoder") else 0.0
            gn_lm = _param_norm(m.lm.parameters()) if hasattr(m, "lm") else 0.0

            if hasattr(m, "encoder"):
                torch.nn.utils.clip_grad_norm_(m.encoder.parameters(), self.max_grad_norm)
            if hasattr(m, "decoder"):
                torch.nn.utils.clip_grad_norm_(m.decoder.parameters(), self.max_grad_norm)
            if hasattr(m, "lm"):
                torch.nn.utils.clip_grad_norm_(m.lm.parameters(), self.max_grad_norm)
            for name in ("enc_proj", "dec_proj"):
                mod = getattr(m, name, None)
                if isinstance(mod, torch.nn.Module) and not isinstance(
                    mod, torch.nn.Identity
                ):
                    torch.nn.utils.clip_grad_norm_(mod.parameters(), self.max_grad_norm)

            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if hasattr(self, "_last_batch_metrics"):
                self._last_batch_metrics.update(
                    {
                        "gn_total": total_norm.item(),
                        "gn_encoder": gn_encoder,
                        "gn_decoder": gn_decoder,
                        "gn_lm": gn_lm,
                    }
                )
            gn_lm_val = total_norm.item()

        if not torch.isfinite(torch.tensor(gn_lm_val)):
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.update()
            return False

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
        else:
            self.optimizer.step()

        self.scaler.update()
        if self.scheduler:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        return True

    def validation_step(self) -> Dict[str, float]:
        val_metrics = super().validation_step()
        self.run_manifold_diagnostics()
        train_cfg = self.config.get("training", {}) if isinstance(self.config, dict) else {}
        nb = int(train_cfg.get("char_bottleneck_val_batches", 12))
        if nb > 0 and hasattr(self.model, "char_encoder") and hasattr(self.model, "enc_proj"):
            from .miron_bottleneck_stats import char_bottleneck_metrics_on_val

            pad_id = int(getattr(self.model, "pad_id", 0))
            try:
                extra = char_bottleneck_metrics_on_val(
                    self.model, self.val_loader, self.device, nb, pad_id
                )
                val_metrics.update(extra)
            except Exception as e:
                self.logger.warning("char bottleneck metrics skipped: %s", e)
        return val_metrics

    def run_manifold_diagnostics(self):
        self.model.eval()
        lang = self.config.get("language", "en")
        diag_cfg = self.config.get("diagnostics", {})
        base_word = diag_cfg.get("base_word", "человек" if lang == "ru" else "human")
        variants = diag_cfg.get(
            "variants",
            (
                [
                    "человека",
                    "человеком",
                    "человеку",
                    "люди",
                    "человечек",
                    "челюсть",
                    "чемодан",
                    "челябинск",
                    "трактор",
                ]
                if lang == "ru"
                else [
                    "humans",
                    "humanly",
                    "humanity",
                    "people",
                    "humanoid",
                    "humble",
                    "humor",
                    "humid",
                    "tractor",
                ]
            ),
        )
        with torch.no_grad():
            all_words = [base_word] + variants
            max_word_len = self.tokenizer.max_word_length
            word_tensor = torch.full(
                (len(all_words), max_word_len),
                self.model.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for i, w in enumerate(all_words):
                w_ids = self.tokenizer.encode(w)[0]
                word_tensor[i, : min(len(w_ids), max_word_len)] = torch.tensor(
                    w_ids[:max_word_len], dtype=torch.long, device=self.device
                )
            if hasattr(self.model, "encode_batch"):
                z = self.model.encode_batch(word_tensor)
            else:
                z = word_tensor.float()
            z_norm = F.normalize(z, p=2, dim=-1)
            sim_matrix = torch.matmul(z_norm, z_norm.T)
            manifold_metrics = {}
            for i, w in enumerate(variants):
                manifold_metrics[f"Manifold/Sim_{w}"] = sim_matrix[0, i + 1].item()
            n = sim_matrix.size(0)
            manifold_metrics["Manifold/Isotropy"] = (
                (sim_matrix.sum() - n) / (n * (n - 1))
            ).item()
            if hasattr(self, "tensorboard_callback") and self.tensorboard_callback:
                self.tensorboard_callback.log(manifold_metrics, self.global_step)
        self.model.train()

    def _generate_sample(self) -> Optional[str]:
        from src.common.inference import InferenceEngine

        if self.tokenizer is None:
            return None

        self.model.eval()
        engine = InferenceEngine(self.model, self.tokenizer, device=self.device)
        prompt = self._get_generation_seed()

        results = engine.generate([prompt], max_new_tokens=15, temperature=0.7, do_sample=True)
        return prompt + (results[0] if results else "")
