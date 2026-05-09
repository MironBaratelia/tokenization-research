import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Set

class NeuralSegmenterLM(nn.Module):
    """
    End-to-End Joint Training wrapper for Neural Segmenter and SmolLM2Model.
    Receives char_ids, predicts boundaries, aggregates tokens, and computes combined loss.
    """
    def __init__(self, segmenter_model, lm_model, tokenizer, config):
        super().__init__()
        self.segmenter = segmenter_model
        self.lm = lm_model
        self.tokenizer = tokenizer
        
        training_cfg = config.get("training", {})
        self.lambda_len = training_cfg.get("lambda_len", 0.001)  # reduced from 0.01 to prevent initial collapse
        # Penalize boundary collapse (low entropy); do NOT subtract H from loss (that makes loss *rise* as boundaries sharpen).
        self.lambda_entropy = float(training_cfg.get("lambda_entropy", 0.001))
        # length_penalty alone can drive all sigmoids → 0 (no interior cuts) → almost no LM CE; floor pushes mean(p) back up.
        self.min_soft_boundary_mean = float(training_cfg.get("min_soft_boundary_mean", 0.02))
        self.lambda_boundary_floor = float(training_cfg.get("lambda_boundary_floor", 0.05))
        self.lambda_boundary_bce = float(training_cfg.get("lambda_boundary_bce", 0.0))
        self.lambda_interior = float(training_cfg.get("lambda_interior", 0.0))
        self.lambda_boundary_ceiling = float(training_cfg.get("lambda_boundary_ceiling", 0.0))
        self.boundary_ceiling_slack = float(training_cfg.get("boundary_ceiling_slack", 0.0))
        # Same semantics as supervised: cuts that yield char spans not in tokenizer.vocab (byte-fallback at encode).
        self.lambda_oov_segment = float(training_cfg.get("lambda_oov_segment", 0.0))
        self.segmenter_unfreeze_step = int(training_cfg.get("segmenter_unfreeze_step", 0))
        self.boundary_warmup_steps = int(training_cfg.get("boundary_warmup_steps", 3000))
        self.joint_lm_ramp_steps = int(training_cfg.get("joint_lm_ramp_steps", 3000))
        self.ramp_lm_update_interval = int(training_cfg.get("ramp_lm_update_interval", 1))
        self.segmenter_joint_decode_mode = str(training_cfg.get("segmenter_joint_decode_mode", "locked_dp"))
        self.segmenter_force_fp32 = bool(training_cfg.get("segmenter_force_fp32", False))
        self.warmup_boundary_bce_weight = float(training_cfg.get("warmup_boundary_bce_weight", 1.0))
        self.warmup_boundary_floor_weight = float(
            training_cfg.get("warmup_boundary_floor_weight", self.lambda_boundary_floor)
        )
        self.warmup_boundary_mean_weight = float(training_cfg.get("warmup_boundary_mean_weight", 0.0))
        self.warmup_boundary_pos_weight = float(training_cfg.get("warmup_boundary_pos_weight", 1.0))
        self.warmup_interior_weight = float(training_cfg.get("warmup_interior_weight", max(self.lambda_interior, 0.0)))
        self.warmup_oov_weight = float(training_cfg.get("warmup_oov_weight", max(self.lambda_oov_segment, 0.0)))
        self._vocab_seg_bytes: Optional[Set[bytes]] = None
        self.pad_id = getattr(tokenizer, "pad_token_id", 0)
        self.unk_token_id = getattr(tokenizer, "unk_token_id", 0)
        if hasattr(lm_model.config, "pad_token_id") and lm_model.config.pad_token_id is not None:
            self.pad_id = lm_model.config.pad_token_id
            
        self.ignore_index = -100

        # Freeze segmenter if in locked stage
        if getattr(self.tokenizer, "training_stage", "") == "locked":
            for param in self.segmenter.parameters():
                param.requires_grad = False

    def _segmenter_is_frozen(self, step: int) -> bool:
        return self.segmenter_unfreeze_step > 0 and int(step) < self.segmenter_unfreeze_step

    def _segmenter_forward(self, input_ids: torch.Tensor):
        if input_ids.is_cuda and self.segmenter_force_fp32:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                return self.segmenter(input_ids)
        return self.segmenter(input_ids)

    def _vocab_segment_bytes_cached(self) -> Set[bytes]:
        if self._vocab_seg_bytes is None:
            from src.models.neural_segmenter_supervised import _build_vocab_segment_bytes

            self._vocab_seg_bytes = _build_vocab_segment_bytes(self.tokenizer)
        return self._vocab_seg_bytes

    def _phase_schedule(self, step: int) -> Dict[str, float]:
        step_i = int(step)
        warmup = max(0, self.boundary_warmup_steps)
        ramp = max(0, self.joint_lm_ramp_steps)

        if step_i < warmup:
            return {"name": "warmup", "lm_weight": 0.0}
        if ramp <= 0:
            return {"name": "joint", "lm_weight": 1.0}

        progress = min(1.0, max(0.0, float(step_i - warmup + 1) / float(ramp)))
        if progress < 1.0:
            return {"name": "ramp", "lm_weight": progress}
        return {"name": "joint", "lm_weight": 1.0}

    def _forward_frozen_tokens(
        self,
        *,
        frozen_input_ids: torch.Tensor,
        frozen_labels: Optional[torch.Tensor],
        frozen_attention_mask: Optional[torch.Tensor],
        device: torch.device,
        phase_name: str = "freeze_fast",
    ) -> Dict[str, torch.Tensor]:
        if frozen_labels is None:
            frozen_labels = frozen_input_ids.clone()
        if frozen_attention_mask is None:
            frozen_attention_mask = frozen_input_ids != self.pad_id

        if frozen_input_ids.size(1) < 2:
            dup_ids = frozen_input_ids[:, -1:].clone()
            dup_labels = frozen_labels[:, -1:].clone()
            dup_mask = frozen_attention_mask[:, -1:].clone()
            frozen_input_ids = torch.cat([frozen_input_ids, dup_ids], dim=1)
            frozen_labels = torch.cat([frozen_labels, dup_labels], dim=1)
            frozen_attention_mask = torch.cat([frozen_attention_mask, dup_mask], dim=1)

        lm_outputs = self.lm(
            input_ids=frozen_input_ids,
            attention_mask=frozen_attention_mask,
            labels=frozen_labels,
        )
        loss = lm_outputs.get("loss", None)
        if loss is None:
            loss = frozen_input_ids.new_zeros((), dtype=torch.float32)

        n_ce = (frozen_labels[:, 1:] != self.ignore_index).sum()
        num_tokens = frozen_attention_mask.sum()

        lm_outputs["loss"] = loss
        lm_outputs["lm_loss"] = loss.detach()
        lm_outputs["length_penalty"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["boundary_floor_penalty"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["boundary_bce"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["interior_penalty"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["boundary_ceiling_penalty"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["pred_boundary_mean"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["gold_boundary_mean"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["oov_segment_penalty"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["mean_soft_boundary"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["num_lm_ce_targets"] = n_ce.detach().float()
        lm_outputs["merged_lm_len"] = frozen_attention_mask.sum(dim=1).float().mean()
        lm_outputs["num_merged_segments"] = num_tokens.detach().float()
        lm_outputs["num_lm_input_tokens"] = num_tokens.detach().float()
        lm_outputs["num_in_vocab_segments"] = num_tokens.detach().float()
        lm_outputs["num_oov_segments"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["oov_segment_ratio"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["in_vocab_segment_ratio"] = torch.ones((), device=device, dtype=torch.float32)
        lm_outputs["fallback_bpe_tokens"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["avg_segment_chars"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["had_segment_truncation"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["used_lm_loss_fallback"] = torch.zeros((), device=device, dtype=torch.float32)
        lm_outputs["lm_loss_weight"] = torch.tensor(1.0, device=device, dtype=torch.float32)
        lm_outputs["train_phase"] = phase_name
        return lm_outputs

    def _boundary_only_outputs(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        denom: torch.Tensor,
        is_bound: torch.Tensor,
        pred_boundary_mean: torch.Tensor,
        gold_boundary_mean: torch.Tensor,
        gold: Optional[torch.Tensor],
        interior: Optional[torch.Tensor],
        phase_name: str,
        lm_weight: float,
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        boundary_bce = torch.zeros((), device=device, dtype=torch.float32)
        floor_pen = torch.zeros((), device=device, dtype=torch.float32)
        mean_pen = torch.zeros((), device=device, dtype=torch.float32)
        interior_pen = torch.zeros((), device=device, dtype=torch.float32)
        oov_seg_pen = torch.zeros((), device=device, dtype=torch.float32)
        loss = torch.zeros((), device=device, dtype=torch.float32)

        if gold is not None:
            pos_weight = None
            if self.warmup_boundary_pos_weight > 1.0:
                pos_weight = torch.tensor(
                    self.warmup_boundary_pos_weight, device=device, dtype=torch.float32
                )
            boundary_bce = (
                F.binary_cross_entropy_with_logits(
                    logits, gold, reduction="none", pos_weight=pos_weight
                )
                * mask
            ).sum() / denom
            loss = loss + self.warmup_boundary_bce_weight * boundary_bce
            if self.warmup_boundary_floor_weight > 0.0:
                min_allowed = torch.clamp(gold_boundary_mean - self.boundary_ceiling_slack, min=0.0)
                floor_gap = torch.relu(min_allowed - pred_boundary_mean)
                floor_pen = floor_gap ** 2
                loss = loss + self.warmup_boundary_floor_weight * floor_pen
            if self.warmup_boundary_mean_weight > 0.0:
                mean_pen = (pred_boundary_mean - gold_boundary_mean) ** 2
                loss = loss + self.warmup_boundary_mean_weight * mean_pen

        if interior is not None and self.warmup_interior_weight > 0.0:
            p = torch.sigmoid(logits.float())
            interior_pen = (p * interior * mask).sum() / denom
            loss = loss + self.warmup_interior_weight * interior_pen

        if self.warmup_oov_weight > 0.0:
            from src.models.neural_segmenter_supervised import _oov_segment_penalty_batch

            vset = self._vocab_segment_bytes_cached()
            if vset:
                oov_seg_pen = _oov_segment_penalty_batch(logits, input_ids, int(self.pad_id), vset, mask)
                loss = loss + self.warmup_oov_weight * oov_seg_pen

        return {
            "loss": loss,
            "lm_loss": torch.zeros((), device=device, dtype=torch.float32),
            "length_penalty": torch.zeros((), device=device, dtype=torch.float32),
            "boundary_floor_penalty": floor_pen,
            "boundary_bce": boundary_bce,
            "interior_penalty": interior_pen,
            "boundary_ceiling_penalty": mean_pen,
            "pred_boundary_mean": pred_boundary_mean,
            "gold_boundary_mean": gold_boundary_mean,
            "oov_segment_penalty": oov_seg_pen,
            "mean_soft_boundary": pred_boundary_mean,
            "num_lm_ce_targets": torch.zeros((), device=device, dtype=torch.float32),
            "merged_lm_len": torch.zeros((), device=device, dtype=torch.float32),
            "num_merged_segments": (is_bound.float() * mask).sum().detach(),
            "num_lm_input_tokens": torch.zeros((), device=device, dtype=torch.float32),
            "num_in_vocab_segments": torch.zeros((), device=device, dtype=torch.float32),
            "num_oov_segments": torch.zeros((), device=device, dtype=torch.float32),
            "oov_segment_ratio": torch.zeros((), device=device, dtype=torch.float32),
            "in_vocab_segment_ratio": torch.zeros((), device=device, dtype=torch.float32),
            "fallback_bpe_tokens": torch.zeros((), device=device, dtype=torch.float32),
            "avg_segment_chars": torch.zeros((), device=device, dtype=torch.float32),
            "had_segment_truncation": torch.zeros((), device=device, dtype=torch.float32),
            "used_lm_loss_fallback": torch.zeros((), device=device, dtype=torch.float32),
            "lm_loss_weight": torch.tensor(lm_weight, device=device, dtype=torch.float32),
            "train_phase": phase_name,
        }

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # input_ids: char_ids (B, L)
        # Note: labels from the data loader are also char_ids, but we'll compute our own labels based on the segments.
        step = int(kwargs.pop("step", 0))
        boundary_targets = kwargs.pop("boundary_targets", None)
        interior_targets = kwargs.pop("interior_targets", None)
        frozen_input_ids = kwargs.pop("frozen_input_ids", None)
        frozen_labels = kwargs.pop("frozen_labels", None)
        frozen_attention_mask = kwargs.pop("frozen_attention_mask", None)
        force_frozen_path = bool(kwargs.pop("force_frozen_path", False))
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        schedule = self._phase_schedule(step)
        phase_name = schedule["name"]
        lm_weight = float(schedule["lm_weight"])
        segmenter_frozen = self._segmenter_is_frozen(step)
        lm_compute = True
        if phase_name == "ramp" and self.ramp_lm_update_interval > 1:
            ramp_offset = max(0, step - self.boundary_warmup_steps)
            lm_compute = (ramp_offset % self.ramp_lm_update_interval) == 0
        
        if force_frozen_path and frozen_input_ids is not None:
            return self._forward_frozen_tokens(
                frozen_input_ids=frozen_input_ids,
                frozen_labels=frozen_labels,
                frozen_attention_mask=frozen_attention_mask,
                device=device,
                phase_name="joint_sparse_frozen",
            )

        if segmenter_frozen:
            if frozen_input_ids is not None:
                return self._forward_frozen_tokens(
                    frozen_input_ids=frozen_input_ids,
                    frozen_labels=frozen_labels,
                    frozen_attention_mask=frozen_attention_mask,
                    device=device,
                    phase_name="freeze_fast",
                )
            with torch.no_grad():
                boundaries, logits = self._segmenter_forward(input_ids)
            boundaries = boundaries.detach()
            logits = logits.detach()
        else:
            boundaries, logits = self._segmenter_forward(input_ids)
        # boundaries are already in (0, 1) using STE
        
        id_to_char_get = self.tokenizer.char_tokenizer.id_to_char.get
        
        valid_masks = (input_ids != self.pad_id)
        valid_counts = valid_masks.sum(dim=1)
        last_valid_indices = valid_counts - 1
        last_valid_indices = torch.clamp(last_valid_indices, min=0)
        if hasattr(self.tokenizer, "hard_boundary_mask_from_logits"):
            is_bound = self.tokenizer.hard_boundary_mask_from_logits(logits, valid_masks)
        else:
            boundary_threshold = float(getattr(self.tokenizer, "boundary_threshold", 0.5))
            is_bound = (torch.sigmoid(logits.float()) >= boundary_threshold) & valid_masks
        # Boundary semantics are token starts, matching tokenizer._segment_char_ids_batch
        batch_idx_range = torch.arange(batch_size, device=device)
        has_valid = valid_counts > 0
        if seq_len > 0 and has_valid.any():
            is_bound[batch_idx_range[has_valid], 0] = True

        gold = None
        interior = None
        need_boundary_targets = (not segmenter_frozen) and (
            phase_name == "warmup"
            or (
                self.training
                and (
                    self.lambda_boundary_floor > 0.0
                    or self.lambda_boundary_bce > 0.0
                    or self.lambda_interior > 0.0
                    or self.lambda_boundary_ceiling > 0.0
                    or self.lambda_oov_segment > 0.0
                )
            )
        )
        if need_boundary_targets:
            if boundary_targets is None or interior_targets is None:
                from src.models.neural_segmenter_supervised import _gold_boundaries_and_interior

                gold, interior = _gold_boundaries_and_interior(
                    input_ids,
                    self.tokenizer,
                    self.tokenizer.char_tokenizer.id_to_char,
                    int(self.pad_id),
                )
            else:
                gold = boundary_targets.to(device=device, dtype=torch.float32)
                interior = interior_targets.to(device=device, dtype=torch.float32)

        mask = valid_masks.to(logits.dtype)
        denom = mask.sum().clamp(min=1.0)
        pred_boundary_mean = (is_bound.float() * mask).sum() / denom
        gold_boundary_mean = (
            (gold * mask).sum() / denom
            if gold is not None
            else torch.zeros((), device=device, dtype=torch.float32)
        )

        if phase_name == "warmup" or not lm_compute:
            boundary_phase_name = phase_name if phase_name == "warmup" else "ramp_sparse"
            return self._boundary_only_outputs(
                logits=logits,
                input_ids=input_ids,
                mask=mask,
                denom=denom,
                is_bound=is_bound,
                pred_boundary_mean=pred_boundary_mean,
                gold_boundary_mean=gold_boundary_mean,
                gold=gold,
                interior=interior,
                phase_name=boundary_phase_name,
                lm_weight=lm_weight if phase_name != "warmup" else 0.0,
            )

        char_seqs = input_ids.detach().cpu().tolist()
        valid_lens_cpu = [int(v) for v in valid_counts.detach().cpu().tolist()]
        use_greedy_boundaries = self.segmenter_joint_decode_mode == "greedy_boundaries"
        if use_greedy_boundaries:
            boundary_starts_cpu = is_bound.detach().cpu().tolist()
            batch_locked_tokens = self.tokenizer.greedy_tokenize_char_ids_batch_with_boundaries(
                char_seqs,
                boundary_starts_cpu,
                valid_lens_cpu,
            )
        else:
            boundary_probs_cpu = torch.sigmoid(logits.float()).detach().cpu().tolist()
        
        all_lbls = []
        all_b_idx = []
        all_end_idx = []
        seq_lengths = []
        oov_segment_count = 0
        in_vocab_segment_count = 0
        predicted_segment_count = 0
        fallback_bpe_token_count = 0
        total_segment_chars = 0
        
        for b in range(batch_size):
            char_seq = [int(idx) for idx in char_seqs[b]]
            valid_len = int(valid_counts[b].item())
            count = 0
            if valid_len <= 0:
                all_lbls.append(self.ignore_index)
                all_b_idx.append(b)
                all_end_idx.append(0)
                seq_lengths.append(1)
                continue

            if use_greedy_boundaries:
                locked_tokens = batch_locked_tokens[b]
            else:
                locked_tokens = self.tokenizer.locked_tokenize_char_ids_with_offsets_from_probs(
                    char_seq[:valid_len],
                    boundary_probs_cpu[b][:valid_len],
                )

            for tok_id, start_idx_val, end_idx_excl in locked_tokens:
                tok_id = int(tok_id)
                start_idx_val = int(start_idx_val)
                end_idx_excl = int(end_idx_excl)
                if end_idx_excl <= start_idx_val:
                    all_lbls.append(self.ignore_index)
                    all_b_idx.append(b)
                    all_end_idx.append(start_idx_val)
                    count += 1
                    continue

                token_char_len = end_idx_excl - start_idx_val
                total_segment_chars += token_char_len
                predicted_segment_count += 1
                if tok_id == self.unk_token_id or tok_id == self.pad_id:
                    oov_segment_count += 1
                else:
                    in_vocab_segment_count += 1
                all_lbls.append(tok_id)
                all_b_idx.append(b)
                all_end_idx.append(start_idx_val)
                count += 1
                
            if count == 0:
                all_lbls.append(self.ignore_index)
                all_b_idx.append(b)
                all_end_idx.append(0)
                count += 1
                
            seq_lengths.append(count)
            
        all_lbls_tensor = torch.tensor(all_lbls, device=device, dtype=torch.long)
        valid_lbls = torch.where(all_lbls_tensor == self.ignore_index, torch.tensor(self.pad_id, device=device), all_lbls_tensor)
        
        # 1. Get embeddings for all tokens in the batch at once
        flat_embs = self.lm.embed_tokens(valid_lbls) # (total_tokens, hidden_size)
        
        # 2. Gather boundary probabilities for gradient flow
        all_b_idx_tensor = torch.tensor(all_b_idx, device=device, dtype=torch.long)
        all_end_idx_tensor = torch.tensor(all_end_idx, device=device, dtype=torch.long)
        flat_bound_probs = boundaries[all_b_idx_tensor, all_end_idx_tensor] # (total_tokens,)
        
        # 3. Straight-through scaling: forward pass sees the unscaled token embedding,
        # while backward still routes LM gradients into the corresponding boundary prob.
        # Directly multiplying by p corrupts the LM input distribution early in training.
        ste_boundary_scale = 1.0 + (
            flat_bound_probs.clamp(min=1e-4) - flat_bound_probs.clamp(min=1e-4).detach()
        )
        flat_embs = flat_embs * ste_boundary_scale.unsqueeze(-1)
        
        # 4. Pad sequences back into batch_size x max_len
        max_len = max(seq_lengths)
        lm_max_pos = self.lm.config.max_position_embeddings
        max_len = min(max_len, lm_max_pos)
        
        # Determine sequence sizes and offsets
        slens_tensor = torch.tensor(seq_lengths, device=device)
        trunc_lens = torch.clamp(slens_tensor, max=max_len)
        
        seq_embeds = []
        seq_labels = []
        offset = 0
        for b in range(batch_size):
            slen = seq_lengths[b]
            trunc_len = min(slen, max_len)
            
            if trunc_len > 0:
                seq_embeds.append(flat_embs[offset : offset + trunc_len])
                seq_labels.append(all_lbls_tensor[offset : offset + trunc_len])
            else:
                # Fallback for empty sequences to maintain batch size
                seq_embeds.append(torch.zeros(1, self.lm.config.hidden_size, device=device, dtype=flat_embs.dtype, requires_grad=True))
                seq_labels.append(torch.tensor([self.ignore_index], device=device, dtype=torch.long))
                
            offset += slen
            
        padded_embeds = torch.nn.utils.rnn.pad_sequence(seq_embeds, batch_first=True, padding_value=0.0)
        padded_labels = torch.nn.utils.rnn.pad_sequence(seq_labels, batch_first=True, padding_value=self.ignore_index)
        
        # Recreate attention mask based on padded_labels not being ignore_index
        attention_mask = (padded_labels != self.ignore_index)
        
        # Ensure max_len is respected if pad_sequence made it smaller
        if padded_embeds.shape[1] < max_len:
            pad_len = max_len - padded_embeds.shape[1]
            padded_embeds = torch.nn.functional.pad(padded_embeds, (0, 0, 0, pad_len))
            padded_labels = torch.nn.functional.pad(padded_labels, (0, pad_len), value=self.ignore_index)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=False)

        # Causal LM loss needs at least two time steps (shift_logits is length T-1).
        if padded_embeds.shape[1] < 2:
            dup_e = padded_embeds[:, -1:, :].clone()
            dup_y = padded_labels[:, -1:].clone()
            dup_m = attention_mask[:, -1:].clone()
            padded_embeds = torch.cat([padded_embeds, dup_e], dim=1)
            padded_labels = torch.cat([padded_labels, dup_y], dim=1)
            attention_mask = torch.cat([attention_mask, dup_m], dim=1)

        diag_lm_tokens = float(sum(seq_lengths))
        diag_pred_segments = float(predicted_segment_count)
        diag_trunc = float((slens_tensor > max_len).any().item())

        # Forward pass through LM
        # We pass attention_mask=None because padding is on the right and causal mask 
        # is sufficient to prevent valid tokens from attending to padding.
        # This completely avoids SDPA custom mask bugs that cause NaN gradients.
        # Full-precision LM forward on CUDA avoids bf16 CE / attention blow-ups when
        # boundary-scaled embeddings have small dynamic range (train *and* val under autocast).
        use_fp32_lm = padded_embeds.is_cuda and padded_embeds.dtype in (
            torch.bfloat16,
            torch.float16,
        )
        if use_fp32_lm:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                lm_outputs = self.lm(
                    inputs_embeds=padded_embeds.float(),
                    attention_mask=None,
                    labels=padded_labels,
                )
        else:
            lm_outputs = self.lm(inputs_embeds=padded_embeds, attention_mask=None, labels=padded_labels)

        # Diagnostic: positions that contribute to shifted CE (ignore_index excluded on labels[:, 1:])
        n_ce = (padded_labels[:, 1:] != self.ignore_index).sum()
        lm_outputs["num_lm_ce_targets"] = n_ce.detach().float()
        lm_outputs["merged_lm_len"] = torch.tensor(
            float(padded_embeds.size(1)), device=device, dtype=torch.float32
        )
        lm_outputs["num_merged_segments"] = torch.tensor(diag_pred_segments, device=device, dtype=torch.float32)
        lm_outputs["num_lm_input_tokens"] = torch.tensor(diag_lm_tokens, device=device, dtype=torch.float32)
        lm_outputs["num_in_vocab_segments"] = torch.tensor(
            float(in_vocab_segment_count), device=device, dtype=torch.float32
        )
        lm_outputs["num_oov_segments"] = torch.tensor(
            float(oov_segment_count), device=device, dtype=torch.float32
        )
        total_segments = max(1.0, float(in_vocab_segment_count + oov_segment_count))
        lm_outputs["oov_segment_ratio"] = torch.tensor(
            float(oov_segment_count) / total_segments, device=device, dtype=torch.float32
        )
        lm_outputs["in_vocab_segment_ratio"] = torch.tensor(
            float(in_vocab_segment_count) / total_segments, device=device, dtype=torch.float32
        )
        lm_outputs["fallback_bpe_tokens"] = torch.tensor(
            float(fallback_bpe_token_count), device=device, dtype=torch.float32
        )
        avg_segment_chars = float(total_segment_chars) / max(1.0, float(predicted_segment_count))
        lm_outputs["avg_segment_chars"] = torch.tensor(
            avg_segment_chars, device=device, dtype=torch.float32
        )
        lm_outputs["mean_soft_boundary"] = boundaries.detach().float().mean()
        lm_outputs["had_segment_truncation"] = torch.tensor(diag_trunc, device=device, dtype=torch.float32)

        loss = lm_outputs.get("loss", None)
        
        if loss is not None:
            # Optional length penalty. In joint training, minimizing sum(p) often fights LM + OOV and
            # drives all soft boundaries → 0 (collapsed merges); set lambda_len=0 to disable.
            if float(self.lambda_len) > 0.0:
                length_penalty = boundaries.sum() * float(self.lambda_len) / batch_size
            else:
                length_penalty = loss.new_zeros(())

            if not torch.isfinite(loss).all():
                # Rare after SmolLM2 masked CE + min length; keep a finite, differentiable objective.
                loss = length_penalty + boundaries.float().mean() * (self.lambda_len * 10.0)
                lm_outputs["lm_loss"] = torch.zeros((), device=device, dtype=torch.float32)
                lm_outputs["used_lm_loss_fallback"] = torch.tensor(1.0, device=device, dtype=torch.float32)
            else:
                lm_outputs["lm_loss"] = loss.detach()
                loss = loss * lm_weight + length_penalty
                lm_outputs["used_lm_loss_fallback"] = torch.tensor(0.0, device=device, dtype=torch.float32)

            lm_outputs["length_penalty"] = length_penalty

            if self.training and not segmenter_frozen and self.lambda_entropy > 0.0:
                lf = logits.float().clamp(-30.0, 30.0)
                p = torch.sigmoid(lf)
                ent = -(p * (p + 1e-8).log() + (1.0 - p) * (1.0 - p + 1e-8).log()).mean()
                # Max entropy per Bernoulli is ln 2; gap>0 only when boundaries become too decisive (collapse).
                h_max = loss.new_tensor(math.log(2.0))
                collapse_gap = torch.relu(h_max - ent)
                loss = loss + self.lambda_entropy * collapse_gap

            floor_pen = torch.zeros((), device=device, dtype=torch.float32)
            boundary_bce = torch.zeros((), device=device, dtype=torch.float32)
            interior_pen = torch.zeros((), device=device, dtype=torch.float32)
            boundary_ceiling_pen = torch.zeros((), device=device, dtype=torch.float32)
            pred_boundary_mean = torch.zeros((), device=device, dtype=torch.float32)
            gold_boundary_mean = torch.zeros((), device=device, dtype=torch.float32)
            if self.training and not segmenter_frozen and (
                self.lambda_boundary_floor > 0.0
                or self.lambda_boundary_bce > 0.0
                or self.lambda_interior > 0.0
                or self.lambda_boundary_ceiling > 0.0
            ):
                p = torch.sigmoid(logits.float())

                if self.lambda_boundary_floor > 0.0:
                    # Keep boundary density near the BPE reference for the current batch,
                    # instead of a fragile global constant. This blocks the "almost no cuts"
                    # collapse while still allowing the segmenter to deviate when LM loss helps.
                    min_allowed = torch.clamp(gold_boundary_mean - self.boundary_ceiling_slack, min=0.0)
                    floor_gap = torch.relu(min_allowed - pred_boundary_mean)
                    floor_pen = floor_gap ** 2
                    loss = loss + self.lambda_boundary_floor * floor_pen

                if self.lambda_boundary_bce > 0.0:
                    boundary_bce = (
                        F.binary_cross_entropy_with_logits(logits, gold, reduction="none") * mask
                    ).sum() / denom
                    loss = loss + self.lambda_boundary_bce * boundary_bce

                if self.lambda_interior > 0.0:
                    interior_pen = (p * interior * mask).sum() / denom
                    loss = loss + self.lambda_interior * interior_pen

                if self.lambda_boundary_ceiling > 0.0:
                    allowed_mean = gold_boundary_mean + self.boundary_ceiling_slack
                    excess = torch.relu(pred_boundary_mean - allowed_mean)
                    boundary_ceiling_pen = excess ** 2
                    loss = loss + self.lambda_boundary_ceiling * boundary_ceiling_pen
            elif self.training and not segmenter_frozen and self.lambda_boundary_floor > 0.0:
                bm = boundaries.float().mean()
                pred_boundary_mean = bm
                gap = torch.relu(self.min_soft_boundary_mean - bm)
                floor_pen = gap ** 2
                loss = loss + self.lambda_boundary_floor * floor_pen

            lm_outputs["boundary_floor_penalty"] = floor_pen
            lm_outputs["boundary_bce"] = boundary_bce
            lm_outputs["interior_penalty"] = interior_pen
            lm_outputs["boundary_ceiling_penalty"] = boundary_ceiling_pen
            lm_outputs["pred_boundary_mean"] = pred_boundary_mean
            lm_outputs["gold_boundary_mean"] = gold_boundary_mean
            lm_outputs["lm_loss_weight"] = torch.tensor(lm_weight, device=device, dtype=torch.float32)
            lm_outputs["train_phase"] = "freeze" if segmenter_frozen else phase_name

            oov_seg_pen = torch.zeros((), device=device, dtype=torch.float32)
            if self.training and not segmenter_frozen and self.lambda_oov_segment > 0.0:
                from src.models.neural_segmenter_supervised import _oov_segment_penalty_batch

                vset = self._vocab_segment_bytes_cached()
                if vset:
                    mask = (input_ids != self.pad_id).to(logits.dtype)
                    oov_seg_pen = _oov_segment_penalty_batch(
                        logits, input_ids, int(self.pad_id), vset, mask
                    )
                    loss = loss + self.lambda_oov_segment * oov_seg_pen
            lm_outputs["oov_segment_penalty"] = oov_seg_pen

            lm_outputs["loss"] = loss

        return lm_outputs
