"""Final MIRON architecture: flattened char states -> CoreLM -> flattened char context."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from transformers import PreTrainedModel

from ..core.configuration import MironConfig
from ..layers.reference_char_codec import ReferenceCharEncoder, ReferenceCharDecoder


class MironPreTrainedModel(PreTrainedModel):
    config_class = MironConfig
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            gain = getattr(self.config, "init_gain", 1.1)
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.zero_()


class MironForCausalLM(MironPreTrainedModel):
    """
    CharEncoder (reference) -> enc_proj -> SmolLM2 -> dec_proj -> CharDecoder (reference).
    Training: next-word LM with shifted windows; loss = CE on target word chars (miron/miron/model/miron.py).
    """

    def __init__(self, config: MironConfig, lm_model: Module):
        super().__init__(config)
        if lm_model is None:
            raise ValueError("lm_model is required (pass SmolLM2Model)")
        self.pad_id = config.pad_token_id
        self.eow_id = config.eow_token_id
        self.bow_id = int(config.bow_token_id)
        self.d_model = config.d_model
        self.word_slot_len = config.max_word_length
        self.pos_cap = max(int(config.encoder_max_pos), 2 * int(config.max_word_length) + 1)

        d_char_enc = int(config.encoder_char_dim)
        d_char_dec = int(config.decoder_char_dim)
        d_core = self.d_model
        self.encoder_pooling = str(getattr(config, "encoder_pooling", "flatten") or "flatten")
        self.decoder_conditioning = str(getattr(config, "decoder_conditioning", "flatten") or "flatten")
        if self.encoder_pooling != "flatten":
            raise ValueError("Final MIRON supports only encoder_pooling='flatten'")
        if self.decoder_conditioning != "flatten":
            raise ValueError("Final MIRON supports only decoder_conditioning='flatten'")
        self.flatten_slot_len = self.word_slot_len + 1
        nhead_enc = config.encoder_nhead
        n_layers_enc = config.encoder_num_layers
        d_ff_enc = config.encoder_d_ff
        drop_enc = config.encoder_dropout

        nhead_dec = nhead_enc
        n_layers_dec = n_layers_enc
        d_ff_dec = d_ff_enc

        self.char_encoder = ReferenceCharEncoder(
            vocab_size=config.vocab_size,
            d_model=d_char_enc,
            n_heads=nhead_enc,
            n_layers=n_layers_enc,
            d_ff=d_ff_enc,
            dropout=drop_enc,
            word_slot_len=self.word_slot_len,
            pad_id=self.pad_id,
            bow_id=self.bow_id,
            max_sequence_positions=self.pos_cap,
        )
        self.encoder = self.char_encoder

        self.enc_proj = nn.Linear(self.flatten_slot_len * d_char_enc, d_core, bias=False)

        self.lm = lm_model

        self.dec_proj = nn.Linear(d_core, self.flatten_slot_len * d_char_dec, bias=False)

        self.char_decoder = ReferenceCharDecoder(
            vocab_size=config.vocab_size,
            d_model=d_char_dec,
            n_heads=nhead_dec,
            n_layers=n_layers_dec,
            d_ff=d_ff_dec,
            dropout=drop_enc,
            word_slot_len=self.word_slot_len,
            pad_id=self.pad_id,
            max_sequence_positions=self.pos_cap,
        )
        self.decoder = self.char_decoder

        for name, m in self.named_modules():
            if name.startswith("char_encoder.") or name.startswith("char_decoder."):
                continue
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.1)

        self.post_init()

    def prepare_for_inference(self) -> None:
        return

    def encode_batch(self, char_ids: torch.Tensor) -> torch.Tensor:
        """char_ids (N, word_slot_len) -> (N, d_core) latent."""
        z_seq = self.char_encoder.forward_sequence(char_ids)
        z_c = self._flatten_char_states(z_seq)
        if z_c.dim() == 1:
            z_c = z_c.unsqueeze(0)
        return self.enc_proj(z_c)

    def _encode_words(self, word_ids: torch.Tensor) -> torch.Tensor:
        z_seq = self.char_encoder.forward_sequence(word_ids)
        z_c = self._flatten_char_states(z_seq)
        return self.enc_proj(z_c)

    def _flatten_char_states(self, z_seq: torch.Tensor) -> torch.Tensor:
        slot_len = z_seq.size(-2)
        if slot_len < self.flatten_slot_len:
            pad_shape = (*z_seq.shape[:-2], self.flatten_slot_len - slot_len, z_seq.size(-1))
            z_seq = torch.cat([z_seq, z_seq.new_zeros(pad_shape)], dim=-2)
        elif slot_len > self.flatten_slot_len:
            z_seq = z_seq[..., : self.flatten_slot_len, :]
        return z_seq.reshape(*z_seq.shape[:-2], self.flatten_slot_len * z_seq.size(-1))

    def _decode_context(self, h: torch.Tensor) -> torch.Tensor:
        h_char = self.dec_proj(h)
        return h_char.reshape(*h_char.shape[:-1], self.flatten_slot_len, self.config.decoder_char_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del labels
        if input_ids.dim() != 3:
            raise ValueError("MironForCausalLM expects input_ids (B, T, word_slot_len)")

        if input_ids.size(1) < 2:
            z0 = next(self.parameters()).sum() * 0.0
            return {
                "loss": z0,
                "lm_char_loss": z0,
                "lm_char_acc": torch.tensor(0.0, device=input_ids.device),
            }

        enc_in = input_ids[:, :-1, :].contiguous()
        target_words = input_ids[:, 1:, :].contiguous()
        B, Tm, L = enc_in.shape

        z = self._encode_words(enc_in)
        lm_out = self.lm(inputs_embeds=z, use_cache=False)
        h = lm_out["hidden_states"]
        h_char = self._decode_context(h)

        bow = torch.full(
            (B, Tm, 1), self.bow_id, device=input_ids.device, dtype=input_ids.dtype
        )
        full_target = torch.cat([bow, target_words], dim=-1)
        dec_input = full_target[:, :, :-1]
        dec_target = full_target[:, :, 1:]

        logits = self.char_decoder(h_char, dec_input)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            dec_target.reshape(-1),
            ignore_index=self.pad_id,
        )

        with torch.no_grad():
            flat_t = dec_target.reshape(-1)
            flat_p = logits.reshape(-1, logits.size(-1)).argmax(-1)
            valid = flat_t != self.pad_id
            acc = (
                (flat_p[valid] == flat_t[valid]).float().mean()
                if valid.any()
                else loss.new_tensor(0.0)
            )

        return {
            "loss": loss,
            "lm_char_loss": loss,
            "lm_char_acc": acc,
        }

    def decode_word(
        self,
        z: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        eow_penalty: float = 0.0,
        return_logits: bool = False,
    ):
        """Decode one word per row of z (B, d_core) or a single (d_core,) vector."""
        del max_len
        self.eval()
        device = z.device
        L = self.flatten_slot_len
        z_core = z.unsqueeze(0) if z.dim() == 1 else z
        h_char = self._decode_context(z_core).unsqueeze(1)
        B = h_char.size(0)
        rows: List[torch.Tensor] = []
        bat_logits: List[torch.Tensor] = []
        for b in range(B):
            hb = h_char[b : b + 1]
            slot_ids: List[int] = [self.bow_id]
            dec_in = torch.full((1, 1, L), self.pad_id, dtype=torch.long, device=device)
            dec_in[0, 0, 0] = self.bow_id
            current_len = 1
            log_chunks: List[torch.Tensor] = []
            for pos in range(L):
                logits = self.char_decoder(hb, dec_in)
                next_logits = logits[0, 0, pos, :].clone()
                if eow_penalty and pos < 5:
                    next_logits[self.eow_id] -= float(eow_penalty)
                if temperature is not None and float(temperature) > 0:
                    next_logits = next_logits / max(float(temperature), 1e-8)
                    if top_k is not None:
                        top_vals, _ = torch.topk(next_logits, min(int(top_k), next_logits.numel()))
                        next_logits = next_logits.masked_fill(
                            next_logits < top_vals[-1], float("-inf")
                        )
                    probs = F.softmax(next_logits, dim=-1)
                    next_id = int(torch.multinomial(probs, 1).item())
                else:
                    next_id = int(next_logits.argmax(dim=-1).item())
                if next_id == self.eow_id:
                    slot_ids.append(next_id)
                    if return_logits:
                        log_chunks.append(logits[0, 0 : pos + 1])
                    break
                slot_ids.append(next_id)
                if current_len < L:
                    dec_in[0, 0, current_len] = next_id
                    current_len += 1
                if return_logits:
                    log_chunks.append(logits[0, 0 : pos + 1])
            body = slot_ids[1:]
            while len(body) < L:
                body.append(self.pad_id)
            rows.append(torch.tensor(body[:L], dtype=torch.long, device=device))
            if return_logits and log_chunks:
                bat_logits.append(torch.cat(log_chunks, dim=0))
        out = torch.stack(rows, dim=0)
        if return_logits and bat_logits:
            return out, torch.stack(bat_logits, dim=0)
        return out, None

    @staticmethod
    def _slice_past_kv(
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
        batch_idx: int,
    ) -> Optional[Tuple[Tuple[torch.Tensor, ...], ...]]:
        """Take one batch row from stacked KV (SmolLM2: tuple of (k, v) per layer)."""
        if past_key_values is None:
            return None
        sliced: List[Tuple[torch.Tensor, ...]] = []
        for layer_kv in past_key_values:
            k, v = layer_kv[0], layer_kv[1]
            sliced.append((k[batch_idx : batch_idx + 1], v[batch_idx : batch_idx + 1]))
        return tuple(sliced)

    @staticmethod
    def _index_past_rows(
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
        row_indices: torch.Tensor,
    ) -> Optional[Tuple[Tuple[torch.Tensor, ...], ...]]:
        if past_key_values is None:
            return None
        idx = row_indices.long()
        out: List[Tuple[torch.Tensor, ...]] = []
        for k, v in past_key_values:
            out.append((k.index_select(0, idx), v.index_select(0, idx)))
        return tuple(out)

    def _decode_one_word_slot(
        self,
        h_last: torch.Tensor,
        word_len: int,
        temperature: float,
        do_sample: bool,
        top_k: Optional[int],
        eos_token_id,
        generator: Optional[torch.Generator] = None,
        initial_char_temperature: Optional[float] = None,
        initial_char_count: int = 0,
    ) -> Tuple[torch.Tensor, bool]:
        """
        h_last: (1, 1, context_len, d_dec). Returns (slot, eos_stops_sequence).
        """
        L = word_len
        dec_buf = torch.full(
            (1, 1, L), self.pad_id, dtype=torch.long, device=h_last.device
        )
        dec_buf[0, 0, 0] = self.bow_id
        slot_ids: List[int] = [self.bow_id]
        current_len = 1
        for pos in range(word_len):
            logits = self.char_decoder(h_last, dec_buf)
            next_logits = logits[0, 0, pos, :]
            step_temperature = temperature
            step_do_sample = do_sample
            if (
                initial_char_temperature is not None
                and int(initial_char_count) > 0
                and pos < int(initial_char_count)
            ):
                step_temperature = float(initial_char_temperature)
                step_do_sample = step_temperature > 0
            if step_do_sample and step_temperature > 0:
                next_logits = next_logits / max(step_temperature, 1e-8)
                if top_k is not None:
                    top_vals, _ = torch.topk(
                        next_logits, min(top_k, next_logits.numel())
                    )
                    next_logits = next_logits.masked_fill(
                        next_logits < top_vals[-1], float("-inf")
                    )
                probs = F.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1, generator=generator).item())
            else:
                next_id = int(next_logits.argmax(dim=-1).item())
            if next_id == self.eow_id or (
                eos_token_id is not None and next_id == eos_token_id
            ):
                slot_ids.append(next_id)
                break
            slot_ids.append(next_id)
            if current_len < L:
                dec_buf[0, 0, current_len] = next_id
                current_len += 1
        body = slot_ids[1:]
        while len(body) < word_len:
            body.append(self.pad_id)
        slot_tensor = torch.tensor(
            body[:word_len], dtype=torch.long, device=h_last.device
        )
        eos_break = eos_token_id is not None and len(slot_ids) > 1 and slot_ids[1] == eos_token_id
        return slot_tensor, eos_break

    def _generate_multi_row_batched_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_prefill: torch.Tensor,
        past_prefill: Tuple[Tuple[torch.Tensor, ...], ...],
        h_last_prefill: torch.Tensor,
        actual_max_new_tokens: int,
        word_len: int,
        temperature: float,
        do_sample: bool,
        top_k: Optional[int],
        eos_token_id,
        row_generators: Optional[List[Optional[torch.Generator]]] = None,
        first_word_temperature: Optional[float] = None,
        initial_char_temperature: Optional[float] = None,
        initial_char_count: int = 0,
    ) -> torch.Tensor:
        """
        B>1: one batched LM forward per new word for all active rows (parallel), char decode row order 0..n-1.
        Stops early per row on eos. Sampling order matches per-word interleaving (not legacy row-major full rows).
        """
        _ = position_ids_prefill
        device = input_ids.device
        b_orig = input_ids.size(0)
        active = torch.arange(b_orig, device=device, dtype=torch.long)
        context = input_ids
        mask = attention_mask
        past_kv = past_prefill
        results: List[Optional[torch.Tensor]] = [None] * b_orig

        with torch.no_grad():
            for wstep in range(actual_max_new_tokens):
                n = active.numel()
                if n == 0:
                    break
                if wstep == 0:
                    h_dec = h_last_prefill.index_select(0, active)
                else:
                    z = self._encode_words(context[:, -1:])
                    new_m = torch.ones((n, 1), dtype=torch.long, device=device)
                    mask = torch.cat([mask, new_m], dim=1)
                    # With right-padded heterogeneous prompts, KV cache width is not the
                    # true per-row sequence length. Pass explicit per-row positions so the
                    # next generated word continues after the last real token, not after
                    # the padded width of the batch.
                    step_position_ids = (mask.sum(dim=1, keepdim=True) - 1).clamp(min=0).long()
                    lm_out = self.lm(
                        inputs_embeds=z,
                        attention_mask=mask,
                        use_cache=True,
                        past_key_values=past_kv,
                        position_ids=step_position_ids,
                    )
                    if isinstance(lm_out, dict):
                        h = lm_out.get("hidden_states", lm_out.get("last_hidden_state"))
                        past_kv = lm_out.get("past_key_values")
                    else:
                        h = getattr(lm_out, "last_hidden_state", lm_out[0])
                        past_kv = getattr(lm_out, "past_key_values", None)
                    h_dec = self._decode_context(h[:, -1:, :])

                new_rows: List[torch.Tensor] = []
                eos_flags: List[bool] = []
                for i in range(n):
                    step_temperature = temperature
                    step_do_sample = do_sample
                    if wstep == 0 and first_word_temperature is not None:
                        step_temperature = float(first_word_temperature)
                        step_do_sample = step_temperature > 0
                    slot_1d, eos_br = self._decode_one_word_slot(
                        h_dec[i : i + 1],
                        word_len,
                        step_temperature,
                        step_do_sample,
                        top_k,
                        eos_token_id,
                        generator=(
                            row_generators[int(active[i].item())]
                            if row_generators is not None
                            else None
                        ),
                        initial_char_temperature=initial_char_temperature,
                        initial_char_count=initial_char_count,
                    )
                    new_rows.append(slot_1d)
                    eos_flags.append(eos_br)

                new_slots = torch.stack(new_rows, dim=0).unsqueeze(1)
                context = torch.cat([context, new_slots], dim=1)

                stop = torch.tensor(eos_flags, device=device, dtype=torch.bool)
                for i in range(n):
                    if stop[i]:
                        orig = int(active[i].item())
                        results[orig] = context[i].clone()

                keep = ~stop
                if not keep.any():
                    break
                keep_idx = torch.where(keep)[0]
                context = context.index_select(0, keep_idx)
                mask = mask.index_select(0, keep_idx)
                active = active.index_select(0, keep_idx)
                past_kv = self._index_past_rows(past_kv, keep_idx)

            for i in range(active.numel()):
                orig = int(active[i].item())
                if results[orig] is None:
                    results[orig] = context[i].clone()

        max_t = max(int(results[i].size(0)) for i in range(b_orig))
        W = word_len
        out = input_ids.new_full((b_orig, max_t, W), self.pad_id)
        for i in range(b_orig):
            r = results[i]
            assert r is not None
            out[i, : r.size(0), :] = r
        return out

    def _generate_single_row(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_len: int,
        actual_max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        do_sample: bool,
        top_p: float,
        top_k: Optional[int],
        eos_token_id,
        position_ids_prefill: torch.Tensor,
        prefill_past: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        h_last_prefill: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        first_word_temperature: Optional[float] = None,
        initial_char_temperature: Optional[float] = None,
        initial_char_count: int = 0,
    ) -> torch.Tensor:
        """Autoregressive word generation for batch size 1; optional batched-prefill continuation."""
        _ = repetition_penalty, top_p
        batch_size, seq_len, _ = input_ids.size()
        context = input_ids
        total_ids = input_ids
        past_key_values = prefill_past
        skip_first_lm = prefill_past is not None

        with torch.no_grad():
            for wstep in range(actual_max_new_tokens):
                if skip_first_lm:
                    h_last = h_last_prefill
                    skip_first_lm = False
                elif past_key_values is not None:
                    z = self._encode_words(context[:, -1:])
                    new_mask = torch.ones(
                        (batch_size, 1), dtype=torch.long, device=input_ids.device
                    )
                    attention_mask = torch.cat([attention_mask, new_mask], dim=1)
                    step_position_ids = (attention_mask.sum(dim=1, keepdim=True) - 1).clamp(
                        min=0
                    ).long()
                    lm_out = self.lm(
                        inputs_embeds=z,
                        attention_mask=attention_mask,
                        use_cache=True,
                        past_key_values=past_key_values,
                        position_ids=step_position_ids,
                    )
                    if isinstance(lm_out, dict):
                        h = lm_out.get("hidden_states", lm_out.get("last_hidden_state"))
                        past_key_values = lm_out.get("past_key_values")
                    else:
                        h = getattr(lm_out, "last_hidden_state", lm_out[0])
                        past_key_values = getattr(lm_out, "past_key_values", None)
                    h_last = self._decode_context(h[:, -1:, :])
                else:
                    z = self._encode_words(context)
                    lm_out = self.lm(
                        inputs_embeds=z,
                        attention_mask=attention_mask,
                        use_cache=True,
                        position_ids=position_ids_prefill,
                    )
                    if isinstance(lm_out, dict):
                        h = lm_out.get("hidden_states", lm_out.get("last_hidden_state"))
                        past_key_values = lm_out.get("past_key_values")
                    else:
                        h = getattr(lm_out, "last_hidden_state", lm_out[0])
                        past_key_values = getattr(lm_out, "past_key_values", None)
                    h_last = self._decode_context(h[:, -1:, :])

                L = word_len
                dec_buf = torch.full(
                    (1, 1, L), self.pad_id, dtype=torch.long, device=context.device
                )
                dec_buf[0, 0, 0] = self.bow_id
                slot_ids: List[int] = [self.bow_id]
                current_len = 1

                for pos in range(word_len):
                    logits = self.char_decoder(h_last, dec_buf)
                    next_logits = logits[0, 0, pos, :]

                    step_temperature = temperature
                    step_do_sample = do_sample
                    if wstep == 0 and first_word_temperature is not None:
                        step_temperature = float(first_word_temperature)
                        step_do_sample = step_temperature > 0
                    if (
                        initial_char_temperature is not None
                        and int(initial_char_count) > 0
                        and pos < int(initial_char_count)
                    ):
                        step_temperature = float(initial_char_temperature)
                        step_do_sample = step_temperature > 0

                    if step_do_sample and step_temperature > 0:
                        next_logits = next_logits / max(step_temperature, 1e-8)
                        if top_k is not None:
                            top_vals, _ = torch.topk(
                                next_logits, min(top_k, next_logits.numel())
                            )
                            next_logits = next_logits.masked_fill(
                                next_logits < top_vals[-1], float("-inf")
                            )
                        probs = F.softmax(next_logits, dim=-1)
                        next_id = int(torch.multinomial(probs, 1, generator=generator).item())
                    else:
                        next_id = int(next_logits.argmax(dim=-1).item())

                    if next_id == self.eow_id or (
                        eos_token_id is not None and next_id == eos_token_id
                    ):
                        slot_ids.append(next_id)
                        break
                    slot_ids.append(next_id)
                    if current_len < L:
                        dec_buf[0, 0, current_len] = next_id
                        current_len += 1

                body = slot_ids[1:]
                while len(body) < word_len:
                    body.append(self.pad_id)
                slot_tensor = torch.tensor(
                    body[:word_len], dtype=torch.long, device=context.device
                )

                new_slot = slot_tensor.unsqueeze(0).unsqueeze(0)
                total_ids = torch.cat([total_ids, new_slot], dim=1)
                context = torch.cat([context, new_slot], dim=1)

                if eos_token_id is not None and len(slot_ids) > 1 and slot_ids[1] == eos_token_id:
                    break

        return total_ids

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 15,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        top_p: float = 0.9,
        top_k: Optional[int] = 50,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("prefill_left_padded", None)
        eos_token_id = kwargs.get("eos_token_id", None)
        row_seeds = kwargs.pop("row_seeds", None)
        first_word_temperature = kwargs.pop("first_word_temperature", None)
        initial_char_temperature = kwargs.pop("initial_char_temperature", None)
        initial_char_count = int(kwargs.pop("initial_char_count", 0) or 0)
        max_pos = getattr(self.config, "max_position_embeddings", 256)

        self.eval()
        batch_size, seq_len, word_len = input_ids.size()

        allowed_new_tokens = max(0, max_pos - seq_len)
        actual_max_new_tokens = min(max_new_tokens, allowed_new_tokens)
        if actual_max_new_tokens <= 0:
            return input_ids

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.long, device=input_ids.device
            )

        am0 = attention_mask.long()
        position_ids_prefill = ((am0.cumsum(dim=1) - 1) * am0).clamp(min=0)

        with torch.no_grad():
            z = self._encode_words(input_ids)
            lm_out = self.lm(
                inputs_embeds=z,
                attention_mask=attention_mask,
                use_cache=True,
                position_ids=position_ids_prefill,
            )
            if isinstance(lm_out, dict):
                h = lm_out.get("hidden_states", lm_out.get("last_hidden_state"))
                past_full = lm_out.get("past_key_values")
            else:
                h = getattr(lm_out, "last_hidden_state", lm_out[0])
                past_full = getattr(lm_out, "past_key_values", None)
            real_lens = attention_mask.sum(dim=1).clamp(min=1).long()
            row_idx = torch.arange(batch_size, device=input_ids.device)
            last_h = h[row_idx, (real_lens - 1), :].unsqueeze(1)
            h_last_all = self._decode_context(last_h)

        row_generators: Optional[List[Optional[torch.Generator]]] = None
        if row_seeds is not None:
            if len(row_seeds) != batch_size:
                raise ValueError(
                    f"row_seeds length {len(row_seeds)} does not match batch_size {batch_size}"
                )
            row_generators = []
            for seed in row_seeds:
                gen = torch.Generator(device=input_ids.device)
                gen.manual_seed(int(seed))
                row_generators.append(gen)

        if batch_size > 1:
            return self._generate_multi_row_batched_lm(
                input_ids,
                attention_mask,
                position_ids_prefill,
                past_full,
                h_last_all,
                actual_max_new_tokens,
                word_len,
                temperature,
                do_sample,
                top_k,
                eos_token_id,
                row_generators=row_generators,
                first_word_temperature=first_word_temperature,
                initial_char_temperature=initial_char_temperature,
                initial_char_count=initial_char_count,
            )

        return self._generate_single_row(
            input_ids,
            attention_mask,
            word_len,
            actual_max_new_tokens,
            temperature,
            repetition_penalty,
            do_sample,
            top_p,
            top_k,
            eos_token_id,
            position_ids_prefill,
            prefill_past=past_full,
            h_last_prefill=h_last_all[:1],
            generator=(row_generators[0] if row_generators is not None else None),
            first_word_temperature=first_word_temperature,
            initial_char_temperature=initial_char_temperature,
            initial_char_count=initial_char_count,
        )
