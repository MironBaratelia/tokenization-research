import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass

LOGIT_SUPPRESS = -1e9  # logit value for token suppression in generation

@dataclass
class SmolLM2Config:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "silu"
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    pad_token_id: Optional[int] = None
    label_smoothing: float = 0.0

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len, :].to(dtype=x.dtype),
            self.sin_cached[:seq_len, :].to(dtype=x.dtype),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1).to(q.dtype)
    sin = sin[position_ids].unsqueeze(1).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SmolLM2Attention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        if attention_mask is not None:
            if q_len > 1:
                causal_mask = torch.triu(torch.ones(q_len, q_len, device=hidden_states.device), diagonal=1).bool()
                combined_mask = attention_mask & (~causal_mask).unsqueeze(0).unsqueeze(1)
                
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=combined_mask,
                    is_causal=False
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=attention_mask,
                    is_causal=False
                )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=(q_len > 1)
            )

        if not self.training:
            attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)
        
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value

class SmolLM2MLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class SmolLM2DecoderLayer(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.self_attn = SmolLM2Attention(config)
        self.mlp = SmolLM2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value, use_cache)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value

class SmolLM2Model(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([SmolLM2DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, use_cache: Optional[bool] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.training: use_cache = False
        elif use_cache is None: use_cache = self.config.use_cache
        
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            # Handle position_ids for KV-caching during generation
            # We offset by the length of the past keys in the first layer
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # PRE-PROCESS MASK TO BE SDPA COMPATIBLE WITHOUT GPU-CPU SYNC
        if attention_mask is not None and attention_mask.dim() == 2:
            # Expand to [bsz, 1, 1, seq_length] boolean mask.
            # SDPA is heavily optimized for boolean masks and is_causal=True.
            attention_mask = attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)

        hidden_states = inputs_embeds
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                # Add requires_grad to hidden_states to ensure gradient flows to embedding
                # Only if it's the first layer and it's a float tensor (which it should be)
                if i == 0 and hidden_states.is_floating_point() and not hidden_states.requires_grad:
                    hidden_states.requires_grad_(True)
                
                # Checkpoint only the module itself
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states, attention_mask, position_ids, None, False,
                    preserve_rng_state=True,
                    use_reentrant=False # use non-reentrant for better performance and compatibility
                )
            else:
                past_key_value = past_key_values[i] if past_key_values is not None else None
                hidden_states, present_key_value = layer(hidden_states, attention_mask, position_ids, past_key_value, use_cache)
                if use_cache: new_past_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if seq_length < 2:
                # Keep a differentiable scalar tied to logits (plain * 0.0 can break the graph / AMP).
                loss = logits.float().square().mean() * 1e-4
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                flat_logits = shift_logits.view(-1, self.config.vocab_size)
                flat_labels = shift_labels.view(-1)
                valid = flat_labels.ne(-100)
                if valid.any():
                    loss = nn.functional.cross_entropy(
                        flat_logits[valid],
                        flat_labels[valid],
                        reduction="mean",
                        label_smoothing=float(getattr(self.config, "label_smoothing", 0.0) or 0.0),
                    )
                else:
                    # No supervised positions: tiny L2 on logits so training does not "freeze" at zero CE.
                    loss = logits.float().square().mean() * 1e-4

        return {
            "loss": loss, "logits": logits, "hidden_states": hidden_states,
            "past_key_values": tuple(new_past_key_values) if use_cache else None
        }

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        self.eval()
        batch_size = input_ids.size(0)
        past_key_values = None
        total_ids = input_ids
        eos_token_id = kwargs.get("eos_token_id") or getattr(self.config, "eos_token_id", None)
        pad_token_id = kwargs.get("pad_token_id") or getattr(self.config, "pad_token_id", 0)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        prefill_left_padded = bool(kwargs.pop("prefill_left_padded", False))

        if attention_mask is not None:
            real_lens = attention_mask.sum(dim=1)
        else:
            real_lens = None

        position_ids_prefill: Optional[torch.Tensor] = None
        if prefill_left_padded and attention_mask is not None:
            am = attention_mask.long()
            position_ids_prefill = ((am.cumsum(dim=1) - 1) * am).clamp(min=0)

        prefill_width = input_ids.size(1)
        if attention_mask is not None:
            base_kv_mask = attention_mask.to(torch.bool)
        else:
            base_kv_mask = torch.ones(
                batch_size, prefill_width, dtype=torch.bool, device=input_ids.device
            )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        with torch.no_grad():
            for step in range(max_new_tokens):
                if past_key_values is not None:
                    # Mask past length +1 so SDPA does not attend to pad keys in batched decode.
                    kv_attn = torch.cat(
                        [
                            base_kv_mask,
                            torch.ones(
                                batch_size,
                                step,
                                dtype=torch.bool,
                                device=input_ids.device,
                            ),
                        ],
                        dim=1,
                    )
                    fwd_kw: Dict[str, Any] = {
                        "input_ids": input_ids[:, -1:],
                        "past_key_values": past_key_values,
                        "use_cache": True,
                        "attention_mask": kv_attn,
                    }
                    if prefill_left_padded and real_lens is not None:
                        fwd_kw["position_ids"] = (real_lens.unsqueeze(1) + step - 1).long()
                    outputs = self.forward(**fwd_kw)
                else:
                    outputs = self.forward(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        position_ids=position_ids_prefill,
                    )

                past_key_values = outputs["past_key_values"]
                logits = outputs["logits"]
                if step == 0 and prefill_left_padded:
                    next_token_logits = logits[:, -1, :].clone()
                elif real_lens is not None and step == 0:
                    idx = (real_lens - 1).clamp(min=0)
                    next_token_logits = logits[torch.arange(batch_size, device=logits.device), idx.long(), :]
                else:
                    next_token_logits = logits[:, -1, :].clone()

                if step == 0:
                    for tid in kwargs.get("bias_against_first_step") or []:
                        next_token_logits[:, tid] = LOGIT_SUPPRESS

                if repetition_penalty != 1.0 and total_ids.size(1) > 0:
                    # Per-token repetition penalty without Python loops over sequence length.
                    V = next_token_logits.size(-1)
                    r = next_token_logits.new_tensor(repetition_penalty)
                    for b in range(batch_size):
                        row = total_ids[b]
                        active = row != pad_token_id
                        if not active.any():
                            continue
                        idx = row[active].long().clamp(0, V - 1)
                        counts = torch.bincount(idx, minlength=V)
                        li = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1)
                        if li.numel() == 0:
                            continue
                        c = counts[li].to(dtype=next_token_logits.dtype)
                        vals = next_token_logits[b, li]
                        pos = vals > 0
                        upd = vals.clone()
                        upd[pos] = vals[pos] / torch.pow(r, c[pos])
                        upd[~pos] = vals[~pos] * torch.pow(r, c[~pos])
                        next_token_logits[b, li] = upd

                if kwargs.get("do_sample", False):
                    temp = kwargs.get("temperature", 1.0)
                    next_token_logits = next_token_logits / max(temp, 1e-5)
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if eos_token_id is not None:
                    eos_mask = (next_token.squeeze(-1) == eos_token_id) & ~finished
                    finished = finished | eos_mask
                    next_token = torch.where(
                        eos_mask.unsqueeze(-1),
                        torch.full_like(next_token, pad_token_id),
                        next_token,
                    )
                    if finished.all():
                        break

                total_ids = torch.cat([total_ids, next_token], dim=-1)
                input_ids = next_token
        return total_ids
