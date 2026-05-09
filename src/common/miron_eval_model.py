"""Construct MironForCausalLM from merged experiment config (same as the training entrypoint)."""

from __future__ import annotations

from typing import Any, Dict

from src.common.tokenizer_utils import get_pad_id
from src.models.smollm2 import SmolLM2Config, SmolLM2Model


def _register_smollm2_with_hf() -> None:
    from transformers import AutoConfig, AutoModel

    if not hasattr(SmolLM2Model, "_from_config"):

        @classmethod
        def _from_config(cls, config, **kwargs):
            return cls(config, **kwargs)

        SmolLM2Model._from_config = _from_config  # type: ignore[attr-defined]

    AutoConfig.register("smollm2", SmolLM2Config)
    AutoModel.register(SmolLM2Config, SmolLM2Model)


def build_miron_causal_lm(tokenizer: Any, full_config: Dict[str, Any]):
    from src.miron_lib import MironConfig, MironForCausalLM

    _register_smollm2_with_hf()

    m = full_config.get("miron", {})
    model_section = full_config.get("model") or {}
    base_cfg = dict(model_section)

    _cti = tokenizer.char_to_id
    _pad = get_pad_id(tokenizer, 0)
    _eow = _cti.get(getattr(tokenizer, "eow_token", "<eow>"))
    if _eow is None:
        _eow = _cti.get("<eow>")
    if _eow is None:
        raise KeyError("eow token not in char_to_id (expected tokenizer.eow_token or '<eow>')")
    _bow = _cti.get(getattr(tokenizer, "bos_token", "[CLS]"))
    if _bow is None:
        _bow = _cti.get("<bow>")
    if _bow is None:
        _bow = _pad

    miron_config = MironConfig(
        base_model_config=base_cfg,
        base_model_type="smollm2",
        vocab_size=tokenizer.vocab_size,
        pad_token_id=int(_pad),
        eow_token_id=int(_eow),
        bow_token_id=int(_bow),
        max_word_length=tokenizer.max_word_length if hasattr(tokenizer, "max_word_length") else 32,
        encoder_char_dim=m.get("encoder_char_dim") or m.get("d_char_emb", 128),
        decoder_char_dim=m.get("decoder_char_dim") or m.get("d_char_emb", 128),
        encoder_nhead=m.get("nhead", 8),
        encoder_num_layers=m.get("nlayers", 2),
        encoder_d_ff=m.get("d_ff", 1024),
        encoder_dropout=m.get("encoder_dropout", 0.1),
        encoder_max_pos=m.get("encoder_max_pos", 64),
        encoder_chunk_size=m.get("encoder_chunk_size", 2048),
        decoder_chunk_size=m.get("decoder_chunk_size", 2048),
        encoder_pooling=m.get("encoder_pooling", "flatten"),
        decoder_conditioning=m.get("decoder_conditioning", "flatten"),
        init_gain=m.get("init_gain", 1.1),
    )
    smol_config = SmolLM2Config(
        hidden_size=miron_config.d_model,
        num_layers=miron_config.core_num_layers,
        num_heads=miron_config.core_num_heads,
        intermediate_size=miron_config.core_intermediate_size,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=miron_config.max_position_embeddings,
        rope_theta=miron_config.rope_theta,
        pad_token_id=miron_config.pad_token_id,
    )
    lm_model = SmolLM2Model(smol_config)
    return MironForCausalLM(miron_config, lm_model=lm_model)
