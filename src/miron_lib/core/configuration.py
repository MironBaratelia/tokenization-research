"""MIRON HuggingFace-style config."""
import copy
from typing import Dict, Any, Optional
from transformers import PretrainedConfig, AutoConfig


class MironConfig(PretrainedConfig):
    model_type = "miron"

    def __init__(
        self,
        base_model_config: Optional[Dict[str, Any]] = None,
        base_model_type: str = "smollm2",
        vocab_size: int = 5000,
        d_model: int = 768,
        pad_token_id: int = 0,
        eow_token_id: int = 1,
        bow_token_id: int = 0,
        max_word_length: int = 32,
        rope_theta: float = 10000.0,
        # Character codec. The final MIRON architecture always projects the full
        # word slot: (max_word_length + 1) * char_dim <-> CoreLM hidden size.
        encoder_char_dim: int = 128,
        decoder_char_dim: int = 128,
        encoder_chunk_size: int = 2048,
        decoder_chunk_size: int = 2048,
        encoder_pooling: str = "flatten",
        decoder_conditioning: str = "flatten",
        encoder_nhead: int = 8,
        encoder_num_layers: int = 2,
        encoder_d_ff: int = 1024,
        encoder_dropout: float = 0.1,
        encoder_max_pos: int = 64,
        # Core LM (injected externally)
        core_num_layers: int = 12,
        core_num_heads: int = 12,
        core_intermediate_size: int = 3072,
        max_position_embeddings: int = 128,
        init_gain: float = 1.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_config = base_model_config or {}
        self.base_model_type = base_model_type
        self.vocab_size = vocab_size
        bc = self.base_model_config
        self.d_model = bc.get("hidden_size", d_model) if bc else d_model
        self.pad_token_id = pad_token_id
        self.eow_token_id = eow_token_id
        self.bow_token_id = bow_token_id
        self.max_word_length = max_word_length
        self.rope_theta = rope_theta

        if encoder_pooling != "flatten":
            raise ValueError("Final MIRON supports only encoder_pooling='flatten'")
        if decoder_conditioning != "flatten":
            raise ValueError("Final MIRON supports only decoder_conditioning='flatten'")

        self.encoder_char_dim = encoder_char_dim
        self.decoder_char_dim = decoder_char_dim
        self.encoder_chunk_size = encoder_chunk_size
        self.decoder_chunk_size = decoder_chunk_size
        self.encoder_pooling = "flatten"
        self.decoder_conditioning = "flatten"
        self.encoder_nhead = encoder_nhead
        self.encoder_num_layers = encoder_num_layers
        self.encoder_d_ff = encoder_d_ff
        self.encoder_dropout = encoder_dropout
        self.encoder_max_pos = encoder_max_pos
        self.core_num_layers = bc.get("num_layers", core_num_layers) if bc else core_num_layers
        self.core_num_heads = bc.get("num_heads", core_num_heads) if bc else core_num_heads
        self.core_intermediate_size = bc.get("intermediate_size", core_intermediate_size) if bc else core_intermediate_size
        self.max_position_embeddings = bc.get("max_position_embeddings", max_position_embeddings) if bc else max_position_embeddings

        self.init_gain = init_gain

    @property
    def core_config(self):
        cfg_dict = copy.deepcopy(self.base_model_config)
        cfg_dict.pop("model_type", None)
        return AutoConfig.for_model(self.base_model_type, **cfg_dict)
