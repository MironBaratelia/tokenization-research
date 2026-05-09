import torch

from src.reporting.checkpoint_param_analysis import (
    analyze_checkpoint_state_dict,
    load_checkpoint_param_stats,
)


def test_tied_embeddings_deduped_total():
    v, h = 100, 32
    emb = torch.nn.Parameter(torch.zeros(v, h))
    sd = {
        "embed_tokens.weight": emb,
        "lm_head.weight": emb,
        "layers.0.weight": torch.randn(16, 16),
    }
    info = analyze_checkpoint_state_dict(sd)
    assert info is not None
    assert info["vocab_size"] == v
    assert info["hidden_size"] == h
    assert info["tied_word_embeddings"] is True
    assert info["embed_head_params_unique"] == v * h


def test_untied_embeddings_count_both():
    v, h = 50, 16
    sd = {
        "embed_tokens.weight": torch.randn(v, h),
        "lm_head.weight": torch.randn(v, h),
    }
    info = analyze_checkpoint_state_dict(sd)
    assert info is not None
    assert info["tied_word_embeddings"] is False
    assert info["embed_head_params_unique"] == 2 * v * h


def test_load_checkpoint_param_stats_wrong_name():
    assert load_checkpoint_param_stats("/nonexistent/file.pt") is None

