"""
Human-eval text generation for tokenizer.type == neural_segmenter.

encode() must yield merged vocab ids (training_stage not joint/adaptation); SmolLM2.generate
consumes those ids. Weights: checkpoint["model_state_dict"] keys lm.* are mandatory — without
them there is no trained language model (supervised_boundaries checkpoints are segmenter-only).
"""

from __future__ import annotations

import glob
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from src.common.generation_speed import (
    benchmark_tokenized_generation_speed,
    build_tokenized_prompt_batches,
)
from src.common.generation_from_config import (
    DEFAULT_BIAS_FIRST_TOKENS,
    base_generate_kwargs,
    merge_generation_layers,
)
from src.common.inference import InferenceEngine
from src.common.tokenizer_utils import get_pad_id
from src.models.smollm2 import SmolLM2Config, SmolLM2Model
from src.evaluation.repro_subsample import reproducible_subsample


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_checkpoint(ckpt_dir: str) -> Optional[str]:
    for name in ("last_checkpoint.pt", "best_model.pt"):
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            return path
    if not os.path.isdir(ckpt_dir):
        return None
    step_files = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not step_files:
        return None

    def step_num(p: str) -> int:
        stem = os.path.basename(p).replace("step_", "").replace(".pt", "")
        return int(stem) if stem.isdigit() else 0

    return max(step_files, key=step_num)


@contextmanager
def _merged_vocab_encode_mode(tokenizer: Any) -> Iterator[None]:
    prev = getattr(tokenizer, "training_stage", "joint")
    tokenizer.training_stage = "locked"
    yield
    tokenizer.training_stage = prev


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _load_smollm_and_segmenter_from_checkpoint(
    tokenizer: Any,
    config: Dict[str, Any],
    device: str,
    experiment_id: str,
) -> Tuple[SmolLM2Model, int]:
    """Load SmolLM2 from checkpoint lm.*; optionally segmenter.* onto tokenizer.segmenter_model."""
    root = _project_root()
    ckpt_dir = os.path.join(root, "outputs", experiment_id, "checkpoints")
    path = _find_checkpoint(ckpt_dir)
    mc = config.get("model") or {}
    tc = config.get("training") or {}
    max_pos = mc.get("max_position_embeddings")
    if max_pos is None:
        max_pos = tc.get("context_length", 512)
    pad_id = int(get_pad_id(tokenizer, 0))
    smcfg = SmolLM2Config(
        vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 0),
        hidden_size=int(mc.get("hidden_size", 768)),
        num_layers=int(mc.get("num_layers", 12)),
        num_heads=int(mc.get("num_heads", 12)),
        intermediate_size=int(mc.get("intermediate_size", 3072)),
        max_position_embeddings=int(max_pos),
        rope_theta=float(mc.get("rope_theta", 10000.0)),
        pad_token_id=pad_id,
    )
    lm = SmolLM2Model(smcfg).to(device)
    lm.eval()

    if not path or not os.path.isfile(path):
        return lm, 0

    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = ckpt["model_state_dict"]

    lm_sd: Dict[str, torch.Tensor] = {}
    seg_sd: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        nk = _strip_module_prefix(k)
        if nk.startswith("lm."):
            lm_sd[nk[3:]] = v
        elif nk.startswith("segmenter."):
            seg_sd[nk[len("segmenter.") :]] = v

    n_lm = 0
    if lm_sd:
        lm.load_state_dict(lm_sd, strict=False)
        n_lm = len(lm_sd)

    seg = getattr(tokenizer, "segmenter_model", None)
    if seg is not None and seg_sd:
        seg.load_state_dict(seg_sd, strict=False)
        seg.eval()

    return lm, n_lm


def run_neural_segmenter_human_eval(
    evaluator: "HumanEvalEvaluator",
    file_path: str,
    batch_size: int,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    tokenizer = evaluator.tokenizer
    exp_id = (evaluator.config.get("experiment") or {}).get("id")
    if not exp_id:
        raise ValueError("Neural human_eval: config.experiment.id is required to locate checkpoints.")

    with _merged_vocab_encode_mode(tokenizer):
        lm, n_lm = _load_smollm_and_segmenter_from_checkpoint(
            tokenizer, evaluator.config, evaluator.device, str(exp_id)
        )
        if n_lm == 0:
            raise RuntimeError(
                "human_eval (neural_segmenter): в checkpoint['model_state_dict'] нет ни одного ключа «lm.*». "
                "Режим supervised_boundaries обучает только сегментер; продолжение текста — это отдельная "
                "языковая модель (SmolLM внутри joint NeuralSegmenterLM). Сохрани чекпойнт после joint-этапа "
                "или не вызывай human_eval до появления lm.* в весах."
            )

        gen_cfg = merge_generation_layers(evaluator.config)
        engine = InferenceEngine(lm, tokenizer, device=evaluator.device)
        kw = base_generate_kwargs(evaluator.config, lm, engine.tokenizer, next_word=False)
        bias = gen_cfg.get("bias_against_first_tokens")
        if bias is None:
            bias = list(DEFAULT_BIAS_FIRST_TOKENS)
        ibs = gen_cfg.get("inference_batch_size")
        eff_bs = int(ibs) if ibs is not None else batch_size

        prompts = evaluator._load_prompts(file_path)
        prompts = reproducible_subsample(prompts, max_samples, evaluator.eval_subsample_seed())

        generated_list = engine.generate(
            prompts=prompts,
            batch_size=eff_bs,
            bias_against_first_tokens=bias,
            suppress_eos_first_step=True,
            desc="HumanEval(neural)",
            seed=evaluator.eval_rng_seed(),
            **kw,
        )

        generations = [{"prompt": p, "generated": g} for p, g in zip(prompts, generated_list)]
        results = {"generations": generations}

        bench = evaluator.config.get("benchmarks") or {}
        runs = max(3, int(bench.get("generation_speed_runs", 5)))
        warmup_runs = max(1, int(bench.get("generation_speed_warmup_runs", 1)))
        target_steps = max(256, int(bench.get("generation_speed_target_generated_steps_per_run", 1024)))
        prompt_ctx = kw.get("max_context_tokens")
        encoded_batches = build_tokenized_prompt_batches(
            engine,
            prompts,
            batch_size=1,
            max_context_tokens=prompt_ctx,
            add_bos_only=True,
        )
        speed_max_new_tokens = int(
            bench.get(
                "generation_speed_max_new_tokens",
                min(int(kw["max_new_tokens"]), 64),
            )
        )
        speed_stats = benchmark_tokenized_generation_speed(
            engine,
            encoded_batches,
            runs=runs,
            warmup_runs=warmup_runs,
            seed=evaluator.eval_rng_seed(),
            max_new_tokens=speed_max_new_tokens,
            target_generated_steps_per_run=target_steps,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            eos_token_id=None,
            device=evaluator.device,
            bias_against_first_tokens=bias,
            suppress_eos_first_step=True,
        )
        speed_stats.update(
            {
                "generation_speed_num_prompts": len(prompts),
                "generation_speed_batch_size": 1,
                "generation_speed_max_new_tokens": speed_max_new_tokens,
                "generation_speed_method": "human_eval_prompts_tokenized_batch1_median_mad",
            }
        )
        results.update(speed_stats)
        return results
