from __future__ import annotations

import os
from typing import List, Optional

from src.common.generation_speed import benchmark_tokenized_generation_speed
from src.common.generation_from_config import base_generate_kwargs
from src.common.inference import InferenceEngine

from .base_evaluator import BaseEvaluator
from .repro_subsample import reproducible_subsample


class GenerationSpeedEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config, device="cuda", data_dir="data/base"):
        super().__init__(model, tokenizer, config, device)
        self.data_dir = data_dir

    def evaluate(self, language="en", batch_size=1):
        max_samples = self.benchmark_max_samples()
        prompts = self._load_prompts(language, max_samples=max_samples or 32)
        if not prompts:
            return {}

        bench = self.config.get("benchmarks") or {}
        runs = max(5, int(bench.get("generation_speed_runs", 7)))
        warmup_runs = max(1, int(bench.get("generation_speed_warmup_runs", 2)))
        max_prompt_chars = max(32, int(bench.get("generation_speed_prompt_chars", 160)))
        max_prompts = max(4, int(bench.get("generation_speed_num_prompts", 16)))
        target_steps = max(512, int(bench.get("generation_speed_target_generated_steps_per_run", 4096)))

        prompts = [p[:max_prompt_chars].rstrip() for p in prompts if p.strip()]
        prompts = [p for p in prompts if p][:max_prompts]
        if not prompts:
            return {}

        engine = InferenceEngine(self.model, self.tokenizer, device=self.device)
        gen_kwargs = base_generate_kwargs(self.config, self.model, engine.tokenizer, next_word=False)
        gen_kwargs["do_sample"] = False
        gen_kwargs["max_new_tokens"] = int(
            bench.get("generation_speed_max_new_tokens", min(int(gen_kwargs["max_new_tokens"]), 32))
        )
        prompt_ctx = gen_kwargs.get("max_context_tokens")

        encoded_prompts = []
        for prompt in prompts:
            enc = engine.tokenizer.encode(
                [prompt],
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
                add_bos_only=True,
                truncation=prompt_ctx is not None,
                max_length=prompt_ctx,
                padding_side="left",
            )
            encoded_prompts.append((enc["input_ids"], enc.get("attention_mask")))

        stats = benchmark_tokenized_generation_speed(
            engine,
            encoded_prompts,
            runs=runs,
            warmup_runs=warmup_runs,
            seed=self.eval_rng_seed(),
            max_new_tokens=int(gen_kwargs["max_new_tokens"]),
            target_generated_steps_per_run=target_steps,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            eos_token_id=None,
            device=self.device,
            **{
                k: v
                for k, v in gen_kwargs.items()
                if k not in ("max_new_tokens", "do_sample", "temperature", "repetition_penalty")
            },
        )
        stats.update(
            {
                "generation_speed_num_prompts": len(prompts),
                "generation_speed_batch_size": 1,
                "generation_speed_max_new_tokens": gen_kwargs["max_new_tokens"],
                "generation_speed_method": "tokenized_prompt_batch1_median_mad",
                "generation_speed_seed_mode": "tokenized",
                "generation_speed_source": self._prompt_source(language),
            }
        )
        return stats

    def _prompt_source(self, language: str) -> str:
        for name in ("test_indomain.txt", "human_eval.txt"):
            path = os.path.join(self.data_dir, language, name)
            if os.path.exists(path):
                return path
        return os.path.join(self.data_dir, language, "test_indomain.txt")

    def _load_prompts(self, language: str, max_samples: Optional[int]) -> List[str]:
        path = self._prompt_source(language)
        if not os.path.exists(path):
            return []
        prompts: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cut = max(1, len(line) // 2)
                prompts.append(line[:cut])
        return reproducible_subsample(prompts, max_samples, self.eval_subsample_seed())
