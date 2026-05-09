import os

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
from src.common.local_model_generation import _apply_miron_benchmark_max_new_words

from .base_evaluator import BaseEvaluator
from .repro_subsample import reproducible_subsample
from .neural_segmenter_human_eval import run_neural_segmenter_human_eval


class HumanEvalEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config, device="cuda", data_dir="data/benchmarks"):
        super().__init__(model, tokenizer, config, device)
        self.data_dir = data_dir

    def evaluate(self, language="en", batch_size=32):
        file_path = os.path.join(self.data_dir, language, "human_eval.txt")
        if not os.path.exists(file_path):
            return {}

        max_samples = self._human_eval_max_samples()
        return self.generate_completions(file_path, batch_size, max_samples=max_samples)

    def generate_completions(self, file_path, batch_size=32, max_samples=None):
        if (self.config.get("tokenizer") or {}).get("type") == "neural_segmenter":
            return run_neural_segmenter_human_eval(self, file_path, batch_size, max_samples)

        prompts = self._load_prompts(file_path)
        prompts = reproducible_subsample(prompts, max_samples, self.eval_subsample_seed())
        gen_cfg = merge_generation_layers(self.config)
        is_miron = (self.config.get("tokenizer") or {}).get("type") == "miron"
        engine = InferenceEngine(
            self.model,
            self.tokenizer,
            device=self.device,
            miron_strict=is_miron,
        )
        kw = base_generate_kwargs(self.config, self.model, engine.tokenizer, next_word=False)
        kw["max_new_tokens"] = _apply_miron_benchmark_max_new_words(
            int(kw["max_new_tokens"]), self.tokenizer, self.config
        )
        bias = gen_cfg.get("bias_against_first_tokens")
        if bias is None:
            bias = list(DEFAULT_BIAS_FIRST_TOKENS)
        ibs = gen_cfg.get("inference_batch_size")
        eff_bs = int(ibs) if ibs is not None else batch_size
        generated_list = engine.generate(
            prompts=prompts,
            batch_size=eff_bs,
            bias_against_first_tokens=bias,
            suppress_eos_first_step=True,
            desc="HumanEval",
            seed=self.eval_rng_seed(),
            **kw,
        )

        generations = []
        for prompt, gen_text in zip(prompts, generated_list):
            generations.append({
                "prompt": prompt,
                "generated": gen_text,
            })

        results = {"generations": generations}
        speed_max_new_tokens = int(
            (self.config.get("benchmarks") or {}).get(
                "generation_speed_max_new_tokens",
                min(int(kw["max_new_tokens"]), 64),
            )
        )
        results.update(
            self._benchmark_generation_speed(
                engine=InferenceEngine(self.model, self.tokenizer, device=self.device),
                prompts=prompts,
                max_new_tokens=speed_max_new_tokens,
                bias_against_first_tokens=bias,
            )
        )
        return results

    def _benchmark_generation_speed(self, *, engine, prompts, max_new_tokens, bias_against_first_tokens):
        if not prompts:
            return {}
        bench = self.config.get("benchmarks") or {}
        runs = max(3, int(bench.get("generation_speed_runs", 5)))
        warmup_runs = max(1, int(bench.get("generation_speed_warmup_runs", 1)))
        target_steps = max(256, int(bench.get("generation_speed_target_generated_steps_per_run", 1024)))
        prompt_ctx = base_generate_kwargs(self.config, self.model, engine.tokenizer, next_word=False).get(
            "max_context_tokens"
        )
        encoded_batches = build_tokenized_prompt_batches(
            engine,
            prompts,
            batch_size=1,
            max_context_tokens=prompt_ctx,
            add_bos_only=True,
        )
        stats = benchmark_tokenized_generation_speed(
            engine,
            encoded_batches,
            runs=runs,
            warmup_runs=warmup_runs,
            seed=self.eval_rng_seed(),
            max_new_tokens=int(max_new_tokens),
            target_generated_steps_per_run=target_steps,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            eos_token_id=None,
            device=self.device,
            bias_against_first_tokens=bias_against_first_tokens,
            suppress_eos_first_step=True,
        )
        stats.update(
            {
                "generation_speed_num_prompts": len(prompts),
                "generation_speed_batch_size": 1,
                "generation_speed_max_new_tokens": int(max_new_tokens),
                "generation_speed_method": "human_eval_prompts_tokenized_batch1_median_mad",
            }
        )
        return stats

    def _load_prompts(self, file_path):
        prompts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if line:
                    prompts.append(line)
        return prompts

    def _human_eval_max_samples(self):
        he = self.config.get("human_eval") or {}
        if he.get("max_samples") is not None:
            return int(he["max_samples"])
        if self.config.get("human_eval_max_samples") is not None:
            return int(self.config["human_eval_max_samples"])
        return None
