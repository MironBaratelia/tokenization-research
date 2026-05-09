import math
import os

from src.common.local_model_generation import (
    generate_continuations_external_style,
    resolve_benchmark_gen_do_sample,
    resolve_benchmark_gen_max_new_cap,
    resolve_benchmark_gen_total_tokens,
)
from src.common.inference import InferenceEngine
from src.common.next_word_extract import first_word_for_scoring
from src.common.word_sequence_score import (
    LEV_RANDOM_MATCH_CUTOFF,
    threshold_random_levenshtein,
    word_sequence_similarity_percent,
)

from .base_evaluator import BaseEvaluator
from .repro_subsample import reproducible_subsample

class MIRONEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config, device="cuda", data_dir="data/benchmarks"):
        super().__init__(model, tokenizer, config, device)
        self.data_dir = data_dir
        self.categories = ["morphology", "facts", "logic", "noise"]

    def evaluate(self, language="en", batch_size=32):
        max_samples = self.benchmark_max_samples()
        results = {}
        for category in self.categories:
            file_path = os.path.join(self.data_dir, language, "miron", f"{category}.tsv")
            if os.path.exists(file_path):
                results[category] = self.evaluate_category(file_path, batch_size, max_samples=max_samples)
        return results

    def evaluate_category(self, file_path, batch_size=32, max_samples=None):
        samples = self._load_samples(file_path)
        samples = reproducible_subsample(samples, max_samples, self.eval_subsample_seed())
        detailed_results = []
        total_lev, total_lev_thresholded, total_exact, total_confidence, count = 0.0, 0.0, 0.0, 0.0, 0

        self.model.eval()
        prompts = [s[0] for s in samples]
        targets = [s[1] for s in samples]
        # Let each tokenizer/model generate the separator itself. For byte-level
        # tokenizers, forcing a trailing space changes the state and can make
        # greedy next-word decoding choose emoji/control byte tokens.
        prompts_for_gen = [p.rstrip() for p in prompts]

        gt = resolve_benchmark_gen_total_tokens(self.config)
        cap = resolve_benchmark_gen_max_new_cap(self.config)
        gen_cfg = dict(self.config)
        benchmarks_cfg = self.config.get("benchmarks") or {}
        _ds = resolve_benchmark_gen_do_sample(self.config)
        gen_cfg["external_gen_do_sample"] = False if _ds is None else _ds
        gen_cfg["external_gen_temperature"] = float(benchmarks_cfg.get("gen_temperature", 1.0))
        gen_cfg["external_gen_repetition_penalty"] = float(benchmarks_cfg.get("gen_repetition_penalty", 1.0))
        generated_list, _, _, self.model = generate_continuations_external_style(
            self.model,
            self.tokenizer,
            gen_cfg,
            self.device,
            prompts_for_gen,
            batch_size=batch_size,
            seed=self.eval_rng_seed(),
            desc=os.path.basename(file_path),
            gen_total_tokens=gt,
            max_new_tokens_cap=cap,
            allow_torch_compile=True,
            use_throughput_prefs=True,
        )

        engine = InferenceEngine(self.model, self.tokenizer, device=self.device)
        score_results = engine.score_continuations(
            prompts,
            targets,
            batch_size=batch_size,
            progress_desc="MIRON score",
        )

        for prompt, target, gen_text, score_res in zip(prompts, targets, generated_list, score_results):
            gen_text_clean = first_word_for_scoring(
                gen_text, unk_token=getattr(self.tokenizer, "unk_token", None)
            )
            target_clean = target.strip()
            lev_score = word_sequence_similarity_percent(gen_text_clean, target_clean)
            lev_score_thresholded = threshold_random_levenshtein(lev_score)
            exact_match = 1.0 if gen_text_clean.lower() == target_clean.lower() else 0.0
            confidence = score_res["confidence"]

            total_lev += lev_score
            total_lev_thresholded += lev_score_thresholded
            total_exact += exact_match
            conf_val = confidence if isinstance(confidence, (int, float)) and math.isfinite(confidence) else 0.0
            total_confidence += conf_val
            count += 1

            detailed_results.append({
                "prompt": prompt,
                "target": target,
                "generated": gen_text_clean,
                "exact_match": exact_match,
                "accuracy": exact_match,
                "lev_score": round(lev_score, 2),
                "lev_score_thresholded": round(lev_score_thresholded, 2),
                "target_confidence": round(conf_val, 2),
            })

        mean_lev = total_lev / count if count > 0 else 0.0
        mean_lev_thresholded = total_lev_thresholded / count if count > 0 else 0.0
        mean_exact = total_exact / count if count > 0 else 0.0
        return {
            "score": mean_lev_thresholded,
            "accuracy": mean_exact,
            "exact_match": mean_exact,
            "lev_score": mean_lev,
            "lev_score_thresholded": mean_lev_thresholded,
            "lev_random_match_cutoff": LEV_RANDOM_MATCH_CUTOFF,
            "confidence": total_confidence / count if count > 0 else 0.0,
            "detailed_results": detailed_results,
        }

    def _load_samples(self, file_path):
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    samples.append((parts[0], parts[1]))
        return samples
