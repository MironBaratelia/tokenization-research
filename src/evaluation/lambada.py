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

class LAMBADAEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config, device="cuda", data_dir="data/benchmarks"):
        super().__init__(model, tokenizer, config, device)
        self.data_dir = data_dir

    def evaluate(self, language="en", batch_size=32):
        if language != "en":
            return {
                "skipped": True,
                "reason": "LAMBADA is English-only (no other lambada.tsv is used).",
            }
        file_path = os.path.join(self.data_dir, language, "lambada.tsv")
        if not os.path.exists(file_path):
            return {"accuracy": 0.0, "note": f"lambada.tsv not found for {language}"}
        max_samples = self.benchmark_max_samples()
        return self.evaluate_file(file_path, batch_size, max_samples=max_samples)

    def evaluate_file(self, file_path, batch_size=32, max_samples=None):
        samples = self._load_samples(file_path)
        samples = reproducible_subsample(samples, max_samples, self.eval_subsample_seed())
        detailed_results = []
        total_accuracy, total_lev_score, total_lev_thresholded, total_confidence, count = 0.0, 0.0, 0.0, 0.0, 0

        self.model.eval()
        prompts = [s[0] for s in samples]
        targets = [s[1] for s in samples]
        prompts_for_gen = [p.rstrip() for p in prompts]

        gt = resolve_benchmark_gen_total_tokens(self.config)
        cap = resolve_benchmark_gen_max_new_cap(self.config)
        gen_cfg = dict(self.config)
        _ds = resolve_benchmark_gen_do_sample(self.config)
        if _ds is not None:
            gen_cfg["external_gen_do_sample"] = _ds
        generated_list, _, _, self.model = generate_continuations_external_style(
            self.model,
            self.tokenizer,
            gen_cfg,
            self.device,
            prompts_for_gen,
            batch_size=batch_size,
            seed=self.eval_rng_seed(),
            desc="LAMBADA",
            gen_total_tokens=gt,
            max_new_tokens_cap=cap,
            allow_torch_compile=True,
            use_throughput_prefs=True,
        )

        engine = InferenceEngine(self.model, self.tokenizer, device=self.device)
        targets_scoring = [t.strip() for t in targets]
        score_results = engine.score_continuations(
            prompts,
            targets_scoring,
            batch_size=batch_size,
            progress_desc="LAMBADA score",
        )

        for prompt, target, gen_text, score_res in zip(prompts, targets, generated_list, score_results):
            gen_text_clean = first_word_for_scoring(
                gen_text, unk_token=getattr(self.tokenizer, "unk_token", None)
            )
            target_clean = target.strip()

            acc = 1.0 if gen_text_clean.lower() == target_clean.lower() else 0.0
            lev_score = word_sequence_similarity_percent(gen_text_clean, target_clean)
            lev_score_thresholded = threshold_random_levenshtein(lev_score)
            confidence = score_res["confidence"]
            conf_val = confidence if isinstance(confidence, (int, float)) and math.isfinite(confidence) else 0.0

            total_accuracy += acc
            total_lev_score += lev_score
            total_lev_thresholded += lev_score_thresholded
            total_confidence += conf_val
            count += 1

            detailed_results.append({
                "prompt": prompt,
                "target": target,
                "generated": gen_text_clean,
                "accuracy": acc,
                "exact_match": acc,
                "lev_score": round(lev_score, 2),
                "lev_score_thresholded": round(lev_score_thresholded, 2),
                "target_confidence": round(conf_val, 2),
            })

        mean_lev = total_lev_score / count if count > 0 else 0.0
        mean_lev_thresholded = total_lev_thresholded / count if count > 0 else 0.0
        mean_accuracy = total_accuracy / count if count > 0 else 0.0
        return {
            "score": mean_lev_thresholded,
            "accuracy": mean_accuracy,
            "exact_match": mean_accuracy,
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
