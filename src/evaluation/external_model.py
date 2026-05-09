from __future__ import annotations

import logging
import math
import os
import json
import time
from typing import Optional

import torch
from src.common.generation_speed import (
    benchmark_tokenized_generation_speed,
    build_tokenized_prompt_batches,
)
from src.common.math_utils import safe_exp
from src.common.tqdm_utils import tqdm
from src.common.generation_from_config import merge_generation_layers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_evaluator import BaseEvaluator
from .repro_subsample import reproducible_subsample
from src.common.local_model_generation import (
    _apply_miron_benchmark_max_new_words,
    gen_context_limits_like_external,
    generate_continuations_external_style,
)
from src.common.inference import InferenceEngine

logger = logging.getLogger(__name__)


class ExternalModelEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config, device="cuda", data_dir="data/base"):
        super().__init__(model, tokenizer, config, device)
        self.data_dir = data_dir
        self.external_model_names = config.get(
            "external_models",
            ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B"]
        )
        self.hf_token = self._load_hf_token(config)
        self.external_model = None
        self.external_tokenizer = None

    def _load_hf_token(self, config):
        token = config.get("external_models_token")
        token_file = config.get("external_models_token_file", ".hf_token.txt")
        if not token:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token and token_file and os.path.exists(token_file):
            with open(token_file, "r", encoding="utf-8") as f:
                token = f.read().strip()
        return token

    def _resolve_gen_seed(self) -> int:
        return self.eval_rng_seed()

    def _external_gen_config(self) -> dict:
        em = dict(self.config.get("external_model_eval") or {})
        gen = merge_generation_layers(self.config)
        for k in (
            "external_gen_batch_size",
            "external_gen_tune_warmup",
            "external_gen_torch_compile",
            "external_gen_reprobe",
            "external_gen_total_tokens",
            "external_gen_plateau_max_batch",
            "external_gen_temperature",
            "external_gen_repetition_penalty",
            "external_gen_do_sample",
            "external_gen_suppress_eos_first_step",
            "trim_repeated_ngram",
            "trim_min_words",
        ):
            if k in self.config:
                em[k] = self.config[k]
        if (
            "do_sample" in gen
            and "external_gen_do_sample" not in self.config
            and "external_gen_do_sample" not in em
        ):
            em["external_gen_do_sample"] = bool(gen["do_sample"])
        if (
            gen.get("temperature") is not None
            and "external_gen_temperature" not in self.config
            and "external_gen_temperature" not in em
        ):
            em["external_gen_temperature"] = float(gen["temperature"])
        if (
            gen.get("repetition_penalty") is not None
            and "external_gen_repetition_penalty" not in self.config
            and "external_gen_repetition_penalty" not in em
        ):
            em["external_gen_repetition_penalty"] = float(gen["repetition_penalty"])
        return em

    def evaluate(self, language="en", batch_size=16, max_samples=None):
        test_file = os.path.join(self.data_dir, language, "test_indomain.txt")
        if not os.path.exists(test_file):
            return {"external_ppl_avg": float('inf'), "examples": []}
        if max_samples is None:
            max_samples = self.benchmark_max_samples() or self.config.get("external_max_samples") or 2048
        samples = self._load_samples(test_file, max_samples=max_samples, min_words=5)
        prompts = [" ".join(s.split()[:5]) for s in samples]

        gen_seed = self._resolve_gen_seed()
        em = self._external_gen_config()
        max_new_cap = em.get("external_gen_max_new_tokens_cap")
        if max_new_cap is not None:
            max_new_cap = max(1, int(max_new_cap))
        gen_total_tokens = em.get("external_gen_total_tokens")
        generated_texts, gen_bs, batch_search_meta, self.model = generate_continuations_external_style(
            self.model,
            self.tokenizer,
            self.config,
            self.device,
            prompts,
            batch_size=batch_size,
            seed=gen_seed,
            desc="External Eval Generation",
            gen_total_tokens=gen_total_tokens,
            max_new_tokens_cap=max_new_cap,
            allow_torch_compile=True,
        )

        results = {
            "models": {},
            "external_ppl_avg": float("inf"),
            "examples": [],
            "local_generation_batch_size": gen_bs,
            "local_generation_batch_search": batch_search_meta,
            "local_generation_config": {
                "external_gen_total_tokens": gen_total_tokens,
                "external_gen_max_new_tokens_cap": max_new_cap,
                "external_gen_do_sample": bool(em.get("external_gen_do_sample", True)),
                "external_gen_temperature": float(em.get("external_gen_temperature", 1.0)),
                "external_gen_repetition_penalty": float(
                    em.get("external_gen_repetition_penalty", 1.0)
                ),
                "trim_repeated_ngram": int(em.get("trim_repeated_ngram", 0) or 0),
                "trim_min_words": int(em.get("trim_min_words", 32) or 32),
                "seed": gen_seed,
            },
        }

        for p, g in zip(prompts, generated_texts):
            results["examples"].append({"prompt": p, "generated": g, "scores": {}})
        trim_n = int(em.get("trim_repeated_ngram", 0) or 0)
        if trim_n > 0:
            results["loop_trim"] = self.trim_examples_at_repeated_ngrams(
                results["examples"],
                n=trim_n,
                min_words=int(em.get("trim_min_words", 32) or 32),
                max_seen=2,
            )
        results["generation_quality"] = self.generation_quality_stats(results["examples"])

        self._save_generation_cache(language, results)
        results.update(self._benchmark_local_generation_speed(language, prompts, gen_bs))

        # External scorer models are much larger than the local model and run
        # on the same GPU. Release local weights from CUDA before loading the
        # scorer to avoid allocator pressure and device_map/offload slowdowns.
        self._release_local_model_from_gpu()

        score_batch_size = self.config.get("external_score_batch_size", 8)

        for model_name in self.external_model_names:
            self._load_external_model(model_name)
            total_ppl, count = 0.0, 0
            model_key = model_name.split("/")[-1]

            for i in tqdm(
                range(0, len(results["examples"]), score_batch_size),
                desc=f"Scoring with {model_key}",
                unit="batch",
                dynamic_ncols=True,
            ):
                batch_examples = results["examples"][i : i + score_batch_size]
                ppl_list = self._calculate_external_ppl_batch(
                    [(ex["prompt"], ex["generated"]) for ex in batch_examples]
                )
                for ex, ppl in zip(batch_examples, ppl_list):
                    ex["scores"][model_key] = ppl
                    if ppl != float('inf'):
                        total_ppl += ppl
                        count += 1

            results["models"][model_name] = {"external_ppl": total_ppl / count if count > 0 else float('inf')}
            self._unload_external_model()

        valid = [v["external_ppl"] for v in results["models"].values() if v["external_ppl"] != float('inf')]
        if valid: results["external_ppl_avg"] = sum(valid) / len(valid)
        self.add_repetition_adjusted_scores(results)

        return results

    @staticmethod
    def generation_quality_stats(examples: list[dict]) -> dict:
        """Cheap degeneracy checks for Oracle PPL results.

        Oracle PPL can become artificially low when a model repeats a very
        predictable phrase. Distinct-n and repeat-n expose that failure mode.
        """
        generated = [str(ex.get("generated") or "") for ex in examples]
        word_lengths = [len(text.split()) for text in generated]
        char_lengths = [len(text) for text in generated]

        def mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
            if len(tokens) < n:
                return []
            return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        distinct_by_n: dict[str, float] = {}
        repeat_by_n: dict[str, float] = {}
        example_repeat_3: list[float] = []
        for n in (1, 2, 3, 4):
            all_ngrams: list[tuple[str, ...]] = []
            repeats: list[float] = []
            for text in generated:
                grams = ngrams(text.split(), n)
                all_ngrams.extend(grams)
                if grams:
                    repeats.append(1.0 - len(set(grams)) / len(grams))
            distinct_by_n[f"distinct_{n}"] = (
                float(len(set(all_ngrams)) / len(all_ngrams)) if all_ngrams else 0.0
            )
            repeat_by_n[f"repeat_{n}"] = mean(repeats)
            if n == 3:
                example_repeat_3 = repeats

        return {
            "num_examples": len(examples),
            "avg_generated_words": mean([float(x) for x in word_lengths]),
            "avg_generated_chars": mean([float(x) for x in char_lengths]),
            "distinct": distinct_by_n,
            "repeat": repeat_by_n,
            "repeat_3_p95": float(
                sorted(example_repeat_3)[int(0.95 * (len(example_repeat_3) - 1))]
            )
            if example_repeat_3
            else 0.0,
        }

    @staticmethod
    def trim_text_at_repeated_ngram(
        text: str,
        *,
        n: int = 3,
        min_words: int = 32,
        max_seen: int = 2,
    ) -> tuple[str, dict]:
        """Cut a continuation when an n-gram starts repeating too often.

        This prevents long loops from receiving artificially good Oracle PPL.
        The raw generation should still be retained by callers when needed.
        """
        words = text.split()
        if n <= 0 or len(words) < max(min_words, n):
            return text, {"trimmed": False}
        seen: dict[tuple[str, ...], int] = {}
        for i in range(0, len(words) - n + 1):
            gram = tuple(words[i : i + n])
            seen[gram] = seen.get(gram, 0) + 1
            if i + n >= min_words and seen[gram] > max_seen:
                trimmed_words = words[:i]
                if len(trimmed_words) < min_words:
                    break
                return " ".join(trimmed_words), {
                    "trimmed": True,
                    "ngram_n": int(n),
                    "min_words": int(min_words),
                    "max_seen": int(max_seen),
                    "original_words": len(words),
                    "trimmed_words": len(trimmed_words),
                    "trigger_ngram": " ".join(gram),
                }
        return text, {"trimmed": False}

    @classmethod
    def trim_examples_at_repeated_ngrams(
        cls,
        examples: list[dict],
        *,
        n: int = 3,
        min_words: int = 32,
        max_seen: int = 2,
    ) -> dict:
        trimmed = 0
        for ex in examples:
            generated = str(ex.get("generated") or "")
            new_text, meta = cls.trim_text_at_repeated_ngram(
                generated, n=n, min_words=min_words, max_seen=max_seen
            )
            if meta.get("trimmed"):
                ex["generated_raw"] = generated
                ex["generated"] = new_text
                trimmed += 1
            ex["loop_trim"] = meta
        return {
            "enabled": True,
            "ngram_n": int(n),
            "min_words": int(min_words),
            "max_seen": int(max_seen),
            "trimmed_examples": int(trimmed),
            "trimmed_ratio": float(trimmed / len(examples)) if examples else 0.0,
        }

    @staticmethod
    def add_repetition_adjusted_scores(
        results: dict,
        *,
        repeat3_floor: float = 0.10,
        repeat3_warn: float = 0.15,
        repeat3_fail: float = 0.30,
        penalty_strength: float = 8.0,
        trim_ratio_floor: float = 0.50,
        trim_ratio_strength: float = 25.0,
    ) -> dict:
        """Add a repetition-aware Oracle score without mutating raw PPL.

        External PPL alone rewards predictable loops: once a model repeats a
        phrase, the next phrase is easy for a strong scorer to predict. The
        adjusted score is a sensitivity check, not the main fluency metric:
        repeated-loop trimming is applied before scoring, so the multiplier
        combines residual repetition after trimming with how often trimming was
        required in the first place.
        """
        quality = results.get("generation_quality") or {}
        repeat = quality.get("repeat") or {}
        repeat3 = float(repeat.get("repeat_3", 0.0) or 0.0)
        excess = max(0.0, repeat3 - float(repeat3_floor))
        repeat_multiplier = math.exp(float(penalty_strength) * excess)
        trim_meta = results.get("loop_trim") or {}
        trim_ratio = float(trim_meta.get("trimmed_ratio", 0.0) or 0.0)
        trim_excess = max(0.0, trim_ratio - float(trim_ratio_floor))
        trim_multiplier_add = float(trim_ratio_strength) * trim_excess
        multiplier = repeat_multiplier + trim_multiplier_add
        status = "ok"
        if repeat3 >= repeat3_fail or trim_ratio >= 0.75:
            status = "fail"
        elif repeat3 >= repeat3_warn or trim_ratio > trim_ratio_floor:
            status = "warn"

        adjusted_models = {}
        for model_name, vals in (results.get("models") or {}).items():
            ppl = vals.get("external_ppl")
            if ppl is None or ppl == float("inf"):
                adjusted = float("inf")
            else:
                adjusted = float(ppl) * multiplier
            vals["external_ppl_repetition_adjusted"] = adjusted
            adjusted_models[model_name] = adjusted

        finite = [v for v in adjusted_models.values() if v != float("inf")]
        results["external_ppl_repetition_adjusted_avg"] = (
            sum(finite) / len(finite) if finite else float("inf")
        )
        results["external_repetition_penalty"] = {
            "status": status,
            "repeat_3": repeat3,
            "repeat3_floor": float(repeat3_floor),
            "repeat3_warn": float(repeat3_warn),
            "repeat3_fail": float(repeat3_fail),
            "penalty_strength": float(penalty_strength),
            "repeat_multiplier": float(repeat_multiplier),
            "trimmed_ratio": float(trim_ratio),
            "trim_ratio_floor": float(trim_ratio_floor),
            "trim_ratio_strength": float(trim_ratio_strength),
            "trim_multiplier_add": float(trim_multiplier_add),
            "multiplier": multiplier,
            "interpretation": (
                "Use raw external_ppl as the primary oracle fluency score. The "
                "adjusted value is a repetition sensitivity check: it penalizes "
                "both residual repeated trigrams and the fraction of generations "
                "that had to be cut before scoring."
            ),
        }
        return results

    def _results_dir(self, language: str) -> str:
        exp_id = (self.config.get("experiment") or {}).get("id")
        if not exp_id:
            exp_id = f"{language}/unknown"
        resolved = self.config.get("resolved_paths") or {}
        outputs_dir = resolved.get("outputs_dir") or os.path.join(os.getcwd(), "outputs")
        return os.path.join(outputs_dir, exp_id, "evaluation")

    def _save_generation_cache(self, language: str, results: dict) -> None:
        path = os.path.join(self._results_dir(language), "external_model_generations_cache.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_examples": len(results.get("examples", [])),
            "local_generation_batch_size": results.get("local_generation_batch_size"),
            "local_generation_batch_search": results.get("local_generation_batch_search"),
            "examples": results.get("examples", []),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _release_local_model_from_gpu(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model.to("cpu")
            except Exception:
                pass
            torch.cuda.empty_cache()

    def _benchmark_local_generation_speed(self, language: str, prompts: list[str], gen_bs: int) -> dict:
        if not prompts:
            return {}
        em = self._external_gen_config()
        max_new, prompt_budget, _ = gen_context_limits_like_external(
            self.tokenizer,
            prompts,
            em,
            gen_total_tokens=None,
            min_new_tokens=32,
        )
        max_new = _apply_miron_benchmark_max_new_words(max_new, self.tokenizer, self.config)
        bench = self.config.get("benchmarks") or {}
        max_new = int(bench.get("generation_speed_max_new_tokens", min(int(max_new), 64)))
        runs = max(3, int(bench.get("generation_speed_runs", 5)))
        warmup_runs = max(1, int(bench.get("generation_speed_warmup_runs", 1)))
        target_steps = max(256, int(bench.get("generation_speed_target_generated_steps_per_run", 1024)))
        engine = InferenceEngine(self.model, self.tokenizer, device=self.device)
        encoded_batches = build_tokenized_prompt_batches(
            engine,
            prompts,
            batch_size=1,
            max_context_tokens=prompt_budget,
            add_bos_only=True,
        )
        stats = benchmark_tokenized_generation_speed(
            engine,
            encoded_batches,
            runs=runs,
            warmup_runs=warmup_runs,
            seed=self.eval_rng_seed(),
            max_new_tokens=int(max_new),
            target_generated_steps_per_run=target_steps,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            eos_token_id=None,
            device=self.device,
            max_context_tokens=prompt_budget,
        )
        stats.update(
            {
                "generation_speed_num_prompts": len(prompts),
                "generation_speed_batch_size": 1,
                "generation_speed_max_new_tokens": int(max_new),
                "generation_speed_method": "external_eval_prompts_tokenized_batch1_median_mad",
                "generation_speed_source": os.path.join(self.data_dir, language, "test_indomain.txt"),
            }
        )
        return stats
    
    def _load_external_model(self, model_name: str):
        token_kwargs = {"token": self.hf_token} if self.hf_token else {}
        self.external_tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
        if self.external_tokenizer.pad_token is None:
            self.external_tokenizer.pad_token = self.external_tokenizer.eos_token
        
        load_kw = {"torch_dtype": torch.float16, "device_map": "auto", **token_kwargs}
        attn_impl = self.config.get("external_attn_implementation", "sdpa")
        if attn_impl:
            load_kw["attn_implementation"] = attn_impl
        try:
            self.external_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
        except Exception:
            load_kw.pop("attn_implementation", None)
            self.external_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
        self.external_model.eval()
    
    def _unload_external_model(self):
        if self.external_model is not None: del self.external_model
        if self.external_tokenizer is not None: del self.external_tokenizer
        self.external_model = self.external_tokenizer = None
        torch.cuda.empty_cache()
    
    def _calculate_external_ppl_batch(self, pairs: list) -> list:
        """Batch PPL: one forward per batch."""
        context_length = self._external_context_length()
        full_texts = []
        prompt_lens = []
        for prompt, generated in pairs:
            if not generated.strip():
                full_texts.append("")
                prompt_lens.append(0)
                continue
            full = (prompt.rstrip() + " " + generated.strip()).strip()
            full_texts.append(full if full else "")
            if not full:
                prompt_lens.append(0)
                continue
            p_out = self.external_tokenizer(prompt, return_tensors="pt")
            f_out = self.external_tokenizer(full, return_tensors="pt")
            p_ids = p_out["input_ids"][0].tolist()
            f_ids = f_out["input_ids"][0].tolist()
            plen = 0
            for start in (0, 1):
                if start + len(p_ids) <= len(f_ids) and f_ids[start : start + len(p_ids)] == p_ids:
                    plen = start + len(p_ids)
                    break
            if plen == 0:
                plen = len(p_ids)
            prompt_lens.append(plen)

        non_empty = [t for t in full_texts if t]
        if not non_empty:
            return [float('inf')] * len(pairs)

        enc = self.external_tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=context_length,
            return_attention_mask=True,
        ).to(self.external_model.device)

        with torch.inference_mode():
            outputs = self.external_model(**enc)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = enc["input_ids"][:, 1:].contiguous()
            attn = enc["attention_mask"][:, 1:].float()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            result = []
            for idx in range(len(pairs)):
                if not full_texts[idx] or prompt_lens[idx] <= 0:
                    result.append(float('inf'))
                    continue
                plen = prompt_lens[idx]
                gen_mask = torch.zeros_like(shift_labels[idx], dtype=torch.float, device=loss.device)
                gen_mask[plen - 1 :] = 1.0
                gen_mask = gen_mask * attn[idx]
                n = gen_mask.sum().item()
                if n <= 0:
                    result.append(float('inf'))
                    continue
                avg_loss = (loss[idx] * gen_mask).sum().item() / n
                result.append(safe_exp(avg_loss))
        return result

    def _external_context_length(self) -> int:
        cfg_len = (self.config.get("external_model_eval") or {}).get("max_length")
        if cfg_len is not None:
            try:
                return max(2, int(cfg_len))
            except (TypeError, ValueError):
                pass
        model_ctx = getattr(self.external_model.config, "max_position_embeddings", None) if self.external_model is not None else None
        if model_ctx is not None:
            try:
                return max(2, int(model_ctx))
            except (TypeError, ValueError):
                pass
        tok_len = getattr(self.external_tokenizer, "model_max_length", None)
        if isinstance(tok_len, int) and 0 < tok_len < 100000:
            return max(2, int(tok_len))
        return 512
    
    def _load_samples(self, file_path, max_samples=100, min_words=5):
        """Filter lines, then reproducible subsample to max_samples."""
        pool: list[str] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and len(s.split()) >= min_words:
                    pool.append(s)
        return reproducible_subsample(pool, max_samples, self.eval_subsample_seed())
