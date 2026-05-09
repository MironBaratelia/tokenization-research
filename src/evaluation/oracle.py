import math
import os
import time

import torch
import torch.nn as nn

from src.common.generation_from_config import base_generate_kwargs
from src.common.generation_speed import benchmark_tokenized_generation_speed
from src.common.inference import InferenceEngine
from src.common.math_utils import safe_exp
from src.common.tqdm_utils import tqdm

from .base_evaluator import BaseEvaluator
from .repro_subsample import reproducible_subsample


class OracleEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config, device="cuda", data_dir="data/base"):
        super().__init__(model, tokenizer, config, device)
        self.data_dir = data_dir

    def evaluate(self, language="en", batch_size=32):
        test_file = os.path.join(self.data_dir, language, "test_indomain.txt")
        if not os.path.exists(test_file):
            return {"oracle_ppl": float("inf"), "throughput_chars_per_sec": 0.0, "oov_rate": 0.0}

        max_samples = self.benchmark_max_samples()
        results = self.evaluate_oracle_metrics(
            test_file, batch_size, max_samples=max_samples if max_samples is not None else 1000
        )
        try:
            results.update(
                self.evaluate_generation_speed(
                    test_file, batch_size, max_samples=max_samples if max_samples is not None else 256
                )
            )
        except Exception:
            pass
        return results

    def evaluate_oracle_metrics(self, file_path, batch_size=32, max_samples=None):
        samples = self._load_samples(file_path, max_samples=max_samples or 1000)

        total_loss = 0.0
        total_tokens = 0
        total_chars = 0
        oov_tokens = 0
        total_tokens_in_corpus = 0

        start_time = time.time()

        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        context_length = self._oracle_context_length()

        for i in tqdm(
            range(0, len(samples), batch_size),
            desc="Oracle",
            unit="batch",
            dynamic_ncols=True,
        ):
            batch_samples = samples[i : i + batch_size]

            inputs = self.text_tokenizer(
                batch_samples,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=context_length,
            )
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            unk_token_id = getattr(self.tokenizer, "unk_token_id", None) or getattr(
                self.text_tokenizer, "unk_token_id", None
            )
            for text in batch_samples:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                ids_flat = token_ids
                if token_ids and isinstance(token_ids[0], list):
                    ids_flat = [tid for word in token_ids for tid in word]
                if unk_token_id is not None:
                    oov_tokens += ids_flat.count(unk_token_id)
                total_tokens_in_corpus += len(ids_flat)

            with torch.no_grad():
                outputs = self.model(input_ids=inputs["input_ids"])
                input_ids = inputs["input_ids"]

                if input_ids.dim() == 3:
                    loss = outputs.get("loss")
                    if loss is None:
                        raise KeyError("MIRON oracle path expected model output to contain 'loss'")
                    pad_id = getattr(self.model, "pad_id", getattr(self.tokenizer, "pad_token_id", 0))
                    bow_id = getattr(self.model, "bow_id", None)
                    enc_in = input_ids[:, :-1, :].contiguous()
                    target_words = input_ids[:, 1:, :].contiguous()
                    bsz, time_steps, _ = enc_in.shape
                    bow_valid = 0
                    if bow_id is not None and int(bow_id) != int(pad_id):
                        bow_valid = bsz * time_steps
                    valid_chars = int((target_words != int(pad_id)).sum().item()) + int(bow_valid)
                    total_loss += float(loss.item()) * float(valid_chars)
                    total_tokens += valid_chars
                else:
                    logits = outputs["logits"]

                    labels = input_ids.clone()
                    labels = labels[:, 1:]

                    shift_logits = logits[:, :-1, :].contiguous()

                    if "attention_mask" in inputs:
                        shift_mask = inputs["attention_mask"][:, 1:]
                        labels = labels.masked_fill(shift_mask == 0, -100)

                    current_vocab_size = shift_logits.size(-1)
                    loss = criterion(shift_logits.view(-1, current_vocab_size), labels.view(-1))

                    total_loss += loss.item()

                    valid_mask = labels != -100
                    num_valid_tokens = valid_mask.sum().item()
                    total_tokens += num_valid_tokens

                for text in batch_samples:
                    total_chars += len(text)

        total_time = time.time() - start_time

        oracle_ppl = safe_exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        throughput_chars_per_sec = total_chars / total_time if total_time > 0 else 0.0
        oov_rate = oov_tokens / total_tokens_in_corpus if total_tokens_in_corpus > 0 else 0.0

        return {
            "oracle_ppl": oracle_ppl,
            "throughput_chars_per_sec": throughput_chars_per_sec,
            "oov_rate": oov_rate,
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "total_time_sec": total_time,
        }

    def evaluate_generation_speed(self, file_path, batch_size=32, max_samples=None):
        prompts = self._load_generation_prompts(file_path, max_samples=max_samples or 256)
        if not prompts:
            return {}

        bench = self.config.get("benchmarks") or {}
        runs = max(5, int(bench.get("generation_speed_runs", 7)))
        warmup_runs = max(1, int(bench.get("generation_speed_warmup_runs", 2)))
        max_prompt_chars = max(32, int(bench.get("generation_speed_prompt_chars", 160)))
        max_prompts = max(4, int(bench.get("generation_speed_num_prompts", 16)))
        target_steps = max(512, int(bench.get("generation_speed_target_generated_steps_per_run", 4096)))

        prompts = [p[:max_prompt_chars].rstrip() for p in prompts if p.strip()]
        prompts = [p for p in prompts if p]
        prompts = prompts[:max_prompts]
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
            seed=None,
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
            }
        )
        return stats

    def _oracle_context_length(self) -> int:
        model_ctx = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if model_ctx is not None:
            try:
                return max(2, int(model_ctx))
            except (TypeError, ValueError):
                pass
        training_ctx = (self.config.get("training") or {}).get("context_length")
        if training_ctx is not None:
            try:
                return max(2, int(training_ctx))
            except (TypeError, ValueError):
                pass
        return 512

    def _load_samples(self, file_path, max_samples=1000):
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line)
        return reproducible_subsample(samples, max_samples, self.eval_subsample_seed())

    def _load_generation_prompts(self, file_path, max_samples=256):
        prompts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cut = max(1, len(line) // 2)
                prompts.append(line[:cut])
        return reproducible_subsample(prompts, max_samples, self.eval_subsample_seed())
