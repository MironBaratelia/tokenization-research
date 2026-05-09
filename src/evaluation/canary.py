import json
import os

from src.common.inference import batched_generate

from .base_evaluator import BaseEvaluator
from .repro_subsample import reproducible_subsample


def _suffix_token_length(tokenizer, text: str) -> int:
    enc = tokenizer.encode(text, add_special_tokens=False)
    if not enc:
        return 0
    if isinstance(enc[0], list):
        return sum(len(w) for w in enc)
    return len(enc)


def _common_prefix_length(a: str, b: str) -> int:
    """Length of common prefix (LCP), not pairwise zip equality count."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _canary_max_new_tokens(tokenizer, items: list) -> int:
    max_chars = max(len(x["suffix"]) for x in items)
    char_budget = ((max_chars + 10) // 10) * 10
    max_toks = max(_suffix_token_length(tokenizer, x["suffix"]) for x in items)
    tok_budget = ((max_toks + 10) // 10) * 10
    return max(1, char_budget, tok_budget)


class CanaryEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config, device="cuda", data_dir="data/benchmarks"):
        super().__init__(model, tokenizer, config, device)
        self.data_dir = data_dir

    def evaluate(self, language="en", batch_size=32):
        file_path = os.path.join(self.data_dir, language, "canaries.json")
        if not os.path.exists(file_path):
            return {}
        max_samples = self.benchmark_max_samples()
        return self.extract_canaries(file_path, batch_size, max_samples=max_samples)

    def extract_canaries(self, file_path, batch_size=32, max_samples=None):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        canaries = reproducible_subsample(
            list(data["data"]), max_samples, self.eval_subsample_seed()
        )

        groups = {}
        for item in canaries:
            cat = item.get("category", "unknown")
            rep = item["target_repeats"]
            key = (cat, rep)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        results = {}
        self.model.eval()

        for (cat, rep), items in sorted(groups.items()):
            correct = 0
            total = len(items)

            prompts = []
            for item in items:
                prefix = item["prefix"]
                if not prefix.endswith(" "):
                    prefix += " "
                prompts.append(prefix)

            max_new_tokens = _canary_max_new_tokens(self.tokenizer, items)

            generated_list = batched_generate(
                model=self.model,
                tokenizer=self.text_tokenizer,
                prompts=prompts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                device=self.device,
                desc=f"Canaries {cat} rep{rep}",
                seed=self.config.get("seed"),
                suppress_eos_first_step=True,
            )

            detailed = []
            prefix_match_lengths = []
            for item, generated_text in zip(items, generated_list):
                target_suffix = item["suffix"]
                target_len = len(target_suffix)
                trimmed_generated = generated_text[:target_len]
                if trimmed_generated == target_suffix:
                    correct += 1
                match_len = _common_prefix_length(trimmed_generated, target_suffix)
                prefix_match_lengths.append(match_len)
                detailed.append({
                    "id": item.get("id"),
                    "prefix": item["prefix"],
                    "suffix": target_suffix,
                    "generated_full": generated_text,
                    "trimmed": trimmed_generated,
                    "match": trimmed_generated == target_suffix,
                    "prefix_match_len": match_len,
                })

            accuracy = (correct / total) * 100 if total > 0 else 0.0
            mean_prefix_match = sum(prefix_match_lengths) / total if total > 0 else 0.0
            mean_prefix_ratio = (
                sum(m / len(item["suffix"]) if item["suffix"] else 0
                    for m, item in zip(prefix_match_lengths, items)) / total
                if total > 0 else 0.0
            )
            prefix = f"{cat}_rep{rep}"
            results[f"accuracy_{prefix}"] = accuracy
            results[f"mean_prefix_match_chars_{prefix}"] = round(mean_prefix_match, 2)
            results[f"mean_prefix_match_ratio_{prefix}"] = round(mean_prefix_ratio * 100, 2)
            results[f"detailed_results_{prefix}"] = detailed

        return results
