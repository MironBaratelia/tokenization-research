import json
import os

from src.common.tokenizer_utils import UniversalTokenizer


class BaseEvaluator:
    def __init__(self, model, tokenizer, config, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.text_tokenizer = UniversalTokenizer(tokenizer)

    def evaluate(self):
        raise NotImplementedError

    def eval_rng_seed(self) -> int:
        s = self.config.get("seed")
        if s is not None:
            return int(s)
        tr = self.config.get("training") or {}
        if tr.get("seed") is not None:
            return int(tr["seed"])
        return 42

    def eval_subsample_seed(self) -> int:
        s = self.config.get("eval_subsample_seed")
        if s is not None:
            return int(s)
        return self.eval_rng_seed()

    def benchmark_max_samples(self):
        b = self.config.get("benchmarks")
        if isinstance(b, dict) and b.get("max_samples") is not None:
            return int(b["max_samples"])
        if self.config.get("max_samples") is not None:
            return int(self.config["max_samples"])
        return None

    def save_results(self, results, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
