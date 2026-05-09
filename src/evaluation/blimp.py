import json
import os

import torch
import torch.nn as nn
from src.common.tqdm_utils import tqdm

from .base_evaluator import BaseEvaluator
from .repro_subsample import reproducible_subsample


def _is_word_slot_miron_model(model) -> bool:
    """True for word-slot MIRON checkpoints (3D input_ids)."""
    name = getattr(model, "__class__", type(model)).__name__
    return name in ("MIRON", "MironForCausalLM")


class BLIMPEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, config, device="cuda", data_dir="data/benchmarks/en/blimp"):
        super().__init__(model, tokenizer, config, device)
        self.data_dir = data_dir

    def evaluate(self, language="en", batch_size=32):
        if language != "en":
            return {
                "skipped": True,
                "reason": "BLiMP is English-only.",
            }

        manifest_path = os.path.join(self.data_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return {}

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        phenomena = manifest["phenomena"]
        results = {}
        all_correct = 0
        all_total = 0

        self.model.eval()
        is_miron = _is_word_slot_miron_model(self.model)

        for phenomenon in tqdm(
            phenomena,
            desc="BLiMP",
            unit="step",
            dynamic_ncols=True,
        ):
            file_path = os.path.join(self.data_dir, f"{phenomenon}.json")
            if not os.path.exists(file_path):
                continue

            phenomenon_results = self.evaluate_phenomenon(
                file_path,
                batch_size=batch_size,
                max_samples=self.benchmark_max_samples(),
                is_miron=is_miron,
            )
            results[phenomenon] = {
                "accuracy": round(phenomenon_results["accuracy"], 1),
                "correct": phenomenon_results["correct"],
                "total": phenomenon_results["total"],
                "detailed_results": phenomenon_results["detailed_results"],
            }
            all_correct += phenomenon_results["correct"]
            all_total += phenomenon_results["total"]

        results["total_accuracy"] = round((all_correct / all_total) * 100, 1) if all_total > 0 else 0.0

        return results

    def evaluate_phenomenon(self, file_path, batch_size=32, max_samples=None, is_miron=False):
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        items = reproducible_subsample(items, max_samples, self.eval_subsample_seed())
        correct = 0
        total = len(items)
        detailed_results = []

        for i in range(0, len(items), batch_size):
            batch_items = items[i : i + batch_size]
            good_sentences = [item["sentence_good"] for item in batch_items]
            bad_sentences = [item["sentence_bad"] for item in batch_items]

            good_losses = self.get_sentence_losses_batch(good_sentences, is_miron)
            bad_losses = self.get_sentence_losses_batch(bad_sentences, is_miron)

            for item, good_loss, bad_loss in zip(batch_items, good_losses, bad_losses):
                pair_ok = good_loss < bad_loss
                if pair_ok:
                    correct += 1
                detailed_results.append({
                    "sentence_good": item["sentence_good"],
                    "sentence_bad": item["sentence_bad"],
                    "loss_good": float(good_loss),
                    "loss_bad": float(bad_loss),
                    "correct": pair_ok,
                })

        accuracy = round((correct / total) * 100, 1) if total > 0 else 0.0
        return {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "detailed_results": detailed_results,
        }

    def get_sentence_loss(self, sentence):
        is_miron = _is_word_slot_miron_model(self.model)
        losses = self.get_sentence_losses_batch([sentence], is_miron)
        return losses[0]

    def get_sentence_losses_batch(self, sentences, is_miron):
        if is_miron:
            losses_list = []
            for sentence in sentences:
                out = self.text_tokenizer.encode(sentence, return_tensors=None, add_special_tokens=True)
                encoded = out["input_ids"][0] if isinstance(out, dict) else out
                if encoded and isinstance(encoded[0], list):
                    max_word_len = max(len(word) for word in encoded)
                    input_ids = torch.full(
                        (1, len(encoded), max_word_len),
                        self.text_tokenizer.pad_token_id,
                        dtype=torch.long,
                    )
                    for j, word in enumerate(encoded):
                        if word:
                            input_ids[0, j, : len(word)] = torch.tensor(word, dtype=torch.long)
                else:
                    input_ids = torch.tensor([encoded], dtype=torch.long)
                input_ids = input_ids.to(self.device)

                with torch.no_grad():
                    out = self.model(input_ids=input_ids, return_dict=True)
                    lt = out.get("total_loss", out.get("loss"))
                    if lt is None:
                        raise KeyError("model forward must return 'loss' or 'total_loss' for BLiMP")
                    losses_list.append(lt.item())
            return losses_list

        batch_ids = []
        for sentence in sentences:
            ids = self.tokenizer.encode(sentence, add_special_tokens=True)
            batch_ids.append(ids)

        max_len = max(len(ids) for ids in batch_ids)
        padded_batch = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.bool)
        for i, ids in enumerate(batch_ids):
            padded_batch[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = True

        padded_batch = padded_batch.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(padded_batch, attention_mask=attention_mask)
            logits = outputs["logits"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = padded_batch[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses = losses.view(len(batch_ids), -1)

            valid_lengths = shift_mask.sum(dim=1).float()
            valid_lengths = valid_lengths.clamp(min=1.0)

            sentence_losses = (losses * shift_mask.float()).sum(dim=1) / valid_lengths

            return sentence_losses.cpu().tolist()
