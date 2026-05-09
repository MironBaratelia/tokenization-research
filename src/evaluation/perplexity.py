import inspect
import math
from collections import Counter

import torch
import torch.nn as nn

from src.common.math_utils import safe_exp
from src.common.tqdm_utils import tqdm

from .base_evaluator import BaseEvaluator


def _decode_ids_preserve_specials(tokenizer, ids):
    """HF tokenizers default skip_special_tokens=True -> BOS/EOS decode to ''; use False for length stats."""
    dec = tokenizer.decode
    try:
        if "skip_special_tokens" in inspect.signature(dec).parameters:
            return dec(ids, skip_special_tokens=False)
    except (TypeError, ValueError):
        pass
    return dec(ids)


class PerplexityEvaluator(BaseEvaluator):
    def evaluate(self, data_loader):
        total_loss, total_tokens, total_chars = 0.0, 0, 0
        oov_tokens, total_tokens_in_corpus = 0, 0
        token_counts = Counter()
        text_token_lengths = []

        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                desc="Perplexity",
                unit="batch",
                dynamic_ncols=True,
            ):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Standard token-level LM path.
                if input_ids.dim() == 2:
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs["logits"]

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    total_loss += loss.item()

                    valid_mask = shift_labels != -100
                    total_tokens += valid_mask.sum().item()

                    batch_ids_cpu = shift_labels.cpu()
                    valid_mask_cpu = valid_mask.cpu()
                    unk_id = getattr(self.tokenizer, "unk_token_id", None)

                    for i in range(batch_ids_cpu.size(0)):
                        m = valid_mask_cpu[i]
                        if not m.any():
                            continue
                        ids = batch_ids_cpu[i][m].tolist()

                        for val in ids:
                            token_counts[val] += 1
                            text_token_lengths.append(
                                len(_decode_ids_preserve_specials(self.tokenizer, [val]))
                            )

                        text = _decode_ids_preserve_specials(self.tokenizer, ids)
                        total_chars += len(text)

                        if unk_id is not None:
                            oov_tokens += ids.count(unk_id)
                            total_tokens_in_corpus += len(ids)
                    continue

                # MIRON path: word-slot character ids.
                if input_ids.dim() != 3:
                    raise ValueError(
                        f"PerplexityEvaluator expected input_ids dim 2 or 3, got {tuple(input_ids.shape)}"
                    )

                outputs = self.model(input_ids=input_ids)
                loss = outputs.get("loss")
                if loss is None:
                    raise KeyError(
                        "MIRON model output has no 'loss'; expected MironForCausalLM-style dict."
                    )

                # Match the valid-character count used inside the MIRON loss.
                pad_id = getattr(self.model, "pad_id", getattr(self.tokenizer, "pad_token_id", 0))
                bow_id = getattr(self.model, "bow_id", None)

                enc_in = input_ids[:, :-1, :].contiguous()
                target_words = input_ids[:, 1:, :].contiguous()
                B, Tm, _L = enc_in.shape

                bow_valid = 0
                if bow_id is not None and int(bow_id) != int(pad_id):
                    bow_valid = B * Tm
                valid_chars = (target_words != int(pad_id)).sum().item() + bow_valid

                # Convert mean loss back to corpus-level sum.
                total_loss += float(loss.item()) * float(valid_chars)
                total_tokens += int(valid_chars)

                # Best-effort char statistics for stable output keys.
                try:
                    flat = target_words.reshape(-1)
                    flat = flat[flat != int(pad_id)].detach().cpu().tolist()
                    if flat:
                        for val in flat:
                            token_counts[int(val)] += 1
                            text_token_lengths.append(
                                len(_decode_ids_preserve_specials(self.tokenizer, [int(val)]))
                            )
                        text = _decode_ids_preserve_specials(self.tokenizer, [int(v) for v in flat])
                        total_chars += len(text)
                except Exception:
                    # Fallback when char ids are not directly decodable.
                    total_chars += int(valid_chars)

        vocab_lens = [
            len(_decode_ids_preserve_specials(self.tokenizer, [i]))
            for i in range(self.tokenizer.vocab_size)
        ]

        _LEN_HIST_LEGEND = (
            "Keys n_chars=k: count of tokens whose decode([id], skip_special_tokens=False) "
            "is a Unicode string of length k. n_chars=0 only for truly empty decode."
        )

        def get_dist(lens):
            c = Counter(lens)
            return {f"n_chars={k}": c[k] for k in sorted(c.keys())}

        if total_tokens == 0:
            return {"ppl": float("inf"), "char_ppl": float("inf")}

        char_ppl = safe_exp(total_loss / total_chars) if total_chars > 0 else float("inf")

        return {
            "ppl": safe_exp(total_loss / total_tokens),
            "char_ppl": char_ppl,
            "compression_ratio": total_chars / total_tokens,
            "oov_rate": oov_tokens / total_tokens_in_corpus if total_tokens_in_corpus > 0 else 0.0,
            "vocab_utilization": len(token_counts) / self.tokenizer.vocab_size,
            "avg_token_length_vocab": sum(vocab_lens) / len(vocab_lens),
            "avg_token_length_text": sum(text_token_lengths) / len(text_token_lengths),
            "token_length_histogram_legend": _LEN_HIST_LEGEND,
            "token_length_dist_vocab": get_dist(vocab_lens),
            "token_length_dist_text": get_dist(text_token_lengths),
        }
