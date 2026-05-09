import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from src.common.tqdm_utils import tqdm

from src.common.continuation_decode import decode_continuation, trim_generated_ids
from src.common.fast_generation import apply_inference_throughput_prefs, restore_inference_prefs
from src.common.tokenizer_utils import UniversalTokenizer, get_prefix_lengths_from_offsets

_GEN_KW_EXCLUDE = frozenset(
    {
        "bias_against_first_tokens",
        "suppress_eos_first_step",
        "add_bos",
        "max_context_tokens",
        "device",
    }
)


def _miron_cuda_fast_inference_enabled() -> bool:
    """MIRON on GPU: fast matmul (AMP+TF32) is default; set MIRON_INFERENCE_STRICT=1 for fp32 regression runs."""
    return os.environ.get("MIRON_INFERENCE_STRICT", "").lower() not in (
        "1",
        "true",
        "yes",
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = "cuda",
        miron_strict: bool = False,
    ):
        self.model = model
        self.tokenizer = UniversalTokenizer(tokenizer)
        self.device = device
        self.miron_strict = bool(miron_strict)
        self.is_miron = bool(
            self.tokenizer.is_miron and callable(getattr(model, "encode_batch", None))
        )

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model, "prepare_for_inference"):
            self.model.prepare_for_inference()

    def _miron_cuda_fast_path(self) -> bool:
        """MIRON on CUDA (other models unchanged)."""
        return (
            bool(getattr(self.tokenizer, "is_miron", False))
            and str(self.device).startswith("cuda")
            and not self.miron_strict
        )

    def _generate_chunk_decoded_and_id_pairs(
        self,
        chunk: Sequence[str],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        pad_id: int,
        **kwargs: Any,
    ) -> Tuple[List[str], List[Tuple[List[int], List[int]]]]:
        add_bos = kwargs.get("add_bos", True)
        max_ctx = kwargs.get("max_context_tokens")
        # MIRON word-slot batches should stay right-padded; left padding is only needed
        # for token-id RoPE models such as SmolLM2.
        pad_side = "right" if self.is_miron else "left"
        inputs = self.tokenizer.encode(
            list(chunk),
            return_tensors="pt",
            padding=True,
            padding_side=pad_side,
            add_special_tokens=False,
            add_bos_only=add_bos,
            truncation=max_ctx is not None,
            max_length=max_ctx,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        width = input_ids.size(1)
        model_kwargs = {k: v for k, v in kwargs.items() if k not in _GEN_KW_EXCLUDE}
        self._apply_first_step_logits_overrides(model_kwargs, kwargs)
        row_seeds = kwargs.get("row_seeds")
        if row_seeds is not None:
            model_kwargs["row_seeds"] = row_seeds

        return self._generate_encoded_batch_and_pairs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            width=width,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_id=pad_id,
            **model_kwargs,
        )

    def _generate_chunk_miron_length_bucketed(
        self,
        chunk: Sequence[str],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        pad_id: int,
        **kwargs: Any,
    ) -> Tuple[List[str], List[Tuple[List[int], List[int]]]]:
        add_bos = kwargs.get("add_bos", True)
        max_ctx = kwargs.get("max_context_tokens")
        indexed: List[Tuple[int, str, int]] = []
        for idx, prompt in enumerate(chunk):
            enc = self.tokenizer.encode(
                [prompt],
                return_tensors="pt",
                padding=False,
                padding_side="right",
                add_special_tokens=False,
                add_bos_only=add_bos,
                truncation=max_ctx is not None,
                max_length=max_ctx,
            )
            seqlen = int(enc["attention_mask"][0].sum().item())
            indexed.append((idx, prompt, seqlen))

        buckets: Dict[int, List[Tuple[int, str]]] = {}
        for idx, prompt, seqlen in indexed:
            buckets.setdefault(seqlen, []).append((idx, prompt))

        out_texts: List[Optional[str]] = [None] * len(chunk)
        out_pairs: List[Optional[Tuple[List[int], List[int]]]] = [None] * len(chunk)
        model_kwargs = {k: v for k, v in kwargs.items() if k not in _GEN_KW_EXCLUDE}
        self._apply_first_step_logits_overrides(model_kwargs, kwargs)

        for _, items in sorted(buckets.items(), key=lambda kv: kv[0]):
            prompts = [p for _, p in items]
            bucket_kwargs = dict(model_kwargs)
            row_seeds = kwargs.get("row_seeds")
            if row_seeds is not None:
                bucket_kwargs["row_seeds"] = [row_seeds[orig_idx] for orig_idx, _ in items]
            enc = self.tokenizer.encode(
                prompts,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
                add_bos_only=add_bos,
                truncation=max_ctx is not None,
                max_length=max_ctx,
            )
            texts, pairs = self._generate_encoded_batch_and_pairs(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
                width=enc["input_ids"].size(1),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_id=pad_id,
                **bucket_kwargs,
            )
            for (orig_idx, _), text, pair in zip(items, texts, pairs):
                out_texts[orig_idx] = text
                out_pairs[orig_idx] = pair

        return [t or "" for t in out_texts], [p or ([], []) for p in out_pairs]

    def _generate_encoded_batch_and_pairs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        width: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        pad_id: int,
        **model_kwargs: Any,
    ) -> Tuple[List[str], List[Tuple[List[int], List[int]]]]:
        pad_token_id = model_kwargs.pop("pad_token_id", self.tokenizer.pad_token_id)
        eos_token_id = model_kwargs.pop("eos_token_id", self.tokenizer.eos_token_id)
        # SmolLM2 token-id batches use left padding and need RoPE correction on prefill.
        # MIRON word-slot batches should remain right-padded and must not enable this path.
        prefill_left = not self.is_miron

        use_amp = (
            self._miron_cuda_fast_path()
            and torch.cuda.is_available()
            and _miron_cuda_fast_inference_enabled()
        )
        with torch.inference_mode():
            if use_amp:
                amp_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        prefill_left_padded=prefill_left,
                        **model_kwargs,
                    )
            else:
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    prefill_left_padded=prefill_left,
                    **model_kwargs,
                )

        tail = output_ids[:, width:]
        mask_bool = attention_mask.to(torch.bool)
        texts: List[str] = []
        pairs: List[Tuple[List[int], List[int]]] = []
        for r in range(tail.size(0)):
            gen_ids = trim_generated_ids(tail[r].tolist(), pad_id)
            prefix_ids = input_ids[r][mask_bool[r]].tolist()
            pairs.append((prefix_ids, list(gen_ids)))
            texts.append(decode_continuation(self.tokenizer, prefix_ids, gen_ids))
        return texts, pairs

    def generate_from_encoded(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        texts, _ = self.generate_from_encoded_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            seed=seed,
            **kwargs,
        )
        return texts[0] if texts else ""

    def generate_from_encoded_batch(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[List[str], List[Tuple[List[int], List[int]]]]:
        if seed is not None:
            _set_seed(seed)

        if attention_mask is None:
            if input_ids.dim() == 3:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).any(dim=-1).long()
            else:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        pad_id = self.tokenizer.pad_token_id or 0
        model_kwargs = {k: v for k, v in kwargs.items() if k not in _GEN_KW_EXCLUDE}
        self._apply_first_step_logits_overrides(model_kwargs, kwargs)
        row_seeds = kwargs.get("row_seeds")
        if row_seeds is not None:
            model_kwargs["row_seeds"] = row_seeds

        with torch.inference_mode():
            return self._generate_encoded_batch_and_pairs(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                width=input_ids.size(1),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_id=pad_id,
                **model_kwargs,
            )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        batch_size: int = 8,
        do_sample: bool = True,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
        desc: str = "Generating",
        **kwargs,
    ) -> List[str]:
        texts, _ = self.generate_with_prefix_gen_id_pairs(
            prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            seed=seed,
            desc=desc,
            **kwargs,
        )
        return texts

    def generate_with_prefix_gen_id_pairs(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        batch_size: int = 8,
        do_sample: bool = True,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
        desc: str = "Generating",
        **kwargs,
    ) -> Tuple[List[str], List[Tuple[List[int], List[int]]]]:
        if seed is not None:
            _set_seed(seed)

        pad_id = self.tokenizer.pad_token_id or 0
        all_texts: List[str] = []
        all_pairs: List[Tuple[List[int], List[int]]] = []

        disable_tqdm = kwargs.pop("disable_tqdm", None)
        tqdm_disable = (len(prompts) <= batch_size) if disable_tqdm is None else bool(disable_tqdm)

        throughput_prefs: Dict[str, Any] = {}
        if (
            self._miron_cuda_fast_path()
            and torch.cuda.is_available()
            and _miron_cuda_fast_inference_enabled()
        ):
            throughput_prefs = apply_inference_throughput_prefs(self.device)
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        try:
            for i in tqdm(
                range(0, len(prompts), batch_size),
                desc=desc,
                unit="batch",
                dynamic_ncols=True,
                disable=tqdm_disable,
            ):
                chunk = prompts[i : i + batch_size]
                chunk_kwargs = dict(kwargs)
                if self.is_miron and seed is not None:
                    chunk_kwargs["row_seeds"] = [
                        int(seed) + row_idx for row_idx in range(i, i + len(chunk))
                    ]
                if self.is_miron:
                    texts, pairs = self._generate_chunk_miron_length_bucketed(
                        chunk,
                        max_new_tokens,
                        do_sample,
                        temperature,
                        repetition_penalty,
                        pad_id,
                        **chunk_kwargs,
                    )
                else:
                    texts, pairs = self._generate_chunk_decoded_and_id_pairs(
                        chunk,
                        max_new_tokens,
                        do_sample,
                        temperature,
                        repetition_penalty,
                        pad_id,
                        **chunk_kwargs,
                    )
                all_texts.extend(texts)
                all_pairs.extend(pairs)
        finally:
            if throughput_prefs:
                restore_inference_prefs(throughput_prefs)

        return all_texts, all_pairs

    def _apply_first_step_logits_overrides(self, model_kwargs: Dict[str, Any], call_kwargs: Dict[str, Any]) -> None:
        bias_src = call_kwargs.get("bias_against_first_tokens")
        if bias_src:
            collected: List[int] = []
            items = bias_src if isinstance(bias_src, (list, tuple)) else [bias_src]
            for s in items:
                collected.extend(self.tokenizer.get_token_ids(str(s)))
            if collected:
                model_kwargs["bias_against_first_step"] = list(dict.fromkeys(collected))
        if call_kwargs.get("suppress_eos_first_step") and self.tokenizer.eos_token_id is not None:
            step_ids = list(model_kwargs.get("bias_against_first_step") or [])
            eos_id = self.tokenizer.eos_token_id
            if eos_id not in step_ids:
                step_ids.append(eos_id)
            model_kwargs["bias_against_first_step"] = step_ids

    def score_continuations(
        self,
        prompts: List[str],
        targets: List[str],
        batch_size: int = 8,
        progress_desc: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        def _join(p, t):
            p, t = p.rstrip(), t.lstrip()
            if p and t and not p.endswith(" ") and not t.startswith(" "):
                return p + " " + t
            return p + t

        results = []
        n = len(prompts)
        starts = range(0, n, batch_size)
        if progress_desc is not None and n > batch_size:
            starts = tqdm(
                starts,
                desc=progress_desc,
                unit="batch",
                dynamic_ncols=True,
            )
        for i in starts:
            b_prompts = prompts[i : i + batch_size]
            b_targets = targets[i : i + batch_size]

            combined = [_join(p, t) for p, t in zip(b_prompts, b_targets)]
            prefix_char_ends = [
                len(p.rstrip()) + (1 if t.lstrip() and not p.rstrip().endswith(" ") else 0)
                for p, t in zip(b_prompts, b_targets)
            ]

            prefix_lens = get_prefix_lengths_from_offsets(
                self.tokenizer, combined, prefix_char_ends, add_bos=True
            )
            if prefix_lens is None:
                if self.is_miron:
                    prefix_texts = [p.rstrip() for p in b_prompts]
                else:
                    prefix_texts = [p.rstrip() + (" " if t.lstrip() and not p.rstrip().endswith(" ") else "") for p, t in zip(b_prompts, b_targets)]
                
                inputs_prefix = self.tokenizer.encode(prefix_texts, return_tensors="pt", padding=True, add_special_tokens=False, add_bos_only=True)
                if self.is_miron:
                    prefix_lens = inputs_prefix["attention_mask"].sum(dim=1)
                else:
                    prefix_lens = (inputs_prefix["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)

            if isinstance(prefix_lens, list):
                prefix_lens = torch.tensor(prefix_lens, dtype=torch.long)

            inputs_full = self.tokenizer.encode(combined, return_tensors="pt", padding=True, add_special_tokens=False, add_bos_only=True)

            ids_full = inputs_full["input_ids"].to(self.device)
            mask_full = inputs_full["attention_mask"].to(self.device)

            with torch.no_grad():
                if self.is_miron:
                    if ids_full.size(1) < 2:
                        results.extend(
                            {"confidence": 0.0, "logprob": float("-inf")}
                            for _ in range(ids_full.size(0))
                        )
                        continue

                    enc_in = ids_full[:, :-1, :].contiguous()
                    target_words = ids_full[:, 1:, :].contiguous()

                    z_all = self.model.encode_batch(enc_in.reshape(-1, enc_in.size(-1)))
                    z_all = z_all.reshape(enc_in.size(0), enc_in.size(1), -1)

                    lm_out = self.model.lm(inputs_embeds=z_all, use_cache=False)
                    h = lm_out["hidden_states"]

                    h_char = self.model._decode_context(h)

                    B, Tm, _ = target_words.shape
                    bow = torch.full(
                        (B, Tm, 1),
                        self.model.bow_id,
                        device=ids_full.device,
                        dtype=ids_full.dtype,
                    )
                    full_target = torch.cat([bow, target_words], dim=-1)

                    dec_input = full_target[:, :, :-1]
                    dec_target = full_target[:, :, 1:]

                    logits = self.model.char_decoder(h_char, dec_input)
                    log_probs = F.log_softmax(logits, dim=-1)

                    for b in range(B):
                        plen = prefix_lens[b].item() if prefix_lens.dim() > 0 else prefix_lens.item()
                        valid_words = int((ids_full[b] != self.model.pad_id).any(dim=-1).sum().item())

                        if plen <= 0 or plen >= valid_words:
                            results.append({"confidence": 0.0, "logprob": float("-inf")})
                            continue

                        first_target_idx = plen - 1
                        target_lps = []
                        for t_idx in range(first_target_idx, valid_words - 1):
                            word_target_chars = dec_target[b, t_idx]
                            word_log_probs = log_probs[b, t_idx]

                            valid_mask = word_target_chars != self.model.pad_id
                            if int(valid_mask.sum().item()) == 0:
                                continue

                            char_lps = word_log_probs.gather(1, word_target_chars.unsqueeze(1)).squeeze(1)
                            target_lps.extend(char_lps[valid_mask].tolist())

                        if not target_lps:
                            results.append({"confidence": 0.0, "logprob": float("-inf")})
                        else:
                            avg_lp = sum(target_lps) / len(target_lps)
                            conf = math.exp(avg_lp) * 100.0
                            if not math.isfinite(conf):
                                conf = 0.0
                            results.append({"confidence": conf, "logprob": avg_lp})
                else:
                    outputs = self.model(ids_full, attention_mask=mask_full)
                    logits = outputs["logits"]

                    for b in range(ids_full.size(0)):
                        plen = prefix_lens[b].item() if prefix_lens.dim() > 0 else prefix_lens.item()
                        if plen <= 0 or plen >= ids_full.size(1):
                            results.append({"confidence": 0.0, "logprob": float("-inf")})
                            continue
                        shift_logits_b = logits[b, plen - 1 : -1, :].unsqueeze(0)
                        target_ids_b = ids_full[b, plen:].unsqueeze(0)
                        log_probs_b = F.log_softmax(shift_logits_b, dim=-1)
                        target_log_probs = log_probs_b.gather(2, target_ids_b.unsqueeze(-1)).squeeze(-1)
                        target_mask_b = (target_ids_b != self.tokenizer.pad_token_id).float()
                        num_valid = target_mask_b.sum().item()
                        if num_valid < 1:
                            results.append({"confidence": 0.0, "logprob": float("-inf")})
                            continue
                        avg_lp = (target_log_probs * target_mask_b).sum().item() / num_valid
                        conf = math.exp(avg_lp) * 100.0
                        if not math.isfinite(conf):
                            conf = 0.0
                        results.append({"confidence": conf, "logprob": avg_lp})

        return results


def limits_for_total_window_generate(
    tokenizer: Any,
    prompts: Sequence[str],
    total_tokens: int,
) -> Tuple[int, int]:
    total = max(2, int(total_tokens))
    prefill_cap = total - 1
    ut = UniversalTokenizer(tokenizer)
    pad_side = "right" if ut.is_miron else "left"
    max_plen = 0
    for p in prompts:
        enc = ut.encode(
            [p],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
            add_bos_only=True,
            truncation=True,
            max_length=prefill_cap,
            padding_side=pad_side,
        )
        mask = enc.get("attention_mask")
        if mask is not None:
            plen = int(mask[0].sum().item())
        else:
            plen = int(enc["input_ids"].shape[1])
        max_plen = max(max_plen, plen)
    max_new = max(1, total - max_plen)
    return max_new, prefill_cap


def batched_generate(model, tokenizer, prompts, **kwargs):
    device = kwargs.pop("device", "cuda")
    miron_strict = bool(kwargs.pop("miron_strict", False))
    engine = InferenceEngine(model, tokenizer, device=device, miron_strict=miron_strict)
    return engine.generate(prompts, **kwargs)
