from __future__ import annotations

import time
from math import ceil
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_tokenized_generation_speed(
    engine: Any,
    encoded_prompts: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    *,
    runs: int = 5,
    warmup_runs: int = 2,
    seed: Optional[int] = None,
    max_new_tokens: int = 32,
    target_generated_steps_per_run: int = 4096,
    do_sample: bool = False,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
    **gen_kwargs: Any,
) -> Dict[str, Any]:
    if not encoded_prompts:
        return {}

    prompts_per_pass = len(encoded_prompts)
    nominal_steps_per_pass = max(1, prompts_per_pass * max(1, int(max_new_tokens)))
    repeat_factor = max(
        1,
        int(
            ceil(
                max(1, int(target_generated_steps_per_run))
                / float(nominal_steps_per_pass)
            )
        ),
    )
    repeated_prompts = list(encoded_prompts) * repeat_factor

    def _run_once() -> Dict[str, Any]:
        _sync(device)
        t0 = time.perf_counter()
        generated_chars = 0
        generated_steps = 0
        for input_ids, attention_mask in repeated_prompts:
            texts, pairs = engine.generate_from_encoded_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                seed=seed,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
                **gen_kwargs,
            )
            text = texts[0] if texts else ""
            generated_chars += len(text or "")
            if pairs:
                _, gen_ids = pairs[0]
                generated_steps += len(gen_ids)
        _sync(device)
        elapsed = time.perf_counter() - t0
        cps = generated_chars / elapsed if elapsed > 0 else 0.0
        sps = generated_steps / elapsed if elapsed > 0 else 0.0
        cps_per_step = generated_chars / generated_steps if generated_steps > 0 else 0.0
        return {
            "elapsed_sec": elapsed,
            "generated_chars": generated_chars,
            "generated_steps": generated_steps,
            "chars_per_sec": cps,
            "steps_per_sec": sps,
            "chars_per_step": cps_per_step,
        }

    for _ in range(max(0, warmup_runs)):
        _run_once()

    stats = [_run_once() for _ in range(max(1, runs))]
    cps_values = [r["chars_per_sec"] for r in stats]
    sps_values = [r["steps_per_sec"] for r in stats]
    cps_step_values = [r["chars_per_step"] for r in stats]

    med_cps = median(cps_values)
    mad_cps = median(abs(v - med_cps) for v in cps_values)
    med_sps = median(sps_values)
    mad_sps = median(abs(v - med_sps) for v in sps_values)
    med_elapsed = median(r["elapsed_sec"] for r in stats)
    med_chars = median(r["generated_chars"] for r in stats)
    med_steps = median(r["generated_steps"] for r in stats)

    return {
        "generation_chars_per_sec_median": med_cps,
        "generation_chars_per_sec_mean": sum(cps_values) / len(cps_values),
        "generation_chars_per_sec_min": min(cps_values),
        "generation_chars_per_sec_max": max(cps_values),
        "generation_chars_per_sec_mad": mad_cps,
        "generation_chars_per_sec_rel_mad": (mad_cps / med_cps) if med_cps > 0 else 0.0,
        "generation_steps_per_sec_median": med_sps,
        "generation_steps_per_sec_mean": sum(sps_values) / len(sps_values),
        "generation_steps_per_sec_min": min(sps_values),
        "generation_steps_per_sec_max": max(sps_values),
        "generation_steps_per_sec_mad": mad_sps,
        "generation_steps_per_sec_rel_mad": (mad_sps / med_sps) if med_sps > 0 else 0.0,
        "generation_chars_per_step_mean": sum(cps_step_values) / len(cps_step_values),
        "generation_speed_elapsed_sec_median": med_elapsed,
        "generation_speed_generated_chars_median": med_chars,
        "generation_speed_generated_steps_median": med_steps,
        "generation_speed_prompt_passes_per_run": repeat_factor,
        "generation_speed_nominal_steps_per_prompt_pass": nominal_steps_per_pass,
        "generation_speed_target_generated_steps_per_run": int(
            target_generated_steps_per_run
        ),
        "generation_speed_runs": stats,
    }


def build_tokenized_prompt_batches(
    engine: Any,
    prompts: Sequence[str],
    *,
    batch_size: int = 1,
    max_context_tokens: Optional[int] = None,
    add_bos_only: bool = True,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    if not prompts:
        return []

    batch_size = max(1, int(batch_size))
    encoded_batches: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i : i + batch_size])
        enc = engine.tokenizer.encode(
            chunk,
            return_tensors="pt",
            padding=(len(chunk) > 1),
            add_special_tokens=False,
            add_bos_only=add_bos_only,
            truncation=max_context_tokens is not None,
            max_length=max_context_tokens,
            padding_side="left",
        )
        encoded_batches.append((enc["input_ids"], enc.get("attention_mask")))
    return encoded_batches
