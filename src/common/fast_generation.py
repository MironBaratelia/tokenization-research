"""Fast local generation helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

PLATEAU_PROBE_POOL = 8
PLATEAU_SLOWDOWN_STOP_RATIO = 3.0
MAX_GEN_BATCH_CAP = 1024


def apply_inference_throughput_prefs(device: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    device_str = str(device)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        state["matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
        state["cudnn_tf32"] = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    state["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    state["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    return state


def restore_inference_prefs(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "matmul_tf32" in state:
        torch.backends.cuda.matmul.allow_tf32 = state["matmul_tf32"]
    if "cudnn_tf32" in state:
        torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
    torch.backends.cudnn.benchmark = state.get("cudnn_benchmark", False)
    torch.backends.cudnn.deterministic = state.get("cudnn_deterministic", True)


def _prompts_for_batch_size(all_prompts: List[str], bs: int) -> List[str]:
    if bs < 1 or not all_prompts:
        return []
    n = len(all_prompts)
    if bs <= n:
        return all_prompts[:bs]
    return [all_prompts[i % n] for i in range(bs)]


def _oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


def find_plateau_generation_batch_size(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    sample_prompts: List[str],
    *,
    max_new_tokens: int,
    max_batch_cap: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: float,
    seed: Optional[int],
    pool_k: int = PLATEAU_PROBE_POOL,
    slowdown_ratio: float = PLATEAU_SLOWDOWN_STOP_RATIO,
    suppress_eos_first_step: bool = False,
    warmup_rounds: int = 1,
    verbose: bool = True,
    max_context_tokens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    from src.common.inference import InferenceEngine, _set_seed

    cap = min(MAX_GEN_BATCH_CAP, max(1, int(max_batch_cap)))
    ratio = float(slowdown_ratio)
    meta: Dict[str, Any] = {
        "max_batch_cap": cap,
        "pool_k": int(pool_k),
        "slowdown_ratio": ratio,
        "sec_per_prompt": {},
    }

    pool = sample_prompts[: max(1, min(int(pool_k), len(sample_prompts)))]
    if not pool:
        return 1, meta

    engine = InferenceEngine(model, tokenizer, device=device)

    def sync() -> None:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_generate(bs: int) -> None:
        chunk = _prompts_for_batch_size(pool, bs)
        if len(chunk) != bs:
            raise RuntimeError("internal: prompt slice size mismatch")
        gen_kw: Dict[str, Any] = dict(
            batch_size=bs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            seed=seed,
            desc="",
            disable_tqdm=True,
            suppress_eos_first_step=suppress_eos_first_step,
        )
        if max_context_tokens is not None:
            gen_kw["max_context_tokens"] = int(max_context_tokens)
        engine.generate(chunk, **gen_kw)
        sync()

    if seed is not None:
        _set_seed(seed)
    wbs = min(4, len(pool), cap)
    for _ in range(max(1, warmup_rounds)):
        try:
            run_generate(max(1, wbs))
        except RuntimeError as e:
            if _oom(e) and device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif not _oom(e):
                raise

    def timed_run(bs: int) -> float:
        if seed is not None:
            _set_seed(seed)
        sync()
        t0 = time.perf_counter()
        run_generate(bs)
        sync()
        return (time.perf_counter() - t0) / max(bs, 1)

    def log_step(bs: int, sp: float) -> None:
        if not verbose:
            return
        pps = 1.0 / max(sp, 1e-12)
        print(
            f"[gen-batch probe] batch_size={bs}  sec_per_prompt={sp:.6f}  prompts_per_sec={pps:.2f}",
            flush=True,
        )

    chosen = 1
    try:
        sp_prev = timed_run(1)
    except RuntimeError as e:
        if _oom(e):
            meta["note"] = "batch_1_oom"
            return 1, meta
        raise
    meta["sec_per_prompt"][1] = sp_prev
    log_step(1, sp_prev)

    b = 2
    while True:
        if chosen >= cap:
            meta["stop_reason"] = "cap"
            break
        if b > cap:
            if chosen < cap:
                try:
                    sp = timed_run(cap)
                except RuntimeError as e:
                    if _oom(e):
                        if device.startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        meta["note"] = "oom_at_batch"
                        meta["oom_batch"] = cap
                        meta["stop_reason"] = "oom"
                        break
                    raise
                meta["sec_per_prompt"][cap] = sp
                log_step(cap, sp)
                if sp > ratio * sp_prev:
                    meta["stop_reason"] = "slowdown"
                    meta["rejected_batch"] = cap
                else:
                    chosen = cap
                    meta["stop_reason"] = "cap"
            else:
                meta["stop_reason"] = "cap"
            break

        try:
            sp = timed_run(b)
        except RuntimeError as e:
            if _oom(e):
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                meta["note"] = "oom_at_batch"
                meta["oom_batch"] = b
                meta["stop_reason"] = "oom"
                break
            raise
        meta["sec_per_prompt"][b] = sp
        log_step(b, sp)
        if sp > ratio * sp_prev:
            meta["stop_reason"] = "slowdown"
            meta["rejected_batch"] = b
            break
        chosen = b
        sp_prev = sp
        if b == cap:
            meta["stop_reason"] = "cap"
            break
        nxt = b * 2
        if nxt > cap:
            b = cap
        else:
            b = nxt

    meta["chosen_batch"] = chosen
    return chosen, meta


def find_max_feasible_generation_batch_size(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    sample_prompts: List[str],
    *,
    max_new_tokens: int,
    max_batch_cap: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: float,
    seed: Optional[int],
    suppress_eos_first_step: bool = False,
    warmup_rounds: int = 1,
    max_context_tokens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    from src.common.inference import InferenceEngine, _set_seed

    cap = min(MAX_GEN_BATCH_CAP, max(1, int(max_batch_cap)))
    meta: Dict[str, Any] = {"max_batch_cap": cap}

    if not sample_prompts:
        return min(8, cap), meta

    engine = InferenceEngine(model, tokenizer, device=device)

    def sync() -> None:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_generate(bs: int) -> None:
        chunk = _prompts_for_batch_size(sample_prompts, bs)
        if len(chunk) != bs:
            raise RuntimeError("internal: prompt slice size mismatch")
        gen_kw: Dict[str, Any] = dict(
            batch_size=bs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            seed=seed,
            desc="",
            disable_tqdm=True,
            suppress_eos_first_step=suppress_eos_first_step,
        )
        if max_context_tokens is not None:
            gen_kw["max_context_tokens"] = int(max_context_tokens)
        engine.generate(chunk, **gen_kw)
        sync()

    def try_bs(bs: int) -> bool:
        try:
            if seed is not None:
                _set_seed(seed)
            sync()
            run_generate(bs)
            return True
        except RuntimeError as e:
            if _oom(e):
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            raise

    def refine(lo: int, hi_fail: int) -> int:
        while lo + 1 < hi_fail:
            mid = (lo + hi_fail) // 2
            if try_bs(mid):
                lo = mid
            else:
                hi_fail = mid
        return lo

    if seed is not None:
        _set_seed(seed)
    wbs = min(8, len(sample_prompts), cap)
    for _ in range(max(1, warmup_rounds)):
        try:
            run_generate(max(1, wbs))
        except RuntimeError as e:
            if _oom(e):
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

    if not try_bs(1):
        meta["max_batch"] = 1
        meta["note"] = "batch_1_failed"
        return 1, meta

    last_ok = 1
    b = 2
    while True:
        if last_ok >= cap:
            break
        if b > cap:
            if try_bs(cap):
                last_ok = cap
            else:
                last_ok = refine(last_ok, cap)
            break

        if try_bs(b):
            last_ok = b
            if b == cap:
                break
            nxt = b * 2
            if nxt > cap:
                if try_bs(cap):
                    last_ok = cap
                else:
                    last_ok = refine(last_ok, cap)
                break
            b = nxt
        else:
            last_ok = refine(last_ok, b)
            break

    meta["max_batch"] = last_ok
    return last_ok, meta
