#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.io.config import get_full_config
from src.common.device_utils import get_device_string
from src.common.miron_eval_model import build_miron_causal_lm
from src.models.smollm2 import SmolLM2Config, SmolLM2Model
from src.datasets.lm_dataset import LanguageModelingDataset, PlainTextLMDataset
from project_tokenizers.implementations import TOKENIZER_REGISTRY
from src.evaluation import (
    PerplexityEvaluator,
    MIRONEvaluator,
    LAMBADAEvaluator,
    CanaryEvaluator,
    OracleEvaluator,
    HumanEvalEvaluator,
    ExternalModelEvaluator,
    GenerationSpeedEvaluator,
    BLIMPEvaluator,
)
from src.scripts.utils.evaluation import build_training_summary


class HFModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, **kwargs):
        outputs = self.model(input_ids, **kwargs)
        return {"logits": outputs.logits, "loss": outputs.loss if hasattr(outputs, "loss") else None}
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


class HFTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.unk_token_id = tokenizer.unk_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.special_tokens = {
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'bos_token_id': tokenizer.bos_token_id
        }
    
    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Find the best available checkpoint."""
    for name in ["last_checkpoint.pt", "best_model.pt"]:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            return path
    
    if not os.path.isdir(ckpt_dir):
        return None
    
    import glob
    step_files = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not step_files:
        return None
    
    def step_num(p):
        try:
            return int(os.path.basename(p).replace("step_", "").replace(".pt", ""))
        except ValueError:
            return 0
    
    return max(step_files, key=step_num)


def load_model_and_tokenizer(experiment_id: str, config: dict, device: str):
    """Load model and tokenizer for evaluation."""
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_type = tokenizer_config.get("type")
    tokenizer_cls = TOKENIZER_REGISTRY.get(tokenizer_type)
    
    if not tokenizer_cls:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
    tokenizer = tokenizer_cls(tokenizer_config)
    tokenizer_id = tokenizer_config.get("id") or experiment_id
    tokenizer_path = os.path.join(project_root, f"project_tokenizers/trained/{tokenizer_id}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path} "
            f"(tokenizer.id={tokenizer_config.get('id')!r}, experiment.id={experiment_id!r})"
        )
    tokenizer.load(tokenizer_path)

    if tokenizer_type == "miron":
        model = build_miron_causal_lm(tokenizer, config)
    else:
        model_config = SmolLM2Config(
            vocab_size=tokenizer.vocab_size,
            **config.get("model", {})
        )
        model = SmolLM2Model(model_config)
    
    ckpt_dir = os.path.join(project_root, f"outputs/{experiment_id}/checkpoints")
    checkpoint_path = find_checkpoint(ckpt_dir)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state_dict[nk] = v

        if tokenizer_type == "neural_segmenter":
            seg_sd = {
                k[len("segmenter.") :]: v
                for k, v in new_state_dict.items()
                if k.startswith("segmenter.")
            }
            lm_sd = {k[len("lm.") :]: v for k, v in new_state_dict.items() if k.startswith("lm.")}
            seg_m = getattr(tokenizer, "segmenter_model", None)
            if seg_sd and seg_m is not None:
                seg_m.load_state_dict(seg_sd, strict=False)
            if lm_sd:
                model.load_state_dict(lm_sd, strict=False)
        else:
            model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()

    if tokenizer_type == "miron" and os.environ.get("MIRON_TORCH_COMPILE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, dynamic=True)  # type: ignore[assignment]
                logging.getLogger(__name__).info(
                    "MIRON: torch.compile enabled (first eval step may be slow)"
                )
            except Exception as e:
                logging.getLogger(__name__).warning("MIRON: torch.compile skipped: %s", e)

    return model, tokenizer


def run_evaluator(evaluator_cls, model, tokenizer, config, device: str, eval_name: str, 
                  language: str, batch_size: int, results_dir: str):
    """Run a single evaluator and save results."""
    evaluator = evaluator_cls(model, tokenizer, config, device=device)
    
    eval_map = {
        "perplexity": _run_perplexity_eval,
        "miron": lambda: evaluator.evaluate(language=language, batch_size=batch_size),
        "lambada": lambda: evaluator.evaluate(language=language, batch_size=batch_size),
        "canary": lambda: evaluator.evaluate(language=language, batch_size=batch_size),
        "oracle": lambda: evaluator.evaluate(language=language, batch_size=batch_size),
        "human_eval": lambda: evaluator.evaluate(language=language, batch_size=batch_size),
        "external_model": lambda: evaluator.evaluate(language=language, batch_size=batch_size),
        "blimp": lambda: evaluator.evaluate(language=language, batch_size=batch_size),
    }
    
    results = eval_map[eval_name]()
    output_path = os.path.join(results_dir, f"{eval_name}_results.json")
    evaluator.save_results(results, output_path)
    
    return results


def _perplexity_context_length(config: dict) -> int:
    m = config.get("model") or {}
    ctx = m.get("max_position_embeddings")
    if ctx is not None:
        return int(ctx)
    tr = config.get("training") or {}
    if tr.get("context_length") is not None:
        return int(tr["context_length"])
    return 512


def _resolve_perplexity_txt_path(config: Optional[dict], language: str, split_name: str) -> str:
    key_map = {
        "test_indomain": "test_indomain_file",
        "ood_science": "ood_science_file",
        "ood_fiction": "ood_fiction_file",
    }
    cfg_key = key_map.get(split_name)
    if config and cfg_key:
        p = (config.get("data") or {}).get(cfg_key)
        if p and os.path.isfile(p):
            return p
    return os.path.join(project_root, f"data/base/{language}/{split_name}.txt")


def _ensure_perplexity_packed_like_val(
    raw_txt_path: str,
    packed_base: str,
    tokenizer,
    config: dict,
    context_length: int,
    logger: logging.Logger,
) -> bool:
    """Create *_packed.arrow next to experiment tokenized/ using the same rules as val in preprocess (without editing 02_preprocess_data)."""
    manifest = packed_base + ".manifest.json"
    if os.path.isfile(manifest):
        return True
    tcfg = config.get("tokenizer") or {}
    tokenizer_type = tcfg.get("type")
    if not tokenizer_type:
        return False
    resolved = config.get("resolved_paths") or {}
    troot = resolved.get("tokenizers_dir") or os.path.join(project_root, "project_tokenizers", "trained")
    exp_id = (config.get("experiment") or {}).get("id") or ""
    tokenizer_path = os.path.normpath(os.path.join(troot, str(exp_id)))

    os.makedirs(os.path.dirname(packed_base), exist_ok=True)
    from src.preprocessing.packing import smart_pack_dataset, smart_pack_dataset_lazy

    logger.info(
        "Perplexity: packing %s like validation -> %s",
        os.path.basename(raw_txt_path),
        os.path.basename(packed_base),
    )
    try:
        if tokenizer_type in ("miron", "char_level"):
            smart_pack_dataset_lazy(raw_txt_path, tokenizer, context_length, packed_base)
        else:
            smart_pack_dataset(
                raw_txt_path,
                tokenizer,
                context_length,
                packed_base,
                num_workers=0,
                tokenizer_type=tokenizer_type,
                tokenizer_path=tokenizer_path,
            )
    except Exception as exc:
        logger.warning("Perplexity: val-style pack failed (%s); falling back to line-wise text.", exc)
        return False
    return os.path.isfile(manifest)


def _run_perplexity_eval(
    evaluator,
    language: str,
    batch_size: int,
    config: dict,
    results_dir: str,
    logger: logging.Logger,
):
    """Full-corpus perplexity: packed splits like val (outputs/.../tokenized/*_packed.arrow) when present."""
    split_names = ("test_indomain", "ood_science", "ood_fiction")
    tokenized_dir = os.path.join(os.path.dirname(results_dir), "tokenized")
    context_length = _perplexity_context_length(config)
    raw_tok = evaluator.tokenizer
    from src.datasets.lm_dataset import DataCollator

    pad_id = getattr(raw_tok, "pad_token_id", None) or 0
    collate = DataCollator(
        pad_id=pad_id,
        max_seq_len=context_length,
        max_word_len=getattr(raw_tok, "max_word_length", 32),
    )

    bench = config.get("benchmarks") or {}
    debug_cap = bench.get("max_samples") if config.get("debug") else None

    results = {}
    for name in split_names:
        path = _resolve_perplexity_txt_path(config, language, name)
        if not os.path.exists(path):
            continue

        stem = os.path.splitext(os.path.basename(path))[0]
        packed_base = os.path.join(tokenized_dir, f"{stem}_packed.arrow")
        packed_manifest = packed_base + ".manifest.json"

        if not os.path.isfile(packed_manifest):
            _ensure_perplexity_packed_like_val(
                path, packed_base, raw_tok, config, context_length, logger
            )

        if os.path.isfile(packed_manifest):
            dataset = LanguageModelingDataset(packed_base, raw_tok, max_length=context_length)
            logger.info(
                "Perplexity [%s]: packed val-style data (%s, n=%s)",
                name,
                os.path.basename(packed_base),
                len(dataset),
            )
        else:
            dataset = PlainTextLMDataset(path, raw_tok, max_length=context_length)
            logger.warning(
                "Perplexity [%s]: val-style pack unavailable; using line-wise text truncated to %s tokens (n=%s lines).",
                name,
                context_length,
                len(dataset),
            )

        if debug_cap is not None:
            try:
                n = min(int(debug_cap), len(dataset))
                from torch.utils.data import Subset

                dataset = Subset(dataset, range(n))
                logger.info("Perplexity [%s]: DEBUG cap enabled (n=%s)", name, n)
            except Exception as exc:
                logger.warning("Perplexity [%s]: DEBUG cap failed (%s); running full split.", name, exc)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
        results[name] = evaluator.evaluate(loader)

    return results


def _print_debug_results(eval_name: str, results: dict, logger, max_examples: int = 5):
    """Print first N examples for debug verification."""
    logger.info(f"=== DEBUG {eval_name.upper()} (first {max_examples} examples) ===")
    if eval_name == "perplexity":
        for split, res in results.items():
            logger.info(
                f"  [{split}] ppl={res.get('ppl', 0):.2f} char_ppl={res.get('char_ppl', 0):.2f}"
            )
    elif isinstance(results, dict):
        detailed = results.get("detailed_results", [])
        if not detailed:
            for k, v in results.items():
                if k.startswith("detailed_results_") and isinstance(v, list):
                    detailed = v
                    break
        generations = results.get("generations", [])
        examples = results.get("examples", [])
        if detailed:
            for i, d in enumerate(detailed[:max_examples]):
                if "prefix" in d and "suffix" in d:
                    logger.info(f"  [{i+1}] prefix={str(d.get('prefix',''))[:50]}... | suffix={str(d.get('suffix',''))[:25]}... | trimmed={str(d.get('trimmed',''))[:25]}... | match={d.get('match')}")
                else:
                    logger.info(f"  [{i+1}] prompt={d.get('prompt', '')[:60]}... target={d.get('target')} generated={d.get('generated')} acc={d.get('accuracy','')} lev={d.get('lev_score','')}")
        elif generations:
            for i, g in enumerate(generations[:max_examples]):
                logger.info(f"  [{i+1}] prompt={g.get('prompt','')[:60]}... generated={g.get('generated','')[:80]}...")
        elif examples:
            for i, ex in enumerate(examples[:max_examples]):
                logger.info(f"  [{i+1}] prompt={ex.get('prompt','')[:60]}... generated={ex.get('generated','')[:80]}...")
        else:
            logger.info(f"  {json.dumps({k: v for k, v in results.items() if k not in ('detailed_results','generations','examples')}, indent=2)[:500]}")
    logger.info("")

def get_results_dir(experiment_id: str, is_hf: bool = False, language: str = None) -> str:
    """Get results directory for evaluation."""
    if is_hf:
        results_dir = os.path.join(project_root, f"outputs/{language}/HuggingFace/evaluation")
    else:
        results_dir = os.path.join(project_root, f"outputs/{experiment_id}/evaluation")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _strip_details_for_report(obj):
    """Drop detailed_results / examples / generations from nested dicts for text reports."""
    if not isinstance(obj, dict):
        return obj
    skip = {"detailed_results", "examples", "generations"}
    out = {}
    if isinstance(obj.get("generations"), list):
        out["num_generations"] = len(obj["generations"])
    if isinstance(obj.get("examples"), list):
        out["num_examples"] = len(obj["examples"])
    for k, v in obj.items():
        if k in skip or k.startswith("detailed_results_"):
            continue
        if isinstance(v, dict):
            out[k] = _strip_details_for_report(v)
        else:
            out[k] = v
    return out


def _split_speed_metrics(results):
    if not isinstance(results, dict):
        return results, None
    speed_keys = {
        k
        for k in results.keys()
        if k.startswith("generation_") or k.startswith("throughput_")
    }
    if not speed_keys:
        return results, None
    main = {k: v for k, v in results.items() if k not in speed_keys}
    speed = {k: v for k, v in results.items() if k in speed_keys}
    return main, speed


def write_report(experiment_id: str, results_dir: str, evaluation_results: dict):
    """Write evaluation report to file."""
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Report: {experiment_id}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        if "training_summary" in evaluation_results:
            s = evaluation_results["training_summary"]
            f.write("--- Training Summary ---\n")
            f.write(f"Last step: {s.get('summary_last_step')}\n")
            f.write(f"Data processed: {s.get('summary_pct_data_processed', 0):.2f}%\n")
            f.write(f"Best val loss: {s.get('summary_best_val_loss', 0):.4f}\n\n")

        if "perplexity" in evaluation_results:
            f.write("--- Perplexity ---\n")
            for split, res in evaluation_results["perplexity"].items():
                f.write(
                    f"{split}: ppl={res.get('ppl', 0):.2f}  char_ppl={res.get('char_ppl', 0):.2f}\n"
                )
            f.write("\n")
        
        for eval_name, results in evaluation_results.items():
            if eval_name not in ["training_summary", "perplexity"]:
                f.write(f"--- {eval_name.upper()} ---\n")
                if isinstance(results, dict):
                    summary_results = _strip_details_for_report(results)
                    f.write(f"{json.dumps(summary_results, indent=2)}\n")
                else:
                    f.write(f"{results}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Professional Model Evaluation")
    parser.add_argument("--experiment", type=str, required=False, help="Experiment ID")
    parser.add_argument("--hf_model", type=str, default=None, help="HuggingFace model name or path")
    parser.add_argument("--evaluators", type=str, default="all", help="Comma-separated list of evaluators or 'all'")
    parser.add_argument("--device", type=str, default=get_device_string())
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Inference batch size; default inference.batch_size from merged config, else 32",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Data language; default from merged config or experiment id prefix (e.g. ru/bpe_32k -> ru)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override benchmarks.max_samples (benchmarks/external_model; perplexity uses full corpus)",
    )
    parser.add_argument("--debug", action="store_true", help="Print full output for first 5 examples of each metric")
    parser.add_argument(
        "--external-gen-batch-size",
        type=int,
        default=None,
        dest="external_gen_batch_size",
        help="Fixed local generation batch for external_model (no probing)",
    )
    parser.add_argument(
        "--external-gen-reprobe",
        action="store_true",
        dest="external_gen_reprobe",
        help="Probe once for gen batch (throughput plateau); else set external_gen_batch_size in yaml",
    )
    parser.add_argument(
        "--external-gen-total-tokens",
        type=int,
        default=None,
        dest="external_gen_total_tokens",
        help="Override external_model local generation total token window",
    )
    parser.add_argument(
        "--external-gen-max-new-tokens",
        type=int,
        default=None,
        dest="external_gen_max_new_tokens_cap",
        help="Override external_model local generation max new tokens/word steps",
    )
    parser.add_argument(
        "--external-gen-temperature",
        type=float,
        default=None,
        dest="external_gen_temperature",
        help="Override external_model local generation temperature",
    )
    parser.add_argument(
        "--external-gen-greedy",
        action="store_true",
        dest="external_gen_greedy",
        help="Use greedy local generation for external_model",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    set_seed(args.seed)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.chdir(project_root)
    
    experiment_id = None
    language = None
    config = None
    model = None
    tokenizer = None

    if args.hf_model:
        experiment_id = args.hf_model.replace("/", "_")
        language = args.language
        
        hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        tokenizer = HFTokenizerWrapper(hf_tokenizer)
            
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            attn_implementation="eager"
        )
        model = HFModelWrapper(hf_model)
        
        config = {
            "language": language,
            "training": {"context_length": 2048},
            "tokenizer": {"vocab_size": tokenizer.vocab_size}
        }
    else:
        if not args.experiment:
            raise ValueError("Either --experiment or --hf_model must be provided")
            
        experiment_id = args.experiment
        config_path = os.path.join(project_root, f"configs/experiments/{args.experiment}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = get_full_config(config_path, project_root)
        
        if args.language:
            language = args.language
        elif config.get("language"):
            language = config.get("language")
        elif "/" in experiment_id:
            language = experiment_id.split("/")[0]
        else:
            language = "en"
        
        model, tokenizer = load_model_and_tokenizer(experiment_id, config, args.device)

    available_evaluators = {
        "perplexity": PerplexityEvaluator,
        "miron": MIRONEvaluator,
        "lambada": LAMBADAEvaluator,
        "canary": CanaryEvaluator,
        "oracle": OracleEvaluator,
        "human_eval": HumanEvalEvaluator,
        "external_model": ExternalModelEvaluator,
        "generation_speed": GenerationSpeedEvaluator,
        "blimp": BLIMPEvaluator
    }
    
    if args.evaluators == "all":
        selected_evaluators = list(available_evaluators.keys())
    else:
        selected_evaluators = args.evaluators.split(",")

    results_dir = get_results_dir(experiment_id, is_hf=bool(args.hf_model), language=language)

    evaluation_results = {}
    if not args.hf_model:
        summary = build_training_summary(project_root, experiment_id, config, model=model)
        if summary:
            summary_path = os.path.join(results_dir, "training_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            evaluation_results["training_summary"] = summary
    
    config["seed"] = args.seed
    _bench = config.setdefault("benchmarks", {})
    if args.max_samples is not None:
        _bench["max_samples"] = args.max_samples
    if args.hf_model:
        _bench.setdefault("max_samples", 2048)
    config["debug"] = args.debug
    if args.external_gen_batch_size is not None:
        config["external_gen_batch_size"] = args.external_gen_batch_size
    if args.external_gen_reprobe:
        config["external_gen_reprobe"] = True
    if args.external_gen_total_tokens is not None:
        config["external_gen_total_tokens"] = args.external_gen_total_tokens
    if args.external_gen_max_new_tokens_cap is not None:
        config["external_gen_max_new_tokens_cap"] = args.external_gen_max_new_tokens_cap
    if args.external_gen_temperature is not None:
        config["external_gen_temperature"] = args.external_gen_temperature
    if args.external_gen_greedy:
        config["external_gen_do_sample"] = False

    _exp = dict(config.get("experiment") or {})
    _exp.setdefault("id", experiment_id)
    config["experiment"] = _exp

    _inf = config.get("inference") or {}
    _cfg_bs = _inf.get("batch_size")
    eval_batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (int(_cfg_bs) if _cfg_bs is not None else 32)
    )

    for eval_name in selected_evaluators:
        if eval_name not in available_evaluators:
            logger.warning(f"Unknown evaluator: {eval_name}")
            continue

        evaluator_cls = available_evaluators[eval_name]
        
        evaluator = evaluator_cls(model, tokenizer, config, device=args.device)
        
        if eval_name == "perplexity":
            ppl_results = _run_perplexity_eval(
                evaluator, language, eval_batch_size, config, results_dir, logger
            )
            evaluation_results[eval_name] = ppl_results
            output_path = os.path.join(results_dir, "perplexity_results.json")
            evaluator.save_results(ppl_results, output_path)
            if config.get("debug"):
                _print_debug_results(eval_name, ppl_results, logger)
        else:
            results = evaluator.evaluate(language=language, batch_size=eval_batch_size)
            main_results, speed_results = _split_speed_metrics(results)
            output_path = os.path.join(results_dir, f"{eval_name}_results.json")
            evaluator.save_results(main_results, output_path)
            evaluation_results[eval_name] = main_results
            if speed_results:
                speed_path = os.path.join(results_dir, f"{eval_name}_speed_results.json")
                evaluator.save_results(speed_results, speed_path)
                evaluation_results[f"{eval_name}_speed"] = speed_results
            if config.get("debug"):
                _print_debug_results(eval_name, main_results, logger)
    
    write_report(experiment_id, results_dir, evaluation_results)


if __name__ == "__main__":
    main()
