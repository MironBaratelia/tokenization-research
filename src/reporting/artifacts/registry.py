"""Named report artifacts. Same pattern as --evaluators in scripts/04_evaluate.py."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from src.reporting.artifacts.eval_json_plots import (
    build_blimp,
    build_canary,
    build_lambada,
    build_perplexity_comparison,
    build_token_dist_text,
    build_token_dist_vocab,
    build_token_length_vocab_vs_text,
    build_training_summary,
)
from src.reporting.artifacts.metrics_plots import build_learning_rate, build_training_curves
from src.reporting.artifacts.cross_tables_latex import build_latex_cross_tables_unscoped
from src.reporting.artifacts.human_eval_latex import build_human_eval_appendix
from src.reporting.artifacts.rebuild_training_summary_json import rebuild_training_summary_json
from src.reporting.artifacts.statistical_significance import build_statistical_significance_unscoped
from src.reporting.artifacts.tables_latex import build_latex_tables

ArtifactFn = Callable[..., bool]

ARTIFACT_ALIASES: Dict[str, str] = {
    "technical_metrics": "learning_rate",
}

# rebuild_training_summary_json first: refreshes training_summary.json (metrics + checkpoint
# stats, epochs, steps) before curves/tables that read that file.
ARTIFACT_REGISTRY: Dict[str, ArtifactFn] = {
    "rebuild_training_summary_json": rebuild_training_summary_json,
    "training_curves": build_training_curves,
    "learning_rate": build_learning_rate,
    "perplexity_comparison": build_perplexity_comparison,
    "token_dist_vocab": build_token_dist_vocab,
    "token_dist_text": build_token_dist_text,
    "token_length_vocab_vs_text": build_token_length_vocab_vs_text,
    "blimp": build_blimp,
    "lambada": build_lambada,
    "canary": build_canary,
    "training_summary": build_training_summary,
    "latex_tables": build_latex_tables,
    "latex_cross_tables": build_latex_cross_tables_unscoped,
    "human_eval_appendix": build_human_eval_appendix,
    "statistical_significance": build_statistical_significance_unscoped,
}

ALL_ARTIFACT_NAMES: Tuple[str, ...] = tuple(ARTIFACT_REGISTRY.keys())


def resolve_artifacts(spec: str) -> List[str]:
    s = (spec or "").strip().lower()
    if s == "all" or not s:
        raw = list(ALL_ARTIFACT_NAMES)
    else:
        canon = {k.lower(): k for k in ARTIFACT_REGISTRY}
        raw = []
        for x in spec.split(","):
            x = x.strip()
            if not x:
                continue
            c = canon.get(x.lower(), x)
            c = ARTIFACT_ALIASES.get(c, c)
            raw.append(c)

    seen = set()
    out: List[str] = []
    for name in raw:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out
