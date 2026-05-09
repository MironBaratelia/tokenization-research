"""Orchestrate building artifacts for one or many experiments."""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional, Sequence, Tuple

from src.reporting.context import ReportContext, ReportOutputDirs
from src.reporting.cross_discovery import discover_languages
from src.reporting.discovery import discover_experiments
from src.reporting.artifacts.cross_tables_latex import run_cross_language_tables
from src.reporting.artifacts.registry import ALL_ARTIFACT_NAMES, ARTIFACT_REGISTRY, resolve_artifacts

logger = logging.getLogger(__name__)

CROSS_ARTIFACTS = frozenset({"latex_cross_tables", "statistical_significance"})


def _resolve_cross_languages(
    project_root: str,
    outputs_root: str,
    cross_arg: Optional[str],
    use_all: bool,
    experiments: List[str],
) -> List[str]:
    if cross_arg:
        return [x.strip() for x in cross_arg.split(",") if x.strip()]
    if use_all:
        return discover_languages(project_root, outputs_root)
    langs = sorted(
        {
            e.replace("\\", "/").split("/")[0]
            for e in experiments
            if "/" in e.replace("\\", "/")
        }
    )
    if langs:
        return langs
    return discover_languages(project_root, outputs_root)


def run_for_experiment(
    project_root: str,
    experiment_id: str,
    artifact_names: Sequence[str],
    outputs_root: str = "outputs",
    plots_parent: str = "reports/plots",
    plots_parent_override: Optional[str] = None,
    plots_in_experiment_output: bool = False,
    tables_parent: str = "reports/tables",
    tables_parent_override: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Returns (attempted, succeeded) counts for selected artifacts.
    Each artifact is attempted independently; missing inputs are skipped without failing others.
    """
    ctx = ReportContext(
        project_root=project_root,
        experiment_id=experiment_id,
        outputs_root=outputs_root,
        plots_parent=plots_parent,
        tables_parent=tables_parent,
    )
    if plots_in_experiment_output:
        plots_dir = os.path.join(ctx.exp_dir, "reports", "plots")
        tables_dir = os.path.join(ctx.exp_dir, "reports", "tables")
    else:
        plots_dir = ctx.plots_dir(override_parent=plots_parent_override)
        tables_dir = ctx.tables_dir(override_parent=tables_parent_override)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    out = ReportOutputDirs(plots_dir=plots_dir, tables_dir=tables_dir)

    attempted = 0
    succeeded = 0
    for name in artifact_names:
        if name in CROSS_ARTIFACTS:
            continue
        fn = ARTIFACT_REGISTRY.get(name)
        if not fn:
            logger.warning("Unknown artifact: %s", name)
            continue
        attempted += 1
        try:
            if fn(ctx, out):
                succeeded += 1
            else:
                logger.info("Skip artifact %s for %s (missing data)", name, experiment_id)
        except Exception:
            logger.exception("Artifact %s failed for %s", name, experiment_id)
    return attempted, succeeded


def run_cli(
    project_root: str,
    experiments: List[str],
    artifacts_spec: str,
    outputs_root: str,
    plots_parent: str,
    plots_parent_override: Optional[str],
    plots_in_experiment_output: bool,
    tables_parent: str,
    tables_parent_override: Optional[str],
    cross_languages: Optional[str],
    cross_tables_root: str,
    all_experiments_flag: bool,
) -> int:
    names = resolve_artifacts(artifacts_spec)
    unknown = [n for n in names if n not in ARTIFACT_REGISTRY]
    for u in unknown:
        logger.warning("Unknown artifact name: %s (ignored)", u)
    names = [n for n in names if n in ARTIFACT_REGISTRY]

    if not names:
        logger.error("No valid artifacts. Choose from: %s", ", ".join(ALL_ARTIFACT_NAMES))
        return 2

    per_exp = [n for n in names if n not in CROSS_ARTIFACTS]
    want_cross = "latex_cross_tables" in names

    total_attempted = 0
    total_ok = 0
    for exp_id in experiments:
        logger.info("--- %s ---", exp_id)
        a, ok = run_for_experiment(
            project_root,
            exp_id,
            per_exp,
            outputs_root=outputs_root,
            plots_parent=plots_parent,
            plots_parent_override=plots_parent_override,
            plots_in_experiment_output=plots_in_experiment_output,
            tables_parent=tables_parent,
            tables_parent_override=tables_parent_override,
        )
        total_attempted += a
        total_ok += ok
        logger.info("Artifacts produced: %s / %s (for this experiment)", ok, a)

    if want_cross:
        langs = _resolve_cross_languages(
            project_root, outputs_root, cross_languages, all_experiments_flag, experiments
        )
        total_attempted += 1
        if not langs:
            logger.warning("Cross-language tables: no languages resolved; try --cross-languages ru,en")
        else:
            cross_root = (
                cross_tables_root
                if os.path.isabs(cross_tables_root)
                else os.path.join(project_root, cross_tables_root)
            )
            n_files, n_lang = run_cross_language_tables(project_root, outputs_root, langs, cross_root)
            if n_files > 0:
                total_ok += 1
                logger.info(
                    "Cross-language tables: %s file(s) for %s language(s) -> %s",
                    n_files,
                    n_lang,
                    cross_root,
                )
            else:
                logger.info("Cross-language tables: nothing written (no evaluation JSON?)")

    if "statistical_significance" in names:
        langs = _resolve_cross_languages(
            project_root, outputs_root, cross_languages, all_experiments_flag, experiments
        )
        total_attempted += 1
        if not langs:
            logger.warning("Statistical significance: no languages resolved; try --cross-languages ru,en")
        else:
            fn = ARTIFACT_REGISTRY["statistical_significance"]
            tables_root_abs = (
                cross_tables_root
                if os.path.isabs(cross_tables_root)
                else os.path.join(project_root, cross_tables_root)
            )
            n_files = fn(project_root, outputs_root, langs, tables_root_abs)
            if n_files > 0:
                total_ok += 1
                logger.info(
                    "Statistical significance: %s file(s) for %s language(s) -> %s",
                    n_files,
                    len(langs),
                    tables_root_abs,
                )
            else:
                logger.info("Statistical significance: nothing written (no supported evaluation JSON?)")

    logger.info("Done: %s / %s artifact builds succeeded overall", total_ok, total_attempted)
    return 0


def _default_project_root() -> str:
    # src/reporting/run.py -> project root is three levels up
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: Optional[Sequence[str]] = None, project_root: Optional[str] = None) -> int:
    import argparse

    root = project_root or _default_project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    parser = argparse.ArgumentParser(
        description="Build thesis/report plots and LaTeX tables from existing outputs (like 04_evaluate.py --evaluators)."
    )
    parser.add_argument(
        "--experiment",
        "--config",
        dest="experiment",
        type=str,
        default=None,
        help="Single experiment id (e.g. ru/morpheme); alias --config for old scripts.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment ids (overrides --experiment when set).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every experiment under outputs/ that has metrics and/or evaluation JSON.",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="all",
        help=f"Comma-separated artifact names or 'all'. Available: {', '.join(ALL_ARTIFACT_NAMES)}",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs",
        help="Directory under project root (default: outputs).",
    )
    parser.add_argument(
        "--plots-root",
        type=str,
        default="reports/plots",
        help="Parent directory under project root; each experiment gets a subfolder (default: reports/plots).",
    )
    parser.add_argument(
        "--plots-output-dir",
        type=str,
        default=None,
        help="Parent directory; each experiment is written to <dir>/<experiment_id_with_underscores>/",
    )
    parser.add_argument(
        "--plots-in-output-dir",
        action="store_true",
        help="Write under outputs/<experiment>/reports/plots and .../reports/tables.",
    )
    parser.add_argument(
        "--tables-root",
        type=str,
        default="reports/tables",
        help="Parent for LaTeX fragments under project root (default: reports/tables).",
    )
    parser.add_argument(
        "--tables-output-dir",
        type=str,
        default=None,
        help="Override parent for tables (same per-experiment subfolders as plots-output-dir).",
    )
    parser.add_argument(
        "--cross-languages",
        type=str,
        default=None,
        help="For latex_cross_tables: comma-separated language codes (default: infer from --experiment or discover if --all).",
    )
    parser.add_argument(
        "--cross-tables-root",
        type=str,
        default="reports/tables/cross",
        help="Output directory for per-language comparison tables (default: reports/tables/cross).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    os.chdir(root)

    experiments: List[str] = []
    if args.all:
        experiments = discover_experiments(root, args.outputs_root)
        if not experiments:
            logger.error("No experiments found under %s", args.outputs_root)
            return 1
        logger.info("Discovered %s experiments", len(experiments))
    elif args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    elif args.experiment:
        experiments = [args.experiment.strip()]
    else:
        parser.error("Specify --experiment, --experiments, or --all")

    plots_override: Optional[str] = None
    if args.plots_output_dir:
        plots_override = os.path.normpath(
            args.plots_output_dir
            if os.path.isabs(args.plots_output_dir)
            else os.path.join(root, args.plots_output_dir)
        )

    tables_override: Optional[str] = None
    if args.tables_output_dir:
        tables_override = os.path.normpath(
            args.tables_output_dir
            if os.path.isabs(args.tables_output_dir)
            else os.path.join(root, args.tables_output_dir)
        )

    return run_cli(
        root,
        experiments,
        args.artifacts,
        args.outputs_root,
        args.plots_root,
        plots_override,
        args.plots_in_output_dir,
        args.tables_root,
        tables_override,
        args.cross_languages,
        args.cross_tables_root,
        args.all,
    )
