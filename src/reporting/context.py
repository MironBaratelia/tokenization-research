"""Paths and ids for one experiment when building report artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReportOutputDirs:
    """Where this run writes plots and LaTeX fragments."""

    plots_dir: str
    tables_dir: str


@dataclass(frozen=True)
class ReportContext:
    project_root: str
    experiment_id: str
    outputs_root: str = "outputs"
    plots_parent: str = "reports/plots"
    tables_parent: str = "reports/tables"

    @property
    def exp_dir(self) -> str:
        return os.path.join(self.project_root, self.outputs_root, self.experiment_id.replace("\\", "/"))

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.exp_dir, "logs")

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.exp_dir, "evaluation")

    @property
    def metrics_jsonl(self) -> str:
        return os.path.join(self.logs_dir, "metrics.jsonl")

    def _safe_exp(self) -> str:
        return self.experiment_id.replace("\\", "/").replace("/", "_")

    def plots_dir(self, override_parent: Optional[str] = None) -> str:
        """One subdirectory per experiment under reports/plots (avoids filename collisions)."""
        if override_parent is not None:
            return os.path.join(override_parent, self._safe_exp())
        return os.path.join(self.project_root, self.plots_parent, self._safe_exp())

    def tables_dir(self, override_parent: Optional[str] = None) -> str:
        if override_parent is not None:
            return os.path.join(override_parent, self._safe_exp())
        return os.path.join(self.project_root, self.tables_parent, self._safe_exp())
