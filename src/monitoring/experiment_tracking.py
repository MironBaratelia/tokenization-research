import os
import json
import hashlib
import sys
import platform
import shutil
import torch
import numpy as np
import random
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False

logger = logging.getLogger(__name__)


@dataclass
class ExperimentState:
    experiment_id: str
    config: Dict[str, Any]
    python_info: Dict[str, str]
    system_info: Dict[str, Any]
    dependencies: Dict[str, str]
    random_seeds: Dict[str, int]
    start_time: str
    checkpoints: List[Dict[str, Any]]
    metrics_history: List[Dict[str, Any]]
    artifacts: List[Dict[str, Any]]
    notes: Optional[str] = None
    end_time: Optional[str] = None
    final_metrics: Optional[Dict[str, Any]] = None


class ExperimentTracker:
    def __init__(self, experiment_id: str, output_dir: str = "outputs"):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.experiment_dir = self.output_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = ExperimentState(
            experiment_id=experiment_id,
            config={},
            python_info={},
            system_info={},
            dependencies={},
            random_seeds={},
            start_time=datetime.now().isoformat(),
            checkpoints=[],
            metrics_history=[],
            artifacts=[],
            notes=None
        )
        
        self._save_state()

    def log_config(self, config: Dict[str, Any]) -> None:
        self.state.config = config
        
        config_file = self.experiment_dir / "config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        config_str = json.dumps(config, sort_keys=True)
        self.state.config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        self._save_state()

    def log_python_info(self) -> None:
        self.state.python_info = {
            'version': sys.version,
            'executable': sys.executable,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor()
        }
        
        self._save_state()

    def log_system_info(self) -> None:
        if not HAS_PSUTIL:
            logger.warning("psutil not available, skipping system info logging")
            return
        
        disk_total = 0
        if os.name == "nt":
            drive = os.environ.get("SystemDrive", "C:")
            disk_total = shutil.disk_usage(drive).total
        else:
            disk_total = shutil.disk_usage("/").total

        self.state.system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': disk_total,
            'platform': os.name
        }
        
        self._save_state()

    def log_dependencies(self) -> None:
        if not HAS_PKG_RESOURCES:
            logger.warning("pkg_resources not available, skipping dependencies logging")
            return
        
        dependencies = {}
        for pkg in pkg_resources.working_set:
            dependencies[pkg.project_name] = pkg.version
        
        self.state.dependencies = dependencies
        
        req_file = self.experiment_dir / "requirements.txt"
        with open(req_file, 'w', encoding='utf-8') as f:
            for pkg, version in dependencies.items():
                f.write(f"{pkg}=={version}\n")
        
        self._save_state()

    def set_random_seeds(self, seed: int) -> None:
        self.state.random_seeds = {
            'numpy': seed,
            'python': seed,
            'torch': seed
        }
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self._save_state()

    def log_checkpoint(self, checkpoint_path: str, 
                      metrics: Optional[Dict[str, Any]] = None,
                      epoch: Optional[int] = None,
                      step: Optional[int] = None,
                      is_best: bool = False) -> None:
        checkpoint_info = {
            'path': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'is_best': is_best,
            'metrics': metrics
        }
        
        self.state.checkpoints.append(checkpoint_info)
        
        if not Path(checkpoint_path).is_relative_to(self.experiment_dir):
            checkpoint_name = Path(checkpoint_path).name
            dest_path = self.experiment_dir / "checkpoints" / checkpoint_name
            dest_path.parent.mkdir(exist_ok=True)
            
            shutil.copy2(checkpoint_path, dest_path)
            checkpoint_info['experiment_path'] = str(dest_path)
        
        self._save_state()

    def log_metrics(self, metrics: Dict[str, Any], 
                   epoch: Optional[int] = None,
                   step: Optional[int] = None,
                   phase: str = "train") -> None:
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'phase': phase,
            'metrics': metrics
        }
        
        self.state.metrics_history.append(metrics_entry)
        
        metrics_file = self.experiment_dir / "metrics.jsonl"
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics_entry) + '\n')
        
        self._save_state()
    
    def log_artifact(self, artifact_path: str, 
                    artifact_type: str = "other",
                    description: Optional[str] = None) -> None:
        artifact_info = {
            'path': artifact_path,
            'type': artifact_type,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        
        artifact_name = Path(artifact_path).name
        dest_path = self.experiment_dir / "artifacts" / artifact_name
        dest_path.parent.mkdir(exist_ok=True)
        
        shutil.copy2(artifact_path, dest_path)
        artifact_info['experiment_path'] = str(dest_path)
        
        self.state.artifacts.append(artifact_info)
        
        self._save_state()

    def add_note(self, note: str) -> None:
        if self.state.notes:
            self.state.notes += f"\n\n{note}"
        else:
            self.state.notes = note
        
        self._save_state()

    def finish_experiment(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        self.state.end_time = datetime.now().isoformat()
        self.state.final_metrics = final_metrics
        
        self._save_state()
        
        self._create_summary()

    def _save_state(self) -> None:
        state_file = self.experiment_dir / "experiment_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
    
    def _create_summary(self) -> None:
        summary = {
            'experiment_id': self.experiment_id,
            'duration': self._calculate_duration(),
            'best_checkpoint': self._get_best_checkpoint(),
            'final_metrics': self.state.final_metrics,
            'total_checkpoints': len(self.state.checkpoints),
            'total_artifacts': len(self.state.artifacts)
        }
        
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _calculate_duration(self) -> str:
        if not self.state.end_time:
            return "N/A"
        
        start = datetime.fromisoformat(self.state.start_time)
        end = datetime.fromisoformat(self.state.end_time)
        duration = end - start
        
        return str(duration)
    
    def _get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        best_checkpoints = [cp for cp in self.state.checkpoints if cp.get('is_best', False)]
        return best_checkpoints[0] if best_checkpoints else None
    
    def get_state(self) -> ExperimentState:
        return self.state
    
    @classmethod
    def load_experiment(cls, experiment_id: str, output_dir: str = "outputs") -> 'ExperimentTracker':
        tracker = cls(experiment_id, output_dir)
        
        state_file = tracker.experiment_dir / "experiment_state.json"
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
            
            tracker.state = ExperimentState(**state_dict)
        
        return tracker
