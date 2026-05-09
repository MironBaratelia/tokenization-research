import logging
import os
import sys
import json
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import traceback

import yaml


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger


@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line: int
    experiment_id: Optional[str] = None
    epoch: Optional[int] = None
    step: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line=record.lineno
        )
        
        if hasattr(record, 'experiment_id'):
            log_entry.experiment_id = record.experiment_id
        if hasattr(record, 'epoch'):
            log_entry.epoch = record.epoch
        if hasattr(record, 'step'):
            log_entry.step = record.step
        
        if hasattr(record, 'metrics'):
            log_entry.metrics = record.metrics
        
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info', 'experiment_id', 'epoch',
                    'step', 'metrics'
                }:
                    extra[key] = value
            
            if extra:
                log_entry.extra = extra
        
        return json.dumps(asdict(log_entry), default=str)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, fmt: Optional[str] = None):
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        super().__init__(fmt)
    
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        formatted = super().format(record)
        
        record.levelname = levelname
        
        return formatted


class MetricsLogger:
    def __init__(self, logger: logging.Logger, experiment_id: Optional[str] = None):
        self.logger = logger
        self.experiment_id = experiment_id
    
    def log_metrics(self, metrics: Dict[str, Any], 
                   epoch: Optional[int] = None, 
                   step: Optional[int] = None,
                   level: int = logging.INFO) -> None:
        self.logger.log(
            level,
            f"Metrics: {json.dumps(metrics)}",
            extra={
                'experiment_id': self.experiment_id,
                'epoch': epoch,
                'step': step,
                'metrics': metrics
            }
        )
    
    def log_loss(self, loss: float, 
                loss_components: Optional[Dict[str, float]] = None,
                epoch: Optional[int] = None, 
                step: Optional[int] = None) -> None:
        metrics = {'loss': loss}
        if loss_components:
            metrics.update(loss_components)
        
        self.log_metrics(metrics, epoch=epoch, step=step)
    
    def log_evaluation_results(self, results: Dict[str, Any],
                              epoch: Optional[int] = None) -> None:
        self.logger.info(
            f"Evaluation results: {json.dumps(results)}",
            extra={
                'experiment_id': self.experiment_id,
                'epoch': epoch,
                'metrics': results
            }
        )


class ExperimentLogger:
    def __init__(self, experiment_id: str, log_dir: str):
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"experiment.{experiment_id}")
        self.logger.setLevel(logging.INFO)
        
        self.logger.handlers.clear()
        
        log_file = self.log_dir / f"{experiment_id}.jsonl"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)
        
        self.metrics_logger = MetricsLogger(self.logger, experiment_id)
        
        self.metadata = {
            'experiment_id': experiment_id,
            'start_time': datetime.now().isoformat(),
            'log_file': str(log_file)
        }
        
        self._save_metadata()
    
    def log_config(self, config: Dict[str, Any]) -> None:
        self.metadata['config'] = config
        self._save_metadata()
        
        self.logger.info(
            f"Experiment configuration loaded",
            extra={'experiment_id': self.experiment_id}
        )
    
    def log_system_info(self, system_info: Dict[str, Any]) -> None:
        self.metadata['system_info'] = system_info
        self._save_metadata()
        
        self.logger.info(
            f"System info: {json.dumps(system_info)}",
            extra={'experiment_id': self.experiment_id}
        )
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        self.metadata['dataset_info'] = dataset_info
        self._save_metadata()
        
        self.logger.info(
            f"Dataset info: {json.dumps(dataset_info)}",
            extra={'experiment_id': self.experiment_id}
        )
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        self.metadata['model_info'] = model_info
        self._save_metadata()
        
        self.logger.info(
            f"Model info: {json.dumps(model_info)}",
            extra={'experiment_id': self.experiment_id}
        )
    
    def log_checkpoint(self, checkpoint_path: str, 
                      metrics: Optional[Dict[str, Any]] = None,
                      epoch: Optional[int] = None) -> None:
        self.logger.info(
            f"Checkpoint saved: {checkpoint_path}",
            extra={
                'experiment_id': self.experiment_id,
                'epoch': epoch,
                'checkpoint_path': checkpoint_path,
                'metrics': metrics
            }
        )
    
    def log_error(self, error: Exception, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc()
        }
        
        if context:
            error_info['context'] = context
        
        self.logger.error(
            f"Error occurred: {json.dumps(error_info)}",
            extra={
                'experiment_id': self.experiment_id,
                'error_info': error_info
            }
        )
    
    def finish_experiment(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        self.metadata['end_time'] = datetime.now().isoformat()
        if final_metrics:
            self.metadata['final_metrics'] = final_metrics
        
        self._save_metadata()
        
        self.logger.info(
            f"Experiment finished",
            extra={
                'experiment_id': self.experiment_id,
                'final_metrics': final_metrics
            }
        )
    
    def _save_metadata(self) -> None:
        metadata_file = self.log_dir / f"{self.experiment_id}_metadata.yaml"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.metadata, f, default_flow_style=False)


class LoggingManager:
    @staticmethod
    def setup_logging(log_level: str = "INFO", 
                     log_dir: Optional[str] = None,
                     structured: bool = False) -> None:
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        root_logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        if structured:
            console_handler.setFormatter(StructuredFormatter(include_extra=False))
        else:
            console_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(console_handler)
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            
            if structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            
            root_logger.addHandler(file_handler)
    
    @staticmethod
    def get_experiment_logger(experiment_id: str, 
                             log_dir: str = "logs/experiments") -> ExperimentLogger:
        return ExperimentLogger(experiment_id, log_dir)
    
    @staticmethod
    def get_metrics_logger(logger: logging.Logger, 
                          experiment_id: Optional[str] = None) -> MetricsLogger:
        return MetricsLogger(logger, experiment_id)


def setup_enhanced_logging(log_level: str = "INFO", 
                          log_dir: Optional[str] = None,
                          structured: bool = False) -> None:
    LoggingManager.setup_logging(log_level, log_dir, structured)


def get_experiment_logger(experiment_id: str, 
                         log_dir: str = "logs/experiments") -> ExperimentLogger:
    return LoggingManager.get_experiment_logger(experiment_id, log_dir)


def get_metrics_logger(logger: logging.Logger, 
                      experiment_id: Optional[str] = None) -> MetricsLogger:
    return LoggingManager.get_metrics_logger(logger, experiment_id)