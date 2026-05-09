"""Performance monitoring and optimization utilities."""
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np


try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None
    disk_io_read: Optional[float] = None
    disk_io_write: Optional[float] = None
    network_io_sent: Optional[float] = None
    network_io_recv: Optional[float] = None


class PerformanceMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
        
    def start_monitoring(self) -> None:
        if self.monitoring:
            return
        self.monitoring = True

    def stop_monitoring(self) -> None:
        self.monitoring = False

    def get_average_metrics(self) -> Optional[PerformanceMetrics]:
        return None


class MemoryOptimizer:
    @staticmethod
    def optimize_model_memory(
        model: nn.Module, 
        enable_mixed_precision: bool = True,
        enable_gradient_checkpointing: bool = False
    ) -> nn.Module:
        if enable_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            elif hasattr(model, 'lm') and hasattr(model.lm, 'gradient_checkpointing_enable'):
                model.lm.gradient_checkpointing_enable()
        return model
    
    @staticmethod
    def optimize_dataloader_memory(
        dataloader: DataLoader,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ) -> DataLoader:
        return dataloader
    
    @staticmethod
    def clear_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage() -> dict:
        memory_info = {}
        try:
            import psutil
            cpu_memory = psutil.virtual_memory()
            memory_info['cpu_memory_used_gb'] = cpu_memory.used / (1024**3)
            memory_info['cpu_memory_total_gb'] = cpu_memory.total / (1024**3)
            memory_info['cpu_memory_percent'] = cpu_memory.percent
        except ImportError:
            pass
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_info['gpu_memory_used_gb'] = gpu_memory / (1024**3)
            memory_info['gpu_memory_total_gb'] = gpu_memory_total / (1024**3)
            memory_info['gpu_memory_percent'] = (gpu_memory / gpu_memory_total) * 100
        
        return memory_info


class TrainingOptimizer:
    @staticmethod
    def optimize_for_training(
        model: nn.Module,
        enable_compile: bool = True,
        enable_cudnn_benchmark: bool = True
    ) -> nn.Module:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available() and enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        # torch.compile: inductor cache has WinError 183 on Windows; skip there
        if enable_compile and hasattr(torch, 'compile') and os.name != "nt":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                logger.warning("torch.compile skipped: %s", e)
        return model
    
    @staticmethod
    def create_efficient_dataloader(
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ) -> DataLoader:
        if num_workers is None:
            num_workers = min(4, os.cpu_count() or 1)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2
        )
    
    @staticmethod
    def get_optimal_batch_size(
        model: nn.Module,
        input_shape: tuple,
        max_batch_size: int = 256,
        memory_fraction: float = 0.8
    ) -> int:
        if not torch.cuda.is_available():
            return min(32, max_batch_size)
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = total_memory * memory_fraction
        
        low, high = 1, max_batch_size
        optimal_batch_size = 1
        
        model.eval()
        
        while low <= high:
            mid = (low + high) // 2
            torch.cuda.empty_cache()
            
            test_input = torch.randn(mid, *input_shape).cuda()
            
            with torch.no_grad():
                _ = model(test_input)
            
            current_memory = torch.cuda.memory_allocated()
            
            if current_memory < target_memory:
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return optimal_batch_size


@contextmanager
def performance_context(monitor: Optional[PerformanceMonitor] = None):
    if monitor is None:
        monitor = PerformanceMonitor()
    monitor.start_monitoring()
    yield monitor
    monitor.stop_monitoring()


@contextmanager
def memory_context():
    initial_memory = MemoryOptimizer.get_memory_usage()
    yield
    MemoryOptimizer.clear_cache()


def optimize_memory(
    model: nn.Module, 
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = False
) -> nn.Module:
    return MemoryOptimizer.optimize_model_memory(
        model, enable_mixed_precision, enable_gradient_checkpointing
    )


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: tuple,
    max_batch_size: int = 256,
    memory_fraction: float = 0.8
) -> int:
    return TrainingOptimizer.get_optimal_batch_size(
        model, input_shape, max_batch_size, memory_fraction
    )
