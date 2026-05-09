import math
import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter

from .metrics import tensor_to_float_for_logging


class EarlyStoppingCallback:
    def __init__(self, patience=5, min_delta=0.0, min_steps=100000):
        self.patience = patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, current_step=0):
        if current_step < self.min_steps:
            return

        if val_loss is None or (isinstance(val_loss, float) and not math.isfinite(val_loss)):
            return

        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class CheckpointCallback:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, step, loss, is_best=False):
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)


class TensorBoardCallback:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, metrics, step):
        for key, value in metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                for i, v in enumerate(value):
                    self.writer.add_scalar(f"{key}_{i}", v, step)
    
    def close(self):
        self.writer.close()


class MetricsLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, metrics):
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                serializable[k] = tensor_to_float_for_logging(v)
            elif isinstance(v, (list, tuple)):
                serializable[k] = [x.item() if isinstance(x, torch.Tensor) else x for x in v]
            else:
                serializable[k] = v
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(serializable) + '\n')
