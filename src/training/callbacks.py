"""
Training callbacks for monitoring and control
"""
import os
import json
import time
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: Any, batch: int) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict) -> None:
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)
    
    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)
    
    def on_batch_begin(self, trainer: Any, batch: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch)
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch, logs)


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path template for saving (e.g., 'model_epoch_{epoch}.pt')
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_freq: Save every N epochs
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.save_best_only:
            if self.mode == 'min' and current < self.best_score:
                self.best_score = current
                self._save(trainer, epoch, logs)
            elif self.mode == 'max' and current > self.best_score:
                self.best_score = current
                self._save(trainer, epoch, logs)
        else:
            if (epoch + 1) % self.save_freq == 0:
                self._save(trainer, epoch, logs)
    
    def _save(self, trainer: Any, epoch: int, logs: Dict) -> None:
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': logs
        }, filepath)
        
        logger.info(f"Saved checkpoint: {filepath}")


class EarlyStopping(Callback):
    """Stop training when metric stops improving."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0
    
    def on_train_begin(self, trainer: Any) -> None:
        self.counter = 0
        self.best_score = None
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.best_score is None:
            self.best_score = current
            self.best_weights = trainer.model.state_dict().copy()
            return
        
        if self.mode == 'min':
            improved = current < self.best_score - self.min_delta
        else:
            improved = current > self.best_score + self.min_delta
        
        if improved:
            self.best_score = current
            self.best_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.stop_training = True
                self.stopped_epoch = epoch
                logger.info(f"Early stopping at epoch {epoch + 1}")
                
                if self.restore_best and self.best_weights:
                    trainer.model.load_state_dict(self.best_weights)
                    logger.info("Restored best weights")


class LearningRateScheduler(Callback):
    """Adjust learning rate during training."""
    
    def __init__(self, scheduler: Any):
        """
        Initialize scheduler callback.
        
        Args:
            scheduler: PyTorch learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        # Handle ReduceLROnPlateau separately
        if hasattr(self.scheduler, 'step'):
            if 'val_loss' in logs:
                self.scheduler.step(logs['val_loss'])
            else:
                self.scheduler.step()


class ProgressLogger(Callback):
    """Log training progress."""
    
    def __init__(self, log_freq: int = 10):
        """
        Initialize progress logger.
        
        Args:
            log_freq: Log every N batches
        """
        self.log_freq = log_freq
        self.epoch_start_time = None
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        self.epoch_start_time = time.time()
        logger.info(f"\nEpoch {epoch + 1}")
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict) -> None:
        if (batch + 1) % self.log_freq == 0:
            loss = logs.get('loss', 0)
            logger.info(f"  Batch {batch + 1}: loss = {loss:.4f}")
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        elapsed = time.time() - self.epoch_start_time
        
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        logger.info(f"Epoch {epoch + 1} completed in {elapsed:.2f}s - {metrics_str}")


class HistoryCallback(Callback):
    """Record training history."""
    
    def __init__(self):
        self.history = {'train': [], 'val': []}
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        train_metrics = {k: v for k, v in logs.items() if not k.startswith('val_')}
        val_metrics = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        
        self.history['train'].append(train_metrics)
        self.history['val'].append(val_metrics)
    
    def save(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            self.history = json.load(f)


class WandbCallback(Callback):
    """Log to Weights & Biases."""
    
    def __init__(self, project: str, config: Optional[Dict] = None):
        """
        Initialize W&B callback.
        
        Args:
            project: W&B project name
            config: Configuration to log
        """
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, config=config)
        except ImportError:
            logger.warning("wandb not installed, logging disabled")
            self.wandb = None
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        if self.wandb:
            self.wandb.log(logs, step=epoch)
    
    def on_train_end(self, trainer: Any) -> None:
        if self.wandb:
            self.wandb.finish()


class TensorBoardCallback(Callback):
    """Log to TensorBoard."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            logger.warning("TensorBoard not installed")
            self.writer = None
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        if self.writer:
            for key, value in logs.items():
                self.writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, trainer: Any) -> None:
        if self.writer:
            self.writer.close()


class GradientMonitor(Callback):
    """Monitor gradient statistics during training."""
    
    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.batch_count = 0
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict) -> None:
        self.batch_count += 1
        
        if self.batch_count % self.log_freq == 0:
            grad_norms = []
            for param in trainer.model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            
            if grad_norms:
                logger.info(
                    f"Gradient stats - mean: {np.mean(grad_norms):.4f}, "
                    f"max: {np.max(grad_norms):.4f}"
                )
