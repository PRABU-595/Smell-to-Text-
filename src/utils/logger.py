"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "smell-to-molecule",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup and configure logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, log_dir: str = "experiments") -> logging.Logger:
    """Get logger for specific experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / experiment_name / f"training_{timestamp}.log"
    return setup_logger(experiment_name, str(log_file))


class TrainingLogger:
    """Structured logging for training."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.history = []
    
    def log_epoch(self, epoch: int, metrics: dict):
        self.history.append({'epoch': epoch, **metrics})
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")
    
    def log_batch(self, batch: int, loss: float, interval: int = 50):
        if batch % interval == 0:
            self.logger.debug(f"Batch {batch}: loss = {loss:.4f}")
    
    def log_evaluation(self, results: dict):
        self.logger.info("Evaluation Results:")
        for k, v in results.items():
            self.logger.info(f"  {k}: {v:.4f}")
