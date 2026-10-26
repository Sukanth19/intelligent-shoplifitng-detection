"""
Logging utilities for training
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_str: Optional custom format string
    
    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_str)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricTracker:
    """
    Track metrics during training/validation.
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_average(self, key: str) -> float:
        """Get average value for a metric."""
        if key not in self.metrics or self.counts[key] == 0:
            return 0.0
        return self.metrics[key] / self.counts[key]
    
    def get_all_averages(self) -> dict:
        """Get all averaged metrics."""
        return {
            key: self.get_average(key)
            for key in self.metrics.keys()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()


class TensorBoardLogger:
    """
    Simple TensorBoard logger wrapper.
    """
    
    def __init__(self, log_dir: Path, enabled: bool = True):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether to enable logging
        """
        self.enabled = enabled
        self.writer = None
        
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(log_dir))
                logging.info(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                logging.warning("TensorBoard not available. Install with: pip install tensorboard")
                self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close the writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()


def print_training_header(config: dict):
    """Print training configuration header."""
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    
    # Model
    if 'model' in config:
        print("\nModel:")
        for key, value in config['model'].items():
            print(f"  {key}: {value}")
    
    # Training
    if 'training' in config:
        print("\nTraining:")
        for key, value in config['training'].items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
    
    # Loss
    if 'loss' in config:
        print("\nLoss:")
        for key, value in config['loss'].items():
            print(f"  {key}: {value}")
    
    # Data
    if 'data' in config:
        print("\nData:")
        for key, value in config['data'].items():
            if key in ['feature_dir', 'batch_size', 'num_workers']:
                print(f"  {key}: {value}")
    
    print("=" * 80 + "\n")


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_metrics: dict,
    val_metrics: Optional[dict] = None,
    elapsed_time: Optional[float] = None
):
    """Print summary for an epoch."""
    print(f"\nEpoch [{epoch}/{total_epochs}]")
    print("-" * 60)
    
    # Training metrics
    print("Training:")
    for key, value in train_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Validation metrics
    if val_metrics:
        print("\nValidation:")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Time
    if elapsed_time:
        print(f"\nElapsed: {elapsed_time:.2f}s")
    
    print("-" * 60)
