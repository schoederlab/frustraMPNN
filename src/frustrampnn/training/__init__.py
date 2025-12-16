"""
FrustraMPNN Training Module.

This module provides training functionality for FrustraMPNN models,
including dataset handling, PyTorch Lightning integration, and
training utilities.

Example:
    >>> from frustrampnn.training import Trainer, TrainingConfig
    >>> config = TrainingConfig.from_yaml("config.yaml")
    >>> trainer = Trainer(config)
    >>> trainer.fit()
"""

from frustrampnn.training.config import (
    DataLocConfig,
    ModelArchConfig,
    PlatformConfig,
    TrainingConfig,
    TrainingHyperparamsConfig,
)
from frustrampnn.training.metrics import (
    MetricTracker,
    aggregate_epoch_metrics,
    compute_all_metrics,
    create_metric_dict,
    get_metrics,
)
from frustrampnn.training.trainer import Trainer

__all__ = [
    # Config classes
    "TrainingConfig",
    "TrainingHyperparamsConfig",
    "ModelArchConfig",
    "DataLocConfig",
    "PlatformConfig",
    # Metrics
    "get_metrics",
    "compute_all_metrics",
    "aggregate_epoch_metrics",
    "create_metric_dict",
    "MetricTracker",
    # Trainer
    "Trainer",
]


