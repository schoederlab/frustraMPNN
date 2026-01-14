"""
Metrics for FrustraMPNN training.

This module provides metric functions and utilities for tracking
training progress and model performance.

Original source: test_data/training/train_thermompnn_refac.py (lines 51-57)
Backup: test_data/training/train_thermompnn_refac.py.bak
"""

from __future__ import annotations

import torch
from torchmetrics import MeanSquaredError, Metric, SpearmanCorrCoef
from torchmetrics.regression import R2Score


def get_metrics() -> dict[str, Metric]:
    """
    Create a dictionary of metrics for training.

    This function matches the original implementation exactly from
    test_data/training/train_thermompnn_refac.py (lines 51-57).

    Returns:
        Dictionary mapping metric name to torchmetrics instance

    Example:
        >>> metrics = get_metrics()
        >>> metrics['mse'].update(pred, target)
        >>> value = metrics['mse'].compute()
    """
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


def create_metric_dict(
    prefixes: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Metric]]]:
    """
    Create nested metric dictionary for multiple splits and tasks.

    This function creates a structure matching the original
    TransferModelPL.metrics dictionary from train_thermompnn_refac.py.

    Args:
        prefixes: Split prefixes (default: ["train", "val", "test"])
        tasks: Task names (default: ["frustration"])

    Returns:
        Nested dictionary: prefix_metrics -> task -> metric_name -> Metric

    Example:
        >>> metrics = create_metric_dict()
        >>> metrics["train_metrics"]["frustration"]["mse"].update(pred, target)
    """
    if prefixes is None:
        prefixes = ["train", "val", "test"]
    if tasks is None:
        tasks = ["frustration"]

    result: dict[str, dict[str, dict[str, Metric]]] = {}
    for prefix in prefixes:
        result[f"{prefix}_metrics"] = {}
        for task in tasks:
            result[f"{prefix}_metrics"][task] = get_metrics()

    return result


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """
    Compute all metrics for a batch of predictions.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]

    Returns:
        Dictionary of metric values

    Example:
        >>> results = compute_all_metrics(pred, target)
        >>> print(f"R²: {results['r2']:.3f}")
    """
    metrics = get_metrics()
    results: dict[str, float] = {}

    for name, metric in metrics.items():
        metric.update(predictions, targets)
        results[name] = metric.compute().item()
        metric.reset()

    return results


def aggregate_epoch_metrics(
    all_predictions: list[torch.Tensor],
    all_targets: list[torch.Tensor],
) -> dict[str, float]:
    """
    Aggregate metrics over an entire epoch.

    Args:
        all_predictions: List of prediction tensors
        all_targets: List of target tensors

    Returns:
        Dictionary of aggregated metric values

    Example:
        >>> preds = [torch.randn(10) for _ in range(5)]
        >>> targets = [torch.randn(10) for _ in range(5)]
        >>> results = aggregate_epoch_metrics(preds, targets)
    """
    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return compute_all_metrics(predictions, targets)


class MetricTracker:
    """
    Track metrics across training epochs.

    This class provides a convenient interface for tracking and
    logging metrics during training.

    Args:
        prefixes: Split prefixes to track
        tasks: Task names to track

    Example:
        >>> tracker = MetricTracker()
        >>> tracker.update("train", "frustration", pred, target)
        >>> results = tracker.compute("train", "frustration")
        >>> tracker.reset("train", "frustration")
    """

    def __init__(
        self,
        prefixes: list[str] | None = None,
        tasks: list[str] | None = None,
    ) -> None:
        """
        Initialize MetricTracker.

        Args:
            prefixes: Split prefixes (default: ["train", "val", "test"])
            tasks: Task names (default: ["frustration"])
        """
        self.metrics = create_metric_dict(prefixes, tasks)

    def update(
        self,
        prefix: str,
        task: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Update metrics with new predictions.

        Args:
            prefix: Split prefix ("train", "val", "test")
            task: Task name ("frustration")
            predictions: Model predictions
            targets: Ground truth values
        """
        metrics = self.metrics[f"{prefix}_metrics"][task]
        for metric in metrics.values():
            metric.update(predictions, targets)

    def compute(self, prefix: str, task: str) -> dict[str, float]:
        """
        Compute current metric values.

        Args:
            prefix: Split prefix ("train", "val", "test")
            task: Task name ("frustration")

        Returns:
            Dictionary of metric values
        """
        metrics = self.metrics[f"{prefix}_metrics"][task]
        return {name: m.compute().item() for name, m in metrics.items()}

    def reset(self, prefix: str, task: str) -> None:
        """
        Reset metrics for a split/task.

        Args:
            prefix: Split prefix ("train", "val", "test")
            task: Task name ("frustration")
        """
        metrics = self.metrics[f"{prefix}_metrics"][task]
        for metric in metrics.values():
            metric.reset()

    def reset_all(self) -> None:
        """Reset all metrics."""
        for prefix_metrics in self.metrics.values():
            for task_metrics in prefix_metrics.values():
                for metric in task_metrics.values():
                    metric.reset()

    def to(self, device: torch.device) -> MetricTracker:
        """
        Move all metrics to a device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        for prefix_metrics in self.metrics.values():
            for task_metrics in prefix_metrics.values():
                for metric in task_metrics.values():
                    metric.to(device)
        return self


__all__ = [
    "get_metrics",
    "create_metric_dict",
    "compute_all_metrics",
    "aggregate_epoch_metrics",
    "MetricTracker",
]
