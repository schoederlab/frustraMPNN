"""
MSE loss functions for FrustraMPNN training.

This module provides MSE loss variants with optional sample weighting.
The loss computation matches the original implementation from
test_data/training/train_thermompnn_refac.py (lines 100-115).

Original source: test_data/training/train_thermompnn_refac.py
Backup: test_data/training/train_thermompnn_refac.py.bak
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from frustrampnn.training.datasets.base import TrainingMutation


def mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute MSE loss.

    Args:
        predictions: Model predictions [N] or [N, 1]
        targets: Ground truth values [N] or [N, 1]
        reduction: "mean", "sum", or "none"

    Returns:
        Loss tensor

    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([1.1, 2.1, 3.1])
        >>> loss = mse_loss(pred, target)
    """
    return F.mse_loss(predictions, targets, reduction=reduction)


def weighted_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    normalize_weights: bool = True,
) -> torch.Tensor:
    """
    Compute weighted MSE loss.

    This function matches the reweighting logic from the original
    train_thermompnn_refac.py (lines 103-106).

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        weights: Sample weights [N]
        normalize_weights: Whether to normalize weights to sum to 1

    Returns:
        Weighted loss tensor

    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([1.1, 2.1, 3.1])
        >>> weights = torch.tensor([1.0, 2.0, 1.0])
        >>> loss = weighted_mse_loss(pred, target, weights)
    """
    # Compute per-sample MSE
    per_sample_loss = F.mse_loss(predictions, targets, reduction="none")

    # Normalize weights if requested
    if normalize_weights:
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum

    # Apply weights
    weighted_loss = (per_sample_loss * weights).sum()

    return weighted_loss


def batch_mse_loss(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute MSE loss for a batch of predictions.

    This function handles the case where predictions and targets
    are lists (one per mutation in a protein). It matches the
    original implementation from train_thermompnn_refac.py (lines 95-115).

    Args:
        predictions: List of prediction tensors
        targets: List of target tensors
        weights: Optional list of sample weights
        device: Target device

    Returns:
        Mean loss tensor

    Example:
        >>> preds = [torch.tensor([1.0]), torch.tensor([2.0])]
        >>> targets = [torch.tensor([1.1]), torch.tensor([2.1])]
        >>> loss = batch_mse_loss(preds, targets)
    """
    if not predictions or not targets:
        return torch.tensor(0.0, requires_grad=True, device=device)

    losses = []
    weight_sum = 0.0

    # Calculate weight sum for normalization (matches original line 97)
    if weights is not None:
        weight_sum = sum(w for w in weights if w is not None) or 1.0

    for i, (pred, target) in enumerate(zip(predictions, targets)):
        if target is None:
            continue

        # Move to device if specified
        if device is not None:
            target = target.to(device)

        # Compute loss (matches original line 105 or 108)
        loss = F.mse_loss(pred, target)

        # Apply weight if available (matches original line 106)
        if weights is not None and weights[i] is not None:
            loss = loss * (weights[i] / weight_sum)

        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=device)

    # Return mean loss (matches original line 115)
    return torch.stack(losses).mean()


class FrustrationLoss:
    """
    Loss function for frustration prediction.

    This class encapsulates the loss computation logic with
    optional sample reweighting. It provides a clean interface
    that matches the original shared_eval logic.

    Args:
        reweighting: Whether to apply sample weights
        reduction: Loss reduction method

    Example:
        >>> loss_fn = FrustrationLoss(reweighting=True)
        >>> loss = loss_fn(predictions, mutations)
    """

    def __init__(
        self,
        reweighting: bool = False,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize FrustrationLoss.

        Args:
            reweighting: Whether to apply sample weights
            reduction: Loss reduction method
        """
        self.reweighting = reweighting
        self.reduction = reduction

    def __call__(
        self,
        predictions: List[dict],
        mutations: List["TrainingMutation"],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute loss for predictions.

        This method matches the loss computation in shared_eval
        from train_thermompnn_refac.py (lines 100-115).

        Args:
            predictions: List of prediction dicts with "frustration" key
            mutations: List of TrainingMutation objects
            device: Target device

        Returns:
            Loss tensor
        """
        pred_values = []
        target_values = []
        weights = []

        for pred, mut in zip(predictions, mutations):
            if mut.frustration is not None:
                pred_values.append(pred["frustration"])
                target_values.append(mut.frustration)
                weights.append(mut.weight if self.reweighting else None)

        if not pred_values:
            return torch.tensor(0.0, requires_grad=True, device=device)

        return batch_mse_loss(
            pred_values,
            target_values,
            weights if self.reweighting else None,
            device,
        )


__all__ = [
    "mse_loss",
    "weighted_mse_loss",
    "batch_mse_loss",
    "FrustrationLoss",
]

