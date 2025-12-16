"""
Training utilities for FrustraMPNN.

This module provides utility functions used across the training pipeline.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get the best available device.

    Returns:
        torch.device: CUDA if available, else CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

