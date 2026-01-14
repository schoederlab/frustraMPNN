"""
Loss functions for FrustraMPNN training.

Provides MSE loss variants and sample reweighting strategies.

Example:
    >>> from frustrampnn.training.loss import mse_loss, FrustrationLoss
    >>> loss = mse_loss(predictions, targets)
    >>> loss_fn = FrustrationLoss(reweighting=True)
"""

from frustrampnn.training.loss.mse import (
    FrustrationLoss,
    batch_mse_loss,
    mse_loss,
    weighted_mse_loss,
)
from frustrampnn.training.loss.reweighting import (
    VALID_WEIGHT_METHODS,
    SampleReweighter,
    compute_bin_weights,
    compute_lds_weights,
    get_weight_method,
    validate_weight_method,
)

__all__ = [
    # Loss functions
    "mse_loss",
    "weighted_mse_loss",
    "batch_mse_loss",
    "FrustrationLoss",
    # Reweighting
    "compute_bin_weights",
    "compute_lds_weights",
    "get_weight_method",
    "SampleReweighter",
    "VALID_WEIGHT_METHODS",
    "validate_weight_method",
]
