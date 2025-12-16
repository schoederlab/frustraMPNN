"""
Sample reweighting strategies for FrustraMPNN training.

This module provides methods for computing sample weights to handle
imbalanced frustration value distributions. The reweighting methods
are used during training to give more weight to underrepresented
frustration values.

Available methods:
- weight_bin_inverse: Inverse of bin frequency
- weight_lds_inverse: LDS smoothed inverse
- weight_bin_inverse_sqrt: Square root of inverse
- weight_lds_inverse_sqrt: Square root of LDS

Original source: test_data/training/train_thermompnn_refac.py (lines 276-283)
Backup: test_data/training/train_thermompnn_refac.py.bak
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def compute_bin_weights(
    values: np.ndarray,
    num_bins: int = 50,
    method: str = "inverse",
) -> np.ndarray:
    """
    Compute sample weights based on bin frequency.

    This method assigns weights inversely proportional to the
    frequency of values in each bin. Values in rare bins get
    higher weights.

    Args:
        values: Array of target values
        num_bins: Number of bins for histogram
        method: "inverse" or "inverse_sqrt"

    Returns:
        Array of sample weights

    Example:
        >>> values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        >>> weights = compute_bin_weights(values)
    """
    # Compute histogram
    hist, bin_edges = np.histogram(values, bins=num_bins)

    # Assign each value to a bin
    bin_indices = np.digitize(values, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Compute weights
    bin_counts = hist[bin_indices]
    bin_counts = np.maximum(bin_counts, 1)  # Avoid division by zero

    if method == "inverse":
        weights = 1.0 / bin_counts
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(bin_counts)
    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


def compute_lds_weights(
    values: np.ndarray,
    num_bins: int = 50,
    kernel_size: int = 5,
    sigma: float = 2.0,
    method: str = "inverse",
) -> np.ndarray:
    """
    Compute sample weights using Label Distribution Smoothing.

    LDS applies a Gaussian kernel to smooth bin counts before
    computing inverse weights, reducing noise in weight estimates.
    This is particularly useful when the distribution has many
    empty or near-empty bins.

    Args:
        values: Array of target values
        num_bins: Number of bins for histogram
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
        method: "inverse" or "inverse_sqrt"

    Returns:
        Array of sample weights

    Example:
        >>> values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        >>> weights = compute_lds_weights(values)
    """
    # Compute histogram
    hist, bin_edges = np.histogram(values, bins=num_bins)

    # Create Gaussian kernel
    kernel = _gaussian_kernel(kernel_size, sigma)

    # Smooth histogram with convolution
    smoothed_hist = np.convolve(hist, kernel, mode="same")
    smoothed_hist = np.maximum(smoothed_hist, 1e-6)  # Avoid division by zero

    # Assign each value to a bin
    bin_indices = np.digitize(values, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Compute weights from smoothed counts
    smoothed_counts = smoothed_hist[bin_indices]

    if method == "inverse":
        weights = 1.0 / smoothed_counts
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(smoothed_counts)
    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create a 1D Gaussian kernel.

    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation

    Returns:
        Normalized Gaussian kernel
    """
    x = np.arange(size) - size // 2
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def get_weight_method(method_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get weight computation function by name.

    This function returns the appropriate weight computation function
    based on the method name. The method names match those used in
    the original training configuration.

    Args:
        method_name: One of:
            - "weight_bin_inverse"
            - "weight_lds_inverse"
            - "weight_bin_inverse_sqrt"
            - "weight_lds_inverse_sqrt"

    Returns:
        Weight computation function

    Raises:
        ValueError: If method_name is not recognized

    Example:
        >>> compute_weights = get_weight_method("weight_lds_inverse")
        >>> weights = compute_weights(values)
    """
    methods = {
        "weight_bin_inverse": lambda v: compute_bin_weights(v, method="inverse"),
        "weight_lds_inverse": lambda v: compute_lds_weights(v, method="inverse"),
        "weight_bin_inverse_sqrt": lambda v: compute_bin_weights(
            v, method="inverse_sqrt"
        ),
        "weight_lds_inverse_sqrt": lambda v: compute_lds_weights(
            v, method="inverse_sqrt"
        ),
    }

    if method_name not in methods:
        raise ValueError(
            f"Unknown weight method: {method_name}. " f"Available: {list(methods.keys())}"
        )

    return methods[method_name]


# List of valid reweighting methods (matches original train_thermompnn_refac.py lines 277-278)
VALID_WEIGHT_METHODS = [
    "weight_bin_inverse",
    "weight_lds_inverse",
    "weight_bin_inverse_sqrt",
    "weight_lds_inverse_sqrt",
]


class SampleReweighter:
    """
    Sample reweighting utility class.

    This class provides a convenient interface for computing and
    applying sample weights during training.

    Args:
        method: Reweighting method name
        num_bins: Number of bins for histogram
        kernel_size: LDS kernel size (for LDS methods)
        sigma: LDS kernel sigma (for LDS methods)

    Example:
        >>> reweighter = SampleReweighter("weight_lds_inverse")
        >>> weights = reweighter.compute_weights(frustration_values)
    """

    def __init__(
        self,
        method: str = "weight_lds_inverse",
        num_bins: int = 50,
        kernel_size: int = 5,
        sigma: float = 2.0,
    ) -> None:
        """
        Initialize SampleReweighter.

        Args:
            method: Reweighting method name
            num_bins: Number of bins for histogram
            kernel_size: LDS kernel size (for LDS methods)
            sigma: LDS kernel sigma (for LDS methods)
        """
        if method not in VALID_WEIGHT_METHODS:
            raise ValueError(
                f"Unknown weight method: {method}. "
                f"Available: {VALID_WEIGHT_METHODS}"
            )

        self.method = method
        self.num_bins = num_bins
        self.kernel_size = kernel_size
        self.sigma = sigma

    def compute_weights(self, values: np.ndarray) -> np.ndarray:
        """
        Compute sample weights for given values.

        Args:
            values: Array of target values

        Returns:
            Array of sample weights
        """
        if "lds" in self.method:
            base_method = "inverse_sqrt" if "sqrt" in self.method else "inverse"
            return compute_lds_weights(
                values,
                num_bins=self.num_bins,
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                method=base_method,
            )
        else:
            base_method = "inverse_sqrt" if "sqrt" in self.method else "inverse"
            return compute_bin_weights(
                values,
                num_bins=self.num_bins,
                method=base_method,
            )

    def normalize_weights(
        self,
        weights: np.ndarray,
        target_sum: float = 1.0,
    ) -> np.ndarray:
        """
        Normalize weights to sum to target value.

        Args:
            weights: Sample weights
            target_sum: Target sum for weights

        Returns:
            Normalized weights
        """
        weight_sum = weights.sum()
        if weight_sum > 0:
            return weights * (target_sum / weight_sum)
        return weights


def validate_weight_method(method_name: str) -> bool:
    """
    Validate that a weight method name is valid.

    This function matches the validation in train_thermompnn_refac.py
    (lines 276-280).

    Args:
        method_name: Weight method name to validate

    Returns:
        True if valid, False otherwise
    """
    return method_name in VALID_WEIGHT_METHODS


__all__ = [
    "compute_bin_weights",
    "compute_lds_weights",
    "get_weight_method",
    "SampleReweighter",
    "VALID_WEIGHT_METHODS",
    "validate_weight_method",
]

