"""
Statistical metrics for validation comparisons.

This module provides functions for computing correlation and error metrics
between FrustraMPNN predictions and frustrapy calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "compute_spearman",
    "compute_pearson",
    "compute_rmse",
    "compute_mae",
    "compute_all_metrics",
]


def compute_spearman(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> float:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        x: First array of values
        y: Second array of values

    Returns:
        Spearman correlation coefficient

    Raises:
        ValueError: If arrays have different lengths or are too short
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")

    if len(x) < 2:
        raise ValueError("Need at least 2 values to compute correlation")

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    try:
        from scipy.stats import spearmanr

        corr, _ = spearmanr(x, y)
        return float(corr)
    except ImportError:
        # Fallback implementation without scipy
        return _spearman_fallback(x, y)


def _spearman_fallback(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation without scipy."""
    # Rank the values
    def rank_data(arr: np.ndarray) -> np.ndarray:
        sorted_indices = np.argsort(arr)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(1, len(arr) + 1)
        return ranks

    rank_x = rank_data(x)
    rank_y = rank_data(y)

    # Compute Pearson correlation on ranks
    return _pearson_impl(rank_x, rank_y)


def compute_pearson(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        x: First array of values
        y: Second array of values

    Returns:
        Pearson correlation coefficient

    Raises:
        ValueError: If arrays have different lengths or are too short
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")

    if len(x) < 2:
        raise ValueError("Need at least 2 values to compute correlation")

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    return _pearson_impl(x, y)


def _pearson_impl(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation (implementation)."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def compute_rmse(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> float:
    """
    Compute Root Mean Square Error.

    Args:
        x: First array of values (e.g., predictions)
        y: Second array of values (e.g., ground truth)

    Returns:
        RMSE value

    Raises:
        ValueError: If arrays have different lengths
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")

    if len(x) == 0:
        return np.nan

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    return float(np.sqrt(np.mean((x - y) ** 2)))


def compute_mae(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        x: First array of values (e.g., predictions)
        y: Second array of values (e.g., ground truth)

    Returns:
        MAE value

    Raises:
        ValueError: If arrays have different lengths
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")

    if len(x) == 0:
        return np.nan

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    return float(np.mean(np.abs(x - y)))


def compute_all_metrics(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> dict[str, float]:
    """
    Compute all metrics at once.

    Args:
        x: First array of values (e.g., FrustraMPNN predictions)
        y: Second array of values (e.g., frustrapy calculations)

    Returns:
        Dictionary with keys: spearman, pearson, rmse, mae

    Example:
        >>> metrics = compute_all_metrics(predictions, ground_truth)
        >>> print(f"Spearman: {metrics['spearman']:.3f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN values once for all metrics
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return {
            "spearman": np.nan,
            "pearson": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "n_valid": len(x_clean),
        }

    return {
        "spearman": compute_spearman(x_clean, y_clean),
        "pearson": compute_pearson(x_clean, y_clean),
        "rmse": compute_rmse(x_clean, y_clean),
        "mae": compute_mae(x_clean, y_clean),
        "n_valid": len(x_clean),
    }


def compute_per_position_metrics(
    df: pd.DataFrame,
    position_col: str = "position",
    pred_col: str = "frustrampnn",
    truth_col: str = "frustrapy",
) -> list[dict[str, float]]:
    """
    Compute metrics for each position separately.

    Args:
        df: DataFrame with comparison data
        position_col: Column name for position
        pred_col: Column name for predictions
        truth_col: Column name for ground truth

    Returns:
        List of dictionaries with metrics per position
    """
    results = []

    for position in df[position_col].unique():
        pos_data = df[df[position_col] == position]
        metrics = compute_all_metrics(
            pos_data[pred_col].values,
            pos_data[truth_col].values,
        )
        metrics["position"] = position
        results.append(metrics)

    return results
