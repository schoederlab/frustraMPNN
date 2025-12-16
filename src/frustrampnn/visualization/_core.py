"""
Core utilities for frustration visualization.

This module provides shared utilities for frustration classification
and data preparation used by all visualization functions.
"""

from __future__ import annotations

import logging

import pandas as pd

from frustrampnn.constants import (
    ALPHABET,
    FRUSTRATION_COLORS,
    FRUSTRATION_THRESHOLDS,
)

__all__ = [
    "classify_frustration",
    "get_color_for_row",
    "prepare_single_residue_data",
]

logger = logging.getLogger(__name__)


def classify_frustration(value: float) -> str:
    """
    Classify frustration value into category.

    Uses the same thresholds as frustrapy:
        - highly: FrstIndex <= -1.0
        - minimally: FrstIndex >= 0.58
        - neutral: -1.0 < FrstIndex < 0.58

    Args:
        value: Frustration index value

    Returns:
        str: 'highly', 'neutral', or 'minimally'

    Example:
        >>> classify_frustration(-2.0)
        'highly'
        >>> classify_frustration(0.0)
        'neutral'
        >>> classify_frustration(1.0)
        'minimally'
    """
    if value <= FRUSTRATION_THRESHOLDS["highly"]:
        return "highly"
    elif value >= FRUSTRATION_THRESHOLDS["minimally"]:
        return "minimally"
    else:
        return "neutral"


def get_color_for_row(
    row: pd.Series,
    wildtype: str,
    frustration_col: str = "frustration_pred",
) -> str:
    """
    Get color for a data row based on frustration and native status.

    Args:
        row: DataFrame row with frustration data
        wildtype: Wild-type amino acid at this position
        frustration_col: Column name for frustration values

    Returns:
        str: Color name ('red', 'gray', 'green', or 'blue')
    """
    if row["mutation"] == wildtype:
        return FRUSTRATION_COLORS["native"]

    category = classify_frustration(row[frustration_col])
    return FRUSTRATION_COLORS[category]


def prepare_single_residue_data(
    results: pd.DataFrame,
    position: int,
    chain: str | None = None,
    frustration_col: str = "frustration_pred",
) -> tuple[pd.DataFrame, str]:
    """
    Prepare data for single-residue plot.

    Args:
        results: DataFrame with frustration predictions
        position: 0-indexed position to plot
        chain: Chain ID (required if multiple chains)
        frustration_col: Column name for frustration values

    Returns:
        Tuple of (filtered DataFrame, wildtype amino acid)

    Raises:
        ValueError: If no data found for position or chain
    """
    df = results[results["position"] == position].copy()

    if chain is not None:
        df = df[df["chain"] == chain]

    if df.empty:
        chain_str = f" chain {chain}" if chain else ""
        raise ValueError(f"No data found for position {position}{chain_str}")

    # Get wildtype
    wildtype = df["wildtype"].iloc[0]

    # Sort by amino acid order (standard ALPHABET order)
    aa_order = list(ALPHABET[:-1])  # Exclude X

    def get_aa_order(aa: str) -> int:
        try:
            return aa_order.index(aa)
        except ValueError:
            return 99

    df["aa_order"] = df["mutation"].apply(get_aa_order)
    df = df.sort_values("aa_order")

    # Add colors
    df["color"] = df.apply(lambda row: get_color_for_row(row, wildtype, frustration_col), axis=1)

    return df, wildtype
