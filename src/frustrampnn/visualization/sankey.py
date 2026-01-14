"""
Sankey diagram for frustration category flow visualization.

This module provides functions for creating Sankey diagrams that show the flow
of residue counts between calculated (FrustratometeR) and predicted (FrustraMPNN)
frustration categories, as shown in Figure 3 of the manuscript.

The Sankey diagram visualizes:
- Left side: Calculated frustration category counts
- Right side: Predicted frustration category counts
- Flows: How residues transition between categories
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure
    import plotly.graph_objects as go

__all__ = [
    "plot_frustration_sankey",
    "plot_frustration_sankey_matplotlib",
    "compute_category_flows",
]

logger = logging.getLogger(__name__)

# Sankey color scheme matching the manuscript
SANKEY_COLORS = {
    "minimally": "rgba(144, 238, 144, 0.8)",  # Light green with transparency
    "neutral": "rgba(211, 211, 211, 0.8)",  # Light gray with transparency
    "highly": "rgba(255, 182, 193, 0.8)",  # Light red/pink with transparency
}

# Node colors (solid)
NODE_COLORS = {
    "minimally": "#90EE90",  # Light green
    "neutral": "#D3D3D3",  # Light gray
    "highly": "#FFB6C1",  # Light red/pink
}


def compute_category_flows(
    calculated: pd.DataFrame,
    predicted: pd.DataFrame,
    position_col: str = "position",
    category_col: str = "category",
) -> tuple[dict[str, int], dict[str, int], dict[tuple[str, str], int]]:
    """
    Compute category counts and flows between calculated and predicted.

    Args:
        calculated: DataFrame with calculated frustration categories
        predicted: DataFrame with predicted frustration categories
        position_col: Column name for positions
        category_col: Column name for categories

    Returns:
        Tuple of:
        - calc_counts: Dict mapping category to count for calculated
        - pred_counts: Dict mapping category to count for predicted
        - flows: Dict mapping (calc_cat, pred_cat) to count

    Example:
        >>> calc_counts, pred_counts, flows = compute_category_flows(calc_df, pred_df)
        >>> print(f"Highly frustrated: {calc_counts['highly']} -> {pred_counts['highly']}")
    """
    # Merge on position
    merged = calculated[[position_col, category_col]].merge(
        predicted[[position_col, category_col]],
        on=position_col,
        suffixes=("_calc", "_pred"),
    )

    # Count categories
    calc_counts = merged[f"{category_col}_calc"].value_counts().to_dict()
    pred_counts = merged[f"{category_col}_pred"].value_counts().to_dict()

    # Ensure all categories are present
    for cat in ["minimally", "neutral", "highly"]:
        calc_counts.setdefault(cat, 0)
        pred_counts.setdefault(cat, 0)

    # Compute flows
    flows = {}
    for calc_cat in ["minimally", "neutral", "highly"]:
        for pred_cat in ["minimally", "neutral", "highly"]:
            count = len(
                merged[
                    (merged[f"{category_col}_calc"] == calc_cat)
                    & (merged[f"{category_col}_pred"] == pred_cat)
                ]
            )
            if count > 0:
                flows[(calc_cat, pred_cat)] = count

    return calc_counts, pred_counts, flows


def plot_frustration_sankey(
    calculated: pd.DataFrame,
    predicted: pd.DataFrame,
    title: str | None = None,
    width: int = 600,
    height: int = 400,
    show_counts: bool = True,
    position_col: str = "position",
    category_col: str = "category",
) -> go.Figure:
    """
    Create Sankey diagram showing frustration category flow.

    Creates an interactive Sankey diagram showing how residues flow between
    calculated and predicted frustration categories.

    Args:
        calculated: DataFrame with calculated frustration categories.
            Must contain: position, category
        predicted: DataFrame with predicted frustration categories.
            Must contain: position, category
        title: Plot title (auto-generated if None)
        width: Figure width in pixels
        height: Figure height in pixels
        show_counts: Whether to show counts in node labels
        position_col: Column name for positions
        category_col: Column name for categories

    Returns:
        plotly.graph_objects.Figure: Interactive Sankey diagram

    Example:
        >>> # Get native frustration from predictions and calculations
        >>> pred_native = get_native_frustration_per_position(results, chain="A")
        >>> calc_native = get_native_frustration_per_position(calc_results, chain="A")
        >>>
        >>> fig = plot_frustration_sankey(calc_native, pred_native, title="4BLM")
        >>> fig.show()
    """
    import plotly.graph_objects as go

    # Compute flows
    calc_counts, pred_counts, flows = compute_category_flows(
        calculated, predicted, position_col, category_col
    )

    # Define nodes (left: Calc, right: Pred)
    # Order: Highly, Minimal, Neutral (top to bottom, matching manuscript)
    categories_order = ["highly", "minimally", "neutral"]

    # Node indices: 0-2 for Calc, 3-5 for Pred
    node_labels = []
    node_colors = []

    # Calculated nodes (left side)
    for cat in categories_order:
        count = calc_counts.get(cat, 0)
        label = f"{cat.capitalize()}\n{count}" if show_counts else cat.capitalize()
        node_labels.append(label)
        node_colors.append(NODE_COLORS[cat])

    # Predicted nodes (right side)
    for cat in categories_order:
        count = pred_counts.get(cat, 0)
        label = f"{cat.capitalize()}\n{count}" if show_counts else cat.capitalize()
        node_labels.append(label)
        node_colors.append(NODE_COLORS[cat])

    # Build links
    source = []
    target = []
    value = []
    link_colors = []

    cat_to_idx = {cat: i for i, cat in enumerate(categories_order)}

    for (calc_cat, pred_cat), count in flows.items():
        source.append(cat_to_idx[calc_cat])
        target.append(cat_to_idx[pred_cat] + 3)  # +3 for right side
        value.append(count)
        # Color link by source category
        link_colors.append(SANKEY_COLORS[calc_cat])

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors,
                    x=[0.01, 0.01, 0.01, 0.99, 0.99, 0.99],  # Left and right positions
                    y=[0.1, 0.5, 0.9, 0.1, 0.5, 0.9],  # Vertical positions
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_colors,
                ),
            )
        ]
    )

    # Update layout
    if title is None:
        title = "Frustration Category Flow: Calculated → Predicted"

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, family="Arial"),
            x=0.5,
        ),
        font=dict(size=12, family="Arial"),
        width=width,
        height=height,
        paper_bgcolor="white",
        annotations=[
            dict(
                x=0.0,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Calc",
                showarrow=False,
                font=dict(size=12, family="Arial", color="gray"),
            ),
            dict(
                x=1.0,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Pred",
                showarrow=False,
                font=dict(size=12, family="Arial", color="gray"),
            ),
        ],
    )

    return fig


def plot_frustration_sankey_matplotlib(
    calculated: pd.DataFrame,
    predicted: pd.DataFrame,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    show_counts: bool = True,
    position_col: str = "position",
    category_col: str = "category",
) -> matplotlib.figure.Figure:
    """
    Create Sankey-style diagram using matplotlib (simplified version).

    Creates a static visualization showing frustration category flows.
    This is a simplified bar-based representation since matplotlib doesn't
    have native Sankey support.

    Args:
        calculated: DataFrame with calculated frustration categories
        predicted: DataFrame with predicted frustration categories
        title: Plot title
        figsize: Figure size as (width, height)
        show_counts: Whether to show counts in labels
        position_col: Column name for positions
        category_col: Column name for categories

    Returns:
        matplotlib.figure.Figure: The figure

    Note:
        For full Sankey diagram functionality, use plot_frustration_sankey()
        which uses Plotly.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # Compute flows
    calc_counts, pred_counts, flows = compute_category_flows(
        calculated, predicted, position_col, category_col
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Categories order (top to bottom)
    categories = ["highly", "minimally", "neutral"]

    # Bar positions
    left_x = 0.2
    right_x = 0.8
    bar_width = 0.15

    # Calculate total for normalization
    total = sum(calc_counts.values())

    # Draw bars and connections
    left_positions = {}
    right_positions = {}

    y_offset = 0.9
    for cat in categories:
        # Left bar (calculated)
        calc_height = calc_counts.get(cat, 0) / total * 0.7
        left_y = y_offset - calc_height / 2
        rect = plt.Rectangle(
            (left_x - bar_width / 2, left_y - calc_height / 2),
            bar_width,
            calc_height,
            facecolor=NODE_COLORS[cat],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        left_positions[cat] = (left_x, left_y)

        # Label
        label = (
            f"{cat.capitalize()}\n{calc_counts.get(cat, 0)}" if show_counts else cat.capitalize()
        )
        ax.text(left_x - bar_width / 2 - 0.02, left_y, label, ha="right", va="center", fontsize=9)

        # Right bar (predicted)
        pred_height = pred_counts.get(cat, 0) / total * 0.7
        right_y = y_offset - pred_height / 2
        rect = plt.Rectangle(
            (right_x - bar_width / 2, right_y - pred_height / 2),
            bar_width,
            pred_height,
            facecolor=NODE_COLORS[cat],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        right_positions[cat] = (right_x, right_y)

        # Label
        label = (
            f"{cat.capitalize()}\n{pred_counts.get(cat, 0)}" if show_counts else cat.capitalize()
        )
        ax.text(right_x + bar_width / 2 + 0.02, right_y, label, ha="left", va="center", fontsize=9)

        y_offset -= 0.3

    # Draw flow connections (simplified as lines)
    for (calc_cat, pred_cat), count in flows.items():
        if count > 0:
            left_pos = left_positions[calc_cat]
            right_pos = right_positions[pred_cat]

            # Line width proportional to count
            line_width = max(1, count / total * 20)

            # Draw bezier-like curve
            import matplotlib.patches as mpatches
            from matplotlib.path import Path

            # Control points for smooth curve
            mid_x = (left_pos[0] + right_pos[0]) / 2

            verts = [
                (left_pos[0] + bar_width / 2, left_pos[1]),
                (mid_x, left_pos[1]),
                (mid_x, right_pos[1]),
                (right_pos[0] - bar_width / 2, right_pos[1]),
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)

            patch = mpatches.PathPatch(
                path,
                facecolor="none",
                edgecolor=NODE_COLORS[calc_cat],
                linewidth=line_width,
                alpha=0.5,
            )
            ax.add_patch(patch)

    # Configure axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    if title is None:
        title = "Frustration Category Flow"
    ax.set_title(title, fontsize=12, fontweight="bold", y=1.02)

    # Column headers
    ax.text(left_x, 0.98, "Calc", ha="center", va="bottom", fontsize=10, color="gray")
    ax.text(right_x, 0.98, "Pred", ha="center", va="bottom", fontsize=10, color="gray")

    plt.tight_layout()
    return fig
