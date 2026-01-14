"""
Sequence map / frustration profile visualization.

This module provides functions for creating sequence-level frustration profile
visualizations that compare calculated (FrustratometeR) vs predicted (FrustraMPNN)
frustration categories, as shown in Figure 3 of the manuscript.

The sequence map displays:
- Row 1: FrustratometeR calculated frustration categories
- Row 2: FrustraMPNN predicted frustration categories
- Row 3: Disagreement markers (dark blue where predictions differ)
- Row 4 (optional): Secondary structure elements (H=Helix, E=Sheet, L=Loop)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from frustrampnn.visualization._core import classify_frustration

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import plotly.graph_objects as go

__all__ = [
    "plot_sequence_map",
    "plot_sequence_map_plotly",
    "get_native_frustration_per_position",
]

logger = logging.getLogger(__name__)

# Color mapping for sequence map
SEQUENCE_MAP_COLORS = {
    "minimally": "#90EE90",  # Light green
    "neutral": "#D3D3D3",  # Light gray
    "highly": "#FFB6C1",  # Light red/pink
    "disagreement": "#00008B",  # Dark blue
    "gap": "#FFFFFF",  # White for gaps
}


def get_native_frustration_per_position(
    results: pd.DataFrame,
    chain: str | None = None,
    frustration_col: str = "frustration_pred",
) -> pd.DataFrame:
    """
    Extract native (wildtype) frustration values for each position.

    Args:
        results: DataFrame with frustration predictions containing columns:
            position, wildtype, mutation, frustration_pred
        chain: Chain ID to filter (optional)
        frustration_col: Column name for frustration values

    Returns:
        DataFrame with columns: position, wildtype, frustration, category
    """
    df = results.copy()
    if chain is not None:
        df = df[df["chain"] == chain]

    # Filter to native residues only (where mutation == wildtype)
    native_df = df[df["mutation"] == df["wildtype"]].copy()

    if native_df.empty:
        raise ValueError("No native residue data found in results")

    # Classify frustration
    native_df["category"] = native_df[frustration_col].apply(classify_frustration)

    # Select relevant columns
    result = native_df[["position", "wildtype", frustration_col, "category"]].copy()
    result = result.rename(columns={frustration_col: "frustration"})
    result = result.sort_values("position").reset_index(drop=True)

    return result


def _prepare_comparison_data(
    predicted: pd.DataFrame,
    calculated: pd.DataFrame | None = None,
    position_col: str = "position",
    category_col: str = "category",
) -> tuple[pd.DataFrame, list[int]]:
    """
    Prepare data for sequence map comparison.

    Args:
        predicted: DataFrame with predicted frustration categories
        calculated: DataFrame with calculated frustration categories (optional)
        position_col: Column name for positions
        category_col: Column name for categories

    Returns:
        Tuple of (merged DataFrame, list of positions)
    """
    # Get all positions
    positions = sorted(predicted[position_col].unique())

    # Create base dataframe
    data = pd.DataFrame({position_col: positions})

    # Merge predicted
    pred_subset = predicted[[position_col, category_col]].copy()
    pred_subset = pred_subset.rename(columns={category_col: "predicted"})
    data = data.merge(pred_subset, on=position_col, how="left")

    # Merge calculated if provided
    if calculated is not None:
        calc_subset = calculated[[position_col, category_col]].copy()
        calc_subset = calc_subset.rename(columns={category_col: "calculated"})
        data = data.merge(calc_subset, on=position_col, how="left")

        # Mark disagreements
        data["disagreement"] = data["predicted"] != data["calculated"]
    else:
        data["calculated"] = None
        data["disagreement"] = False

    return data, positions


def plot_sequence_map(
    predicted: pd.DataFrame,
    calculated: pd.DataFrame | None = None,
    secondary_structure: pd.DataFrame | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    show_position_labels: bool = True,
    label_interval: int = 20,
    surface_positions: list[int] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot sequence map comparing calculated vs predicted frustration profiles.

    Creates a horizontal bar visualization showing frustration categories for
    each residue position, with optional disagreement markers and secondary
    structure annotation.

    Args:
        predicted: DataFrame with predicted frustration data. Must contain:
            - position: 0-indexed residue position
            - category: frustration category ('highly', 'neutral', 'minimally')
        calculated: DataFrame with calculated (FrustratometeR) frustration data.
            Same format as predicted. If None, only predicted is shown.
        secondary_structure: DataFrame with secondary structure assignments.
            Must contain: position, ss (H=Helix, E=Sheet, L=Loop)
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height). Auto-calculated if None.
        show_position_labels: Whether to show position numbers on x-axis
        label_interval: Interval for position labels (e.g., every 20 residues)
        surface_positions: List of surface residue positions to mark with stars
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib.figure.Figure: The sequence map figure

    Example:
        >>> # Get native frustration from predictions
        >>> pred_native = get_native_frustration_per_position(results, chain="A")
        >>>
        >>> # Plot comparison with calculated values
        >>> fig = plot_sequence_map(pred_native, calc_native, title="4BLM Chain A")
        >>> fig.savefig("sequence_map.png", dpi=300, bbox_inches="tight")
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Prepare data
    data, positions = _prepare_comparison_data(predicted, calculated)
    n_positions = len(positions)

    # Determine number of rows
    n_rows = 2 if calculated is not None else 1
    if calculated is not None:
        n_rows += 1  # Disagreement row
    if secondary_structure is not None:
        n_rows += 1

    # Calculate figure size if not provided
    if figsize is None:
        width = max(12, n_positions * 0.08)
        height = 1.5 + n_rows * 0.5
        figsize = (width, height)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Category to numeric mapping
    cat_to_num = {"minimally": 0, "neutral": 1, "highly": 2}
    colors = [
        SEQUENCE_MAP_COLORS["minimally"],
        SEQUENCE_MAP_COLORS["neutral"],
        SEQUENCE_MAP_COLORS["highly"],
    ]
    cmap = ListedColormap(colors)

    # Build matrix for visualization
    row_labels = []
    matrix_data = []

    # Row 1: Calculated (if provided)
    if calculated is not None:
        calc_row = [cat_to_num.get(c, 1) if pd.notna(c) else np.nan for c in data["calculated"]]
        matrix_data.append(calc_row)
        row_labels.append("FrustratometeR")

    # Row 2: Predicted
    pred_row = [cat_to_num.get(c, 1) if pd.notna(c) else np.nan for c in data["predicted"]]
    matrix_data.append(pred_row)
    row_labels.append("FrustraMPNN")

    # Row 3: Disagreement (if calculated provided)
    if calculated is not None:
        disagree_row = [1 if d else 0 for d in data["disagreement"]]
        matrix_data.append(disagree_row)
        row_labels.append("Disagree")

    # Convert to numpy array
    matrix = np.array(matrix_data, dtype=float)

    # Plot main frustration rows
    n_main_rows = 2 if calculated is not None else 1
    ax.imshow(
        matrix[:n_main_rows],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest",
        extent=[0, n_positions, n_main_rows, 0],
    )

    # Plot disagreement row separately (dark blue)
    if calculated is not None:
        disagree_cmap = ListedColormap(["white", SEQUENCE_MAP_COLORS["disagreement"]])
        ax.imshow(
            matrix[n_main_rows : n_main_rows + 1],
            aspect="auto",
            cmap=disagree_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=[0, n_positions, n_main_rows + 1, n_main_rows],
        )

    # Add secondary structure row if provided
    if secondary_structure is not None:
        ss_row_y = len(matrix_data)
        ss_colors = {"H": "#4169E1", "E": "#FFD700", "L": "#808080"}  # Blue, Gold, Gray

        for _, row in secondary_structure.iterrows():
            pos = row["position"]
            ss = row.get("ss", "L")
            if pos in positions:
                x_idx = positions.index(pos)
                color = ss_colors.get(ss, "#808080")
                ax.add_patch(
                    plt.Rectangle((x_idx, ss_row_y), 1, 1, facecolor=color, edgecolor="none")
                )
        row_labels.append("SecStruct")

    # Add surface markers (stars) if provided
    if surface_positions is not None:
        for pos in surface_positions:
            if pos in positions:
                x_idx = positions.index(pos)
                ax.text(
                    x_idx + 0.5,
                    len(row_labels) + 0.3,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    # Configure axes
    ax.set_xlim(0, n_positions)
    ax.set_ylim(len(row_labels), 0)

    # Y-axis labels
    ax.set_yticks([i + 0.5 for i in range(len(row_labels))])
    ax.set_yticklabels(row_labels, fontsize=10)

    # X-axis labels
    if show_position_labels:
        # Show labels at intervals
        tick_positions = list(range(0, n_positions, label_interval))
        tick_labels = [positions[i] + 1 for i in tick_positions]  # 1-indexed
        ax.set_xticks([p + 0.5 for p in tick_positions])
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Position", fontsize=10)
    else:
        ax.set_xticks([])

    # Title
    if title is None:
        title = "Frustration Sequence Map"
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add legend
    legend_patches = [
        mpatches.Patch(color=SEQUENCE_MAP_COLORS["minimally"], label="Minimal"),
        mpatches.Patch(color=SEQUENCE_MAP_COLORS["neutral"], label="Neutral"),
        mpatches.Patch(color=SEQUENCE_MAP_COLORS["highly"], label="Highly"),
    ]
    if calculated is not None:
        legend_patches.append(
            mpatches.Patch(color=SEQUENCE_MAP_COLORS["disagreement"], label="Disagree")
        )
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize=8,
    )

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


def plot_sequence_map_plotly(
    predicted: pd.DataFrame,
    calculated: pd.DataFrame | None = None,
    secondary_structure: pd.DataFrame | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int = 300,
    show_position_labels: bool = True,
    label_interval: int = 20,
    surface_positions: list[int] | None = None,
) -> go.Figure:
    """
    Plot interactive sequence map comparing calculated vs predicted frustration.

    Creates an interactive horizontal bar visualization using Plotly.

    Args:
        predicted: DataFrame with predicted frustration data
        calculated: DataFrame with calculated frustration data (optional)
        secondary_structure: DataFrame with secondary structure (optional)
        title: Plot title
        width: Figure width in pixels (auto-calculated if None)
        height: Figure height in pixels
        show_position_labels: Whether to show position numbers
        label_interval: Interval for position labels
        surface_positions: List of surface residue positions to mark

    Returns:
        plotly.graph_objects.Figure: Interactive sequence map

    Example:
        >>> fig = plot_sequence_map_plotly(pred_native, calc_native)
        >>> fig.show()
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Prepare data
    data, positions = _prepare_comparison_data(predicted, calculated)
    n_positions = len(positions)

    # Calculate width if not provided
    if width is None:
        width = max(800, n_positions * 6)

    # Category to numeric and color mapping
    cat_to_num = {"minimally": 2, "neutral": 1, "highly": 0}

    # Custom colorscale: red (0) -> gray (1) -> green (2)
    colorscale = [
        [0, SEQUENCE_MAP_COLORS["highly"]],
        [0.5, SEQUENCE_MAP_COLORS["neutral"]],
        [1, SEQUENCE_MAP_COLORS["minimally"]],
    ]

    # Determine number of rows
    n_rows = 1
    row_titles = ["FrustraMPNN"]
    if calculated is not None:
        n_rows = 3
        row_titles = ["FrustratometeR", "FrustraMPNN", "Disagree"]
    if secondary_structure is not None:
        n_rows += 1
        row_titles.append("SecStruct")

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[1] * n_rows,
    )

    # Position labels for hover
    pos_labels = [f"Pos {p + 1}" for p in positions]

    row_idx = 1

    # Row 1: Calculated (if provided)
    if calculated is not None:
        calc_values = [cat_to_num.get(c, 1) if pd.notna(c) else None for c in data["calculated"]]
        calc_text = [c if pd.notna(c) else "gap" for c in data["calculated"]]

        fig.add_trace(
            go.Heatmap(
                z=[calc_values],
                x=pos_labels,
                colorscale=colorscale,
                zmin=0,
                zmax=2,
                showscale=False,
                text=[calc_text],
                hovertemplate="Position: %{x}<br>Category: %{text}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        row_idx += 1

    # Row 2: Predicted
    pred_values = [cat_to_num.get(c, 1) if pd.notna(c) else None for c in data["predicted"]]
    pred_text = [c if pd.notna(c) else "gap" for c in data["predicted"]]

    fig.add_trace(
        go.Heatmap(
            z=[pred_values],
            x=pos_labels,
            colorscale=colorscale,
            zmin=0,
            zmax=2,
            showscale=False,
            text=[pred_text],
            hovertemplate="Position: %{x}<br>Category: %{text}<extra></extra>",
        ),
        row=row_idx,
        col=1,
    )
    row_idx += 1

    # Row 3: Disagreement (if calculated provided)
    if calculated is not None:
        disagree_values = [1 if d else 0 for d in data["disagreement"]]
        disagree_colorscale = [
            [0, "white"],
            [1, SEQUENCE_MAP_COLORS["disagreement"]],
        ]

        fig.add_trace(
            go.Heatmap(
                z=[disagree_values],
                x=pos_labels,
                colorscale=disagree_colorscale,
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate="Position: %{x}<br>Disagree: %{z}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        row_idx += 1

    # Row 4: Secondary structure (if provided)
    if secondary_structure is not None:
        ss_map = {"H": 2, "E": 1, "L": 0}
        ss_colorscale = [
            [0, "#808080"],  # Loop - gray
            [0.5, "#FFD700"],  # Sheet - gold
            [1, "#4169E1"],  # Helix - blue
        ]

        # Create mapping from position to ss
        ss_dict = dict(
            zip(
                secondary_structure["position"],
                secondary_structure["ss"],
                strict=False,
            )
        )
        ss_values = [ss_map.get(ss_dict.get(p, "L"), 0) for p in positions]
        ss_text = [ss_dict.get(p, "L") for p in positions]

        fig.add_trace(
            go.Heatmap(
                z=[ss_values],
                x=pos_labels,
                colorscale=ss_colorscale,
                zmin=0,
                zmax=2,
                showscale=False,
                text=[ss_text],
                hovertemplate="Position: %{x}<br>SecStruct: %{text}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )

    # Update layout
    if title is None:
        title = "Frustration Sequence Map"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        width=width,
        height=height,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Update y-axes with row labels
    for i, label in enumerate(row_titles):
        fig.update_yaxes(
            title_text=label,
            row=i + 1,
            col=1,
            showticklabels=False,
            title_font=dict(size=10),
        )

    # Update x-axis
    if show_position_labels:
        tick_indices = list(range(0, n_positions, label_interval))
        fig.update_xaxes(
            tickmode="array",
            tickvals=[pos_labels[i] for i in tick_indices],
            ticktext=[str(positions[i] + 1) for i in tick_indices],
            tickfont={"size": 8},
            row=n_rows,
            col=1,
        )

    return fig
