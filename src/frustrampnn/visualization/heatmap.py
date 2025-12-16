"""
Frustration heatmap plots.

This module provides functions for plotting frustration heatmaps showing
values for all amino acid variants across all positions in a protein.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from frustrampnn.constants import ALPHABET

if TYPE_CHECKING:
    import matplotlib.figure
    import plotly.graph_objects as go

__all__ = [
    "plot_frustration_heatmap",
    "plot_frustration_heatmap_plotly",
]

logger = logging.getLogger(__name__)


def plot_frustration_heatmap(
    results: pd.DataFrame,
    chain: str | None = None,
    figsize: tuple[int, int] = (14, 8),
    cmap: str = "RdYlGn_r",
    vmin: float = -3,
    vmax: float = 3,
    frustration_col: str = "frustration_pred",
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot frustration heatmap for all positions and mutations (matplotlib).

    Creates a heatmap showing frustration values for all amino acid variants
    across all positions in the protein.

    Args:
        results: DataFrame with frustration predictions
        chain: Chain ID (required if multiple chains)
        figsize: Figure size as (width, height)
        cmap: Matplotlib colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        frustration_col: Column name for frustration values
        title: Plot title (auto-generated if None)

    Returns:
        matplotlib.figure.Figure: The heatmap figure

    Example:
        >>> results = model.predict("protein.pdb", chains=["A"])
        >>> fig = plot_frustration_heatmap(results, chain="A")
        >>> fig.savefig("heatmap.png", dpi=300)
    """
    import matplotlib.pyplot as plt

    df = results.copy()
    if chain is not None:
        df = df[df["chain"] == chain]

    if df.empty:
        raise ValueError(f"No data found for chain {chain}")

    # Pivot to matrix
    aa_order = list(ALPHABET[:-1])  # Exclude X
    pivot = df.pivot_table(
        index="mutation",
        columns="position",
        values=frustration_col,
        aggfunc="mean",
    )

    # Reorder rows by standard amino acid order
    pivot = pivot.reindex(aa_order)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Labels
    ax.set_yticks(range(len(aa_order)))
    ax.set_yticklabels(aa_order)
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Mutation", fontsize=12)

    # X-axis ticks (show every 10th position for readability)
    positions = pivot.columns.tolist()
    if len(positions) > 20:
        tick_step = max(1, len(positions) // 10)
        tick_positions = range(0, len(positions), tick_step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([positions[i] + 1 for i in tick_positions])  # 1-indexed
    else:
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([p + 1 for p in positions])  # 1-indexed

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Frustration Index", fontsize=12)

    # Title
    if title is None:
        pdb = df["pdb"].iloc[0] if "pdb" in df.columns else "Unknown"
        chain_str = f" Chain {chain}" if chain else ""
        title = f"Frustration Heatmap{chain_str} ({pdb})"
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_frustration_heatmap_plotly(
    results: pd.DataFrame,
    chain: str | None = None,
    width: int = 1200,
    height: int = 600,
    colorscale: str = "RdYlGn_r",
    zmin: float = -3,
    zmax: float = 3,
    frustration_col: str = "frustration_pred",
    title: str | None = None,
) -> go.Figure:
    """
    Plot frustration heatmap for all positions and mutations (plotly).

    Creates an interactive heatmap showing frustration values for all amino
    acid variants across all positions in the protein.

    Args:
        results: DataFrame with frustration predictions
        chain: Chain ID (required if multiple chains)
        width: Figure width in pixels
        height: Figure height in pixels
        colorscale: Plotly colorscale name
        zmin: Minimum value for color scale
        zmax: Maximum value for color scale
        frustration_col: Column name for frustration values
        title: Plot title (auto-generated if None)

    Returns:
        plotly.graph_objects.Figure: Interactive heatmap figure

    Example:
        >>> results = model.predict("protein.pdb", chains=["A"])
        >>> fig = plot_frustration_heatmap_plotly(results, chain="A")
        >>> fig.show()
    """
    import plotly.graph_objects as go

    df = results.copy()
    if chain is not None:
        df = df[df["chain"] == chain]

    if df.empty:
        raise ValueError(f"No data found for chain {chain}")

    # Pivot to matrix
    aa_order = list(ALPHABET[:-1])  # Exclude X
    pivot = df.pivot_table(
        index="mutation",
        columns="position",
        values=frustration_col,
        aggfunc="mean",
    )

    # Reorder rows by standard amino acid order
    pivot = pivot.reindex(aa_order)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[p + 1 for p in pivot.columns.tolist()],  # 1-indexed
            y=aa_order,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Frustration Index"),
        )
    )

    # Title
    if title is None:
        pdb = df["pdb"].iloc[0] if "pdb" in df.columns else "Unknown"
        chain_str = f" Chain {chain}" if chain else ""
        title = f"Frustration Heatmap{chain_str} ({pdb})"

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Position", font=dict(size=14, family="Arial")),
            tickfont=dict(size=10, family="Arial"),
        ),
        yaxis=dict(
            title=dict(text="Mutation", font=dict(size=14, family="Arial")),
            tickfont=dict(size=12, family="Arial"),
        ),
        width=width,
        height=height,
        font=dict(family="Arial", size=12, color="black"),
    )

    return fig
