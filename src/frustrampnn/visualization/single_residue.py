"""
Single-residue frustration plots.

This module provides functions for plotting frustration values for all 20
amino acid variants at a specific position, matching frustrapy's style.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from frustrampnn.constants import AA_1_TO_3, FRUSTRATION_THRESHOLDS
from frustrampnn.visualization._core import prepare_single_residue_data

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import plotly.graph_objects as go

__all__ = [
    "plot_single_residue",
    "plot_single_residue_plotly",
]

logger = logging.getLogger(__name__)


def _generate_title(
    df: pd.DataFrame,
    position: int,
    chain: str | None,
    wildtype: str,
    title: str | None,
) -> str:
    """Generate plot title."""
    if title is not None:
        return title

    pdb = df["pdb"].iloc[0] if "pdb" in df.columns else "Unknown"
    chain_str = f" Chain {chain}" if chain else ""

    # Convert wildtype to 3-letter code for display
    wt_3letter = AA_1_TO_3.get(wildtype, wildtype)

    # Position is 0-indexed internally, display as 1-indexed
    return f"Single-Residue Frustration: {wt_3letter}{position + 1}{chain_str} ({pdb})"


def plot_single_residue(
    results: pd.DataFrame,
    position: int,
    chain: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show_thresholds: bool = True,
    frustration_col: str = "frustration_pred",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot frustration values for all amino acids at a single position (matplotlib).

    Creates a bar chart showing the predicted frustration index for all 20
    amino acid variants at a specific position. Colors indicate frustration
    category (red=highly, gray=neutral, green=minimally, blue=native).

    Args:
        results: DataFrame with frustration predictions (from FrustraMPNN.predict())
        position: 0-indexed position to plot
        chain: Chain ID (required if multiple chains in results)
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height)
        show_thresholds: Show horizontal threshold lines at -1.0 and 0.58
        frustration_col: Column name for frustration values
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib.figure.Figure: The plot figure

    Example:
        >>> results = model.predict("protein.pdb", chains=["A"])
        >>> fig = plot_single_residue(results, position=72, chain="A")
        >>> fig.savefig("position_73.png", dpi=300)

    Note:
        Position is 0-indexed internally but displayed as 1-indexed in the title.
    """
    import matplotlib.pyplot as plt

    # Prepare data
    df, wildtype = prepare_single_residue_data(results, position, chain, frustration_col)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot bars
    x = range(len(df))
    ax.bar(
        x,
        df[frustration_col],
        color=df["color"].tolist(),
        edgecolor="black",
        linewidth=0.5,
    )

    # Add threshold lines
    if show_thresholds:
        ax.axhline(
            y=FRUSTRATION_THRESHOLDS["highly"],
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Highly frustrated (≤ -1.0)",
        )
        ax.axhline(
            y=FRUSTRATION_THRESHOLDS["minimally"],
            color="green",
            linestyle="--",
            alpha=0.7,
            label="Minimally frustrated (≥ 0.58)",
        )

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(df["mutation"].tolist())
    ax.set_xlabel("Amino Acid", fontsize=12)
    ax.set_ylabel("Frustration Index", fontsize=12)

    # Title
    plot_title = _generate_title(df, position, chain, wildtype, title)
    ax.set_title(plot_title, fontsize=14)

    # Legend
    if show_thresholds:
        ax.legend(loc="upper right", fontsize=10)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    return fig


def plot_single_residue_plotly(
    results: pd.DataFrame,
    position: int,
    chain: str | None = None,
    title: str | None = None,
    width: int = 800,
    height: int = 500,
    show_thresholds: bool = True,
    frustration_col: str = "frustration_pred",
) -> go.Figure:
    """
    Plot frustration values for all amino acids at a single position (plotly).

    Creates an interactive bar chart showing the predicted frustration index
    for all 20 amino acid variants at a specific position. Matches frustrapy's
    plot_mutate_res() style.

    Args:
        results: DataFrame with frustration predictions (from FrustraMPNN.predict())
        position: 0-indexed position to plot
        chain: Chain ID (required if multiple chains in results)
        title: Plot title (auto-generated if None)
        width: Figure width in pixels
        height: Figure height in pixels
        show_thresholds: Show horizontal threshold lines at -1.0 and 0.58
        frustration_col: Column name for frustration values

    Returns:
        plotly.graph_objects.Figure: Interactive plot figure

    Example:
        >>> results = model.predict("protein.pdb", chains=["A"])
        >>> fig = plot_single_residue_plotly(results, position=72, chain="A")
        >>> fig.show()
        >>> fig.write_html("position_73.html")
    """
    import plotly.graph_objects as go

    # Prepare data
    df, wildtype = prepare_single_residue_data(results, position, chain, frustration_col)

    # Create figure
    fig = go.Figure()

    # Add traces for each frustration state (for legend)
    color_order = [
        ("green", "Minimally frustrated"),
        ("gray", "Neutral"),
        ("red", "Highly frustrated"),
        ("blue", "Native"),
    ]

    for color, label in color_order:
        sub = df[df["color"] == color]
        if not sub.empty:
            fig.add_trace(
                go.Bar(
                    x=sub["mutation"].tolist(),
                    y=sub[frustration_col].tolist(),
                    marker_color=color,
                    marker_line_color="black",
                    marker_line_width=0.5,
                    name=label,
                    showlegend=True,
                )
            )

    # Add threshold lines
    if show_thresholds:
        fig.add_hline(
            y=FRUSTRATION_THRESHOLDS["highly"],
            line_dash="dash",
            line_color="rgba(128,128,128,0.5)",
            line_width=1,
            annotation_text="Highly (≤ -1.0)",
            annotation_position="bottom right",
        )
        fig.add_hline(
            y=FRUSTRATION_THRESHOLDS["minimally"],
            line_dash="dash",
            line_color="rgba(128,128,128,0.5)",
            line_width=1,
            annotation_text="Minimally (≥ 0.58)",
            annotation_position="top right",
        )

    # Title
    plot_title = _generate_title(df, position, chain, wildtype, title)

    # Update layout (matching frustrapy style)
    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(size=16, family="Arial"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Amino Acid", font=dict(size=14, family="Arial")),
            tickfont=dict(size=12, family="Arial"),
        ),
        yaxis=dict(
            title=dict(text="Frustration Index", font=dict(size=14, family="Arial")),
            tickfont=dict(size=12, family="Arial"),
            range=[-4, 4],
            tickvals=list(np.arange(-4, 4.5, 0.5)),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Arial"),
        ),
        width=width,
        height=height,
        barmode="overlay",
    )

    # Add box around plot
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    return fig


