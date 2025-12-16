"""
Visualization tools for frustration analysis.

This module provides plotting functions matching frustrapy's style for
single-residue frustration analysis.

Color Scheme (from frustrapy):
    - highly frustrated: red (FrstIndex <= -1.0)
    - neutral: gray (-1.0 < FrstIndex < 0.58)
    - minimally frustrated: green (FrstIndex >= 0.58)
    - native: blue (wild-type residue highlight)

This module re-exports functions from submodules for backward compatibility.
For new code, you can import directly from the submodules:
    - frustrampnn.visualization._core
    - frustrampnn.visualization.single_residue
    - frustrampnn.visualization.heatmap
"""

from __future__ import annotations

# Re-export core utilities
from frustrampnn.visualization._core import (
    classify_frustration,
)

# Re-export heatmap plots
from frustrampnn.visualization.heatmap import (
    plot_frustration_heatmap,
    plot_frustration_heatmap_plotly,
)

# Re-export single residue plots
from frustrampnn.visualization.single_residue import (
    plot_single_residue,
    plot_single_residue_plotly,
)

__all__ = [
    "plot_single_residue",
    "plot_single_residue_plotly",
    "plot_frustration_heatmap",
    "plot_frustration_heatmap_plotly",
    "classify_frustration",
]


