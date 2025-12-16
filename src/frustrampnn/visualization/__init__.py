"""
Visualization tools for frustration analysis.

This module provides plotting functions matching frustrapy's style for
single-residue frustration analysis, including:

- Single-residue plots: Bar charts showing frustration for all 20 AA variants
- Heatmaps: Full frustration landscape across all positions
- Sequence maps: Horizontal profile comparing calculated vs predicted
- Sankey diagrams: Flow visualization of category transitions

Color Scheme (from frustrapy):
    - highly frustrated: red (FrstIndex <= -1.0)
    - neutral: gray (-1.0 < FrstIndex < 0.58)
    - minimally frustrated: green (FrstIndex >= 0.58)
    - native: blue (wild-type residue highlight)

Example:
    >>> from frustrampnn import FrustraMPNN
    >>> from frustrampnn.visualization import (
    ...     plot_single_residue,
    ...     plot_frustration_heatmap,
    ...     plot_sequence_map,
    ...     plot_frustration_sankey,
    ... )
    >>>
    >>> model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
    >>> results = model.predict("protein.pdb", chains=["A"])
    >>>
    >>> # Single-residue plot at position 73 (0-indexed: 72)
    >>> fig = plot_single_residue(results, position=72, chain="A")
    >>>
    >>> # Full heatmap
    >>> fig = plot_frustration_heatmap(results, chain="A")
    >>>
    >>> # Sequence map comparison (requires calculated data)
    >>> from frustrampnn.visualization import get_native_frustration_per_position
    >>> pred_native = get_native_frustration_per_position(results, chain="A")
    >>> fig = plot_sequence_map(pred_native, calc_native)
"""

# Import from submodules
from frustrampnn.visualization._core import classify_frustration
from frustrampnn.visualization.heatmap import (
    plot_frustration_heatmap,
    plot_frustration_heatmap_plotly,
)
from frustrampnn.visualization.sankey import (
    compute_category_flows,
    plot_frustration_sankey,
    plot_frustration_sankey_matplotlib,
)
from frustrampnn.visualization.sequence_map import (
    get_native_frustration_per_position,
    plot_sequence_map,
    plot_sequence_map_plotly,
)
from frustrampnn.visualization.single_residue import (
    plot_single_residue,
    plot_single_residue_plotly,
)

__all__ = [
    # Core utilities
    "classify_frustration",
    # Single-residue plots
    "plot_single_residue",
    "plot_single_residue_plotly",
    # Heatmaps
    "plot_frustration_heatmap",
    "plot_frustration_heatmap_plotly",
    # Sequence maps
    "plot_sequence_map",
    "plot_sequence_map_plotly",
    "get_native_frustration_per_position",
    # Sankey diagrams
    "plot_frustration_sankey",
    "plot_frustration_sankey_matplotlib",
    "compute_category_flows",
]

