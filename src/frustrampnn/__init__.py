"""
FrustraMPNN: Ultra-fast deep learning prediction of single-residue local energetic frustration.

This package provides tools for predicting frustration profiles in proteins using
a message-passing neural network trained via transfer learning from ProteinMPNN.

Example:
    >>> from frustrampnn import FrustraMPNN
    >>> model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
    >>> results = model.predict("protein.pdb")

Visualization:
    >>> from frustrampnn.visualization import plot_single_residue
    >>> fig = plot_single_residue(results, position=72, chain="A")
    >>> fig.savefig("position_73.png")
"""

__version__ = "1.0.0"
__author__ = (
    "Max Beining, Felipe Engelberger, Clara T. Schoeder, César A. Ramírez-Sarmiento, Jens Meiler"
)

from frustrampnn.constants import (
    AA_1_TO_3,
    AA_3_TO_1,
    ALPHABET,
    FRUSTRATION_COLORS,
    FRUSTRATION_THRESHOLDS,
    VOCAB_DIM,
)
from frustrampnn.data import Mutation, generate_ssm_mutations, parse_mutation_string
from frustrampnn.inference import FrustraMPNN

# Validation imports (optional - requires frustrapy for full functionality)
from frustrampnn.validation import (
    ComparisonResult,
    compare_with_frustrapy,
)

# Visualization imports (optional - requires matplotlib/plotly)
# These are imported lazily to avoid requiring viz dependencies for basic usage
from frustrampnn.visualization import (
    classify_frustration,
    compute_category_flows,
    get_native_frustration_per_position,
    plot_frustration_heatmap,
    plot_frustration_heatmap_plotly,
    plot_frustration_sankey,
    plot_frustration_sankey_matplotlib,
    plot_sequence_map,
    plot_sequence_map_plotly,
    plot_single_residue,
    plot_single_residue_plotly,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "FrustraMPNN",
    "Mutation",
    "parse_mutation_string",
    "generate_ssm_mutations",
    # Constants
    "ALPHABET",
    "VOCAB_DIM",
    "AA_3_TO_1",
    "AA_1_TO_3",
    "FRUSTRATION_THRESHOLDS",
    "FRUSTRATION_COLORS",
    # Visualization - Single residue
    "plot_single_residue",
    "plot_single_residue_plotly",
    # Visualization - Heatmaps
    "plot_frustration_heatmap",
    "plot_frustration_heatmap_plotly",
    # Visualization - Sequence maps
    "plot_sequence_map",
    "plot_sequence_map_plotly",
    "get_native_frustration_per_position",
    # Visualization - Sankey diagrams
    "plot_frustration_sankey",
    "plot_frustration_sankey_matplotlib",
    "compute_category_flows",
    # Visualization - Utilities
    "classify_frustration",
    # Validation
    "compare_with_frustrapy",
    "ComparisonResult",
]
