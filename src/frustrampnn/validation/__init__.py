"""
Validation tools for comparing FrustraMPNN with frustrapy.

This module provides tools for validating FrustraMPNN predictions against
the physics-based frustrapy (FrustratometeR wrapper) calculations.

Example:
    >>> from frustrampnn import FrustraMPNN
    >>> from frustrampnn.validation import compare_with_frustrapy
    >>>
    >>> model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
    >>> results = model.predict("protein.pdb", chains=["A"])
    >>>
    >>> # Compare with frustrapy (only a few positions - it's slow!)
    >>> comparison = compare_with_frustrapy(
    ...     pdb_path="protein.pdb",
    ...     chain="A",
    ...     frustrampnn_results=results,
    ...     positions=[50, 73, 120],
    ... )
    >>>
    >>> print(f"Spearman: {comparison.spearman:.3f}")
    >>> print(f"RMSE: {comparison.rmse:.3f}")
    >>> comparison.plot()

Note:
    frustrapy is an optional dependency. If not installed, importing
    comparison functions will raise FrustrapyNotInstalledError when called.
"""

from frustrampnn.validation.compare import (
    align_predictions,
    compare_with_frustrapy,
    quick_validate,
)
from frustrampnn.validation.frustrapy_wrapper import (
    FrustrapyNotInstalledError,
    convert_pdb_numbering_to_positions,
    convert_positions_to_pdb_numbering,
    extract_single_residue_data,
    get_pdb_residue_mapping,
    get_reverse_residue_mapping,
    run_frustrapy_single_residue,
)
from frustrampnn.validation.metrics import (
    compute_all_metrics,
    compute_mae,
    compute_pearson,
    compute_per_position_metrics,
    compute_rmse,
    compute_spearman,
)
from frustrampnn.validation.results import ComparisonResult, PositionComparison

__all__ = [
    # Main comparison function
    "compare_with_frustrapy",
    "quick_validate",
    "align_predictions",
    # Result containers
    "ComparisonResult",
    "PositionComparison",
    # frustrapy wrapper
    "run_frustrapy_single_residue",
    "extract_single_residue_data",
    "FrustrapyNotInstalledError",
    # Position mapping utilities
    "get_pdb_residue_mapping",
    "get_reverse_residue_mapping",
    "convert_positions_to_pdb_numbering",
    "convert_pdb_numbering_to_positions",
    # Metrics
    "compute_spearman",
    "compute_pearson",
    "compute_rmse",
    "compute_mae",
    "compute_all_metrics",
    "compute_per_position_metrics",
]
