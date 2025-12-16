"""
Result containers for validation comparisons.

This module provides dataclasses for storing comparison results between
FrustraMPNN predictions and frustrapy (physics-based) calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "ComparisonResult",
    "PositionComparison",
]


@dataclass
class PositionComparison:
    """
    Comparison results for a single position.

    Stores the frustration values for all 20 amino acid variants
    at a specific position from both FrustraMPNN and frustrapy.

    Attributes:
        position: 0-indexed position in the sequence
        pdb_residue_num: PDB residue number (may differ from position)
        chain: Chain identifier
        wildtype: Wild-type amino acid (1-letter code)
        frustrampnn_values: Dict mapping mutation -> predicted frustration
        frustrapy_values: Dict mapping mutation -> calculated frustration
        spearman: Spearman correlation for this position
        pearson: Pearson correlation for this position
        rmse: RMSE for this position
        mae: MAE for this position
    """

    position: int
    pdb_residue_num: int
    chain: str
    wildtype: str
    frustrampnn_values: dict[str, float] = field(default_factory=dict)
    frustrapy_values: dict[str, float] = field(default_factory=dict)
    spearman: float | None = None
    pearson: float | None = None
    rmse: float | None = None
    mae: float | None = None

    @property
    def n_mutations(self) -> int:
        """Number of mutations compared at this position."""
        common_keys = set(self.frustrampnn_values.keys()) & set(
            self.frustrapy_values.keys()
        )
        return len(common_keys)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position": self.position,
            "pdb_residue_num": self.pdb_residue_num,
            "chain": self.chain,
            "wildtype": self.wildtype,
            "n_mutations": self.n_mutations,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "rmse": self.rmse,
            "mae": self.mae,
        }


@dataclass
class ComparisonResult:
    """
    Container for comparison results between FrustraMPNN and frustrapy.

    This class stores the full comparison data including per-position
    results and aggregate metrics.

    Attributes:
        pdb_path: Path to the PDB file used
        chain: Chain identifier
        positions: List of 0-indexed positions compared
        position_results: Per-position comparison results
        merged_data: DataFrame with aligned FrustraMPNN and frustrapy values
        spearman: Overall Spearman correlation
        pearson: Overall Pearson correlation
        rmse: Overall RMSE
        mae: Overall MAE
        n_comparisons: Total number of mutation comparisons
        frustrapy_time_seconds: Time taken for frustrapy calculations

    Example:
        >>> comparison = compare_with_frustrapy(pdb_path, chain, results)
        >>> print(f"Spearman: {comparison.spearman:.3f}")
        >>> print(f"RMSE: {comparison.rmse:.3f}")
        >>> comparison.plot()
    """

    pdb_path: str
    chain: str
    positions: list[int]
    position_results: list[PositionComparison] = field(default_factory=list)
    merged_data: pd.DataFrame | None = None

    # Aggregate metrics
    spearman: float | None = None
    pearson: float | None = None
    rmse: float | None = None
    mae: float | None = None

    # Metadata
    n_comparisons: int = 0
    frustrapy_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate and compute derived attributes."""
        if self.merged_data is not None:
            self.n_comparisons = len(self.merged_data)

    @property
    def n_positions(self) -> int:
        """Number of positions compared."""
        return len(self.positions)

    def summary(self) -> str:
        """Return a summary string of the comparison results."""
        lines = [
            f"Comparison Results for {self.pdb_path} chain {self.chain}",
            f"  Positions compared: {self.n_positions}",
            f"  Total comparisons: {self.n_comparisons}",
            f"  Spearman correlation: {self.spearman:.4f}" if self.spearman else "",
            f"  Pearson correlation: {self.pearson:.4f}" if self.pearson else "",
            f"  RMSE: {self.rmse:.4f}" if self.rmse else "",
            f"  MAE: {self.mae:.4f}" if self.mae else "",
            f"  frustrapy time: {self.frustrapy_time_seconds:.1f}s",
        ]
        return "\n".join(line for line in lines if line)

    def __repr__(self) -> str:
        return (
            f"ComparisonResult(pdb={self.pdb_path!r}, chain={self.chain!r}, "
            f"n_positions={self.n_positions}, spearman={self.spearman})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pdb_path": self.pdb_path,
            "chain": self.chain,
            "positions": self.positions,
            "n_positions": self.n_positions,
            "n_comparisons": self.n_comparisons,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "rmse": self.rmse,
            "mae": self.mae,
            "frustrapy_time_seconds": self.frustrapy_time_seconds,
            "position_results": [p.to_dict() for p in self.position_results],
        }

    def plot(
        self,
        figsize: tuple = (10, 8),
        show_identity: bool = True,
        show_regression: bool = True,
    ) -> Any:
        """
        Create a scatter plot comparing FrustraMPNN vs frustrapy values.

        Args:
            figsize: Figure size (width, height)
            show_identity: Show y=x identity line
            show_regression: Show linear regression line

        Returns:
            matplotlib Figure object

        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If no comparison data available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from e

        if self.merged_data is None or len(self.merged_data) == 0:
            raise ValueError("No comparison data available for plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(
            self.merged_data["frustrapy"],
            self.merged_data["frustrampnn"],
            alpha=0.5,
            s=20,
            label="Mutations",
        )

        # Get axis limits
        all_values = list(self.merged_data["frustrapy"]) + list(
            self.merged_data["frustrampnn"]
        )
        min_val = min(all_values) - 0.5
        max_val = max(all_values) + 0.5

        # Identity line
        if show_identity:
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                alpha=0.5,
                label="Identity (y=x)",
            )

        # Regression line
        if show_regression:
            try:
                import numpy as np
                from scipy import stats

                slope, intercept, _, _, _ = stats.linregress(
                    self.merged_data["frustrapy"], self.merged_data["frustrampnn"]
                )
                x_line = np.array([min_val, max_val])
                y_line = slope * x_line + intercept
                ax.plot(
                    x_line,
                    y_line,
                    "r-",
                    alpha=0.7,
                    label=f"Regression (slope={slope:.2f})",
                )
            except ImportError:
                pass  # scipy not available

        ax.set_xlabel("frustrapy (physics-based)", fontsize=12)
        ax.set_ylabel("FrustraMPNN (predicted)", fontsize=12)
        ax.set_title(
            f"FrustraMPNN vs frustrapy Comparison\n"
            f"Spearman={self.spearman:.3f}, RMSE={self.rmse:.3f}",
            fontsize=14,
        )
        ax.legend()
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_per_position(self, figsize: tuple = (12, 4)) -> Any:
        """
        Create a bar plot showing correlation per position.

        Args:
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from e

        if not self.position_results:
            raise ValueError("No per-position results available")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        positions = [p.pdb_residue_num for p in self.position_results]
        spearmans = [p.spearman or 0 for p in self.position_results]
        rmses = [p.rmse or 0 for p in self.position_results]

        # Spearman per position
        axes[0].bar(range(len(positions)), spearmans, color="steelblue")
        axes[0].set_xticks(range(len(positions)))
        axes[0].set_xticklabels(positions)
        axes[0].set_xlabel("Position (PDB numbering)")
        axes[0].set_ylabel("Spearman correlation")
        axes[0].set_title("Correlation per Position")
        axes[0].axhline(y=self.spearman or 0, color="red", linestyle="--", label="Overall")
        axes[0].legend()

        # RMSE per position
        axes[1].bar(range(len(positions)), rmses, color="coral")
        axes[1].set_xticks(range(len(positions)))
        axes[1].set_xticklabels(positions)
        axes[1].set_xlabel("Position (PDB numbering)")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("RMSE per Position")
        axes[1].axhline(y=self.rmse or 0, color="red", linestyle="--", label="Overall")
        axes[1].legend()

        plt.tight_layout()
        return fig
