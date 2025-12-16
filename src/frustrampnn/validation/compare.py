"""
Comparison tools for validating FrustraMPNN against frustrapy.

This module provides the main comparison functions for validating
FrustraMPNN predictions against physics-based frustrapy calculations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from frustrampnn.validation.metrics import compute_all_metrics
from frustrampnn.validation.results import ComparisonResult, PositionComparison

__all__ = [
    "compare_with_frustrapy",
    "align_predictions",
]

logger = logging.getLogger(__name__)


def compare_with_frustrapy(
    pdb_path: str | Path,
    chain: str,
    frustrampnn_results: pd.DataFrame,
    positions: list[int] | None = None,
    n_cpus: int | None = None,
    results_dir: str | None = None,
    cleanup: bool = True,
) -> ComparisonResult:
    """
    Compare FrustraMPNN predictions with frustrapy calculations.

    This function runs frustrapy's single-residue frustration calculation
    for the specified positions and compares the results with FrustraMPNN
    predictions.

    **Warning**: frustrapy is slow (~3-4 minutes per residue). Only compare
    a few positions for validation, not the entire protein.

    Args:
        pdb_path: Path to PDB file
        chain: Chain identifier
        frustrampnn_results: DataFrame from FrustraMPNN.predict()
        positions: List of 0-indexed positions to compare (None = all in results)
        n_cpus: Number of CPUs for frustrapy (None = all available)
        results_dir: Directory for frustrapy results (temp if None)
        cleanup: Whether to clean up temporary files

    Returns:
        ComparisonResult: Container with comparison metrics and data

    Raises:
        FrustrapyNotInstalledError: If frustrapy is not installed
        FileNotFoundError: If PDB file not found
        ValueError: If no valid positions to compare

    Example:
        >>> from frustrampnn import FrustraMPNN
        >>> from frustrampnn.validation import compare_with_frustrapy
        >>>
        >>> model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
        >>> results = model.predict("protein.pdb", chains=["A"])
        >>>
        >>> # Compare only a few positions (frustrapy is slow!)
        >>> comparison = compare_with_frustrapy(
        ...     pdb_path="protein.pdb",
        ...     chain="A",
        ...     frustrampnn_results=results,
        ...     positions=[50, 73, 120],
        ... )
        >>>
        >>> print(f"Spearman: {comparison.spearman:.3f}")
        >>> print(f"RMSE: {comparison.rmse:.3f}")
    """
    from frustrampnn.validation.frustrapy_wrapper import (
        get_pdb_residue_mapping,
        run_frustrapy_single_residue,
    )

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Filter results for the specified chain
    chain_results = frustrampnn_results[frustrampnn_results["chain"] == chain].copy()

    if len(chain_results) == 0:
        raise ValueError(f"No FrustraMPNN results found for chain {chain}")

    # Determine positions to compare
    if positions is None:
        positions = sorted(chain_results["position"].unique().tolist())
        logger.warning(
            f"No positions specified. Using all {len(positions)} positions from results. "
            "This may take a very long time with frustrapy!"
        )

    # Validate positions exist in FrustraMPNN results
    available_positions = set(chain_results["position"].unique())
    valid_positions = [p for p in positions if p in available_positions]

    if not valid_positions:
        raise ValueError(
            f"None of the specified positions {positions} found in FrustraMPNN results. "
            f"Available positions: {sorted(available_positions)[:10]}..."
        )

    if len(valid_positions) < len(positions):
        missing = set(positions) - set(valid_positions)
        logger.warning(f"Positions not in FrustraMPNN results: {missing}")

    # Get PDB residue number mapping
    pos_to_pdb = get_pdb_residue_mapping(str(pdb_path), chain)
    pdb_positions = [pos_to_pdb.get(p, p + 1) for p in valid_positions]

    logger.info(f"Comparing {len(valid_positions)} positions with frustrapy")
    logger.info(f"0-indexed positions: {valid_positions}")
    logger.info(f"PDB residue numbers: {pdb_positions}")

    # Run frustrapy
    frustrapy_results, elapsed_time = run_frustrapy_single_residue(
        pdb_path=str(pdb_path),
        chain=chain,
        positions=pdb_positions,
        results_dir=results_dir,
        cleanup=cleanup,
        n_cpus=n_cpus,
    )

    # Align and compare results
    position_comparisons = []
    all_merged_rows = []

    for seq_pos, pdb_pos in zip(valid_positions, pdb_positions, strict=False):
        # Get FrustraMPNN predictions for this position
        pos_mpnn = chain_results[chain_results["position"] == seq_pos]

        if pdb_pos not in frustrapy_results:
            logger.warning(f"Position {pdb_pos} not in frustrapy results")
            continue

        frustrapy_pos = frustrapy_results[pdb_pos]

        # Get wildtype
        wildtype = pos_mpnn["wildtype"].iloc[0] if len(pos_mpnn) > 0 else "X"

        # Build comparison for this position
        mpnn_values: dict[str, float] = {}
        fpy_values: dict[str, float] = {}

        for _, row in pos_mpnn.iterrows():
            mut = row["mutation"]
            mpnn_values[mut] = row["frustration_pred"]

            if mut in frustrapy_pos:
                fpy_values[mut] = frustrapy_pos[mut]

                all_merged_rows.append(
                    {
                        "position": seq_pos,
                        "pdb_residue_num": pdb_pos,
                        "chain": chain,
                        "wildtype": wildtype,
                        "mutation": mut,
                        "frustrampnn": row["frustration_pred"],
                        "frustrapy": frustrapy_pos[mut],
                    }
                )

        # Compute per-position metrics
        if fpy_values:
            common_muts = set(mpnn_values.keys()) & set(fpy_values.keys())
            mpnn_arr = [mpnn_values[m] for m in common_muts]
            fpy_arr = [fpy_values[m] for m in common_muts]

            pos_metrics = compute_all_metrics(mpnn_arr, fpy_arr)

            pos_comparison = PositionComparison(
                position=seq_pos,
                pdb_residue_num=pdb_pos,
                chain=chain,
                wildtype=wildtype,
                frustrampnn_values=mpnn_values,
                frustrapy_values=fpy_values,
                spearman=pos_metrics["spearman"],
                pearson=pos_metrics["pearson"],
                rmse=pos_metrics["rmse"],
                mae=pos_metrics["mae"],
            )
            position_comparisons.append(pos_comparison)

    # Create merged DataFrame
    merged_df = pd.DataFrame(all_merged_rows) if all_merged_rows else pd.DataFrame()

    # Compute overall metrics
    if len(merged_df) > 0:
        overall_metrics = compute_all_metrics(
            merged_df["frustrampnn"].values,
            merged_df["frustrapy"].values,
        )
    else:
        overall_metrics = {
            "spearman": None,
            "pearson": None,
            "rmse": None,
            "mae": None,
        }

    return ComparisonResult(
        pdb_path=str(pdb_path),
        chain=chain,
        positions=valid_positions,
        position_results=position_comparisons,
        merged_data=merged_df,
        spearman=overall_metrics["spearman"],
        pearson=overall_metrics["pearson"],
        rmse=overall_metrics["rmse"],
        mae=overall_metrics["mae"],
        n_comparisons=len(merged_df),
        frustrapy_time_seconds=elapsed_time,
    )


def align_predictions(
    frustrampnn_results: pd.DataFrame,
    frustrapy_results: dict[int, dict[str, float]],
    chain: str,
    positions: list[int],
    pos_to_pdb: dict[int, int],
) -> pd.DataFrame:
    """
    Align FrustraMPNN predictions with frustrapy results.

    Args:
        frustrampnn_results: DataFrame from FrustraMPNN.predict()
        frustrapy_results: Dict from run_frustrapy_single_residue()
        chain: Chain identifier
        positions: List of 0-indexed positions
        pos_to_pdb: Mapping from 0-indexed position to PDB residue number

    Returns:
        DataFrame with aligned predictions
    """
    rows = []

    chain_results = frustrampnn_results[frustrampnn_results["chain"] == chain]

    for seq_pos in positions:
        pdb_pos = pos_to_pdb.get(seq_pos, seq_pos + 1)

        if pdb_pos not in frustrapy_results:
            continue

        pos_mpnn = chain_results[chain_results["position"] == seq_pos]
        frustrapy_pos = frustrapy_results[pdb_pos]

        for _, row in pos_mpnn.iterrows():
            mut = row["mutation"]
            if mut in frustrapy_pos:
                rows.append(
                    {
                        "position": seq_pos,
                        "pdb_residue_num": pdb_pos,
                        "chain": chain,
                        "wildtype": row["wildtype"],
                        "mutation": mut,
                        "frustrampnn": row["frustration_pred"],
                        "frustrapy": frustrapy_pos[mut],
                    }
                )

    return pd.DataFrame(rows)


def quick_validate(
    model: FrustraMPNN,  # noqa: F821
    pdb_path: str | Path,
    chain: str = "A",
    n_positions: int = 3,
    seed: int = 42,
) -> ComparisonResult:
    """
    Quick validation of FrustraMPNN against frustrapy.

    Selects a few random positions and compares predictions.
    Useful for sanity checking model performance.

    Args:
        model: FrustraMPNN model instance
        pdb_path: Path to PDB file
        chain: Chain identifier
        n_positions: Number of positions to compare
        seed: Random seed for position selection

    Returns:
        ComparisonResult: Comparison results

    Example:
        >>> from frustrampnn import FrustraMPNN
        >>> from frustrampnn.validation import quick_validate
        >>>
        >>> model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
        >>> result = quick_validate(model, "protein.pdb", n_positions=3)
        >>> print(f"Quick validation Spearman: {result.spearman:.3f}")
    """
    import random

    from frustrampnn.validation.frustrapy_wrapper import get_pdb_residue_mapping

    pdb_path = Path(pdb_path)

    # Get all positions
    pos_mapping = get_pdb_residue_mapping(str(pdb_path), chain)
    all_positions = list(pos_mapping.keys())

    # Select random positions
    random.seed(seed)
    selected_positions = random.sample(
        all_positions, min(n_positions, len(all_positions))
    )

    logger.info(f"Quick validation with positions: {selected_positions}")

    # Run FrustraMPNN prediction
    results = model.predict(pdb_path, chains=[chain], positions=selected_positions)

    # Compare with frustrapy
    return compare_with_frustrapy(
        pdb_path=pdb_path,
        chain=chain,
        frustrampnn_results=results,
        positions=selected_positions,
    )
