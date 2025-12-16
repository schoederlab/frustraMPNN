"""
Wrapper for frustrapy single-residue frustration calculations.

This module provides a clean interface to frustrapy's calculate_frustration
function, handling temporary directories, amino acid code conversion,
and result extraction.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from frustrampnn.constants import AA_3_TO_1

__all__ = [
    "run_frustrapy_single_residue",
    "FrustrapyNotInstalledError",
    "extract_single_residue_data",
]

logger = logging.getLogger(__name__)


class FrustrapyNotInstalledError(ImportError):
    """Raised when frustrapy is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "frustrapy is not installed. Install with: pip install frustrapy\n"
            "Or install from source: pip install git+https://github.com/engelberger/frustrapy.git"
        )


def _check_frustrapy_available() -> bool:
    """Check if frustrapy is available."""
    try:
        import frustrapy  # noqa: F401

        return True
    except ImportError:
        return False


def run_frustrapy_single_residue(
    pdb_path: str,
    chain: str,
    positions: list[int],
    results_dir: str | None = None,
    cleanup: bool = True,
    n_cpus: int | None = None,
    suppress_output: bool = True,
) -> tuple[dict[int, dict[str, float]], float]:
    """
    Run frustrapy single-residue frustration calculation.

    This function wraps frustrapy's calculate_frustration in single-residue mode
    for specific positions. Note that frustrapy is slow (~3-4 minutes per residue).

    Args:
        pdb_path: Path to PDB file
        chain: Chain identifier
        positions: List of PDB residue numbers (1-indexed) to analyze
        results_dir: Directory for results (temp dir if None)
        cleanup: Whether to clean up temporary files
        n_cpus: Number of CPUs for parallel processing (None = all)
        suppress_output: Suppress frustrapy console output

    Returns:
        Tuple of:
            - Dict mapping position -> {mutation_1letter: frustration_value}
            - Time taken in seconds

    Raises:
        FrustrapyNotInstalledError: If frustrapy is not installed
        FileNotFoundError: If PDB file not found
        ValueError: If no valid positions provided

    Example:
        >>> results, time_taken = run_frustrapy_single_residue(
        ...     "protein.pdb", "A", [50, 73, 120]
        ... )
        >>> print(results[73])  # {'A': -0.5, 'C': 0.3, ...}
    """
    if not _check_frustrapy_available():
        raise FrustrapyNotInstalledError()

    # Import frustrapy here to avoid import errors when not installed
    from frustrapy import calculate_frustration

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    if not positions:
        raise ValueError("No positions provided for analysis")

    # Create temporary directory if needed
    use_temp_dir = results_dir is None
    if use_temp_dir:
        results_dir = tempfile.mkdtemp(prefix="frustrapy_")
    else:
        os.makedirs(results_dir, exist_ok=True)

    logger.info(
        f"Running frustrapy single-residue analysis for {len(positions)} positions"
    )
    logger.info(f"Positions: {positions}")
    logger.info(f"Results directory: {results_dir}")

    start_time = time.time()

    try:
        # Prepare residues dict for frustrapy
        residues = {chain: positions}

        # Suppress output if requested
        if suppress_output:
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

        try:
            # Run frustrapy calculation
            pdb_obj, plots, density_results, single_residue_data = calculate_frustration(
                pdb_file=str(pdb_path),
                chain=chain,
                residues=residues,
                mode="singleresidue",
                graphics=False,
                visualization=False,
                results_dir=results_dir,
                debug=False,
                overwrite=True,
                n_cpus=n_cpus,
            )
        finally:
            if suppress_output:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        elapsed_time = time.time() - start_time
        logger.info(f"frustrapy calculation completed in {elapsed_time:.1f}s")

        # Extract results
        results = extract_single_residue_data(
            pdb_obj, single_residue_data, chain, positions
        )

        return results, elapsed_time

    finally:
        # Cleanup temporary directory
        if use_temp_dir and cleanup:
            try:
                shutil.rmtree(results_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")


def extract_single_residue_data(
    pdb_obj: Any,
    single_residue_data: dict | None,
    chain: str,
    positions: list[int],
) -> dict[int, dict[str, float]]:
    """
    Extract single-residue frustration data from frustrapy results.

    Converts 3-letter amino acid codes to 1-letter codes.

    Args:
        pdb_obj: frustrapy Pdb object
        single_residue_data: Single residue data from calculate_frustration
        chain: Chain identifier
        positions: List of positions analyzed

    Returns:
        Dict mapping position -> {mutation_1letter: frustration_value}
    """
    results: dict[int, dict[str, float]] = {}

    if single_residue_data is None:
        logger.warning("No single residue data returned from frustrapy")
        return results

    # Get data for the specified chain
    chain_data = single_residue_data.get(chain, {})

    for position in positions:
        if position not in chain_data:
            logger.warning(f"Position {position} not found in frustrapy results")
            continue

        res_data = chain_data[position]

        # Convert 3-letter codes to 1-letter codes
        position_results: dict[str, float] = {}
        for aa_3letter, frustration in res_data.mutations.items():
            # Handle both 3-letter and 1-letter codes
            if len(aa_3letter) == 3:
                aa_1letter = AA_3_TO_1.get(aa_3letter.upper())
                if aa_1letter is None:
                    logger.warning(f"Unknown amino acid code: {aa_3letter}")
                    continue
            else:
                aa_1letter = aa_3letter.upper()

            position_results[aa_1letter] = float(frustration)

        results[position] = position_results

    return results


def get_pdb_residue_mapping(
    pdb_path: str,
    chain: str,
) -> dict[int, int]:
    """
    Get mapping from 0-indexed sequence position to PDB residue number.

    Args:
        pdb_path: Path to PDB file
        chain: Chain identifier

    Returns:
        Dict mapping 0-indexed position -> PDB residue number
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError as e:
        raise ImportError(
            "BioPython is required for PDB parsing. "
            "Install with: pip install biopython"
        ) from e

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)

    mapping: dict[int, int] = {}
    seq_idx = 0

    for model in structure:
        for pdb_chain in model:
            if pdb_chain.id != chain:
                continue

            for residue in pdb_chain:
                # Skip hetero atoms and water
                if residue.id[0] != " ":
                    continue

                pdb_res_num = residue.id[1]
                mapping[seq_idx] = pdb_res_num
                seq_idx += 1

    return mapping


def get_reverse_residue_mapping(
    pdb_path: str,
    chain: str,
) -> dict[int, int]:
    """
    Get mapping from PDB residue number to 0-indexed sequence position.

    Args:
        pdb_path: Path to PDB file
        chain: Chain identifier

    Returns:
        Dict mapping PDB residue number -> 0-indexed position
    """
    forward_mapping = get_pdb_residue_mapping(pdb_path, chain)
    return {v: k for k, v in forward_mapping.items()}


def convert_positions_to_pdb_numbering(
    positions: list[int],
    pdb_path: str,
    chain: str,
) -> list[int]:
    """
    Convert 0-indexed positions to PDB residue numbers.

    Args:
        positions: List of 0-indexed positions
        pdb_path: Path to PDB file
        chain: Chain identifier

    Returns:
        List of PDB residue numbers
    """
    mapping = get_pdb_residue_mapping(pdb_path, chain)
    return [mapping.get(pos, pos + 1) for pos in positions]


def convert_pdb_numbering_to_positions(
    pdb_residue_nums: list[int],
    pdb_path: str,
    chain: str,
) -> list[int]:
    """
    Convert PDB residue numbers to 0-indexed positions.

    Args:
        pdb_residue_nums: List of PDB residue numbers
        pdb_path: Path to PDB file
        chain: Chain identifier

    Returns:
        List of 0-indexed positions
    """
    mapping = get_reverse_residue_mapping(pdb_path, chain)
    return [mapping.get(num, num - 1) for num in pdb_residue_nums]
