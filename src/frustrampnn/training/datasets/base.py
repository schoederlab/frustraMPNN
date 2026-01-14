"""
Base dataset class for FrustraMPNN training.

This module provides the abstract base class and common utilities
for all training datasets.

Original source: test_data/training/datasets.py
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from frustrampnn.training.config import TrainingConfig


# Amino acid alphabet (matches original)
ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"


@dataclass
class TrainingMutation:
    """
    Mutation data for training.

    This dataclass matches the original Mutation class from
    test_data/training/datasets.py exactly.

    Attributes:
        position: 0-indexed position in sequence
        wildtype: Single-letter wildtype amino acid
        mutation: Single-letter mutant amino acid
        frustration: Target frustration value (as tensor)
        pdb: PDB identifier
        weight: Sample weight for loss reweighting
    """

    position: int
    wildtype: str
    mutation: str
    frustration: torch.Tensor | None = None
    pdb: str | None = ""
    weight: float | None = None


def seq1_index_to_seq2_index(align: Any, index: int) -> int | None:
    """
    Convert index from sequence 1 to sequence 2 after alignment.

    This function matches the original implementation exactly from
    test_data/training/datasets.py. It handles Bio.pairwise2 alignment
    objects which have seqA and seqB attributes.

    Args:
        align: Alignment object from pairwise2.align.globalxx with
            seqA and seqB attributes
        index: Position in sequence 1 (0-indexed)

    Returns:
        Corresponding position in sequence 2 (0-indexed), or None if
        the position maps to a gap in sequence 2

    Example:
        >>> from Bio import pairwise2
        >>> align, *rest = pairwise2.align.globalxx("ACDEF", "AC-EF")
        >>> seq1_index_to_seq2_index(align, 3)  # Position of E in seq1
        2  # Position of E in seq2
    """
    cur_seq1_index = 0

    # First find the aligned index
    aln_idx = 0
    for aln_idx, char in enumerate(align.seqA):  # noqa: B007
        if char != "-":
            cur_seq1_index += 1
        if cur_seq1_index > index:
            break

    # Now the index in seq 2 corresponding to aligned index
    if align.seqB[aln_idx] == "-":
        return None

    seq2_to_idx = align.seqB[: aln_idx + 1]
    seq2_idx = aln_idx
    for char in seq2_to_idx:
        if char == "-":
            seq2_idx -= 1

    if seq2_idx < 0:
        return None

    return seq2_idx


def align_sequences(seq1: str, seq2: str) -> Any:
    """
    Align two sequences using global alignment.

    Args:
        seq1: First sequence
        seq2: Second sequence (gaps replaced with X)

    Returns:
        First alignment object from pairwise2
    """
    from Bio import pairwise2

    alignments = pairwise2.align.globalxx(seq1, seq2.replace("-", "X"))
    if alignments:
        return alignments[0]
    return None


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for FrustraMPNN training datasets.

    This class provides common functionality for all training datasets:
    - Configuration handling
    - PDB parsing and caching
    - Split management
    - Mutation data loading

    Subclasses must implement:
    - _load_data(): Load dataset-specific data
    - __getitem__(): Return (pdb_dict, mutations) for an index
    - __len__(): Return dataset size

    Args:
        config: Training configuration (TrainingConfig or DictConfig)
        split: Data split ("train", "val", "test", "all")
    """

    def __init__(
        self,
        config: TrainingConfig | DictConfig,
        split: str = "train",
    ) -> None:
        self.cfg = config  # Use cfg to match original
        self.config = config  # Also expose as config
        self.split = split

        # Data storage (matches original)
        self.wt_names: list[str] = []
        self.mut_rows: dict[str, Any] = {}
        self.wt_seqs: dict[str, str] = {}

        # Load data
        self._load_data()

    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset-specific data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[list[dict], list[TrainingMutation]]:
        """
        Get item at index.

        Args:
            index: Dataset index

        Returns:
            Tuple of (pdb_dict_list, list of TrainingMutation)
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass

    def _load_splits(self, splits_path: str | Path) -> dict[str, list[str]]:
        """
        Load train/val/test splits from pickle file.

        Args:
            splits_path: Path to splits pickle file

        Returns:
            Dictionary mapping split name to list of protein names
        """
        with open(splits_path, "rb") as f:
            return pickle.load(f)

    def _get_pdb_path(self, pdb_dir: str, pdb_name: str) -> Path:
        """
        Get path to PDB file.

        Args:
            pdb_dir: Directory containing PDB files
            pdb_name: PDB identifier

        Returns:
            Path to PDB file
        """
        pdb_path = Path(pdb_dir) / f"{pdb_name}.pdb"
        if not pdb_path.exists():
            # Try without extension
            pdb_path = Path(pdb_dir) / pdb_name
        return pdb_path

    def _parse_pdb_cached(self, pdb_file: str) -> list[dict]:
        """
        Parse PDB file with caching.

        Uses the cached PDB parser from the cache module.

        Args:
            pdb_file: Path to PDB file

        Returns:
            List of parsed PDB dictionaries
        """
        from frustrampnn.model.pdb_parsing import parse_PDB
        from frustrampnn.training.cache import cache

        @cache(lambda cfg, pdb_file: pdb_file)
        def parse_pdb_cached_inner(cfg: Any, pdb_file: str) -> list[dict]:
            return parse_PDB(pdb_file)

        return parse_pdb_cached_inner(self.cfg, pdb_file)


def get_dataset(
    config: TrainingConfig | DictConfig,
    split: str,
) -> Dataset:
    """
    Factory function to get dataset based on configuration.

    Args:
        config: Training configuration
        split: Data split ("train", "val", "test")

    Returns:
        Dataset instance

    Example:
        >>> from frustrampnn.training.datasets import get_dataset
        >>> train_dataset = get_dataset(config, "train")
    """
    from frustrampnn.training.datasets.combo import ComboDataset
    from frustrampnn.training.datasets.fireprot import FireProtDataset
    from frustrampnn.training.datasets.megascale import MegaScaleDataset

    datasets_name = getattr(config, "datasets", "fireprot")

    if datasets_name == "fireprot":
        return FireProtDataset(config, split)
    elif datasets_name == "megascale":
        return MegaScaleDataset(config, split)
    elif datasets_name == "combo":
        return ComboDataset(config, split)
    else:
        raise ValueError(f"Unknown dataset: {datasets_name}")
