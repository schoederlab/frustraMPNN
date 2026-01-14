"""
Dataset classes for FrustraMPNN training.

Provides dataset implementations for various frustration prediction datasets:
- FireProtDataset: FireProt database mutations
- MegaScaleDataset: MegaScale database mutations
- ComboDataset: Combined dataset wrapper

Example:
    >>> from frustrampnn.training.datasets import get_dataset
    >>> train_dataset = get_dataset(config, "train")

    >>> # Or use specific datasets
    >>> from frustrampnn.training.datasets import FireProtDataset
    >>> dataset = FireProtDataset(config, "train")
"""

from frustrampnn.training.datasets.base import (
    ALPHABET,
    BaseDataset,
    TrainingMutation,
    align_sequences,
    get_dataset,
    seq1_index_to_seq2_index,
)
from frustrampnn.training.datasets.combo import ComboDataset
from frustrampnn.training.datasets.fireprot import FireProtDataset
from frustrampnn.training.datasets.megascale import MegaScaleDataset

__all__ = [
    # Constants
    "ALPHABET",
    # Base classes
    "BaseDataset",
    "TrainingMutation",
    # Dataset classes
    "FireProtDataset",
    "MegaScaleDataset",
    "ComboDataset",
    # Factory function
    "get_dataset",
    # Utilities
    "seq1_index_to_seq2_index",
    "align_sequences",
]
