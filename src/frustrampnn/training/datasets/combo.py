"""
Combined dataset for FrustraMPNN training.

This module implements the ComboDataset class that combines multiple
datasets for training on diverse data sources.

Original source: test_data/training/datasets.py (lines 351-368)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import ConcatDataset, Dataset

from frustrampnn.training.datasets.base import TrainingMutation
from frustrampnn.training.datasets.fireprot import FireProtDataset
from frustrampnn.training.datasets.megascale import MegaScaleDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from frustrampnn.training.config import TrainingConfig


class ComboDataset(Dataset):
    """
    Combined dataset for training on multiple data sources.

    This dataset wraps multiple datasets (FireProt, MegaScale) into
    a single unified dataset for training. It uses PyTorch's
    ConcatDataset internally.

    This implementation matches the original ComboDataset from
    test_data/training/datasets.py exactly.

    The datasets to include are determined by the config.datasets field:
    - "fireprot": Include FireProt dataset
    - "megascale": Include MegaScale dataset
    - "combo" or "fireprot,megascale": Include both

    Args:
        config: Training configuration
        split: Data split ("train", "val", "test")

    Example:
        >>> # Config with datasets="combo" or datasets="fireprot,megascale"
        >>> dataset = ComboDataset(config, "train")
        >>> pdb, mutations = dataset[0]
    """

    def __init__(
        self,
        config: TrainingConfig | DictConfig,
        split: str = "train",
    ) -> None:
        self.cfg = config
        self.config = config
        self.split = split

        # Determine which datasets to include (matches original)
        datasets_str = getattr(config, "datasets", "combo")
        datasets_list = []

        # Create datasets (matches original logic)
        if "fireprot" in datasets_str:
            fireprot = FireProtDataset(config, split)
            datasets_list.append(fireprot)

        if "megascale" in datasets_str:
            mega_scale = MegaScaleDataset(config, split)
            datasets_list.append(mega_scale)

        # Combine datasets (matches original)
        self.mut_dataset = ConcatDataset(datasets_list)

        # Store individual datasets for introspection
        self._datasets = datasets_list

    def __getitem__(self, index: int) -> tuple[list[dict], list[TrainingMutation]]:
        """
        Get item at index.

        Args:
            index: Dataset index

        Returns:
            Tuple of (pdb_dict_list, mutations_list)
        """
        return self.mut_dataset[index]

    def __len__(self) -> int:
        """Return total size of combined dataset."""
        return len(self.mut_dataset)

    def get_dataset_sizes(self) -> dict[str, int]:
        """
        Get sizes of individual datasets.

        Returns:
            Dictionary mapping dataset name to size
        """
        sizes = {}
        for ds in self._datasets:
            name = ds.__class__.__name__
            sizes[name] = len(ds)
        return sizes

    def get_total_mutations(self) -> int:
        """
        Get total number of mutations across all datasets.

        Returns:
            Total mutation count
        """
        total = 0
        for ds in self._datasets:
            if hasattr(ds, "get_mutation_count"):
                total += ds.get_mutation_count()
        return total

