"""
FireProt dataset for FrustraMPNN training.

This module implements the FireProtDataset class for loading mutations
from the FireProt database with frustration values.

Original source: test_data/training/datasets.py (lines 170-276)
"""

from __future__ import annotations

import os
from math import isnan
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from Bio import pairwise2

from frustrampnn.training.datasets.base import (
    BaseDataset,
    TrainingMutation,
    seq1_index_to_seq2_index,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from frustrampnn.training.config import TrainingConfig


class FireProtDataset(BaseDataset):
    """
    Dataset for FireProt database mutations.

    The FireProt database contains experimentally measured thermostability
    data for protein mutations. This dataset loads mutations with their
    corresponding frustration values for training.

    This implementation matches the original FireProtDataset from
    test_data/training/datasets.py exactly.

    CSV Schema:
        - pdb_id_corrected: PDB identifier
        - pdb_sequence: Wild-type sequence
        - pdb_position: Mutation position (0-indexed)
        - wild_type: Wild-type amino acid
        - mutation: Mutant amino acid
        - frustration: Target frustration value
        - weight_*: Sample weights for reweighting

    Args:
        config: Training configuration
        split: Data split ("train", "val", "test", "homologue-free", "all")

    Example:
        >>> dataset = FireProtDataset(config, "train")
        >>> pdb, mutations = dataset[0]
        >>> print(f"Protein: {mutations[0].pdb}, Mutations: {len(mutations)}")
    """

    def __init__(
        self,
        config: TrainingConfig | DictConfig,
        split: str = "train",
    ) -> None:
        # Initialize storage before calling parent __init__
        self._df: pd.DataFrame = None
        self._splits: dict[str, list[str]] = {}
        self.seq_to_data: dict[str, pd.DataFrame] = {}
        self.split_wt_names: dict[str, list[str]] = {}
        self.reweighting: bool = False
        self.weight_method: str = "weight_lds_inverse"
        self.weight_rows: dict[str, Any] = {}

        super().__init__(config, split)

    def _load_data(self) -> None:
        """Load FireProt data from CSV and splits."""
        cfg = self.cfg

        # Get data paths
        csv_path = getattr(cfg.data_loc, "fireprot_csv", "")
        splits_path = getattr(cfg.data_loc, "fireprot_splits", "")

        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"FireProt CSV not found: {csv_path}")

        if not splits_path or not os.path.exists(splits_path):
            raise FileNotFoundError(f"FireProt splits not found: {splits_path}")

        # Load CSV (matches original exactly)
        df = pd.read_csv(csv_path).dropna(subset=["frustration"])
        df = df.where(pd.notnull(df), None)

        # Build seq_to_data mapping (matches original)
        seq_key = "pdb_sequence"
        for wt_seq in df[seq_key].unique():
            self.seq_to_data[wt_seq] = df.query(f"{seq_key} == @wt_seq").reset_index(
                drop=True
            )

        self._df = df

        # Load splits (matches original)
        self._splits = self._load_splits(splits_path)

        # Initialize split_wt_names (matches original)
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "homologue-free": [],
            "all": [],
        }

        # Get proteins for this split (matches original)
        if self.split == "all":
            all_names = list(self._splits.values())
            all_names = [j for sub in all_names for j in sub]
            self.split_wt_names[self.split] = all_names
        else:
            if self.split not in self._splits:
                raise ValueError(
                    f"Split '{self.split}' not found. "
                    f"Available: {list(self._splits.keys())}"
                )
            self.split_wt_names[self.split] = self._splits[self.split]

        # Get reweighting settings (matches original)
        self.reweighting = getattr(cfg.training, "reweighting", False)
        self.weight_method = getattr(cfg.training, "weight_method", "weight_lds_inverse")

        self.wt_names = self.split_wt_names[self.split]

        # Group mutations by protein (matches original)
        for wt_name in self.wt_names:
            self.mut_rows[wt_name] = df.query(
                "pdb_id_corrected == @wt_name"
            ).reset_index(drop=True)

            if self.split == "test":
                print(f"TEST wt_name: {wt_name}")

            self.wt_seqs[wt_name] = self.mut_rows[wt_name].pdb_sequence[0]

            if self.reweighting:
                self.weight_rows[wt_name] = self.mut_rows[wt_name][self.weight_method]
            else:
                self.weight_rows[wt_name] = 1

    def __len__(self) -> int:
        """Return number of proteins in dataset."""
        return len(self.wt_names)

    def __getitem__(self, index: int) -> tuple[list[dict], list[TrainingMutation]]:
        """
        Get protein and its mutations.

        This method matches the original __getitem__ from
        test_data/training/datasets.py exactly.

        Args:
            index: Dataset index

        Returns:
            Tuple of (pdb_dict_list, mutations_list)
        """
        wt_name = self.wt_names[index]
        seq = self.wt_seqs[wt_name]
        data = self.seq_to_data[seq]

        # Get PDB path (matches original)
        pdb_dir = getattr(self.cfg.data_loc, "fireprot_pdbs", "")
        pdb_file = os.path.join(pdb_dir, f"{data.pdb_id_corrected[0]}.pdb")

        # Parse PDB with caching
        pdb = self._parse_pdb_cached(pdb_file)

        # Create mutations (matches original exactly)
        mutations = []
        for i, row in data.iterrows():  # noqa: B007
            try:
                pdb_idx = row.pdb_position
                assert (
                    pdb[0]["seq"][pdb_idx]
                    == row.wild_type
                    == row.pdb_sequence[row.pdb_position]
                )

            except AssertionError:  # contingency for mis-alignments
                align, *rest = pairwise2.align.globalxx(
                    seq, pdb[0]["seq"].replace("-", "X")
                )
                pdb_idx = seq1_index_to_seq2_index(align, row.pdb_position)
                if pdb_idx is None:
                    continue

                assert (
                    pdb[0]["seq"][pdb_idx]
                    == row.wild_type
                    == row.pdb_sequence[row.pdb_position]
                )

            # Handle frustration value (matches original)
            frustration = (
                None
                if row.frustration is None or isnan(row.frustration)
                else torch.tensor([row.frustration], dtype=torch.float32)
            )

            # Create mutation with weight (matches original)
            if self.reweighting:
                weight = row[self.weight_method]
                mut = TrainingMutation(
                    pdb_idx,
                    pdb[0]["seq"][pdb_idx],
                    row.mutation,
                    frustration,
                    wt_name,
                    weight,
                )
            else:
                mut = TrainingMutation(
                    pdb_idx,
                    pdb[0]["seq"][pdb_idx],
                    row.mutation,
                    frustration,
                    wt_name,
                    1,
                )
            mutations.append(mut)

        return pdb, mutations

    def get_protein_names(self) -> list[str]:
        """Get list of protein names in dataset."""
        return self.wt_names.copy()

    def get_mutation_count(self) -> int:
        """Get total number of mutations across all proteins."""
        return sum(len(self.mut_rows[name]) for name in self.wt_names)

