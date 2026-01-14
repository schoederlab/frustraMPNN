"""
MegaScale dataset for FrustraMPNN training.

This module implements the MegaScaleDataset class for loading mutations
from the MegaScale database with frustration values.

Original source: test_data/training/datasets.py (lines 62-167)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from frustrampnn.training.datasets.base import (
    BaseDataset,
    TrainingMutation,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from frustrampnn.training.config import TrainingConfig


class MegaScaleDataset(BaseDataset):
    """
    Dataset for MegaScale database mutations.

    The MegaScale database contains a large collection of protein mutations
    with stability measurements. This dataset uses AlphaFold-predicted
    structures for the wild-type proteins.

    This implementation matches the original MegaScaleDataset from
    test_data/training/datasets.py exactly.

    CSV Schema:
        - WT_name: Protein identifier
        - wt_seq: Wild-type sequence
        - aa_seq: Mutant sequence
        - mut_type: Mutation notation (e.g., "A123G")
        - frustration: Target frustration value

    Filtering:
        - Removes double mutants (containing ':')
        - Removes insertions (containing 'ins')
        - Removes deletions (containing 'del')
        - Removes invalid frustration values ('-')

    Args:
        config: Training configuration
        split: Data split ("train", "val", "test", "cv_train_0", etc.)

    Example:
        >>> dataset = MegaScaleDataset(config, "train")
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
        self.split_wt_names: dict[str, list[str]] = {}

        super().__init__(config, split)

    def _load_data(self) -> None:
        """Load MegaScale data from CSV and splits."""
        cfg = self.cfg

        # Get data paths
        csv_path = getattr(cfg.data_loc, "megascale_csv", "")
        splits_path = getattr(cfg.data_loc, "megascale_splits", "")

        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"MegaScale CSV not found: {csv_path}")

        if not splits_path or not os.path.exists(splits_path):
            raise FileNotFoundError(f"MegaScale splits not found: {splits_path}")

        # Load CSV with specific columns (matches original)
        df = pd.read_csv(
            csv_path,
            usecols=["frustration", "mut_type", "WT_name", "aa_seq", "wt_seq"],
        )

        # Remove unreliable data and more complicated mutations (matches original)
        df = df.loc[df.frustration != "-", :].reset_index(drop=True)
        df = df.loc[
            ~df.mut_type.str.contains("ins")
            & ~df.mut_type.str.contains("del")
            & ~df.mut_type.str.contains(":"),
            :,
        ].reset_index(drop=True)

        self._df = df

        # Load splits (matches original)
        self._splits = self._load_splits(splits_path)

        # Initialize split_wt_names (matches original exactly)
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "train_s669": [],
            "all": [],
            "cv_train_0": [],
            "cv_train_1": [],
            "cv_train_2": [],
            "cv_train_3": [],
            "cv_train_4": [],
            "cv_val_0": [],
            "cv_val_1": [],
            "cv_val_2": [],
            "cv_val_3": [],
            "cv_val_4": [],
            "cv_test_0": [],
            "cv_test_1": [],
            "cv_test_2": [],
            "cv_test_3": [],
            "cv_test_4": [],
        }

        # Handle reduce config (matches original)
        if not hasattr(cfg, "reduce"):
            cfg.reduce = ""

        # Get proteins for this split (matches original)
        if self.split == "all":
            all_names = self._splits["train"] + self._splits["val"] + self._splits["test"]
            self.split_wt_names[self.split] = all_names
        else:
            if getattr(cfg, "reduce", "") == "prot" and self.split == "train":
                n_prots_reduced = 58
                self.split_wt_names[self.split] = np.random.choice(
                    self._splits["train"], n_prots_reduced
                ).tolist()
            else:
                if self.split not in self._splits:
                    raise ValueError(
                        f"Split '{self.split}' not found. Available: {list(self._splits.keys())}"
                    )
                self.split_wt_names[self.split] = self._splits[self.split]

        self.wt_names = self.split_wt_names[self.split]

        # Group mutations by protein (matches original with tqdm)
        for wt_name in tqdm(self.wt_names):
            wt_rows = df.query("WT_name == @wt_name").reset_index(drop=True)
            self.mut_rows[wt_name] = df.query(
                'WT_name == @wt_name and mut_type != "wt"'
            ).reset_index(drop=True)

            # Handle reduce for mutations (matches original)
            reduce_val = getattr(cfg, "reduce", "")
            if isinstance(reduce_val, float) and self.split == "train":
                self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(
                    frac=float(reduce_val), replace=False
                )

            self.wt_seqs[wt_name] = wt_rows.wt_seq[0]

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
        mut_data = self.mut_rows[wt_name]
        wt_seq = self.wt_seqs[wt_name]

        # Process wt_name for PDB file (matches original)
        wt_name_pdb = wt_name.split(".pdb")[0].replace("|", ":")

        # Get PDB path (matches original)
        pdb_dir = getattr(self.cfg.data_loc, "megascale_pdbs", "")
        pdb_file = os.path.join(pdb_dir, f"{wt_name_pdb}.pdb")

        # Parse PDB with caching
        pdb = self._parse_pdb_cached(pdb_file)

        # Verify sequence length and update (matches original)
        assert len(pdb[0]["seq"]) == len(wt_seq)
        pdb[0]["seq"] = wt_seq

        # Create mutations (matches original exactly)
        mutations = []
        for i, row in mut_data.iterrows():  # noqa: B007
            # No insertions, deletions, or double mutants (matches original)
            if "ins" in row.mut_type or "del" in row.mut_type or ":" in row.mut_type:
                continue

            assert len(row.aa_seq) == len(wt_seq)
            wt = row.mut_type[0]
            mut = row.mut_type[-1]
            idx = int(row.mut_type[1:-1]) - 1

            assert wt_seq[idx] == wt
            assert row.aa_seq[idx] == mut

            if row.frustration == "-":
                continue  # filter out any unreliable data

            # Create frustration tensor (matches original)
            ddG = torch.tensor([float(row.frustration)], dtype=torch.float32)
            mutations.append(TrainingMutation(idx, wt, mut, ddG, wt_name_pdb))

        return pdb, mutations

    def get_protein_names(self) -> list[str]:
        """Get list of protein names in dataset."""
        return self.wt_names.copy()

    def get_mutation_count(self) -> int:
        """Get total number of mutations across all proteins."""
        return sum(len(self.mut_rows[name]) for name in self.wt_names)
