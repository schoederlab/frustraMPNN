"""
High-level inference API for FrustraMPNN.

This module provides the main FrustraMPNN class for easy frustration prediction.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from frustrampnn.constants import ALPHABET
from frustrampnn.data.mutation import Mutation
from frustrampnn.model.pdb_parsing import alt_parse_PDB

if TYPE_CHECKING:
    from frustrampnn.model.transfer_model import TransferModel

__all__ = ["FrustraMPNN"]


class FrustraMPNN:
    """
    High-level interface for frustration prediction.

    This class provides a simple API for predicting single-residue local
    energetic frustration values for proteins.

    Example:
        >>> model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
        >>> results = model.predict("protein.pdb", chains=["A"])
        >>> print(results.head())
    """

    def __init__(
        self,
        model: TransferModel,
        cfg: OmegaConf,
        device: str | None = None,
    ) -> None:
        """
        Initialize FrustraMPNN.

        Args:
            model: Loaded TransferModel
            cfg: Model configuration
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.cfg = cfg

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str | None = None,
    ) -> FrustraMPNN:
        """
        Load a pretrained FrustraMPNN model.

        Args:
            checkpoint_path: Path to model checkpoint (.ckpt file)
            config_path: Optional path to config file (for old format checkpoints)
            device: Device to use ('cuda', 'cpu', or None for auto)

        Returns:
            FrustraMPNN: Loaded model ready for prediction

        Example:
            >>> model = FrustraMPNN.from_pretrained("model.ckpt")
        """
        # Import here to avoid circular imports and allow optional lightning
        from frustrampnn.model.transfer_model import TransferModelPL

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine device for loading
        if device is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            map_location = device

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Handle old vs new checkpoint format
        if config_path is not None:
            # Old format: config from separate file
            cfg = OmegaConf.load(config_path)
        elif "cfg" in checkpoint:
            # New format: config embedded in checkpoint
            cfg = OmegaConf.create(checkpoint["cfg"])
        elif "hyper_parameters" in checkpoint and "cfg" in checkpoint["hyper_parameters"]:
            # PyTorch Lightning format: config in hyper_parameters
            cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])
        else:
            # Try to find config.yaml in common locations
            possible_config_paths = [
                checkpoint_path.parent / "config.yaml",
                checkpoint_path.parent.parent / "config.yaml",
                checkpoint_path.parent.parent / "inference" / "config.yaml",
                Path(__file__).parent.parent.parent.parent / "inference" / "config.yaml",
            ]

            config_found = None
            for cfg_path in possible_config_paths:
                if cfg_path.exists():
                    config_found = cfg_path
                    break

            if config_found is not None:
                cfg = OmegaConf.load(config_found)
            else:
                raise ValueError(
                    "Checkpoint does not contain 'cfg' and no config.yaml found. "
                    f"Searched: {[str(p) for p in possible_config_paths]}. "
                    "For old format checkpoints, provide config_path explicitly."
                )

        # Set thermompnn_dir for weight loading
        # Use the directory containing the checkpoint
        cfg.platform.thermompnn_dir = str(checkpoint_path.parent)

        # Check for vanilla_model_weights in common locations
        possible_weight_dirs = [
            checkpoint_path.parent / "vanilla_model_weights",
            checkpoint_path.parent.parent / "inference" / "vanilla_model_weights",
            Path(__file__).parent.parent.parent.parent / "inference" / "vanilla_model_weights",
        ]

        for weight_dir in possible_weight_dirs:
            if weight_dir.exists():
                cfg.platform.thermompnn_dir = str(weight_dir.parent)
                break

        # Load model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model_pl = TransferModelPL.load_from_checkpoint(
                str(checkpoint_path),
                cfg=cfg,
                strict=False,
                map_location=map_location,
                weights_only=False
            )

        return cls(model_pl.model, cfg, device)

    def predict(
        self,
        pdb_path: str | Path,
        chains: list[str] | None = None,
        positions: list[int] | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Predict frustration values for a protein structure.

        Args:
            pdb_path: Path to PDB file
            chains: List of chain IDs to analyze (None = all chains)
            positions: List of positions to analyze (None = all positions)
            show_progress: Show progress bar

        Returns:
            pd.DataFrame: Predictions with columns:
                - frustration_pred: Predicted frustration value
                - position: 0-indexed position
                - wildtype: Wild-type amino acid
                - mutation: Mutant amino acid
                - pdb: PDB identifier
                - chain: Chain identifier

        Example:
            >>> results = model.predict("protein.pdb", chains=["A"])
            >>> print(results[results['position'] == 73])
        """
        pdb_path = Path(pdb_path)

        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        # Get chains if not specified
        if chains is None:
            chains = self._get_chains(pdb_path)

        pdb_id = pdb_path.stem
        all_results = []

        for chain in chains:
            chain_results = self._predict_chain(pdb_path, chain, positions, show_progress)
            for result in chain_results:
                result["chain"] = chain
                result["pdb"] = pdb_id
            all_results.extend(chain_results)

        return pd.DataFrame(all_results)

    def predict_batch(
        self,
        pdb_paths: list[str | Path],
        chains: list[str] | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Predict frustration for multiple PDB files.

        Args:
            pdb_paths: List of paths to PDB files
            chains: Chain IDs to analyze (applied to all PDBs)
            show_progress: Show progress bar

        Returns:
            pd.DataFrame: Combined predictions for all PDBs
        """
        all_results = []

        iterator = tqdm(pdb_paths, disable=not show_progress, desc="Processing PDBs")

        for pdb_path in iterator:
            try:
                results = self.predict(pdb_path, chains=chains, show_progress=False)
                all_results.append(results)
            except Exception as e:
                warnings.warn(f"Failed to process {pdb_path}: {e}", stacklevel=2)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def _get_chains(self, pdb_path: Path) -> list[str]:
        """Get chain IDs from PDB file."""
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("", str(pdb_path))
        return [c.id for c in structure.get_chains()]

    def _predict_chain(
        self,
        pdb_path: Path,
        chain: str,
        positions: list[int] | None,
        show_progress: bool,
    ) -> list[dict]:
        """Predict frustration for a single chain."""
        # Parse PDB
        pdb = alt_parse_PDB(str(pdb_path), [chain])

        if not pdb:
            return []

        # Generate mutations
        mutations = self._generate_mutations(pdb[0], positions)

        if not mutations:
            return []

        # Run inference
        results = []

        with torch.no_grad():
            iterator = tqdm(mutations, disable=not show_progress, desc=f"Chain {chain}")

            for mut in iterator:
                if mut is None:
                    continue

                pred, _ = self.model([pdb[0]], [mut])

                results.append(
                    {
                        "frustration_pred": pred[0]["frustration"].item(),
                        "position": mut.position,
                        "wildtype": mut.wildtype,
                        "mutation": mut.mutation,
                    }
                )

        return results

    def _generate_mutations(
        self,
        pdb: dict,
        positions: list[int] | None,
    ) -> list[Mutation | None]:
        """Generate site-saturation mutagenesis mutations."""
        mutations: list[Mutation | None] = []
        seq = pdb["seq"]

        for seq_pos in range(len(seq)):
            if positions is not None and seq_pos not in positions:
                continue

            wt_aa = seq[seq_pos]

            if wt_aa == "-" or wt_aa not in ALPHABET:
                mutations.append(None)
                continue

            for mut_aa in ALPHABET[:-1]:  # Exclude 'X'
                mutations.append(
                    Mutation(
                        position=seq_pos,
                        wildtype=wt_aa,
                        mutation=mut_aa,
                    )
                )

        return mutations

    def __repr__(self) -> str:
        return f"FrustraMPNN(device={self.device})"
