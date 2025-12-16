"""
Custom callbacks for FrustraMPNN training.

This module provides custom PyTorch Lightning callbacks for
checkpoint management and training monitoring.

Original source: test_data/training/train_thermompnn_refac.py (lines 289-296)
Backup: test_data/training/train_thermompnn_refac.py.bak
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None
    ModelCheckpoint = object  # type: ignore[misc, assignment]


if HAS_LIGHTNING:

    class FrustraMPNNCheckpoint(ModelCheckpoint):
        """
        Custom checkpoint callback for FrustraMPNN.

        Extends ModelCheckpoint with:
        - Custom filename formatting
        - Automatic best model tracking
        - Checkpoint validation

        This callback matches the checkpoint configuration from
        test_data/training/train_thermompnn_refac.py (lines 289-296).

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename template
            monitor: Metric to monitor
            mode: "min" or "max"
            save_top_k: Number of best checkpoints to keep
            **kwargs: Additional arguments passed to ModelCheckpoint

        Example:
            >>> callback = FrustraMPNNCheckpoint(
            ...     dirpath="./checkpoints",
            ...     monitor="val_frustration_spearman",
            ...     mode="max",
            ... )
            >>> trainer = pl.Trainer(callbacks=[callback])
        """

        def __init__(
            self,
            dirpath: Optional[str] = None,
            filename: Optional[str] = None,
            monitor: str = "val_frustration_spearman",
            mode: str = "max",
            save_top_k: int = 3,
            **kwargs: Any,
        ) -> None:
            """
            Initialize FrustraMPNNCheckpoint.

            Default filename format matches original:
            {name}_{epoch:02d}_{val_frustration_spearman:.02}

            Args:
                dirpath: Directory to save checkpoints
                filename: Checkpoint filename template
                monitor: Metric to monitor (default: val_frustration_spearman)
                mode: "min" or "max" (default: max for spearman)
                save_top_k: Number of best checkpoints to keep
                **kwargs: Additional arguments
            """
            if filename is None:
                filename = "{epoch:02d}_{val_frustration_spearman:.02}"

            super().__init__(
                dirpath=dirpath,
                filename=filename,
                monitor=monitor,
                mode=mode,
                save_top_k=save_top_k,
                **kwargs,
            )

        def on_save_checkpoint(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            checkpoint: Dict[str, Any],
        ) -> None:
            """
            Add custom metadata to checkpoint.

            Args:
                trainer: PyTorch Lightning trainer
                pl_module: Lightning module being trained
                checkpoint: Checkpoint dictionary
            """
            super().on_save_checkpoint(trainer, pl_module, checkpoint)

            # Add training info
            checkpoint["training_info"] = {
                "best_score": self.best_model_score,
                "best_path": self.best_model_path,
            }

else:

    class FrustraMPNNCheckpoint:  # type: ignore[no-redef]
        """Placeholder for FrustraMPNNCheckpoint when pytorch_lightning is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "FrustraMPNNCheckpoint requires pytorch_lightning. "
                "Install with: pip install pytorch-lightning"
            )


def load_checkpoint_safe(
    checkpoint_path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Safely load a checkpoint with error handling.

    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to (default: cpu)

    Returns:
        Checkpoint dictionary

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is corrupted

    Example:
        >>> checkpoint = load_checkpoint_safe("model.ckpt")
        >>> print(checkpoint.keys())
    """
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    # Validate checkpoint structure
    if "state_dict" not in checkpoint:
        raise RuntimeError("Invalid checkpoint: missing state_dict")

    return checkpoint


def get_checkpoint_callback(
    name: str,
    weights_dir: str,
    monitor: str = "val_frustration_spearman",
    mode: str = "max",
) -> "FrustraMPNNCheckpoint":
    """
    Create a checkpoint callback with standard settings.

    This function matches the checkpoint setup from
    test_data/training/train_thermompnn_refac.py (lines 289-296).

    Args:
        name: Experiment name (used in filename)
        weights_dir: Directory to save checkpoints
        monitor: Metric to monitor
        mode: "min" or "max"

    Returns:
        Configured FrustraMPNNCheckpoint callback

    Example:
        >>> callback = get_checkpoint_callback("experiment_1", "./weights")
        >>> trainer = pl.Trainer(callbacks=[callback])
    """
    filename = f"{name}_{{epoch:02d}}_{{val_frustration_spearman:.02}}"
    return FrustraMPNNCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=weights_dir,
        filename=filename,
    )


__all__ = [
    "FrustraMPNNCheckpoint",
    "load_checkpoint_safe",
    "get_checkpoint_callback",
]


