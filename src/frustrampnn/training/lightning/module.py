"""
PyTorch Lightning module for FrustraMPNN training.

This module provides the TransferModelPL class that wraps the
TransferModel for training with PyTorch Lightning.

Original source: test_data/training/train_thermompnn_refac.py (lines 60-211)
Backup: test_data/training/train_thermompnn_refac.py.bak
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl

    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

from torchmetrics import MeanSquaredError, SpearmanCorrCoef
from torchmetrics.regression import R2Score

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from frustrampnn.training.config import TrainingConfig
    from frustrampnn.training.datasets.base import TrainingMutation


def get_metrics() -> Dict[str, Any]:
    """
    Create a dictionary of metrics for training.

    This function matches the original implementation exactly from
    test_data/training/train_thermompnn_refac.py (lines 51-57).

    Returns:
        Dictionary with metric name -> metric instance

    Example:
        >>> metrics = get_metrics()
        >>> metrics['mse'].update(pred, target)
        >>> value = metrics['mse'].compute()
    """
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


if HAS_LIGHTNING:

    class TransferModelPL(pl.LightningModule):
        """
        PyTorch Lightning module for FrustraMPNN training.

        This class wraps the TransferModel and provides:
        - Training/validation/test step implementations
        - Optimizer and scheduler configuration
        - Metric computation and logging
        - Checkpoint save/load handling

        This implementation matches the original exactly from
        test_data/training/train_thermompnn_refac.py (lines 60-211).

        Attributes:
            stage: Training stage (1 or 2 for two-stage training)
            model: The underlying TransferModel
            cfg: Training configuration

        Args:
            cfg: Training configuration (TrainingConfig or DictConfig)

        Example:
            >>> model = TransferModelPL(cfg)
            >>> trainer = pl.Trainer(max_epochs=100)
            >>> trainer.fit(model, train_loader, val_loader)
        """

        stage: int = 1  # For two-stage training

        def __init__(
            self,
            cfg: Union["TrainingConfig", "DictConfig"],
        ) -> None:
            """
            Initialize TransferModelPL.

            Matches original implementation from train_thermompnn_refac.py lines 64-84.

            Args:
                cfg: Training configuration
            """
            super().__init__()
            self.cfg = cfg

            # Import here to avoid circular imports
            from frustrampnn.model import TransferModel

            # Initialize model
            self.model = TransferModel(cfg)

            # Training parameters - match original exactly
            self.learn_rate = cfg.training.learn_rate
            self.mpnn_learn_rate = (
                cfg.training.mpnn_learn_rate
                if "mpnn_learn_rate" in cfg.training
                else None
            )
            self.lr_schedule = (
                cfg.training.lr_schedule if "lr_schedule" in cfg.training else False
            )

            # Set up metrics dictionary - match original exactly (lines 73-80)
            self.metrics = nn.ModuleDict()
            for split in ("train_metrics", "val_metrics", "test_metrics"):
                self.metrics[split] = nn.ModuleDict()
                out = "frustration"
                self.metrics[split][out] = nn.ModuleDict()
                for name, metric in get_metrics().items():
                    self.metrics[split][out][name] = metric

            # Loss reweighting settings
            self.reweighting_loss = cfg.training.reweighting
            self.weight_method = cfg.training.weight_method

        def forward(self, *args: Any) -> Any:
            """
            Forward pass through the model.

            Matches original implementation from train_thermompnn_refac.py lines 86-87.

            Args:
                *args: Arguments passed to TransferModel

            Returns:
                Model output
            """
            return self.model(*args)

        def shared_eval(
            self,
            batch: List[Tuple],
            batch_idx: int,
            prefix: str,
        ) -> Optional[torch.Tensor]:
            """
            Shared evaluation logic for train/val/test.

            This method matches the original implementation exactly from
            test_data/training/train_thermompnn_refac.py (lines 89-130).

            Args:
                batch: Batch of (pdb, mutations) tuples
                batch_idx: Batch index
                prefix: Metric prefix ("train", "val", "test")

            Returns:
                Loss tensor or None if no valid mutations
            """
            # Batch contains one protein (batch size = 1) - match original line 91
            assert len(batch) == 1
            mut_pdb, mutations = batch[0]
            pred, _ = self(mut_pdb, mutations)

            frustration_mses = []
            if self.reweighting_loss:
                # Calculate weight sum for normalization - match original line 97
                weight_sum = sum([mut.weight for mut in mutations])

            for mut, out in zip(mutations, pred):
                if mut.frustration is not None:
                    # Reweight loss if specified - match original lines 102-108
                    if self.reweighting_loss:
                        weight = mut.weight
                        loss = F.mse_loss(out["frustration"], mut.frustration)
                        loss = loss * (weight / weight_sum)
                    else:
                        loss = F.mse_loss(out["frustration"], mut.frustration)

                    frustration_mses.append(loss)
                    # Update metrics - match original lines 111-112
                    for metric in self.metrics[f"{prefix}_metrics"][
                        "frustration"
                    ].values():
                        metric.update(out["frustration"], mut.frustration)

            # Compute mean loss - match original line 115
            loss = (
                0.0 if len(frustration_mses) == 0 else torch.stack(frustration_mses).mean()
            )
            on_step = False
            on_epoch = not on_step

            # Log metrics - match original lines 119-126
            output = "frustration"
            for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
                try:
                    metric.compute()
                except ValueError:
                    continue
                self.log(
                    f"{prefix}_{output}_{name}",
                    metric,
                    prog_bar=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=len(batch),
                )

            # Return None if no loss - match original lines 127-130
            if loss == 0.0:
                return None

            return loss

        def training_step(
            self,
            batch: List[Tuple],
            batch_idx: int,
        ) -> Optional[torch.Tensor]:
            """
            Training step.

            Matches original implementation from train_thermompnn_refac.py lines 132-133.

            Args:
                batch: Batch of (pdb, mutations) tuples
                batch_idx: Batch index

            Returns:
                Loss tensor or None
            """
            return self.shared_eval(batch, batch_idx, "train")

        def validation_step(
            self,
            batch: List[Tuple],
            batch_idx: int,
        ) -> Optional[torch.Tensor]:
            """
            Validation step.

            Matches original implementation from train_thermompnn_refac.py lines 135-136.

            Args:
                batch: Batch of (pdb, mutations) tuples
                batch_idx: Batch index

            Returns:
                Loss tensor or None
            """
            return self.shared_eval(batch, batch_idx, "val")

        def test_step(
            self,
            batch: List[Tuple],
            batch_idx: int,
        ) -> Optional[torch.Tensor]:
            """
            Test step.

            Matches original implementation from train_thermompnn_refac.py lines 138-139.

            Args:
                batch: Batch of (pdb, mutations) tuples
                batch_idx: Batch index

            Returns:
                Loss tensor or None
            """
            return self.shared_eval(batch, batch_idx, "test")

        def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict]:
            """
            Configure optimizers and schedulers.

            This method matches the original implementation exactly from
            test_data/training/train_thermompnn_refac.py (lines 141-182).

            Returns:
                Optimizer or dictionary with optimizer and scheduler
            """
            # For second stage, drop LR by factor of 10 - match original lines 142-144
            if self.stage == 2:
                self.learn_rate /= 10.0
                print("New second-stage learning rate: ", self.learn_rate)

            # ProteinMPNN parameters (if not frozen) - match original lines 146-152
            if not self.cfg.model.freeze_weights:
                param_list = [
                    {
                        "params": self.model.prot_mpnn.parameters(),
                        "lr": self.mpnn_learn_rate,
                    }
                ]
            else:
                param_list = []

            # LightAttention parameters - match original lines 154-160
            if self.model.lightattn:
                if self.stage == 2:
                    param_list.append(
                        {"params": self.model.light_attention.parameters(), "lr": 0.0}
                    )
                else:
                    param_list.append(
                        {"params": self.model.light_attention_mpnn.parameters()}
                    )
                    if self.cfg.training.add_esm_embeddings:
                        param_list.append(
                            {"params": self.model.light_attention_esm.parameters()}
                        )

            # MLP parameters - match original lines 163-166
            mlp_params = [
                {"params": self.model.both_out.parameters()},
                {"params": self.model.frustration_out.parameters()},
            ]

            param_list = param_list + mlp_params

            # Create optimizer - match original line 172
            opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

            # Add scheduler if enabled - match original lines 174-182
            if self.lr_schedule:
                lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=opt, verbose=True, mode="min", factor=0.5
                )
                return {
                    "optimizer": opt,
                    "lr_scheduler": lr_sched,
                    "monitor": "val_frustration_mse",
                }
            else:
                return opt

        def on_save_checkpoint(self, checkpoint: Dict) -> None:
            """
            Customize checkpoint saving.

            Saves only trainable parameters and configuration.
            This method matches the original implementation exactly from
            test_data/training/train_thermompnn_refac.py (lines 185-192).

            Args:
                checkpoint: Checkpoint dictionary to modify
            """
            # Store only trainable parameters - match original lines 188-191
            checkpoint["state_dict"] = {
                k: v
                for k, v in self.state_dict().items()
                if k in dict(self.named_parameters())
                and self.get_parameter(k).requires_grad
            }
            checkpoint["cfg"] = self.cfg

        def on_load_checkpoint(self, checkpoint: Dict) -> None:
            """
            Customize checkpoint loading.

            Handles missing keys by using current model state.
            This method matches the original implementation exactly from
            test_data/training/train_thermompnn_refac.py (lines 194-211).

            Args:
                checkpoint: Checkpoint dictionary
            """
            # Get checkpoint and model keys - match original lines 198-199
            state_dict_checkpoint = set(checkpoint["state_dict"].keys())
            state_dict_model = set(self.state_dict().keys())

            # Find missing keys - match original line 202
            missing_keys = state_dict_model - state_dict_checkpoint

            # Silently fill in missing keys - match original lines 205-211
            if missing_keys:
                for key in missing_keys:
                    checkpoint["state_dict"][key] = self.state_dict()[key]

else:
    # Placeholder if pytorch_lightning is not available
    class TransferModelPL:  # type: ignore[no-redef]
        """Placeholder for TransferModelPL when pytorch_lightning is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "TransferModelPL requires pytorch_lightning. "
                "Install with: pip install pytorch-lightning"
            )


__all__ = [
    "TransferModelPL",
    "get_metrics",
]


