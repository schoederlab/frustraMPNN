"""
High-level Trainer class for FrustraMPNN training.

This module provides a simplified interface for training FrustraMPNN models,
wrapping PyTorch Lightning's Trainer with sensible defaults.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from frustrampnn.training.config import TrainingConfig


class Trainer:
    """
    High-level trainer for FrustraMPNN models.

    This class orchestrates the training process, including:
    - Configuration management
    - Dataset loading
    - Model initialization
    - Training loop execution
    - Checkpoint management
    - Logging setup

    Example:
        >>> from frustrampnn.training import Trainer, TrainingConfig
        >>> config = TrainingConfig.from_yaml("config.yaml")
        >>> trainer = Trainer(config)
        >>> trainer.fit()

    Args:
        config: Training configuration (TrainingConfig or OmegaConf DictConfig)
        resume_from: Path to checkpoint to resume from (optional)
    """

    def __init__(
        self,
        config: TrainingConfig | DictConfig,
        resume_from: str | Path | None = None,
    ) -> None:
        self.config = config
        self.resume_from = Path(resume_from) if resume_from else None

        # Will be initialized in setup()
        self._model: pl.LightningModule | None = None
        self._trainer: pl.Trainer | None = None
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None

        # Set random seeds
        self._set_seeds()

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = getattr(self.config.training, "seed", 0)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _setup_logger(self) -> WandbLogger | CSVLogger:
        """Setup experiment logger based on configuration."""
        logger_type = getattr(self.config, "logger", "csv")

        if logger_type == "wandb":
            return WandbLogger(
                project=self.config.project,
                name=self.config.name,
                log_model=False,
            )
        else:
            log_dir = getattr(self.config.data_loc, "log_dir", "./logs")
            return CSVLogger(log_dir, name=self.config.name)

    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []

        # Checkpoint callback
        filename = f"{self.config.name}_{{epoch:02d}}_{{val_frustration_spearman:.02f}}"
        weights_dir = getattr(self.config.data_loc, "weights_dir", "./weights")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_frustration_spearman",
            mode="max",
            dirpath=weights_dir,
            filename=filename,
            save_top_k=3,
        )
        callbacks.append(checkpoint_callback)

        return callbacks

    def _get_accelerator_config(self) -> dict:
        """Get accelerator configuration for Lightning Trainer."""
        cfg = self.config

        # Determine accelerator
        accel = getattr(cfg.platform, "accel", "auto")
        use_tpu = getattr(cfg.platform, "use_tpu", False)

        if use_tpu:
            return {"accelerator": "tpu", "devices": 8}

        # GPU configuration
        if accel == "gpu" and torch.cuda.is_available():
            ddp = getattr(cfg.training, "ddp", False)
            if ddp:
                return {
                    "accelerator": "gpu",
                    "devices": torch.cuda.device_count(),
                    "strategy": "ddp",
                }
            return {"accelerator": "gpu", "devices": 1}

        return {"accelerator": "cpu"}

    def setup(self) -> None:
        """
        Setup training components.

        This method initializes:
        - Datasets and dataloaders
        - Model
        - Lightning trainer

        Call this before fit() if you need access to components.
        """
        # Import here to avoid circular imports
        from frustrampnn.training.datasets import get_dataset
        from frustrampnn.training.lightning import TransferModelPL

        cfg = self.config

        # Setup datasets
        train_dataset = get_dataset(cfg, "train")
        val_dataset = get_dataset(cfg, "val")

        # Calculate worker distribution
        num_workers = getattr(cfg.training, "num_workers", 4)
        train_workers = max(2, int(num_workers * 0.75))
        val_workers = max(1, int(num_workers * 0.25))

        # Create dataloaders
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # One protein per batch
            shuffle=True,
            num_workers=train_workers,
            collate_fn=lambda x: x,  # Pass through as-is
        )

        self._val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=val_workers,
            collate_fn=lambda x: x,
        )

        # Setup test loader if testing enabled
        if getattr(cfg.training, "testing", False):
            test_dataset = get_dataset(cfg, "test")
            self._test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=val_workers,
                collate_fn=lambda x: x,
            )

        # Initialize model
        self._model = TransferModelPL(cfg)

        # Setup Lightning trainer
        logger = self._setup_logger()
        callbacks = self._setup_callbacks()
        accel_config = self._get_accelerator_config()

        self._trainer = pl.Trainer(
            max_epochs=cfg.training.epochs,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            **accel_config,
        )

    def fit(self) -> None:
        """
        Run the training loop.

        This method:
        1. Calls setup() if not already done
        2. Runs trainer.fit()
        3. Optionally runs testing
        """
        if self._trainer is None:
            self.setup()

        # Run training
        self._trainer.fit(
            self._model,
            self._train_loader,
            self._val_loader,
            ckpt_path=str(self.resume_from) if self.resume_from else None,
        )

        # Run testing if enabled
        if self._test_loader is not None:
            self._trainer.test(self._model, self._test_loader)

    def test(self, checkpoint_path: str | Path | None = None) -> dict:
        """
        Run evaluation on test set.

        Args:
            checkpoint_path: Path to checkpoint to evaluate (optional)

        Returns:
            Dictionary of test metrics
        """
        if self._trainer is None:
            self.setup()

        ckpt = str(checkpoint_path) if checkpoint_path else None
        results = self._trainer.test(self._model, self._test_loader, ckpt_path=ckpt)
        return results[0] if results else {}

    @property
    def model(self) -> pl.LightningModule | None:
        """Get the Lightning module."""
        return self._model

    @property
    def lightning_trainer(self) -> pl.Trainer | None:
        """Get the underlying Lightning Trainer."""
        return self._trainer

