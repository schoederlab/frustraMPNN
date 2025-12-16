"""
PyTorch Lightning components for FrustraMPNN training.

Provides Lightning module wrappers and callbacks for training.

Example:
    >>> from frustrampnn.training.lightning import TransferModelPL
    >>> model = TransferModelPL(cfg)
    >>> trainer = pl.Trainer(max_epochs=100)
    >>> trainer.fit(model, train_loader, val_loader)
"""

from frustrampnn.training.lightning.callbacks import (
    FrustraMPNNCheckpoint,
    get_checkpoint_callback,
    load_checkpoint_safe,
)
from frustrampnn.training.lightning.module import TransferModelPL, get_metrics

__all__ = [
    # Module
    "TransferModelPL",
    "get_metrics",
    # Callbacks
    "FrustraMPNNCheckpoint",
    "get_checkpoint_callback",
    "load_checkpoint_safe",
]
