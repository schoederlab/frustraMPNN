"""
Model utilities for FrustraMPNN.

This module contains utility functions for model loading, ESM integration,
and metric computation.
"""

import os

import torch

from frustrampnn.constants import ALPHABET, VOCAB_DIM

__all__ = [
    "ALPHABET",
    "VOCAB_DIM",
    "get_protein_mpnn",
    "get_esm_model",
    "get_metrics",
]


def get_metrics():
    """Create dictionary of evaluation metrics.

    Requires torchmetrics to be installed (part of 'train' extras).

    Returns:
        dict: Dictionary with keys 'r2', 'mse', 'rmse', 'spearman'

    Raises:
        ImportError: If torchmetrics is not installed
    """
    try:
        from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef
    except ImportError as e:
        raise ImportError(
            "torchmetrics is required for get_metrics(). "
            "Install with: pip install frustrampnn[train]"
        ) from e

    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


def get_esm_model(esm_model_name: str):
    """Load a pretrained ESM model.

    Args:
        esm_model_name: Name of ESM model to load

    Returns:
        tuple: (model, alphabet) from ESM

    Raises:
        ValueError: If unknown ESM model name
    """
    import esm

    esm_models = {
        "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
        "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm1b_t33_650M_UR50S": esm.pretrained.esm1b_t33_650M_UR50S,
        "esm1_t34_670M_UR50S": esm.pretrained.esm1_t34_670M_UR50S,
        "esm1_t34_670M_UR50D": esm.pretrained.esm1_t34_670M_UR50D,
        "esm1_t34_670M_UR100": esm.pretrained.esm1_t34_670M_UR100,
        "esm1_t12_85M_UR50S": esm.pretrained.esm1_t12_85M_UR50S,
        "esm1_t6_43M_UR50S": esm.pretrained.esm1_t6_43M_UR50S,
    }

    if esm_model_name not in esm_models:
        raise ValueError(
            f"Unknown ESM model: {esm_model_name}. Available: {list(esm_models.keys())}"
        )

    return esm_models[esm_model_name]()


def get_protein_mpnn(cfg, version: str = "v_48_020.pt"):
    """Load pretrained ProteinMPNN model.

    Args:
        cfg: Configuration object with platform.thermompnn_dir
        version: Weight file version (default: v_48_020.pt)

    Returns:
        ProteinMPNN: Loaded model
    """
    from frustrampnn.model.protein_mpnn import ProteinMPNN

    hidden_dim = 128
    num_layers = 3

    model_weight_dir = os.path.join(cfg.platform.thermompnn_dir, "vanilla_model_weights")

    checkpoint_path = os.path.join(model_weight_dir, version)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        k_neighbors=checkpoint["num_edges"],
        augment_eps=0.0,
    )

    if cfg.model.load_pretrained:
        model.load_state_dict(checkpoint["model_state_dict"])

    if cfg.model.freeze_weights:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model
