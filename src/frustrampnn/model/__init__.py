"""Model architecture components for FrustraMPNN.

This module provides the neural network architecture including:
- ProteinMPNN: Base message-passing neural network
- TransferModel: Frustration prediction head
- LightAttention: Attention mechanism for feature aggregation
"""

from frustrampnn.model.features import CA_ProteinFeatures, ProteinFeatures
from frustrampnn.model.featurization import tied_featurize
from frustrampnn.model.layers import (
    DecLayer,
    EncLayer,
    PositionalEncodings,
    PositionWiseFeedForward,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
    gather_nodes_t,
)
from frustrampnn.model.model_utils import (
    ALPHABET,
    VOCAB_DIM,
    get_esm_model,
    get_metrics,
    get_protein_mpnn,
)
from frustrampnn.model.pdb_parsing import (
    alt_parse_PDB,
    alt_parse_PDB_biounits,
    parse_PDB,
    parse_PDB_biounits,
)
from frustrampnn.model.protein_mpnn import (
    ProteinMPNN,
    _S_to_seq,
    _scores,
    loss_nll,
    loss_smoothed,
)
from frustrampnn.model.transfer_model import LightAttention, TransferModel, TransferModelPL

__all__ = [
    # ProteinMPNN
    "ProteinMPNN",
    "ProteinFeatures",
    "CA_ProteinFeatures",
    "EncLayer",
    "DecLayer",
    "PositionWiseFeedForward",
    "PositionalEncodings",
    # PDB parsing
    "parse_PDB",
    "parse_PDB_biounits",
    "alt_parse_PDB",
    "alt_parse_PDB_biounits",
    # Featurization
    "tied_featurize",
    # Gather functions
    "gather_edges",
    "gather_nodes",
    "gather_nodes_t",
    "cat_neighbors_nodes",
    # Loss functions
    "loss_nll",
    "loss_smoothed",
    "_scores",
    "_S_to_seq",
    # TransferModel
    "TransferModel",
    "TransferModelPL",
    "LightAttention",
    # Utils
    "ALPHABET",
    "VOCAB_DIM",
    "get_protein_mpnn",
    "get_esm_model",
    "get_metrics",
]
