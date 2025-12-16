"""Tests for ProteinMPNN module."""

import pytest


def test_protein_mpnn_import():
    """Test that ProteinMPNN can be imported."""
    from frustrampnn.model import ProteinMPNN

    assert ProteinMPNN is not None


def test_parse_pdb_import():
    """Test that PDB parsing functions can be imported."""
    from frustrampnn.model import alt_parse_PDB, parse_PDB

    assert parse_PDB is not None
    assert alt_parse_PDB is not None


def test_tied_featurize_import():
    """Test that featurization function can be imported."""
    from frustrampnn.model import tied_featurize

    assert tied_featurize is not None


def test_layers_import():
    """Test that layer classes can be imported."""
    from frustrampnn.model import DecLayer, EncLayer, PositionalEncodings, PositionWiseFeedForward

    assert EncLayer is not None
    assert DecLayer is not None
    assert PositionWiseFeedForward is not None
    assert PositionalEncodings is not None


def test_features_import():
    """Test that feature classes can be imported."""
    from frustrampnn.model import CA_ProteinFeatures, ProteinFeatures

    assert ProteinFeatures is not None
    assert CA_ProteinFeatures is not None


def test_gather_functions_import():
    """Test that gather functions can be imported."""
    from frustrampnn.model import cat_neighbors_nodes, gather_edges, gather_nodes, gather_nodes_t

    assert gather_edges is not None
    assert gather_nodes is not None
    assert gather_nodes_t is not None
    assert cat_neighbors_nodes is not None


def test_loss_functions_import():
    """Test that loss functions can be imported."""
    from frustrampnn.model import loss_nll, loss_smoothed

    assert loss_nll is not None
    assert loss_smoothed is not None


def test_alt_parse_pdb(test_pdb_path):
    """Test PDB parsing with test file."""
    from frustrampnn.model import alt_parse_PDB

    if not test_pdb_path.exists():
        pytest.skip("Test PDB file not found")

    result = alt_parse_PDB(str(test_pdb_path))

    assert isinstance(result, list)
    assert len(result) > 0
    assert "seq" in result[0]
    assert "name" in result[0]
    assert "resn_list" in result[0]


def test_parse_pdb(test_pdb_path):
    """Test standard PDB parsing."""
    from frustrampnn.model import parse_PDB

    if not test_pdb_path.exists():
        pytest.skip("Test PDB file not found")

    result = parse_PDB(str(test_pdb_path))

    assert isinstance(result, list)
    assert len(result) > 0
    assert "seq" in result[0]
    assert "name" in result[0]


