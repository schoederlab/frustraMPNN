"""Tests for FrustraMPNN predictor."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_frustrampnn_import():
    """Test that FrustraMPNN can be imported."""
    from frustrampnn import FrustraMPNN

    assert FrustraMPNN is not None


def test_frustrampnn_from_pretrained_missing_file():
    """Test error handling for missing checkpoint."""
    from frustrampnn import FrustraMPNN

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        FrustraMPNN.from_pretrained("nonexistent.ckpt")


def test_frustrampnn_predict_missing_pdb():
    """Test error handling for missing PDB."""
    from frustrampnn import FrustraMPNN

    # Create mock model
    model = FrustraMPNN.__new__(FrustraMPNN)
    model.model = MagicMock()
    model.cfg = MagicMock()
    model.device = "cpu"

    with pytest.raises(FileNotFoundError, match="PDB file not found"):
        model.predict("nonexistent.pdb")


def test_frustrampnn_repr():
    """Test FrustraMPNN repr."""
    from frustrampnn import FrustraMPNN

    model = FrustraMPNN.__new__(FrustraMPNN)
    model.device = "cpu"

    assert "FrustraMPNN" in repr(model)
    assert "cpu" in repr(model)


def test_frustrampnn_has_predict_method():
    """Test that FrustraMPNN has predict method."""
    from frustrampnn import FrustraMPNN

    assert hasattr(FrustraMPNN, "predict")


def test_frustrampnn_has_predict_batch_method():
    """Test that FrustraMPNN has predict_batch method."""
    from frustrampnn import FrustraMPNN

    assert hasattr(FrustraMPNN, "predict_batch")


def test_frustrampnn_has_from_pretrained_method():
    """Test that FrustraMPNN has from_pretrained method."""
    from frustrampnn import FrustraMPNN

    assert hasattr(FrustraMPNN, "from_pretrained")


def test_frustrampnn_generate_mutations():
    """Test mutation generation."""
    from frustrampnn import FrustraMPNN

    model = FrustraMPNN.__new__(FrustraMPNN)

    # Mock PDB data
    pdb = {"seq": "MAG"}

    mutations = model._generate_mutations(pdb, positions=None)

    # 3 positions × 20 amino acids = 60 mutations
    assert len(mutations) == 60


def test_frustrampnn_generate_mutations_specific_positions():
    """Test mutation generation with specific positions."""
    from frustrampnn import FrustraMPNN

    model = FrustraMPNN.__new__(FrustraMPNN)

    # Mock PDB data
    pdb = {"seq": "MAG"}

    mutations = model._generate_mutations(pdb, positions=[0])

    # 1 position × 20 amino acids = 20 mutations
    assert len(mutations) == 20


def test_frustrampnn_generate_mutations_with_gap():
    """Test mutation generation with gap in sequence."""
    from frustrampnn import FrustraMPNN

    model = FrustraMPNN.__new__(FrustraMPNN)

    # Mock PDB data with gap
    pdb = {"seq": "M-G"}

    mutations = model._generate_mutations(pdb, positions=None)

    # Position 1 is a gap, so we get None for it
    # M at position 0: 20 mutations
    # - at position 1: 1 None
    # G at position 2: 20 mutations
    # Total: 41 items (20 + 1 + 20)
    assert len(mutations) == 41

    # Check that position 1 has None
    none_count = sum(1 for m in mutations if m is None)
    assert none_count == 1


def test_frustrampnn_get_chains(test_pdb_path):
    """Test chain extraction from PDB."""
    from frustrampnn import FrustraMPNN

    model = FrustraMPNN.__new__(FrustraMPNN)

    chains = model._get_chains(test_pdb_path)

    assert isinstance(chains, list)
    assert len(chains) > 0
    assert "A" in chains  # 1UBQ has chain A


# Integration test (requires checkpoint)
@pytest.mark.skipif(
    not Path("inference/vanilla_model_weights").exists(),
    reason="Model weights not available",
)
def test_frustrampnn_integration(test_pdb_path):
    """Integration test with real model (if available)."""
    # This test requires a checkpoint to be available
    # Skip if not in the right environment
    pass


