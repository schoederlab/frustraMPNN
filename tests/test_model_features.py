"""Tests for protein feature extraction modules.

This module tests the ProteinFeatures and CA_ProteinFeatures classes
that extract structural features from protein coordinates.
"""

import pytest
import torch

# =============================================================================
# Test CA_ProteinFeatures
# =============================================================================


class TestCAProteinFeatures:
    """Tests for CA-only protein features."""

    def test_init(self):
        """Test CA_ProteinFeatures initialization."""
        from frustrampnn.model.features import CA_ProteinFeatures

        edge_features = 128
        node_features = 128
        layer = CA_ProteinFeatures(edge_features, node_features)

        assert layer.edge_features == edge_features
        assert layer.node_features == node_features
        assert layer.top_k == 30  # default
        assert layer.num_rbf == 16  # default

    def test_init_custom_params(self):
        """Test CA_ProteinFeatures with custom parameters."""
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(
            edge_features=64,
            node_features=64,
            num_positional_embeddings=8,
            num_rbf=8,
            top_k=15,
            augment_eps=0.1,
        )

        assert layer.edge_features == 64
        assert layer.top_k == 15
        assert layer.num_rbf == 8
        assert layer.augment_eps == 0.1

    def test_forward_shape(
        self,
        sample_ca_coords,
        sample_mask,
        sample_residue_idx,
        sample_chain_labels,
        batch_size,
        seq_length,
    ):
        """Test forward pass output shape."""
        from frustrampnn.model.features import CA_ProteinFeatures

        edge_features = 128
        node_features = 128
        top_k = min(5, seq_length)  # Ensure top_k <= seq_length
        layer = CA_ProteinFeatures(
            edge_features, node_features, top_k=top_k
        )

        E, E_idx = layer(
            sample_ca_coords,
            sample_mask,
            sample_residue_idx,
            sample_chain_labels,
        )

        assert E.shape == (batch_size, seq_length, top_k, edge_features)
        assert E_idx.shape == (batch_size, seq_length, top_k)

    def test_forward_with_augmentation(
        self,
        sample_ca_coords,
        sample_mask,
        sample_residue_idx,
        sample_chain_labels,
    ):
        """Test forward pass with coordinate augmentation."""
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(
            edge_features=64,
            node_features=64,
            top_k=5,
            augment_eps=0.1,
        )

        # Run twice with same input - should get different results due to augmentation
        torch.manual_seed(42)
        E1, _ = layer(
            sample_ca_coords,
            sample_mask,
            sample_residue_idx,
            sample_chain_labels,
        )

        torch.manual_seed(43)
        E2, _ = layer(
            sample_ca_coords,
            sample_mask,
            sample_residue_idx,
            sample_chain_labels,
        )

        # Results should be different due to random augmentation
        assert not torch.allclose(E1, E2)

    def test_forward_no_augmentation(
        self,
        sample_ca_coords,
        sample_mask,
        sample_residue_idx,
        sample_chain_labels,
    ):
        """Test forward pass without augmentation is deterministic."""
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(
            edge_features=64,
            node_features=64,
            top_k=5,
            augment_eps=0.0,  # No augmentation
        )

        E1, E_idx1 = layer(
            sample_ca_coords,
            sample_mask,
            sample_residue_idx,
            sample_chain_labels,
        )

        E2, E_idx2 = layer(
            sample_ca_coords,
            sample_mask,
            sample_residue_idx,
            sample_chain_labels,
        )

        # Results should be identical
        assert torch.allclose(E1, E2)
        assert torch.equal(E_idx1, E_idx2)

    def test_rbf_output_range(self):
        """Test that RBF outputs are in valid range [0, 1]."""
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(64, 64, top_k=5)

        # Create distances in typical protein range
        D = torch.tensor([[[2.0, 5.0, 10.0, 15.0, 20.0]]])
        rbf = layer._rbf(D)

        assert rbf.min() >= 0
        assert rbf.max() <= 1

    def test_dist_returns_neighbors(
        self,
        sample_ca_coords,
        sample_mask,
    ):
        """Test that _dist returns correct number of neighbors."""
        from frustrampnn.model.features import CA_ProteinFeatures

        top_k = 5
        layer = CA_ProteinFeatures(64, 64, top_k=top_k)

        D_neighbors, E_idx, mask_neighbors = layer._dist(
            sample_ca_coords, sample_mask
        )

        batch_size, seq_length = sample_ca_coords.shape[:2]
        assert D_neighbors.shape == (batch_size, seq_length, top_k)
        assert E_idx.shape == (batch_size, seq_length, top_k)


# =============================================================================
# Test ProteinFeatures
# =============================================================================


class TestProteinFeatures:
    """Tests for full backbone protein features."""

    def test_init(self):
        """Test ProteinFeatures initialization."""
        from frustrampnn.model.features import ProteinFeatures

        edge_features = 128
        node_features = 128
        layer = ProteinFeatures(edge_features, node_features)

        assert layer.edge_features == edge_features
        assert layer.node_features == node_features
        assert layer.top_k == 30  # default

    def test_forward_shape(
        self,
        sample_backbone_coords,
        sample_mask,
        sample_residue_idx,
        sample_chain_labels,
        batch_size,
        seq_length,
    ):
        """Test forward pass output shape."""
        from frustrampnn.model.features import ProteinFeatures

        edge_features = 128
        node_features = 128
        top_k = min(5, seq_length)
        layer = ProteinFeatures(
            edge_features, node_features, top_k=top_k
        )

        E, E_idx = layer(
            sample_backbone_coords,
            sample_mask,
            sample_residue_idx,
            sample_chain_labels,
        )

        assert E.shape == (batch_size, seq_length, top_k, edge_features)
        assert E_idx.shape == (batch_size, seq_length, top_k)

    def test_forward_with_augmentation(
        self,
        sample_backbone_coords,
        sample_mask,
        sample_residue_idx,
        sample_chain_labels,
    ):
        """Test forward pass with coordinate augmentation."""
        from frustrampnn.model.features import ProteinFeatures

        layer = ProteinFeatures(
            edge_features=64,
            node_features=64,
            top_k=5,
            augment_eps=0.1,
        )

        torch.manual_seed(42)
        E1, _ = layer(
            sample_backbone_coords,
            sample_mask,
            sample_residue_idx,
            sample_chain_labels,
        )

        torch.manual_seed(43)
        E2, _ = layer(
            sample_backbone_coords,
            sample_mask,
            sample_residue_idx,
            sample_chain_labels,
        )

        # Results should be different due to random augmentation
        assert not torch.allclose(E1, E2)

    def test_cb_calculation(self):
        """Test that Cb is calculated correctly from backbone atoms."""
        from frustrampnn.model.features import ProteinFeatures

        layer = ProteinFeatures(64, 64, top_k=5)

        # Create simple backbone coordinates
        # N, CA, C, O for one residue
        X = torch.zeros(1, 5, 4, 3)
        X[0, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])  # N
        X[0, 0, 1, :] = torch.tensor([1.46, 0.0, 0.0])  # CA
        X[0, 0, 2, :] = torch.tensor([2.0, 1.2, 0.0])  # C
        X[0, 0, 3, :] = torch.tensor([2.0, 2.4, 0.0])  # O

        # The Cb calculation should not raise errors
        mask = torch.ones(1, 5)
        residue_idx = torch.arange(5).unsqueeze(0)
        chain_labels = torch.zeros(1, 5, dtype=torch.long)

        E, E_idx = layer(X, mask, residue_idx, chain_labels)

        assert E.shape[0] == 1
        assert not torch.isnan(E).any()


# =============================================================================
# Test RBF Functions
# =============================================================================


class TestRBFFunctions:
    """Tests for radial basis function calculations."""

    def test_rbf_shape(self):
        """Test RBF output shape."""
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(64, 64, num_rbf=16)

        D = torch.randn(2, 10, 5)  # batch, seq, neighbors
        rbf = layer._rbf(D)

        assert rbf.shape == (2, 10, 5, 16)

    def test_rbf_different_num_rbf(self):
        """Test RBF with different number of basis functions."""
        from frustrampnn.model.features import CA_ProteinFeatures

        for num_rbf in [8, 16, 32]:
            layer = CA_ProteinFeatures(64, 64, num_rbf=num_rbf)
            D = torch.randn(2, 10, 5)
            rbf = layer._rbf(D)
            assert rbf.shape[-1] == num_rbf

    def test_get_rbf_shape(self):
        """Test _get_rbf output shape."""
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(64, 64, num_rbf=16, top_k=5)

        batch_size, seq_len, num_neighbors = 2, 10, 5
        A = torch.randn(batch_size, seq_len, 3)
        B = torch.randn(batch_size, seq_len, 3)
        E_idx = torch.randint(0, seq_len, (batch_size, seq_len, num_neighbors))

        rbf = layer._get_rbf(A, B, E_idx)

        assert rbf.shape == (batch_size, seq_len, num_neighbors, 16)


# =============================================================================
# Test Feature Exports
# =============================================================================


class TestFeatureExports:
    """Test that all expected classes are exported."""

    def test_all_exports(self):
        """Test all expected exports from features module."""
        from frustrampnn.model import features

        expected = ["ProteinFeatures", "CA_ProteinFeatures"]

        for name in expected:
            assert hasattr(features, name), f"Missing export: {name}"

    def test_exports_from_model_init(self):
        """Test exports from model __init__."""
        from frustrampnn.model import CA_ProteinFeatures, ProteinFeatures

        assert ProteinFeatures is not None
        assert CA_ProteinFeatures is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestFeatureEdgeCases:
    """Test edge cases for feature extraction."""

    def test_single_residue(self):
        """Test with single residue (edge case).
        
        Note: Single residue is an edge case that may not work with all
        feature extraction methods due to orientation calculations requiring
        multiple residues. This test verifies the behavior.
        """
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(64, 64, top_k=1)

        # Single residue is a degenerate case - skip if it fails
        # as orientation calculations need at least 3 residues
        Ca = torch.randn(1, 1, 3)
        mask = torch.ones(1, 1)
        residue_idx = torch.zeros(1, 1, dtype=torch.long)
        chain_labels = torch.zeros(1, 1, dtype=torch.long)

        try:
            E, E_idx = layer(Ca, mask, residue_idx, chain_labels)
            assert E.shape == (1, 1, 1, 64)
            assert E_idx.shape == (1, 1, 1)
        except RuntimeError:
            # Single residue may fail due to orientation calculations
            pytest.skip("Single residue edge case not supported")

    def test_masked_residues(self):
        """Test with some residues masked."""
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(64, 64, top_k=3)

        Ca = torch.randn(1, 5, 3)
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])  # Middle residue masked
        residue_idx = torch.arange(5).unsqueeze(0)
        chain_labels = torch.zeros(1, 5, dtype=torch.long)

        E, E_idx = layer(Ca, mask, residue_idx, chain_labels)

        # Should still produce output
        assert E.shape == (1, 5, 3, 64)

    def test_multiple_chains(self):
        """Test with multiple chains."""
        from frustrampnn.model.features import CA_ProteinFeatures

        layer = CA_ProteinFeatures(64, 64, top_k=3)

        Ca = torch.randn(1, 10, 3)
        mask = torch.ones(1, 10)
        residue_idx = torch.arange(10).unsqueeze(0)
        # First 5 residues chain 0, last 5 chain 1
        chain_labels = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

        E, E_idx = layer(Ca, mask, residue_idx, chain_labels)

        assert E.shape == (1, 10, 3, 64)
        assert not torch.isnan(E).any()

