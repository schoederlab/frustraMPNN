"""Tests for neural network layers in ProteinMPNN.

This module tests the encoder and decoder layers, positional encodings,
and utility functions for the message-passing neural network.
"""

import torch

# =============================================================================
# Test Gather Functions
# =============================================================================


class TestGatherFunctions:
    """Tests for gather utility functions."""

    def test_gather_edges_shape(self):
        """Test gather_edges output shape."""
        from frustrampnn.model.layers import gather_edges

        batch_size, seq_len, num_neighbors, channels = 2, 10, 5, 32
        edges = torch.randn(batch_size, seq_len, seq_len, channels)
        neighbor_idx = torch.randint(0, seq_len, (batch_size, seq_len, num_neighbors))

        result = gather_edges(edges, neighbor_idx)

        assert result.shape == (batch_size, seq_len, num_neighbors, channels)

    def test_gather_edges_values(self):
        """Test gather_edges gathers correct values."""
        from frustrampnn.model.layers import gather_edges

        # Simple case: 1 batch, 3 nodes, 2 neighbors
        edges = torch.arange(27).float().reshape(1, 3, 3, 3)
        neighbor_idx = torch.tensor([[[0, 1], [1, 2], [0, 2]]])

        result = gather_edges(edges, neighbor_idx)

        # Check first node's neighbors
        assert result[0, 0, 0].tolist() == edges[0, 0, 0].tolist()  # neighbor 0
        assert result[0, 0, 1].tolist() == edges[0, 0, 1].tolist()  # neighbor 1

    def test_gather_nodes_shape(self):
        """Test gather_nodes output shape."""
        from frustrampnn.model.layers import gather_nodes

        batch_size, seq_len, num_neighbors, channels = 2, 10, 5, 32
        nodes = torch.randn(batch_size, seq_len, channels)
        neighbor_idx = torch.randint(0, seq_len, (batch_size, seq_len, num_neighbors))

        result = gather_nodes(nodes, neighbor_idx)

        assert result.shape == (batch_size, seq_len, num_neighbors, channels)

    def test_gather_nodes_values(self):
        """Test gather_nodes gathers correct values."""
        from frustrampnn.model.layers import gather_nodes

        # Simple case
        nodes = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        neighbor_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])

        result = gather_nodes(nodes, neighbor_idx)

        # Node 0's neighbors are nodes 1 and 2
        assert result[0, 0, 0].tolist() == [3.0, 4.0]  # node 1
        assert result[0, 0, 1].tolist() == [5.0, 6.0]  # node 2

    def test_gather_nodes_t_shape(self):
        """Test gather_nodes_t output shape."""
        from frustrampnn.model.layers import gather_nodes_t

        batch_size, seq_len, num_neighbors, channels = 2, 10, 5, 32
        nodes = torch.randn(batch_size, seq_len, channels)
        neighbor_idx = torch.randint(0, seq_len, (batch_size, num_neighbors))

        result = gather_nodes_t(nodes, neighbor_idx)

        assert result.shape == (batch_size, num_neighbors, channels)

    def test_cat_neighbors_nodes_shape(self):
        """Test cat_neighbors_nodes output shape."""
        from frustrampnn.model.layers import cat_neighbors_nodes

        batch_size, seq_len, num_neighbors, channels = 2, 10, 5, 32
        h_nodes = torch.randn(batch_size, seq_len, channels)
        h_neighbors = torch.randn(batch_size, seq_len, num_neighbors, channels)
        E_idx = torch.randint(0, seq_len, (batch_size, seq_len, num_neighbors))

        result = cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)

        # Output should have 2*channels in last dimension
        assert result.shape == (batch_size, seq_len, num_neighbors, 2 * channels)


# =============================================================================
# Test PositionWiseFeedForward
# =============================================================================


class TestPositionWiseFeedForward:
    """Tests for PositionWiseFeedForward layer."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        from frustrampnn.model.layers import PositionWiseFeedForward

        num_hidden, num_ff = 32, 128
        layer = PositionWiseFeedForward(num_hidden, num_ff)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, num_hidden)

        result = layer(x)

        assert result.shape == x.shape

    def test_forward_different_sizes(self):
        """Test forward pass with different hidden sizes."""
        from frustrampnn.model.layers import PositionWiseFeedForward

        for num_hidden in [16, 32, 64, 128]:
            num_ff = num_hidden * 4
            layer = PositionWiseFeedForward(num_hidden, num_ff)

            x = torch.randn(2, 10, num_hidden)
            result = layer(x)

            assert result.shape == x.shape

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the layer."""
        from frustrampnn.model.layers import PositionWiseFeedForward

        layer = PositionWiseFeedForward(32, 128)
        x = torch.randn(2, 10, 32, requires_grad=True)

        result = layer(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


# =============================================================================
# Test PositionalEncodings
# =============================================================================


class TestPositionalEncodings:
    """Tests for PositionalEncodings layer."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        from frustrampnn.model.layers import PositionalEncodings

        num_embeddings = 16
        max_relative_feature = 32
        layer = PositionalEncodings(num_embeddings, max_relative_feature)

        batch_size, seq_len, num_neighbors = 2, 10, 5
        # Use long tensor for offset (required for one_hot)
        offset = torch.randint(-10, 10, (batch_size, seq_len, num_neighbors), dtype=torch.long)
        mask = torch.ones(batch_size, seq_len, num_neighbors, dtype=torch.long)

        result = layer(offset, mask)

        assert result.shape == (batch_size, seq_len, num_neighbors, num_embeddings)

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        from frustrampnn.model.layers import PositionalEncodings

        layer = PositionalEncodings(16, 32)

        offset = torch.tensor([[[0, 1, 2], [1, 0, -1]]], dtype=torch.long)
        mask = torch.tensor([[[1, 1, 0], [1, 1, 1]]], dtype=torch.long)

        result = layer(offset, mask)

        # Masked positions should have different encoding
        assert result.shape == (1, 2, 3, 16)

    def test_forward_clipping(self):
        """Test that offsets are clipped to valid range."""
        from frustrampnn.model.layers import PositionalEncodings

        max_relative_feature = 32
        layer = PositionalEncodings(16, max_relative_feature)

        # Offsets outside range should be clipped
        offset = torch.tensor([[[100, -100, 0]]], dtype=torch.long)
        mask = torch.ones(1, 1, 3, dtype=torch.long)

        result = layer(offset, mask)

        # Should not raise error and produce valid output
        assert result.shape == (1, 1, 3, 16)
        assert not torch.isnan(result).any()


# =============================================================================
# Test EncLayer
# =============================================================================


class TestEncLayer:
    """Tests for encoder layer."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        from frustrampnn.model.layers import EncLayer

        batch_size, seq_len, num_neighbors = 2, 10, 5
        hidden_dim = 32
        # EncLayer: W1 expects (num_hidden + num_in) input
        # In forward: h_EV = cat([h_V_expand, cat_neighbors_nodes(h_V, h_E, E_idx)])
        # cat_neighbors_nodes returns cat([h_E, h_V_gathered]) = 2 * hidden_dim
        # So total input to W1 = hidden_dim + 2*hidden_dim = 3*hidden_dim
        # Therefore num_in should be 2*hidden_dim (the h_EV part before h_V_expand)
        num_in = hidden_dim * 2
        layer = EncLayer(hidden_dim, num_in, dropout=0.0)

        h_V = torch.randn(batch_size, seq_len, hidden_dim)
        h_E = torch.randn(batch_size, seq_len, num_neighbors, hidden_dim)
        E_idx = torch.randint(0, seq_len, (batch_size, seq_len, num_neighbors))

        h_V_out, h_E_out = layer(h_V, h_E, E_idx)

        assert h_V_out.shape == h_V.shape
        assert h_E_out.shape == h_E.shape

    def test_forward_with_mask(self):
        """Test forward pass with masks."""
        from frustrampnn.model.layers import EncLayer

        batch_size, seq_len, num_neighbors = 2, 10, 5
        hidden_dim = 32
        num_in = hidden_dim * 2
        layer = EncLayer(hidden_dim, num_in, dropout=0.0)

        h_V = torch.randn(batch_size, seq_len, hidden_dim)
        h_E = torch.randn(batch_size, seq_len, num_neighbors, hidden_dim)
        E_idx = torch.randint(0, seq_len, (batch_size, seq_len, num_neighbors))
        mask_V = torch.ones(batch_size, seq_len)
        mask_attend = torch.ones(batch_size, seq_len, num_neighbors)

        h_V_out, h_E_out = layer(h_V, h_E, E_idx, mask_V=mask_V, mask_attend=mask_attend)

        assert h_V_out.shape == h_V.shape
        assert h_E_out.shape == h_E.shape

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the layer."""
        from frustrampnn.model.layers import EncLayer

        batch_size, seq_len, num_neighbors = 2, 10, 5
        hidden_dim = 32
        num_in = hidden_dim * 2
        layer = EncLayer(hidden_dim, num_in, dropout=0.0)

        h_V = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        h_E = torch.randn(batch_size, seq_len, num_neighbors, hidden_dim, requires_grad=True)
        E_idx = torch.randint(0, seq_len, (batch_size, seq_len, num_neighbors))

        h_V_out, h_E_out = layer(h_V, h_E, E_idx)
        loss = h_V_out.sum() + h_E_out.sum()
        loss.backward()

        assert h_V.grad is not None
        assert h_E.grad is not None


# =============================================================================
# Test DecLayer
# =============================================================================


class TestDecLayer:
    """Tests for decoder layer."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        from frustrampnn.model.layers import DecLayer

        batch_size, seq_len, num_neighbors = 2, 10, 5
        hidden_dim = 32
        # DecLayer: W1 expects (num_hidden + num_in) input
        # In forward: h_EV = cat([h_V_expand, h_E]) = 2 * hidden_dim
        # So num_in should be hidden_dim (the h_E part)
        num_in = hidden_dim
        layer = DecLayer(hidden_dim, num_in, dropout=0.0)

        h_V = torch.randn(batch_size, seq_len, hidden_dim)
        h_E = torch.randn(batch_size, seq_len, num_neighbors, hidden_dim)

        h_V_out = layer(h_V, h_E)

        assert h_V_out.shape == h_V.shape

    def test_forward_with_mask(self):
        """Test forward pass with masks."""
        from frustrampnn.model.layers import DecLayer

        batch_size, seq_len, num_neighbors = 2, 10, 5
        hidden_dim = 32
        num_in = hidden_dim
        layer = DecLayer(hidden_dim, num_in, dropout=0.0)

        h_V = torch.randn(batch_size, seq_len, hidden_dim)
        h_E = torch.randn(batch_size, seq_len, num_neighbors, hidden_dim)
        mask_V = torch.ones(batch_size, seq_len)
        mask_attend = torch.ones(batch_size, seq_len, num_neighbors)

        h_V_out = layer(h_V, h_E, mask_V=mask_V, mask_attend=mask_attend)

        assert h_V_out.shape == h_V.shape

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the layer."""
        from frustrampnn.model.layers import DecLayer

        batch_size, seq_len, num_neighbors = 2, 10, 5
        hidden_dim = 32
        num_in = hidden_dim
        layer = DecLayer(hidden_dim, num_in, dropout=0.0)

        h_V = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        h_E = torch.randn(batch_size, seq_len, num_neighbors, hidden_dim, requires_grad=True)

        h_V_out = layer(h_V, h_E)
        loss = h_V_out.sum()
        loss.backward()

        assert h_V.grad is not None
        assert h_E.grad is not None


# =============================================================================
# Test Layer Exports
# =============================================================================


class TestLayerExports:
    """Test that all expected classes are exported."""

    def test_all_exports(self):
        """Test all expected exports from layers module."""
        from frustrampnn.model import layers

        expected = [
            "EncLayer",
            "DecLayer",
            "PositionWiseFeedForward",
            "PositionalEncodings",
            "gather_edges",
            "gather_nodes",
            "gather_nodes_t",
            "cat_neighbors_nodes",
        ]

        for name in expected:
            assert hasattr(layers, name), f"Missing export: {name}"

    def test_exports_from_model_init(self):
        """Test exports from model __init__."""
        from frustrampnn.model import (
            DecLayer,
            EncLayer,
            PositionalEncodings,
            PositionWiseFeedForward,
            cat_neighbors_nodes,
            gather_edges,
            gather_nodes,
            gather_nodes_t,
        )

        assert EncLayer is not None
        assert DecLayer is not None
        assert PositionWiseFeedForward is not None
        assert PositionalEncodings is not None
        assert gather_edges is not None
        assert gather_nodes is not None
        assert gather_nodes_t is not None
        assert cat_neighbors_nodes is not None
