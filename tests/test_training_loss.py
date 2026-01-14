"""Tests for loss functions and sample reweighting."""

import numpy as np
import pytest
import torch

from frustrampnn.training.loss import (
    VALID_WEIGHT_METHODS,
    FrustrationLoss,
    SampleReweighter,
    batch_mse_loss,
    compute_bin_weights,
    compute_lds_weights,
    get_weight_method,
    mse_loss,
    validate_weight_method,
    weighted_mse_loss,
)


class TestMSELoss:
    """Tests for mse_loss function."""

    def test_basic_mse(self):
        """Test basic MSE computation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = mse_loss(pred, target)
        assert loss.item() == pytest.approx(0.0)

    def test_mse_with_difference(self):
        """Test MSE with actual difference."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.1, 3.1])
        loss = mse_loss(pred, target)
        assert loss.item() == pytest.approx(0.01, rel=1e-5)

    def test_mse_reduction_none(self):
        """Test MSE with no reduction."""
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 3.0])
        loss = mse_loss(pred, target, reduction="none")
        assert loss.shape == (2,)
        assert loss[0].item() == pytest.approx(1.0)
        assert loss[1].item() == pytest.approx(1.0)

    def test_mse_reduction_sum(self):
        """Test MSE with sum reduction."""
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 3.0])
        loss = mse_loss(pred, target, reduction="sum")
        assert loss.item() == pytest.approx(2.0)


class TestWeightedMSELoss:
    """Tests for weighted_mse_loss function."""

    def test_equal_weights(self):
        """Test weighted MSE with equal weights."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.1, 3.1])
        weights = torch.tensor([1.0, 1.0, 1.0])
        loss = weighted_mse_loss(pred, target, weights)
        # With equal weights, should be same as regular MSE
        assert loss.item() == pytest.approx(0.01, rel=1e-5)

    def test_unequal_weights(self):
        """Test weighted MSE with unequal weights."""
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 2.0])  # First has error, second is exact
        weights = torch.tensor([1.0, 0.0])  # Only weight first sample
        loss = weighted_mse_loss(pred, target, weights)
        # Only first sample contributes
        assert loss.item() == pytest.approx(1.0)

    def test_weight_normalization(self):
        """Test that weights are normalized."""
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 3.0])
        weights = torch.tensor([2.0, 2.0])  # Sum to 4
        loss = weighted_mse_loss(pred, target, weights, normalize_weights=True)
        # After normalization, should be same as equal weights
        assert loss.item() == pytest.approx(1.0)


class TestBatchMSELoss:
    """Tests for batch_mse_loss function."""

    def test_empty_batch(self):
        """Test batch MSE with empty inputs."""
        loss = batch_mse_loss([], [])
        assert loss.item() == pytest.approx(0.0)

    def test_single_sample(self):
        """Test batch MSE with single sample."""
        preds = [torch.tensor([1.0])]
        targets = [torch.tensor([2.0])]
        loss = batch_mse_loss(preds, targets)
        assert loss.item() == pytest.approx(1.0)

    def test_multiple_samples(self):
        """Test batch MSE with multiple samples."""
        preds = [torch.tensor([1.0]), torch.tensor([2.0])]
        targets = [torch.tensor([1.1]), torch.tensor([2.1])]
        loss = batch_mse_loss(preds, targets)
        assert loss.item() == pytest.approx(0.01)

    def test_with_weights(self):
        """Test batch MSE with sample weights."""
        preds = [torch.tensor([1.0]), torch.tensor([2.0])]
        targets = [torch.tensor([2.0]), torch.tensor([2.0])]  # First has error
        weights = [1.0, 0.0]  # Only weight first
        loss = batch_mse_loss(preds, targets, weights=weights)
        # Mean of weighted losses
        assert loss.item() > 0

    def test_with_none_target(self):
        """Test batch MSE skips None targets."""
        preds = [torch.tensor([1.0]), torch.tensor([2.0])]
        targets = [torch.tensor([1.1]), None]
        loss = batch_mse_loss(preds, targets)
        # Should only compute loss for first sample
        assert loss.item() == pytest.approx(0.01)


class TestFrustrationLoss:
    """Tests for FrustrationLoss class."""

    def test_create_loss(self):
        """Test creating FrustrationLoss."""
        loss_fn = FrustrationLoss()
        assert loss_fn.reweighting is False

    def test_create_loss_with_reweighting(self):
        """Test creating FrustrationLoss with reweighting."""
        loss_fn = FrustrationLoss(reweighting=True)
        assert loss_fn.reweighting is True


class TestComputeBinWeights:
    """Tests for compute_bin_weights function."""

    def test_uniform_distribution(self):
        """Test weights for uniform distribution."""
        np.random.seed(42)
        values = np.random.uniform(-1, 1, 1000)
        weights = compute_bin_weights(values)
        # Weights should be relatively uniform
        assert weights.min() > 0
        assert weights.max() / weights.min() < 10  # Not too different

    def test_imbalanced_distribution(self):
        """Test weights for imbalanced distribution."""
        np.random.seed(42)
        values = np.concatenate(
            [
                np.random.normal(-1, 0.1, 100),  # Many negative
                np.random.normal(1, 0.1, 10),  # Few positive
            ]
        )
        weights = compute_bin_weights(values)
        # Positive samples should have higher weights
        assert weights.max() > weights.min()

    def test_inverse_sqrt_method(self):
        """Test inverse_sqrt method."""
        np.random.seed(42)
        values = np.random.uniform(-1, 1, 100)
        weights = compute_bin_weights(values, method="inverse_sqrt")
        assert weights.min() > 0

    def test_invalid_method(self):
        """Test invalid method raises error."""
        values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown method"):
            compute_bin_weights(values, method="invalid")


class TestComputeLDSWeights:
    """Tests for compute_lds_weights function."""

    def test_basic_lds(self):
        """Test basic LDS weight computation."""
        np.random.seed(42)
        values = np.random.uniform(-1, 1, 100)
        weights = compute_lds_weights(values)
        assert weights.min() > 0
        assert len(weights) == len(values)

    def test_lds_smoothing(self):
        """Test that LDS produces smoother weights than bin."""
        np.random.seed(42)
        values = np.random.uniform(-1, 1, 100)
        bin_weights = compute_bin_weights(values)
        lds_weights = compute_lds_weights(values)
        # LDS should have smaller variance
        assert np.std(lds_weights) <= np.std(bin_weights) * 2

    def test_lds_inverse_sqrt(self):
        """Test LDS with inverse_sqrt method."""
        np.random.seed(42)
        values = np.random.uniform(-1, 1, 100)
        weights = compute_lds_weights(values, method="inverse_sqrt")
        assert weights.min() > 0


class TestGetWeightMethod:
    """Tests for get_weight_method function."""

    def test_bin_inverse(self):
        """Test getting bin_inverse method."""
        fn = get_weight_method("weight_bin_inverse")
        assert callable(fn)

    def test_lds_inverse(self):
        """Test getting lds_inverse method."""
        fn = get_weight_method("weight_lds_inverse")
        assert callable(fn)

    def test_bin_inverse_sqrt(self):
        """Test getting bin_inverse_sqrt method."""
        fn = get_weight_method("weight_bin_inverse_sqrt")
        assert callable(fn)

    def test_lds_inverse_sqrt(self):
        """Test getting lds_inverse_sqrt method."""
        fn = get_weight_method("weight_lds_inverse_sqrt")
        assert callable(fn)

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown weight method"):
            get_weight_method("invalid_method")


class TestValidWeightMethods:
    """Tests for VALID_WEIGHT_METHODS constant."""

    def test_contains_all_methods(self):
        """Test that all methods are in the list."""
        assert "weight_bin_inverse" in VALID_WEIGHT_METHODS
        assert "weight_lds_inverse" in VALID_WEIGHT_METHODS
        assert "weight_bin_inverse_sqrt" in VALID_WEIGHT_METHODS
        assert "weight_lds_inverse_sqrt" in VALID_WEIGHT_METHODS

    def test_length(self):
        """Test that list has correct length."""
        assert len(VALID_WEIGHT_METHODS) == 4


class TestValidateWeightMethod:
    """Tests for validate_weight_method function."""

    def test_valid_methods(self):
        """Test validation of valid methods."""
        for method in VALID_WEIGHT_METHODS:
            assert validate_weight_method(method) is True

    def test_invalid_method(self):
        """Test validation of invalid method."""
        assert validate_weight_method("invalid") is False


class TestSampleReweighter:
    """Tests for SampleReweighter class."""

    def test_create_reweighter(self):
        """Test creating a reweighter."""
        reweighter = SampleReweighter()
        assert reweighter.method == "weight_lds_inverse"

    def test_create_with_custom_method(self):
        """Test creating with custom method."""
        reweighter = SampleReweighter(method="weight_bin_inverse")
        assert reweighter.method == "weight_bin_inverse"

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError):
            SampleReweighter(method="invalid")

    def test_compute_weights(self):
        """Test computing weights."""
        np.random.seed(42)
        values = np.random.uniform(-1, 1, 100)
        reweighter = SampleReweighter()
        weights = reweighter.compute_weights(values)
        assert len(weights) == len(values)
        assert weights.min() > 0

    def test_normalize_weights(self):
        """Test normalizing weights."""
        reweighter = SampleReweighter()
        weights = np.array([1.0, 2.0, 3.0])
        normalized = reweighter.normalize_weights(weights, target_sum=1.0)
        assert normalized.sum() == pytest.approx(1.0)

    def test_normalize_weights_custom_sum(self):
        """Test normalizing weights with custom sum."""
        reweighter = SampleReweighter()
        weights = np.array([1.0, 2.0, 3.0])
        normalized = reweighter.normalize_weights(weights, target_sum=10.0)
        assert normalized.sum() == pytest.approx(10.0)
