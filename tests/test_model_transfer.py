"""Tests for TransferModel module."""


def test_transfer_model_import():
    """Test that TransferModel can be imported."""
    from frustrampnn.model import TransferModel

    assert TransferModel is not None


def test_transfer_model_pl_import():
    """Test that TransferModelPL can be imported."""
    from frustrampnn.model import TransferModelPL

    assert TransferModelPL is not None


def test_light_attention_import():
    """Test that LightAttention can be imported."""
    from frustrampnn.model import LightAttention

    assert LightAttention is not None


def test_utility_functions_import():
    """Test that utility functions can be imported."""
    from frustrampnn.model import get_esm_model, get_metrics, get_protein_mpnn

    assert callable(get_protein_mpnn)
    assert callable(get_esm_model)
    assert callable(get_metrics)


def test_get_metrics():
    """Test that get_metrics returns expected metrics."""
    from frustrampnn.model import get_metrics

    metrics = get_metrics()

    assert "r2" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "spearman" in metrics
