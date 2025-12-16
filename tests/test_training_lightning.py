"""Tests for PyTorch Lightning training module."""

import pytest

from frustrampnn.training.lightning import (
    FrustraMPNNCheckpoint,
    TransferModelPL,
    get_checkpoint_callback,
    get_metrics,
    load_checkpoint_safe,
)
from frustrampnn.training.metrics import MetricTracker


class TestGetMetrics:
    """Tests for get_metrics function."""

    def test_returns_dict(self):
        """Test that get_metrics returns a dictionary."""
        metrics = get_metrics()
        assert isinstance(metrics, dict)

    def test_contains_required_metrics(self):
        """Test that all required metrics are present."""
        metrics = get_metrics()
        assert "r2" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "spearman" in metrics

    def test_metrics_are_torchmetrics(self):
        """Test that metrics are torchmetrics instances."""
        from torchmetrics import Metric

        metrics = get_metrics()
        for name, metric in metrics.items():
            assert isinstance(metric, Metric), f"{name} is not a Metric"


class TestTransferModelPL:
    """Tests for TransferModelPL class."""

    def test_class_exists(self):
        """Test that TransferModelPL class exists."""
        assert TransferModelPL is not None

    def test_has_stage_attribute(self):
        """Test that class has stage attribute."""
        assert hasattr(TransferModelPL, "stage")
        assert TransferModelPL.stage == 1

    def test_has_required_methods(self):
        """Test that class has all required methods."""
        required_methods = [
            "forward",
            "training_step",
            "validation_step",
            "test_step",
            "shared_eval",
            "configure_optimizers",
            "on_save_checkpoint",
            "on_load_checkpoint",
        ]
        for method in required_methods:
            assert hasattr(TransferModelPL, method), f"Missing method: {method}"

    def test_inherits_from_lightning_module(self):
        """Test that class inherits from LightningModule."""
        import pytorch_lightning as pl

        assert issubclass(TransferModelPL, pl.LightningModule)


class TestFrustraMPNNCheckpoint:
    """Tests for FrustraMPNNCheckpoint callback."""

    def test_class_exists(self):
        """Test that FrustraMPNNCheckpoint class exists."""
        assert FrustraMPNNCheckpoint is not None

    def test_default_monitor(self):
        """Test default monitor metric."""
        callback = FrustraMPNNCheckpoint()
        assert callback.monitor == "val_frustration_spearman"

    def test_default_mode(self):
        """Test default mode is max."""
        callback = FrustraMPNNCheckpoint()
        assert callback.mode == "max"

    def test_custom_dirpath(self):
        """Test custom directory path."""
        callback = FrustraMPNNCheckpoint(dirpath="/tmp/checkpoints")
        assert callback.dirpath == "/tmp/checkpoints"

    def test_custom_monitor(self):
        """Test custom monitor metric."""
        callback = FrustraMPNNCheckpoint(monitor="val_loss", mode="min")
        assert callback.monitor == "val_loss"
        assert callback.mode == "min"


class TestGetCheckpointCallback:
    """Tests for get_checkpoint_callback function."""

    def test_returns_callback(self):
        """Test that function returns a callback."""
        callback = get_checkpoint_callback("test", "./weights")
        assert isinstance(callback, FrustraMPNNCheckpoint)

    def test_filename_includes_name(self):
        """Test that filename includes experiment name."""
        callback = get_checkpoint_callback("my_experiment", "./weights")
        assert "my_experiment" in callback.filename

    def test_dirpath_set_correctly(self):
        """Test that dirpath is set correctly."""
        callback = get_checkpoint_callback("test", "/custom/path")
        assert callback.dirpath == "/custom/path"


class TestLoadCheckpointSafe:
    """Tests for load_checkpoint_safe function."""

    def test_raises_on_missing_file(self):
        """Test that function raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint_safe("/nonexistent/path/model.ckpt")


class TestMetricTracker:
    """Tests for MetricTracker class."""

    def test_create_tracker(self):
        """Test creating a metric tracker."""
        tracker = MetricTracker()
        assert tracker is not None

    def test_default_prefixes(self):
        """Test default prefixes are created."""
        tracker = MetricTracker()
        assert "train_metrics" in tracker.metrics
        assert "val_metrics" in tracker.metrics
        assert "test_metrics" in tracker.metrics

    def test_default_tasks(self):
        """Test default tasks are created."""
        tracker = MetricTracker()
        assert "frustration" in tracker.metrics["train_metrics"]

    def test_update_and_compute(self):
        """Test updating and computing metrics."""
        import torch

        tracker = MetricTracker()
        pred = torch.randn(10)
        target = pred + torch.randn(10) * 0.1

        tracker.update("train", "frustration", pred, target)
        results = tracker.compute("train", "frustration")

        assert "r2" in results
        assert "mse" in results
        assert "spearman" in results

    def test_reset(self):
        """Test resetting metrics."""
        import torch

        tracker = MetricTracker()
        pred = torch.randn(10)
        target = torch.randn(10)

        tracker.update("train", "frustration", pred, target)
        tracker.reset("train", "frustration")

        # After reset, compute should work but may have different values
        # This just tests that reset doesn't crash
        tracker.update("train", "frustration", pred, target)

    def test_reset_all(self):
        """Test resetting all metrics."""
        import torch

        tracker = MetricTracker()
        pred = torch.randn(10)
        target = torch.randn(10)

        tracker.update("train", "frustration", pred, target)
        tracker.update("val", "frustration", pred, target)
        tracker.reset_all()

        # Should not crash
        tracker.update("train", "frustration", pred, target)

    def test_custom_prefixes(self):
        """Test custom prefixes."""
        tracker = MetricTracker(prefixes=["custom1", "custom2"])
        assert "custom1_metrics" in tracker.metrics
        assert "custom2_metrics" in tracker.metrics

