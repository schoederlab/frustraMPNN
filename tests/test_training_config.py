"""Tests for training configuration classes."""

import tempfile
from pathlib import Path

from frustrampnn.training.config import (
    DataLocConfig,
    ModelArchConfig,
    PlatformConfig,
    TrainingConfig,
    TrainingHyperparamsConfig,
)


class TestPlatformConfig:
    """Tests for PlatformConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PlatformConfig()
        assert config.accel == "gpu"
        assert config.cache_dir == "cache"
        assert config.thermompnn_dir == "./"
        assert config.use_tpu is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PlatformConfig(
            accel="cpu",
            cache_dir="/tmp/cache",
            use_tpu=True,
        )
        assert config.accel == "cpu"
        assert config.cache_dir == "/tmp/cache"
        assert config.use_tpu is True


class TestDataLocConfig:
    """Tests for DataLocConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataLocConfig()
        assert config.weights_dir == "./weights"
        assert config.torch_hub == "./esm_models"
        assert config.log_dir == "./logs"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataLocConfig(
            fireprot_csv="/data/fireprot.csv",
            weights_dir="/models/weights",
        )
        assert config.fireprot_csv == "/data/fireprot.csv"
        assert config.weights_dir == "/models/weights"


class TestTrainingHyperparamsConfig:
    """Tests for TrainingHyperparamsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingHyperparamsConfig()
        assert config.num_workers == 4
        assert config.learn_rate == 0.001
        assert config.epochs == 100
        assert config.lr_schedule is True
        assert config.seed == 0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingHyperparamsConfig(
            epochs=50,
            learn_rate=0.0001,
            seed=42,
        )
        assert config.epochs == 50
        assert config.learn_rate == 0.0001
        assert config.seed == 42


class TestModelArchConfig:
    """Tests for ModelArchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelArchConfig()
        assert config.hidden_dims == [64, 32]
        assert config.subtract_mut is True
        assert config.num_final_layers == 3
        assert config.freeze_weights is True
        assert config.lightattn is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelArchConfig(
            hidden_dims=[128, 64, 32],
            freeze_weights=False,
        )
        assert config.hidden_dims == [128, 64, 32]
        assert config.freeze_weights is False


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.project == "frustrampnn"
        assert config.name == "training_run"
        assert config.logger == "csv"
        assert config.datasets == "fireprot"
        assert isinstance(config.platform, PlatformConfig)
        assert isinstance(config.data_loc, DataLocConfig)
        assert isinstance(config.training, TrainingHyperparamsConfig)
        assert isinstance(config.model, ModelArchConfig)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            project="my_project",
            name="experiment_1",
            datasets="megascale",
            training=TrainingHyperparamsConfig(epochs=50),
        )
        assert config.project == "my_project"
        assert config.name == "experiment_1"
        assert config.datasets == "megascale"
        assert config.training.epochs == 50

    def test_to_dictconfig(self):
        """Test conversion to OmegaConf DictConfig."""
        config = TrainingConfig()
        dictconfig = config.to_dictconfig()
        assert dictconfig.project == "frustrampnn"
        assert dictconfig.training.epochs == 100

    def test_save_and_load_yaml(self):
        """Test saving and loading from YAML."""
        config = TrainingConfig(
            project="test_project",
            training=TrainingHyperparamsConfig(epochs=25, seed=123),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.save_yaml(yaml_path)

            loaded = TrainingConfig.from_yaml(yaml_path)
            assert loaded.project == "test_project"
            assert loaded.training.epochs == 25
            assert loaded.training.seed == 123

    def test_from_yaml_with_overrides(self):
        """Test loading from YAML with CLI overrides."""
        config = TrainingConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.save_yaml(yaml_path)

            loaded = TrainingConfig.from_yaml(
                yaml_path,
                overrides=["training.epochs=10", "training.seed=42"],
            )
            assert loaded.training.epochs == 10
            assert loaded.training.seed == 42


class TestTrainingConfigFromDictConfig:
    """Tests for TrainingConfig.from_dictconfig."""

    def test_from_dictconfig(self):
        """Test creating config from OmegaConf DictConfig."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "project": "test",
            "name": "run1",
            "logger": "wandb",
            "datasets": "combo",
            "platform": {"accel": "cpu"},
            "training": {"epochs": 50, "seed": 42},
            "model": {"hidden_dims": [128, 64]},
        })

        config = TrainingConfig.from_dictconfig(cfg)
        assert config.project == "test"
        assert config.name == "run1"
        assert config.logger == "wandb"
        assert config.datasets == "combo"
        assert config.platform.accel == "cpu"
        assert config.training.epochs == 50
        assert config.training.seed == 42
        assert config.model.hidden_dims == [128, 64]

    def test_from_dictconfig_with_missing_keys(self):
        """Test creating config from partial DictConfig."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "project": "test",
        })

        config = TrainingConfig.from_dictconfig(cfg)
        assert config.project == "test"
        # Should use defaults for missing keys
        assert config.training.epochs == 100
        assert config.model.hidden_dims == [64, 32]

