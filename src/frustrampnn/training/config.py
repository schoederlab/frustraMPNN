"""
Configuration dataclasses for FrustraMPNN training.

This module provides type-safe configuration classes that are compatible
with OmegaConf YAML files. The classes mirror the structure of the
original config.yaml format.

Example:
    >>> from frustrampnn.training.config import TrainingConfig
    >>> config = TrainingConfig.from_yaml("config.yaml")
    >>> print(config.training.learn_rate)
    0.001
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class PlatformConfig:
    """
    Platform and hardware configuration.

    Attributes:
        accel: Accelerator type ("gpu", "cpu", "auto")
        cache_dir: Directory for caching parsed PDBs
        thermompnn_dir: Base directory for model weights
        use_tpu: Whether to use TPU (for Google Cloud)
    """

    accel: str = "gpu"
    cache_dir: str = "cache"
    thermompnn_dir: str = "./"
    use_tpu: bool = False


@dataclass
class DataLocConfig:
    """
    Data location configuration.

    Attributes:
        megascale_csv: Path to MegaScale CSV file
        megascale_splits: Path to MegaScale splits pickle
        megascale_pdbs: Directory containing MegaScale PDB files
        fireprot_csv: Path to FireProt CSV file
        fireprot_splits: Path to FireProt splits pickle
        fireprot_pdbs: Directory containing FireProt PDB files
        weights_dir: Directory for saving model checkpoints
        torch_hub: Directory for ESM model cache
        log_dir: Directory for training logs
    """

    megascale_csv: str = ""
    megascale_splits: str = ""
    megascale_pdbs: str = ""
    fireprot_csv: str = ""
    fireprot_splits: str = ""
    fireprot_pdbs: str = ""
    weights_dir: str = "./weights"
    torch_hub: str = "./esm_models"
    log_dir: str = "./logs"


@dataclass
class TrainingHyperparamsConfig:
    """
    Training hyperparameters configuration.

    Attributes:
        num_workers: Number of dataloader workers
        learn_rate: Learning rate for MLP layers
        epochs: Maximum training epochs
        lr_schedule: Whether to use learning rate scheduling
        mpnn_learn_rate: Learning rate for ProteinMPNN (if unfrozen)
        two_stage: Whether to use two-stage training
        testing: Whether to run test evaluation after training
        ddp: Whether to use distributed data parallel
        add_esm_embeddings: Whether to use ESM embeddings
        esm_model: ESM model name (if using ESM)
        reweighting: Whether to use sample reweighting
        weight_method: Reweighting method name
        seed: Random seed for reproducibility
    """

    num_workers: int = 4
    learn_rate: float = 0.001
    epochs: int = 100
    lr_schedule: bool = True
    mpnn_learn_rate: float = 0.001
    two_stage: bool = False
    testing: bool = True
    ddp: bool = False
    add_esm_embeddings: bool = False
    esm_model: str = "esm2_t33_650M_UR50D"
    reweighting: bool = False
    weight_method: str = "weight_lds_inverse"
    seed: int | None = 0


@dataclass
class ModelArchConfig:
    """
    Model architecture configuration.

    Attributes:
        hidden_dims: Hidden layer dimensions for MLP
        subtract_mut: Whether to subtract mutation embedding
        num_final_layers: Number of final MLP layers
        freeze_weights: Whether to freeze ProteinMPNN weights
        load_pretrained: Whether to load pretrained ProteinMPNN
        lightattn: Whether to use LightAttention
    """

    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    subtract_mut: bool = True
    num_final_layers: int = 3
    freeze_weights: bool = True
    load_pretrained: bool = True
    lightattn: bool = True


@dataclass
class TrainingConfig:
    """
    Complete training configuration.

    This is the top-level configuration class that contains all
    training settings. It can be loaded from YAML files or created
    programmatically.

    Attributes:
        project: Project name for logging
        name: Experiment name
        logger: Logger type ("wandb" or "csv")
        datasets: Dataset name ("fireprot", "megascale", "combo")
        platform: Platform configuration
        data_loc: Data location configuration
        training: Training hyperparameters
        model: Model architecture configuration

    Example:
        >>> config = TrainingConfig.from_yaml("config.yaml")
        >>> config = TrainingConfig(
        ...     project="my_project",
        ...     name="experiment_1",
        ...     training=TrainingHyperparamsConfig(epochs=50),
        ... )
    """

    project: str = "frustrampnn"
    name: str = "training_run"
    logger: str = "csv"  # "wandb" or "csv"
    datasets: str = "fireprot"

    platform: PlatformConfig = field(default_factory=PlatformConfig)
    data_loc: DataLocConfig = field(default_factory=DataLocConfig)
    training: TrainingHyperparamsConfig = field(default_factory=TrainingHyperparamsConfig)
    model: ModelArchConfig = field(default_factory=ModelArchConfig)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        overrides: list[str] | None = None,
    ) -> TrainingConfig:
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file
            overrides: List of CLI overrides (e.g., ["training.epochs=50"])

        Returns:
            TrainingConfig instance

        Example:
            >>> config = TrainingConfig.from_yaml("config.yaml")
            >>> config = TrainingConfig.from_yaml(
            ...     "config.yaml",
            ...     overrides=["training.epochs=50", "training.seed=42"]
            ... )
        """
        cfg = OmegaConf.load(yaml_path)

        if overrides:
            cli_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, cli_cfg)

        return cls.from_dictconfig(cfg)

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> TrainingConfig:
        """
        Create TrainingConfig from OmegaConf DictConfig.

        This method provides backward compatibility with existing
        OmegaConf configurations.

        Args:
            cfg: OmegaConf DictConfig

        Returns:
            TrainingConfig instance
        """
        # Extract nested configs
        platform = PlatformConfig(
            accel=cfg.get("platform", {}).get("accel", "gpu"),
            cache_dir=cfg.get("platform", {}).get("cache_dir", "cache"),
            thermompnn_dir=cfg.get("platform", {}).get("thermompnn_dir", "./"),
            use_tpu=cfg.get("platform", {}).get("use_tpu", False),
        )

        data_loc = DataLocConfig(
            megascale_csv=cfg.get("data_loc", {}).get("megascale_csv", ""),
            megascale_splits=cfg.get("data_loc", {}).get("megascale_splits", ""),
            megascale_pdbs=cfg.get("data_loc", {}).get("megascale_pdbs", ""),
            fireprot_csv=cfg.get("data_loc", {}).get("fireprot_csv", ""),
            fireprot_splits=cfg.get("data_loc", {}).get("fireprot_splits", ""),
            fireprot_pdbs=cfg.get("data_loc", {}).get("fireprot_pdbs", ""),
            weights_dir=cfg.get("data_loc", {}).get("weights_dir", "./weights"),
            torch_hub=cfg.get("data_loc", {}).get("torch_hub", "./esm_models"),
            log_dir=cfg.get("data_loc", {}).get("log_dir", "./logs"),
        )

        training = TrainingHyperparamsConfig(
            num_workers=cfg.get("training", {}).get("num_workers", 4),
            learn_rate=cfg.get("training", {}).get("learn_rate", 0.001),
            epochs=cfg.get("training", {}).get("epochs", 100),
            lr_schedule=cfg.get("training", {}).get("lr_schedule", True),
            mpnn_learn_rate=cfg.get("training", {}).get("mpnn_learn_rate", 0.001),
            two_stage=cfg.get("training", {}).get("two_stage", False),
            testing=cfg.get("training", {}).get("testing", True),
            ddp=cfg.get("training", {}).get("ddp", False),
            add_esm_embeddings=cfg.get("training", {}).get("add_esm_embeddings", False),
            esm_model=cfg.get("training", {}).get("esm_model", "esm2_t33_650M_UR50D"),
            reweighting=cfg.get("training", {}).get("reweighting", False),
            weight_method=cfg.get("training", {}).get("weight_method", "weight_lds_inverse"),
            seed=cfg.get("training", {}).get("seed", 0),
        )

        model = ModelArchConfig(
            hidden_dims=list(cfg.get("model", {}).get("hidden_dims", [64, 32])),
            subtract_mut=cfg.get("model", {}).get("subtract_mut", True),
            num_final_layers=cfg.get("model", {}).get("num_final_layers", 3),
            freeze_weights=cfg.get("model", {}).get("freeze_weights", True),
            load_pretrained=cfg.get("model", {}).get("load_pretrained", True),
            lightattn=cfg.get("model", {}).get("lightattn", True),
        )

        return cls(
            project=cfg.get("project", "frustrampnn"),
            name=cfg.get("name", "training_run"),
            logger=cfg.get("logger", "csv"),
            datasets=cfg.get("datasets", "fireprot"),
            platform=platform,
            data_loc=data_loc,
            training=training,
            model=model,
        )

    def to_dictconfig(self) -> DictConfig:
        """
        Convert to OmegaConf DictConfig.

        This method provides backward compatibility with code
        expecting OmegaConf configurations.

        Returns:
            OmegaConf DictConfig
        """
        return OmegaConf.structured(self)

    def save_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        cfg = self.to_dictconfig()
        OmegaConf.save(cfg, path)
