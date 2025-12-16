# FrustraMPNN Training Guide

This guide covers training FrustraMPNN models from scratch or fine-tuning existing models.

## Quick Start

### 1. Install Dependencies

```bash
pip install frustrampnn[train]
```

### 2. Prepare Configuration

Create a `config.yaml` file:

```yaml
project: frustrampnn
name: my_training_run
logger: csv
datasets: fireprot

platform:
  accel: gpu
  cache_dir: ./cache

data_loc:
  fireprot_csv: /path/to/fireprot.csv
  fireprot_splits: /path/to/splits.pkl
  fireprot_pdbs: /path/to/pdbs/
  weights_dir: ./weights
  log_dir: ./logs

training:
  epochs: 100
  learn_rate: 0.001
  lr_schedule: true
  seed: 42

model:
  hidden_dims: [64, 32]
  freeze_weights: true
  lightattn: true
```

### 3. Run Training

```bash
frustrampnn train --config config.yaml
```

## CLI Commands

### Train Command

```bash
frustrampnn train --config config.yaml [OPTIONS]

Options:
  -c, --config FILE     Path to training config YAML file (required)
  -e, --epochs INTEGER  Override number of training epochs
  -s, --seed INTEGER    Random seed for reproducibility
  -r, --resume FILE     Resume training from checkpoint
  -q, --quiet           Suppress verbose output
```

### Evaluate Command

```bash
frustrampnn evaluate --checkpoint model.ckpt [OPTIONS]

Options:
  -c, --checkpoint FILE     Model checkpoint file (required)
  --config FILE             Config YAML (optional)
  --split [train|val|test]  Data split to evaluate (default: test)
  -o, --output FILE         Output file for results (JSON)
  -q, --quiet               Suppress verbose output
```

## Configuration Reference

### Platform Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `accel` | str | "gpu" | Accelerator type ("gpu", "cpu", "auto") |
| `cache_dir` | str | "cache" | Directory for caching parsed PDBs |
| `use_tpu` | bool | false | Whether to use TPU |

### Data Location Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fireprot_csv` | str | "" | Path to FireProt CSV file |
| `fireprot_splits` | str | "" | Path to FireProt splits pickle |
| `fireprot_pdbs` | str | "" | Directory containing FireProt PDB files |
| `megascale_csv` | str | "" | Path to MegaScale CSV file |
| `megascale_splits` | str | "" | Path to MegaScale splits pickle |
| `megascale_pdbs` | str | "" | Directory containing MegaScale PDB files |
| `weights_dir` | str | "./weights" | Directory for saving checkpoints |
| `log_dir` | str | "./logs" | Directory for training logs |

### Training Hyperparameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `epochs` | int | 100 | Maximum training epochs |
| `learn_rate` | float | 0.001 | Learning rate for MLP layers |
| `mpnn_learn_rate` | float | 0.001 | Learning rate for ProteinMPNN |
| `lr_schedule` | bool | true | Enable learning rate scheduling |
| `num_workers` | int | 4 | Number of dataloader workers |
| `seed` | int | 0 | Random seed for reproducibility |
| `reweighting` | bool | false | Enable sample reweighting |
| `weight_method` | str | "weight_lds_inverse" | Reweighting method |
| `add_esm_embeddings` | bool | false | Use ESM embeddings |

### Model Architecture

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `hidden_dims` | list | [64, 32] | Hidden layer dimensions |
| `freeze_weights` | bool | true | Freeze ProteinMPNN weights |
| `lightattn` | bool | true | Use LightAttention |
| `num_final_layers` | int | 3 | Number of final MLP layers |

## Datasets

### FireProt Dataset

The FireProt dataset contains thermostability measurements for protein variants.

Required files:
- CSV file with mutation data
- Splits pickle file (train/val/test)
- PDB files directory

### MegaScale Dataset

The MegaScale dataset contains large-scale protein stability measurements.

Required files:
- CSV file with mutation data
- Splits pickle file
- PDB files directory

### Combo Dataset

Combines FireProt and MegaScale datasets for training.

## Sample Reweighting

FrustraMPNN supports sample reweighting to handle imbalanced frustration distributions.

### Available Methods

| Method | Description |
|--------|-------------|
| `weight_bin_inverse` | Inverse of bin frequency |
| `weight_lds_inverse` | LDS smoothed inverse |
| `weight_bin_inverse_sqrt` | Square root of inverse |
| `weight_lds_inverse_sqrt` | Square root of LDS |

### Enable Reweighting

```yaml
training:
  reweighting: true
  weight_method: weight_lds_inverse
```

## Checkpoints

### Checkpoint Format

Checkpoints contain:
- `state_dict`: Trainable parameters only
- `cfg`: Training configuration
- `epoch`: Current epoch
- `global_step`: Global step count

### Resume Training

```bash
frustrampnn train --config config.yaml --resume checkpoint.ckpt
```

### Load Checkpoint in Python

```python
from frustrampnn.training.lightning import load_checkpoint_safe

checkpoint = load_checkpoint_safe("model.ckpt")
config = checkpoint["cfg"]
state_dict = checkpoint["state_dict"]
```

## Metrics

Training tracks the following metrics:

| Metric | Description |
|--------|-------------|
| `train_frustration_r2` | R² score on training set |
| `train_frustration_mse` | MSE on training set |
| `train_frustration_rmse` | RMSE on training set |
| `train_frustration_spearman` | Spearman correlation on training set |
| `val_frustration_*` | Same metrics on validation set |

## Python API

### Using the Trainer Class

```python
from frustrampnn.training import Trainer, TrainingConfig

# Load configuration
config = TrainingConfig.from_yaml("config.yaml")

# Create trainer
trainer = Trainer(config)

# Run training
trainer.fit()

# Evaluate
results = trainer.test(checkpoint_path="best_model.ckpt")
```

### Using TransferModelPL Directly

```python
from frustrampnn.training.lightning import TransferModelPL
from frustrampnn.training import TrainingConfig
import pytorch_lightning as pl

# Create model
config = TrainingConfig.from_yaml("config.yaml")
model = TransferModelPL(config)

# Create Lightning trainer
trainer = pl.Trainer(max_epochs=100)

# Train
trainer.fit(model, train_loader, val_loader)
```

## Troubleshooting

### Out of Memory

- Reduce `num_workers` in config
- Use CPU instead of GPU: `platform.accel: cpu`
- Enable gradient checkpointing (if available)

### Slow Training

- Enable PDB caching: `platform.cache_dir: ./cache`
- Increase `num_workers`
- Use GPU acceleration

### Poor Performance

- Try different learning rates
- Enable sample reweighting
- Increase training epochs
- Check data quality

