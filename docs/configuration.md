# Configuration Reference

This document describes all configuration options for FrustraMPNN.

## Overview

FrustraMPNN uses YAML configuration files for training. Inference typically uses embedded configuration from checkpoints.

## Configuration File Structure

```yaml
# Project identification
project: frustrampnn
name: my_experiment
logger: csv  # csv, wandb, tensorboard

# Dataset selection
datasets: fireprot  # fireprot, megascale, combo

# Platform settings
platform:
  accel: gpu
  cache_dir: ./cache
  use_tpu: false

# Data locations
data_loc:
  fireprot_csv: /path/to/fireprot.csv
  fireprot_splits: /path/to/splits.pkl
  fireprot_pdbs: /path/to/pdbs/
  weights_dir: ./weights
  log_dir: ./logs

# Training hyperparameters
training:
  epochs: 100
  learn_rate: 0.001
  lr_schedule: true
  seed: 42

# Model architecture
model:
  hidden_dims: [64, 32]
  freeze_weights: true
  lightattn: true
```

## Configuration Sections

### Project Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `project` | str | "frustrampnn" | Project name for logging |
| `name` | str | "default" | Experiment name |
| `logger` | str | "csv" | Logger type (csv, wandb, tensorboard) |
| `datasets` | str | "fireprot" | Dataset to use |

### Platform Settings

```yaml
platform:
  accel: gpu           # Accelerator: gpu, cpu, auto
  cache_dir: ./cache   # PDB cache directory
  use_tpu: false       # TPU support (experimental)
  thermompnn_dir: .    # Directory containing vanilla_model_weights
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `accel` | str | "gpu" | Accelerator type |
| `cache_dir` | str | "cache" | Cache directory for parsed PDBs |
| `use_tpu` | bool | false | Enable TPU support |
| `thermompnn_dir` | str | "." | Base directory for model weights |

### Data Locations

```yaml
data_loc:
  # FireProt dataset
  fireprot_csv: /path/to/fireprot.csv
  fireprot_splits: /path/to/splits.pkl
  fireprot_pdbs: /path/to/pdbs/
  
  # MegaScale dataset
  megascale_csv: /path/to/megascale.csv
  megascale_splits: /path/to/splits.pkl
  megascale_pdbs: /path/to/pdbs/
  
  # Output directories
  weights_dir: ./weights
  log_dir: ./logs
```

| Option | Type | Description |
|--------|------|-------------|
| `fireprot_csv` | str | Path to FireProt CSV file |
| `fireprot_splits` | str | Path to FireProt splits pickle |
| `fireprot_pdbs` | str | Directory containing FireProt PDB files |
| `megascale_csv` | str | Path to MegaScale CSV file |
| `megascale_splits` | str | Path to MegaScale splits pickle |
| `megascale_pdbs` | str | Directory containing MegaScale PDB files |
| `weights_dir` | str | Directory for saving checkpoints |
| `log_dir` | str | Directory for training logs |

### Training Hyperparameters

```yaml
training:
  epochs: 100
  learn_rate: 0.001
  mpnn_learn_rate: 0.0001
  lr_schedule: true
  num_workers: 4
  seed: 42
  reweighting: false
  weight_method: weight_lds_inverse
  add_esm_embeddings: false
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `epochs` | int | 100 | Maximum training epochs |
| `learn_rate` | float | 0.001 | Learning rate for MLP layers |
| `mpnn_learn_rate` | float | 0.001 | Learning rate for ProteinMPNN |
| `lr_schedule` | bool | true | Enable learning rate scheduling |
| `num_workers` | int | 4 | DataLoader workers |
| `seed` | int | 0 | Random seed |
| `reweighting` | bool | false | Enable sample reweighting |
| `weight_method` | str | "weight_lds_inverse" | Reweighting method |
| `add_esm_embeddings` | bool | false | Use ESM embeddings |

### Model Architecture

```yaml
model:
  hidden_dims: [64, 32]
  freeze_weights: true
  lightattn: true
  num_final_layers: 3
  dropout: 0.1
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `hidden_dims` | list | [64, 32] | Hidden layer dimensions |
| `freeze_weights` | bool | true | Freeze ProteinMPNN weights |
| `lightattn` | bool | true | Use LightAttention |
| `num_final_layers` | int | 3 | Number of final MLP layers |
| `dropout` | float | 0.1 | Dropout rate |

## Sample Reweighting

FrustraMPNN supports sample reweighting to handle imbalanced frustration distributions.

### Available Methods

| Method | Description |
|--------|-------------|
| `weight_bin_inverse` | Inverse of bin frequency |
| `weight_lds_inverse` | LDS smoothed inverse |
| `weight_bin_inverse_sqrt` | Square root of inverse |
| `weight_lds_inverse_sqrt` | Square root of LDS |

### Configuration

```yaml
training:
  reweighting: true
  weight_method: weight_lds_inverse
```

## ESM Embeddings

Enable ESM-2 language model embeddings:

```yaml
training:
  add_esm_embeddings: true

model:
  esm_model: esm2_t33_650M_UR50D
```

Available ESM models:
- `esm2_t6_8M_UR50D` (smallest)
- `esm2_t12_35M_UR50D`
- `esm2_t30_150M_UR50D`
- `esm2_t33_650M_UR50D` (recommended)
- `esm2_t36_3B_UR50D`
- `esm2_t48_15B_UR50D` (largest)

## Logging

### CSV Logger (default)

```yaml
logger: csv
data_loc:
  log_dir: ./logs
```

### Weights & Biases

```yaml
logger: wandb
wandb:
  project: frustrampnn
  entity: your-username
  tags: [experiment1]
```

### TensorBoard

```yaml
logger: tensorboard
data_loc:
  log_dir: ./logs
```

## Complete Example

```yaml
# Full configuration example
project: frustrampnn
name: fireprot_balanced_esm
logger: wandb

datasets: fireprot

platform:
  accel: gpu
  cache_dir: ./cache
  thermompnn_dir: /path/to/frustrampnn

data_loc:
  fireprot_csv: /data/fireprot/fireprot_balanced.csv
  fireprot_splits: /data/fireprot/splits.pkl
  fireprot_pdbs: /data/fireprot/pdbs/
  weights_dir: ./checkpoints
  log_dir: ./logs

training:
  epochs: 100
  learn_rate: 0.001
  mpnn_learn_rate: 0.0001
  lr_schedule: true
  num_workers: 8
  seed: 42
  reweighting: true
  weight_method: weight_lds_inverse
  add_esm_embeddings: true

model:
  hidden_dims: [64, 32]
  freeze_weights: true
  lightattn: true
  num_final_layers: 3
  dropout: 0.1
  esm_model: esm2_t33_650M_UR50D

wandb:
  project: frustrampnn
  entity: schoederlab
  tags: [fireprot, balanced, esm]
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control GPU visibility |
| `FRUSTRAMPNN_CACHE_DIR` | Override cache directory |
| `WANDB_API_KEY` | Weights & Biases API key |
| `OMP_NUM_THREADS` | OpenMP threads |

## Loading Configuration

### In Python

```python
from frustrampnn.training import TrainingConfig

# From YAML file
config = TrainingConfig.from_yaml("config.yaml")

# Override values
config.training.epochs = 50
config.training.seed = 123
```

### From CLI

```bash
# Use config file
frustrampnn train --config config.yaml

# Override values
frustrampnn train --config config.yaml --epochs 50 --seed 123
```

## See Also

- [Training Guide](training/README.md) - Training custom models
- [CLI Reference](api/cli.md) - Command line usage
- [API Reference](api/reference.md) - Python API

