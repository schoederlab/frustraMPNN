# Scripts

This directory contains utility scripts organized by purpose.

## Directory Structure

```
scripts/
├── analysis/          # Python scripts for data analysis and plotting
├── data/              # Data download and preparation scripts
├── setup/             # Environment setup and installation scripts
└── training/          # Model training scripts
```

## Analysis Scripts

Located in `scripts/analysis/`:

| Script | Description |
|--------|-------------|
| `plot_comparison.py` | Compare FrustraMPNN predictions with frustrapy |
| `plot_aa_error_heatmap.py` | Heatmap of prediction errors by amino acid |
| `plot_secstruct_error.py` | Error analysis by secondary structure |
| `plot_location_error.py` | Error analysis by protein location (surface/core) |

These scripts are used by the Singularity pipeline (`docker/run_singularity.sh`) for validation analysis.

## Data Scripts

Located in `scripts/data/`:

| Script | Description |
|--------|-------------|
| `download_data.sh` | Download training datasets from the data repository |

Usage:
```bash
cd /path/to/frustraMPNN
bash scripts/data/download_data.sh
```

## Setup Scripts

Located in `scripts/setup/`:

| Script | Description |
|--------|-------------|
| `install_frustraMPNN.sh` | Create conda environment with all dependencies |

Usage:
```bash
bash scripts/setup/install_frustraMPNN.sh --pkg_manager conda
bash scripts/setup/install_frustraMPNN.sh --pkg_manager mamba --cuda 12.1
```

Options:
- `--pkg_manager`: Use `conda` or `mamba` (default: conda)
- `--cuda`: Specify CUDA version (e.g., 12.1)
- `--reinstall`: Remove existing environment and reinstall

## Training Scripts

Located in `scripts/training/`:

| Script | Description |
|--------|-------------|
| `run_local_training.sh` | Example script for local model training |

For training, we recommend using the CLI directly:
```bash
frustrampnn train --config config.yaml --epochs 50
```

See `docs/training/README.md` for detailed training documentation.

## Container Scripts

Container-related scripts are in the `docker/` directory:

| Script | Description |
|--------|-------------|
| `docker/build_docker.sh` | Build Docker image |
| `docker/build_singu_from_docker.sh` | Build Singularity image from Docker |
| `docker/run_docker.sh` | Run commands in Docker container |
| `docker/run_singularity.sh` | Run full pipeline in Singularity |

See `docs/containers.md` for detailed container documentation.
