# Installation Guide

This guide covers all installation methods for FrustraMPNN.

## Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended, but CPU works)

## Installation Methods

### Method 1: pip install (Recommended for most users)

Install from PyPI (when available):

```bash
pip install frustrampnn
```

Install with visualization support:

```bash
pip install frustrampnn[viz]
```

Install with all optional dependencies:

```bash
pip install frustrampnn[all]
```

### Method 2: Install from source

Clone the repository and install:

```bash
git clone https://github.com/schoederlab/frustraMPNN.git
cd frustraMPNN
pip install -e .
```

With all extras:

```bash
pip install -e ".[all]"
```

### Method 3: Using uv (Recommended for development)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager (10-100x faster than pip).

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone and install:

```bash
git clone https://github.com/schoederlab/frustraMPNN.git
cd frustraMPNN
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[all]"
```

3. Verify installation:

```bash
python -c "import frustrampnn; print(frustrampnn.__version__)"
```

### Method 4: Docker

For reproducible environments:

```bash
cd docker
bash build_docker.sh
```

Run predictions:

```bash
bash docker/run_docker.sh predict --pdb protein.pdb --checkpoint model.ckpt
```

### Method 5: Singularity (for HPC)

Build from Docker:

```bash
bash docker/build_singu_from_docker.sh
```

Run on cluster:

```bash
singularity exec --nv frustrampnn.sif frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt
```

### Method 6: Google Colab

For zero-installation access, use the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schoederlab/frustraMPNN/blob/main/notebooks/FrustraMPNN_Colab.ipynb)

## Optional Dependencies

FrustraMPNN has several optional dependency groups:

| Group | Dependencies | Use Case |
|-------|--------------|----------|
| `viz` | matplotlib, seaborn, plotly, kaleido, py3Dmol | Visualization |
| `train` | pytorch-lightning, torchmetrics, wandb | Training models |
| `esm` | fair-esm | ESM embeddings |
| `validation` | scipy | Statistical validation |
| `dev` | pytest, ruff, mypy | Development |
| `all` | All of the above | Full installation |

Install specific groups:

```bash
pip install frustrampnn[viz,validation]
```

## Model Weights

FrustraMPNN requires pretrained model weights. Download options:

### Option 1: From checkpoint file

If you have a checkpoint file (`.ckpt`):

```python
from frustrampnn import FrustraMPNN
model = FrustraMPNN.from_pretrained("path/to/checkpoint.ckpt")
```

### Option 2: Bundled weights

The repository includes vanilla ProteinMPNN weights in `inference/vanilla_model_weights/`.

## Verifying Installation

Run the following to verify your installation:

```python
import frustrampnn
print(f"FrustraMPNN version: {frustrampnn.__version__}")

# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

Run the test suite:

```bash
pytest tests/ -v
```

## Troubleshooting

### CUDA not detected

If PyTorch does not detect your GPU:

1. Check CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### Import errors

If you see "ModuleNotFoundError":

1. Ensure you installed the package:
   ```bash
   pip install -e .
   ```

2. Check your Python environment:
   ```bash
   which python
   pip list | grep frustrampnn
   ```

### Memory errors

For large proteins or limited GPU memory:

1. Use CPU instead:
   ```python
   model = FrustraMPNN.from_pretrained("checkpoint.ckpt", device="cpu")
   ```

2. Process chains separately:
   ```python
   results = model.predict("protein.pdb", chains=["A"])
   ```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get running in 5 minutes
- [Python API](api/python-api.md) - Learn the Python interface
- [CLI Reference](api/cli.md) - Command line usage

