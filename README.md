# FrustraMPNN

Ultra-fast deep learning prediction of single-residue local energetic frustration in proteins.

FrustraMPNN is a message-passing neural network trained via transfer learning from ProteinMPNN to predict frustration profiles 1,000-4,500x faster than physics-based methods.

## Table of Contents

- [Installation](#installation)
  - [Quick Install (pip)](#quick-install-pip)
  - [Development Install (uv)](#development-install-uv)
  - [Using Notebooks with uv](#using-notebooks-with-uv)
  - [Docker/Singularity](#dockersingularity)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Python API](#python-api)
  - [Command Line Interface](#command-line-interface)
  - [Google Colab](#google-colab)
- [Visualization](#visualization)
- [Validation](#validation)
- [Citation](#citation)
- [License](#license)

## Installation

### Quick Install (pip)

```bash
pip install frustrampnn
```

With visualization support:

```bash
pip install frustrampnn[viz]
```

With all optional dependencies:

```bash
pip install frustrampnn[all]
```

### Development Install (uv)

We recommend using [uv](https://docs.astral.sh/uv/) for development. It is 10-100x faster than pip.

1. Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```bash
git clone https://github.com/schoederlab/frustraMPNN.git
cd frustraMPNN
```

3. Create virtual environment and install dependencies:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[all]"
```

4. Verify installation:

```bash
python -c "import frustrampnn; print(frustrampnn.__version__)"
```

### Using Notebooks with uv

When using uv with Jupyter notebooks, you need to ensure the notebook uses the correct Python environment. Follow these steps:

#### Step 1: Install ipykernel in the virtual environment

```bash
cd /path/to/frustraMPNN
source .venv/bin/activate
uv pip install ipykernel
```

#### Step 2: Register the kernel

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=frustrampnn --display-name="FrustraMPNN"
```

This creates a kernel named "FrustraMPNN" that will appear in Jupyter and VS Code.

#### Step 3: Select the kernel in your notebook

**In VS Code / Cursor:**
1. Open the notebook file (`.ipynb`)
2. Click the kernel selector in the top-right corner
3. Select "FrustraMPNN" or ".venv (Python 3.10.x)"

**In Jupyter Notebook / JupyterLab:**
1. Open the notebook
2. Go to Kernel > Change Kernel
3. Select "FrustraMPNN"

**From command line:**
```bash
source .venv/bin/activate
jupyter notebook notebooks/FrustraMPNN_Local.ipynb
```

#### Troubleshooting

**"No module named X" errors:**

If you see import errors, the notebook is likely using the wrong Python environment. Verify by running in a notebook cell:

```python
import sys
print(sys.executable)
```

This should print a path containing `.venv`, for example:
```
/path/to/frustraMPNN/.venv/bin/python
```

If it shows a different path (e.g., `/usr/bin/python` or a conda environment), select the correct kernel as described above.

**Kernel not appearing:**

If the "FrustraMPNN" kernel does not appear in the kernel list:

1. Ensure ipykernel is installed in the venv:
   ```bash
   source .venv/bin/activate
   uv pip install ipykernel
   ```

2. Re-register the kernel:
   ```bash
   python -m ipykernel install --user --name=frustrampnn --display-name="FrustraMPNN"
   ```

3. Restart VS Code / Jupyter

**List installed kernels:**

```bash
jupyter kernelspec list
```

**Remove a kernel:**

```bash
jupyter kernelspec remove frustrampnn
```

### Docker/Singularity

For reproducible environments, use Docker or Singularity:

```bash
# Build Docker image
cd docker && bash build_docker.sh

# Run with Docker
bash docker/run_docker.sh predict --pdb protein.pdb --checkpoint model.ckpt

# Build Singularity image (for HPC)
bash docker/build_singu_from_docker.sh

# Run with Singularity
singularity exec --nv frustrampnn.sif frustrampnn predict --pdb protein.pdb
```

## Quick Start

```python
from frustrampnn import FrustraMPNN

# Load model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# Predict frustration
results = model.predict("protein.pdb", chains=["A"])

# View results
print(results.head())
```

## Usage

### Python API

```python
from frustrampnn import FrustraMPNN

# Load pretrained model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# Predict frustration for a single structure
results = model.predict(
    "protein.pdb",
    chains=["A"],           # Specific chains (None = all)
    positions=[10, 20, 30], # Specific positions (None = all)
    show_progress=True
)

# Batch prediction for multiple structures
results = model.predict_batch(
    ["protein1.pdb", "protein2.pdb"],
    chains=["A"],
    show_progress=True
)

# Results DataFrame columns:
# - frustration_pred: Predicted frustration value
# - position: 0-indexed position
# - wildtype: Wild-type amino acid
# - mutation: Mutant amino acid
# - pdb: PDB identifier
# - chain: Chain identifier
```

### Command Line Interface

```bash
# Basic prediction
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt --output results.csv

# Specify chains
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt --chains A B

# Batch processing
frustrampnn predict --pdb-dir ./structures/ --checkpoint model.ckpt --output batch_results.csv

# Show help
frustrampnn --help
frustrampnn predict --help
```

### Google Colab

For zero-installation access, use our Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schoederlab/frustraMPNN/blob/main/notebooks/FrustraMPNN_Colab.ipynb)

The notebook provides:
- One-click installation
- PDB upload or RCSB fetch
- Interactive visualizations
- Results download

## Visualization

```python
from frustrampnn.visualization import (
    plot_single_residue,
    plot_single_residue_plotly,
    plot_frustration_heatmap,
    plot_frustration_heatmap_plotly,
)

# Single-residue plot (matplotlib)
fig = plot_single_residue(results, position=72, chain="A")
fig.savefig("position_73.png", dpi=300)

# Single-residue plot (plotly, interactive)
fig = plot_single_residue_plotly(results, position=72, chain="A")
fig.show()

# Heatmap (matplotlib)
fig = plot_frustration_heatmap(results, chain="A")
fig.savefig("heatmap.png", dpi=300)

# Heatmap (plotly, interactive)
fig = plot_frustration_heatmap_plotly(results, chain="A")
fig.show()
```

Color scheme:
- Red: Highly frustrated (frustration index <= -1.0)
- Gray: Neutral (-1.0 < frustration index < 0.58)
- Green: Minimally frustrated (frustration index >= 0.58)
- Blue: Native (wild-type) residue

## Validation

Compare FrustraMPNN predictions with physics-based frustrapy:

```python
from frustrampnn.validation import compare_with_frustrapy

# Compare at specific positions (frustrapy is slow)
comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
    positions=[50, 73, 120],
)

print(f"Spearman: {comparison.spearman:.3f}")
print(f"RMSE: {comparison.rmse:.3f}")

# Plot comparison
fig = comparison.plot()
fig.show()
```

## Performance

| Protein Size | FrustraMPNN (GPU) | FrustratometeR (CPU) | Speedup |
|--------------|-------------------|----------------------|---------|
| 100 residues | ~20 ms            | ~3.4 min             | 1,200x  |
| 300 residues | ~30 s             | ~19 h                | 2,300x  |
| 500 residues | ~30 s             | ~35 h                | 4,500x  |

## Citation

If you use FrustraMPNN in your research, please cite:

```bibtex
@article{beining2026frustrampnn,
  title={FrustraMPNN: Ultra-fast deep learning prediction of single-residue local energetic frustration},
  author={Beining, Max and Engelberger, Felipe and Schoeder, Clara T. and Ram{\'\i}rez-Sarmiento, C{\'e}sar A. and Meiler, Jens},
  journal={XXXXXXXXXX},
  year={2026},
  doi={10.1101/2026.XX.XX.XXXXXX}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [frustrapy](https://github.com/engelberger/frustrapy) - Python wrapper for FrustratometeR
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Base architecture
- [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) - Transfer learning approach
