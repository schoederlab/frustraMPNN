# Migration Guide

This guide helps you migrate from the old FrustraMPNN scripts to the new package-based interface.

## Quick Comparison

| Feature | Old Way | New Way |
|---------|---------|---------|
| Installation | Clone repo + manual setup | `pip install frustrampnn` |
| CLI | `python inference/custom_inference_refac.py` | `frustrampnn predict` |
| Python API | Direct script imports | `from frustrampnn import FrustraMPNN` |
| Configuration | Separate YAML file | Embedded in checkpoint |

## Command Line Interface

### New Way (recommended)

```bash
frustrampnn predict \
    --pdb protein.pdb \
    --checkpoint checkpoint.ckpt \
    --output results.csv
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--pdb` | `-p` | Input PDB file |
| `--checkpoint` | `-c` | Model checkpoint file |
| `--output` | `-o` | Output CSV file |
| `--chains` | | Comma-separated chain IDs |
| `--device` | | Device selection (cuda/cpu) |
| `--quiet` | `-q` | Suppress progress output |
| `--config` | | Config file for old checkpoints |

### CLI Commands

```bash
# Show help
frustrampnn --help

# Show package info
frustrampnn info

# Predict frustration
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt

# Train a model
frustrampnn train --config config.yaml

# Evaluate a model
frustrampnn evaluate --checkpoint model.ckpt
```

## Python API

### New Way (recommended)

```python
from frustrampnn import FrustraMPNN

# Load model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# Predict
results = model.predict("protein.pdb")

# Results is a pandas DataFrame
print(results.head())
```

### Batch Processing

```python
# Process multiple PDB files
results = model.predict_batch(
    ["protein1.pdb", "protein2.pdb"],
    chains=["A"],
    show_progress=True
)
```

## Output Format

The output CSV format:

```csv
frustration_pred,position,wildtype,mutation,pdb,chain
0.334,0,M,A,1UBQ,A
1.410,0,M,C,1UBQ,A
...
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `frustration_pred` | float | Predicted frustration value |
| `position` | int | 0-indexed residue position |
| `wildtype` | str | Wild-type amino acid (1-letter) |
| `mutation` | str | Mutant amino acid (1-letter) |
| `pdb` | str | PDB identifier |
| `chain` | str | Chain identifier |

**Note:** Positions are 0-indexed internally. To convert to PDB numbering, you may need to add an offset based on the PDB file's residue numbering.

## Checkpoint Compatibility

### New Format (recommended)

New checkpoints embed the configuration:

```python
# Just provide the checkpoint
model = FrustraMPNN.from_pretrained("new_checkpoint.ckpt")
```

### Old Format (still supported)

Old checkpoints require a separate config file:

```python
# Provide both checkpoint and config
model = FrustraMPNN.from_pretrained(
    "old_checkpoint.ckpt",
    config_path="config.yaml"
)
```

Or via CLI:

```bash
frustrampnn predict \
    --pdb protein.pdb \
    --checkpoint old_checkpoint.ckpt \
    --config config.yaml
```

## Docker/Singularity

Container scripts are in the `docker/` directory:

```bash
# Build Docker image
cd docker && bash build_docker.sh

# Run with Docker
bash docker/run_docker.sh predict \
    --pdb protein.pdb --checkpoint checkpoint.ckpt

# Build Singularity image
bash docker/build_singu_from_docker.sh

# Run with Singularity
bash docker/run_singularity.sh predict \
    --pdb protein.pdb --checkpoint checkpoint.ckpt
```

## Common Migration Issues

### Issue: "Module not found" errors

**Solution:** Install the package:

```bash
pip install -e .
# or
pip install frustrampnn
```

### Issue: Old checkpoint doesn't load

**Solution:** Provide the config file:

```python
model = FrustraMPNN.from_pretrained(
    "checkpoint.ckpt",
    config_path="config.yaml"
)
```

### Issue: Position numbering differs

**Note:** Positions are always 0-indexed in the output. PDB residue numbers may differ due to insertion codes or non-standard numbering.

## Getting Help

- **Documentation:** See README.md
- **Issues:** https://github.com/schoederlab/frustraMPNN/issues

## Visualization

```python
from frustrampnn.visualization import (
    plot_single_residue,
    plot_frustration_heatmap,
)

# Single-residue plot
fig = plot_single_residue(results, position=72, chain="A")
fig.savefig("position_73.png")

# Heatmap
fig = plot_frustration_heatmap(results, chain="A")
fig.savefig("heatmap.png")
```

## Validation Against frustrapy

```python
from frustrampnn.validation import compare_with_frustrapy

comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
    positions=[50, 73, 120],
)

print(f"Spearman: {comparison.spearman:.3f}")
print(f"RMSE: {comparison.rmse:.3f}")
```
