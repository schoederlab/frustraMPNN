# Quick Start Guide

Get FrustraMPNN running in 5 minutes.

## Prerequisites

- Python 3.10+
- A PDB file to analyze
- A model checkpoint file

## Step 1: Install FrustraMPNN

```bash
pip install frustrampnn[viz]
```

Or from source:

```bash
git clone https://github.com/schoederlab/frustraMPNN.git
cd frustraMPNN
pip install -e ".[viz]"
```

## Step 2: Download a test structure

Download ubiquitin (1UBQ) from the PDB:

```bash
curl -o 1ubq.pdb "https://files.rcsb.org/download/1UBQ.pdb"
```

Or in Python:

```python
from urllib.request import urlretrieve
urlretrieve("https://files.rcsb.org/download/1UBQ.pdb", "1ubq.pdb")
```

## Step 3: Run prediction

### Using Python

```python
from frustrampnn import FrustraMPNN

# Load model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# Predict frustration for all positions
results = model.predict("1ubq.pdb")

# View results
print(results.head(20))
```

Output:

```
   frustration_pred  position wildtype mutation   pdb chain
0          0.334012         0        M        A  1UBQ     A
1          1.410234         0        M        C  1UBQ     A
2         -0.892341         0        M        D  1UBQ     A
3         -0.123456         0        M        E  1UBQ     A
...
```

### Using the command line

```bash
frustrampnn predict --pdb 1ubq.pdb --checkpoint checkpoint.ckpt --output results.csv
```

## Step 4: Visualize results

### Single-residue plot

Show frustration for all 20 amino acids at a specific position:

```python
from frustrampnn.visualization import plot_single_residue

# Plot position 72 (Leu73 in PDB numbering)
fig = plot_single_residue(results, position=72, chain="A")
fig.savefig("position_73.png", dpi=300)
```

### Interactive plot

```python
from frustrampnn.visualization import plot_single_residue_plotly

fig = plot_single_residue_plotly(results, position=72, chain="A")
fig.show()
```

### Heatmap

Show frustration across all positions:

```python
from frustrampnn.visualization import plot_frustration_heatmap

fig = plot_frustration_heatmap(results, chain="A")
fig.savefig("heatmap.png", dpi=300)
```

## Step 5: Interpret results

### Frustration categories

| Category | Frustration Index | Color | Interpretation |
|----------|-------------------|-------|----------------|
| Highly frustrated | <= -1.0 | Red | Conflicting interactions, often functional |
| Neutral | -1.0 to 0.58 | Gray | Average interactions |
| Minimally frustrated | >= 0.58 | Green | Optimized interactions, stable |

### Output columns

| Column | Description |
|--------|-------------|
| `frustration_pred` | Predicted frustration index |
| `position` | 0-indexed residue position |
| `wildtype` | Original amino acid |
| `mutation` | Substituted amino acid |
| `pdb` | PDB identifier |
| `chain` | Chain identifier |

### Finding highly frustrated positions

```python
# Get native residue frustration (wildtype == mutation)
native = results[results['wildtype'] == results['mutation']]

# Find highly frustrated positions
highly_frustrated = native[native['frustration_pred'] <= -1.0]
print(f"Highly frustrated positions: {highly_frustrated['position'].tolist()}")

# Find minimally frustrated positions
minimally_frustrated = native[native['frustration_pred'] >= 0.58]
print(f"Minimally frustrated positions: {minimally_frustrated['position'].tolist()}")
```

## Complete example

```python
from frustrampnn import FrustraMPNN
from frustrampnn.visualization import (
    plot_single_residue,
    plot_frustration_heatmap,
)
import pandas as pd

# Load model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# Predict
results = model.predict("1ubq.pdb", chains=["A"])

# Save results
results.to_csv("1ubq_frustration.csv", index=False)

# Summary statistics
native = results[results['wildtype'] == results['mutation']]
print(f"Total positions: {len(native)}")
print(f"Mean frustration: {native['frustration_pred'].mean():.3f}")
print(f"Highly frustrated: {(native['frustration_pred'] <= -1.0).sum()}")
print(f"Minimally frustrated: {(native['frustration_pred'] >= 0.58).sum()}")

# Visualize
fig = plot_frustration_heatmap(results, chain="A")
fig.savefig("1ubq_heatmap.png", dpi=300)

# Plot specific position
fig = plot_single_residue(results, position=72, chain="A")
fig.savefig("1ubq_position_73.png", dpi=300)
```

## Next steps

- [Python API Guide](api/python-api.md) - Detailed API documentation
- [Visualization Guide](visualization.md) - Creating publication-quality figures
- [Batch Processing](batch-processing.md) - Processing multiple structures
- [Validation](validation.md) - Comparing with physics-based methods

