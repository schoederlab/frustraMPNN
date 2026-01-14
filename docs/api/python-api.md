# Python API Guide

This guide covers the FrustraMPNN Python API in detail.

## Core Classes

### FrustraMPNN

The main class for frustration prediction.

```python
from frustrampnn import FrustraMPNN
```

#### Loading a model

```python
# From checkpoint file
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# With explicit device
model = FrustraMPNN.from_pretrained("checkpoint.ckpt", device="cuda")
model = FrustraMPNN.from_pretrained("checkpoint.ckpt", device="cpu")

# Old checkpoint format (requires config)
model = FrustraMPNN.from_pretrained(
    "old_checkpoint.ckpt",
    config_path="config.yaml"
)
```

#### Making predictions

```python
# Predict all positions in all chains
results = model.predict("protein.pdb")

# Predict specific chains
results = model.predict("protein.pdb", chains=["A", "B"])

# Predict specific positions (0-indexed)
results = model.predict("protein.pdb", positions=[10, 20, 30])

# With progress bar
results = model.predict("protein.pdb", show_progress=True)
```

#### Batch prediction

```python
# Multiple PDB files
results = model.predict_batch(
    ["protein1.pdb", "protein2.pdb", "protein3.pdb"],
    chains=["A"],
    show_progress=True
)
```

### Mutation

Data class representing a single mutation.

```python
from frustrampnn.data import Mutation

mutation = Mutation(
    position=72,      # 0-indexed position
    wildtype="L",     # Wild-type amino acid
    mutation="A",     # Mutant amino acid
    pdb="1UBQ"        # PDB identifier
)
```

## Results DataFrame

The `predict()` method returns a pandas DataFrame with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `frustration_pred` | float | Predicted frustration index |
| `position` | int | 0-indexed residue position |
| `wildtype` | str | Wild-type amino acid (1-letter code) |
| `mutation` | str | Mutant amino acid (1-letter code) |
| `pdb` | str | PDB identifier |
| `chain` | str | Chain identifier |

### Working with results

```python
import pandas as pd

# Load results
results = model.predict("protein.pdb")

# Filter by chain
chain_a = results[results['chain'] == 'A']

# Get native residue frustration
native = results[results['wildtype'] == results['mutation']]

# Get mutations only (exclude native)
mutations = results[results['wildtype'] != results['mutation']]

# Filter by frustration category
highly_frustrated = results[results['frustration_pred'] <= -1.0]
neutral = results[(results['frustration_pred'] > -1.0) & 
                  (results['frustration_pred'] < 0.58)]
minimally_frustrated = results[results['frustration_pred'] >= 0.58]

# Get frustration for specific position
pos_72 = results[results['position'] == 72]

# Pivot to matrix form (positions x mutations)
matrix = results.pivot_table(
    index='position',
    columns='mutation',
    values='frustration_pred'
)
```

### Aggregating results

```python
# Mean frustration per position
mean_per_position = results.groupby('position')['frustration_pred'].mean()

# Count frustration categories per position
def categorize(x):
    if x <= -1.0:
        return 'highly'
    elif x >= 0.58:
        return 'minimally'
    return 'neutral'

results['category'] = results['frustration_pred'].apply(categorize)
category_counts = results.groupby(['position', 'category']).size().unstack(fill_value=0)
```

## Visualization Module

```python
from frustrampnn.visualization import (
    plot_single_residue,
    plot_single_residue_plotly,
    plot_frustration_heatmap,
    plot_frustration_heatmap_plotly,
)
```

### Single-residue plots

Show frustration for all 20 amino acids at a specific position:

```python
# Matplotlib (static)
fig = plot_single_residue(
    results,
    position=72,           # 0-indexed position
    chain="A",             # Chain identifier
    title="Position 73",   # Optional title
    figsize=(10, 6),       # Figure size
)
fig.savefig("position_73.png", dpi=300)

# Plotly (interactive)
fig = plot_single_residue_plotly(
    results,
    position=72,
    chain="A",
    title="Position 73",
)
fig.show()
fig.write_html("position_73.html")
```

### Heatmaps

Show frustration across all positions and mutations:

```python
# Matplotlib (static)
fig = plot_frustration_heatmap(
    results,
    chain="A",
    figsize=(20, 10),
    cmap="RdYlGn_r",       # Color map
)
fig.savefig("heatmap.png", dpi=300)

# Plotly (interactive)
fig = plot_frustration_heatmap_plotly(
    results,
    chain="A",
    title="Frustration Heatmap",
)
fig.show()
fig.write_html("heatmap.html")
```

## Validation Module

Compare FrustraMPNN predictions with physics-based frustrapy:

```python
from frustrampnn.validation import compare_with_frustrapy

# Compare at specific positions (frustrapy is slow)
comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
    positions=[50, 73, 120],  # Optional: specific positions
)

# Access metrics
print(f"Spearman correlation: {comparison.spearman:.3f}")
print(f"RMSE: {comparison.rmse:.3f}")
print(f"Pearson correlation: {comparison.pearson:.3f}")

# Get detailed results
print(comparison.results)  # DataFrame with both predictions

# Plot comparison
fig = comparison.plot()
fig.show()
```

## Constants

```python
from frustrampnn.constants import (
    ALPHABET,              # 'ACDEFGHIKLMNPQRSTVWYX'
    AMINO_ACIDS,           # List of 20 standard amino acids
    FRUSTRATION_THRESHOLDS,  # {'highly': -1.0, 'minimally': 0.58}
)
```

## Advanced Usage

### Custom device handling

```python
import torch

# Check available devices
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model on specific device
model = FrustraMPNN.from_pretrained("checkpoint.ckpt", device=device)

# Move model to different device
model.to("cpu")
```

### Memory management

```python
import torch

# Clear GPU memory
torch.cuda.empty_cache()

# Process large proteins in chunks
def predict_chunked(model, pdb_path, chunk_size=100):
    """Predict in chunks to manage memory."""
    import pandas as pd
    
    # Get total positions
    results = model.predict(pdb_path, positions=[0])
    total_positions = len(results['position'].unique())
    
    all_results = []
    for start in range(0, total_positions, chunk_size):
        end = min(start + chunk_size, total_positions)
        positions = list(range(start, end))
        chunk_results = model.predict(pdb_path, positions=positions)
        all_results.append(chunk_results)
        torch.cuda.empty_cache()
    
    return pd.concat(all_results, ignore_index=True)
```

### Accessing internal model

```python
# Access the underlying PyTorch model
pytorch_model = model.model

# Access configuration
config = model.cfg

# Get model device
device = model.device
```

## Error Handling

```python
from frustrampnn import FrustraMPNN

try:
    model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
    results = model.predict("protein.pdb")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"Model error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Complete Example

```python
"""Complete FrustraMPNN analysis workflow."""

from frustrampnn import FrustraMPNN
from frustrampnn.visualization import (
    plot_single_residue,
    plot_frustration_heatmap,
)
from frustrampnn.validation import compare_with_frustrapy
import pandas as pd

# Configuration
PDB_FILE = "protein.pdb"
CHECKPOINT = "checkpoint.ckpt"
OUTPUT_DIR = "results"

# Load model
print("Loading model...")
model = FrustraMPNN.from_pretrained(CHECKPOINT)

# Predict frustration
print("Predicting frustration...")
results = model.predict(PDB_FILE, chains=["A"], show_progress=True)

# Save results
results.to_csv(f"{OUTPUT_DIR}/frustration.csv", index=False)

# Summary statistics
native = results[results['wildtype'] == results['mutation']]
print(f"\nSummary for chain A:")
print(f"  Total positions: {len(native)}")
print(f"  Mean frustration: {native['frustration_pred'].mean():.3f}")
print(f"  Std frustration: {native['frustration_pred'].std():.3f}")
print(f"  Highly frustrated: {(native['frustration_pred'] <= -1.0).sum()}")
print(f"  Minimally frustrated: {(native['frustration_pred'] >= 0.58).sum()}")

# Find most frustrated positions
most_frustrated = native.nsmallest(5, 'frustration_pred')
print(f"\nMost frustrated positions:")
for _, row in most_frustrated.iterrows():
    print(f"  Position {row['position']}: {row['wildtype']} "
          f"(frustration: {row['frustration_pred']:.3f})")

# Generate visualizations
print("\nGenerating visualizations...")

# Heatmap
fig = plot_frustration_heatmap(results, chain="A")
fig.savefig(f"{OUTPUT_DIR}/heatmap.png", dpi=300, bbox_inches='tight')

# Single-residue plots for top frustrated positions
for _, row in most_frustrated.iterrows():
    pos = row['position']
    fig = plot_single_residue(results, position=pos, chain="A")
    fig.savefig(f"{OUTPUT_DIR}/position_{pos}.png", dpi=300, bbox_inches='tight')

# Validate against frustrapy (optional, slow)
print("\nValidating against frustrapy...")
comparison = compare_with_frustrapy(
    pdb_path=PDB_FILE,
    chain="A",
    frustrampnn_results=results,
    positions=most_frustrated['position'].tolist()[:3],  # Top 3 only
)
print(f"  Spearman: {comparison.spearman:.3f}")
print(f"  RMSE: {comparison.rmse:.3f}")

print(f"\nResults saved to {OUTPUT_DIR}/")
```

