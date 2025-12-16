# Validation Guide

This guide covers validating FrustraMPNN predictions against physics-based methods.

## Overview

FrustraMPNN predictions can be validated against FrustratometeR, the physics-based frustration calculator. The validation module provides tools for:

- Running frustrapy calculations
- Comparing predictions with calculations
- Computing correlation metrics
- Visualizing agreement

## Installation

Install validation dependencies:

```bash
pip install frustrampnn[validation]
```

For full frustrapy support:

```bash
pip install frustrapy
```

## Quick Comparison

```python
from frustrampnn import FrustraMPNN
from frustrampnn.validation import compare_with_frustrapy

# Load model and predict
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
results = model.predict("protein.pdb", chains=["A"])

# Compare with frustrapy (slow - calculates physics-based frustration)
comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
)

# View metrics
print(f"Spearman correlation: {comparison.spearman:.3f}")
print(f"Pearson correlation: {comparison.pearson:.3f}")
print(f"RMSE: {comparison.rmse:.3f}")
```

## Comparison Options

### Specific positions

FrustratometeR is computationally expensive. For large proteins, compare only specific positions:

```python
comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
    positions=[50, 73, 120, 200],  # Only these positions
)
```

### All positions (slow)

For comprehensive validation:

```python
comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
    positions=None,  # All positions (default)
)
```

### Parallel computation

Speed up frustrapy calculations:

```python
comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
    n_jobs=4,  # Use 4 CPU cores
)
```

## Comparison Results

The `compare_with_frustrapy()` function returns a `ComparisonResult` object:

### Metrics

```python
# Correlation metrics
comparison.spearman      # Spearman correlation coefficient
comparison.pearson       # Pearson correlation coefficient
comparison.rmse          # Root mean squared error
comparison.mae           # Mean absolute error

# Category metrics
comparison.accuracy      # Categorical accuracy
comparison.f1_score      # F1 score (macro average)
comparison.confusion_matrix  # Confusion matrix
```

### Detailed results

```python
# DataFrame with both predictions
df = comparison.results

# Columns:
# - position: Residue position
# - wildtype: Wild-type amino acid
# - mutation: Mutant amino acid
# - frustrampnn: FrustraMPNN prediction
# - frustrapy: FrustratometeR calculation
# - category_pred: Predicted category
# - category_calc: Calculated category
```

### Visualization

```python
# Scatter plot
fig = comparison.plot()
fig.show()

# With options
fig = comparison.plot(
    show_regression=True,   # Show regression line
    show_identity=True,     # Show y=x line
    show_categories=True,   # Color by category
    title="FrustraMPNN vs FrustratometeR",
)
fig.savefig("comparison.png", dpi=300)
```

## Validation Metrics

### Spearman Correlation

Measures rank correlation between predictions and calculations. Values range from -1 to 1, with 1 indicating perfect agreement.

```python
from scipy.stats import spearmanr

spearman, pvalue = spearmanr(
    comparison.results['frustrampnn'],
    comparison.results['frustrapy']
)
print(f"Spearman: {spearman:.3f} (p={pvalue:.2e})")
```

### RMSE

Root mean squared error between predictions and calculations.

```python
import numpy as np

rmse = np.sqrt(np.mean(
    (comparison.results['frustrampnn'] - comparison.results['frustrapy'])**2
))
print(f"RMSE: {rmse:.3f}")
```

### Categorical Accuracy

Accuracy of frustration category predictions.

```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(
    comparison.results['category_calc'],
    comparison.results['category_pred']
)
print(f"Accuracy: {accuracy:.3f}")

# Detailed report
print(classification_report(
    comparison.results['category_calc'],
    comparison.results['category_pred'],
    target_names=['highly', 'neutral', 'minimally']
))
```

## Running frustrapy Directly

For more control, run frustrapy calculations directly:

```python
from frustrampnn.validation import run_frustrapy

# Calculate frustration for specific positions
frustrapy_results = run_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    positions=[50, 73, 120],
    mode="singleresidue",  # Single-residue mode
)

# Returns DataFrame with columns:
# - position, wildtype, mutation, frustration_index
```

### frustrapy modes

| Mode | Description | Speed |
|------|-------------|-------|
| `singleresidue` | All 20 amino acids at each position | Slowest |
| `configurational` | Native contacts only | Fast |
| `mutational` | Specific mutations | Medium |

## Validation Workflow

Complete validation workflow:

```python
from frustrampnn import FrustraMPNN
from frustrampnn.validation import compare_with_frustrapy
import pandas as pd

# Load model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# Predict
results = model.predict("protein.pdb", chains=["A"])

# Get native residue predictions
native = results[results['wildtype'] == results['mutation']]

# Find highly frustrated positions (most interesting to validate)
highly_frustrated = native[native['frustration_pred'] <= -1.0]
positions_to_validate = highly_frustrated['position'].tolist()[:10]  # Top 10

print(f"Validating {len(positions_to_validate)} positions...")

# Compare
comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
    positions=positions_to_validate,
)

# Report
print(f"\nValidation Results:")
print(f"  Spearman: {comparison.spearman:.3f}")
print(f"  RMSE: {comparison.rmse:.3f}")
print(f"  Accuracy: {comparison.accuracy:.3f}")

# Save detailed results
comparison.results.to_csv("validation_results.csv", index=False)

# Plot
fig = comparison.plot(show_regression=True)
fig.savefig("validation_plot.png", dpi=300)
```

## Interpreting Results

### Good agreement

- Spearman > 0.7: Strong correlation
- RMSE < 1.0: Predictions within typical range
- Accuracy > 0.7: Good categorical agreement

### Common discrepancies

1. **Loop regions**: Higher uncertainty in flexible regions
2. **Surface residues**: More variable due to solvent effects
3. **Active sites**: May show systematic differences

### When to trust predictions

FrustraMPNN predictions are most reliable for:
- Core residues
- Helices and sheets
- Conserved positions

Use caution for:
- Highly flexible loops
- Disordered regions
- Unusual amino acids

## Batch Validation

Validate multiple structures:

```python
import os
from pathlib import Path

# Collect results
all_comparisons = []

for pdb_file in Path("structures/").glob("*.pdb"):
    print(f"Processing {pdb_file.name}...")
    
    # Predict
    results = model.predict(str(pdb_file), chains=["A"])
    
    # Compare (sample positions for speed)
    native = results[results['wildtype'] == results['mutation']]
    sample_positions = native.sample(min(20, len(native)))['position'].tolist()
    
    comparison = compare_with_frustrapy(
        pdb_path=str(pdb_file),
        chain="A",
        frustrampnn_results=results,
        positions=sample_positions,
    )
    
    all_comparisons.append({
        'pdb': pdb_file.stem,
        'spearman': comparison.spearman,
        'rmse': comparison.rmse,
        'accuracy': comparison.accuracy,
    })

# Summary
summary = pd.DataFrame(all_comparisons)
print(f"\nOverall Spearman: {summary['spearman'].mean():.3f} +/- {summary['spearman'].std():.3f}")
print(f"Overall RMSE: {summary['rmse'].mean():.3f} +/- {summary['rmse'].std():.3f}")
```

## Computational Considerations

### Time estimates

| Protein Size | FrustraMPNN | FrustratometeR (1 CPU) | Speedup |
|--------------|-------------|------------------------|---------|
| 100 residues | ~20 ms | ~3.4 min | 1,200x |
| 300 residues | ~30 s | ~19 h | 2,300x |
| 500 residues | ~30 s | ~35 h | 4,500x |

### Memory requirements

FrustratometeR requires significant memory for large proteins. Monitor usage:

```python
import psutil

process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB")
```

### Parallelization

Use multiple cores for frustrapy:

```python
comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
    n_jobs=-1,  # Use all available cores
)
```

## See Also

- [Python API](api/python-api.md) - API documentation
- [Visualization](visualization.md) - Creating plots
- [Output Format](output-format.md) - Understanding results

