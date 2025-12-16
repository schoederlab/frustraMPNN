# Tutorial: Basic Frustration Analysis

This tutorial walks through a complete frustration analysis of ubiquitin (PDB: 1UBQ).

## Prerequisites

- FrustraMPNN installed (`pip install frustrampnn[viz]`)
- A model checkpoint file
- Internet connection (to download PDB)

## Step 1: Download the structure

```python
from urllib.request import urlretrieve

# Download ubiquitin from RCSB PDB
urlretrieve("https://files.rcsb.org/download/1UBQ.pdb", "1ubq.pdb")
print("Downloaded 1ubq.pdb")
```

## Step 2: Load the model

```python
from frustrampnn import FrustraMPNN

# Load pretrained model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
print(f"Model loaded on device: {model.device}")
```

## Step 3: Predict frustration

```python
# Predict frustration for all positions in chain A
results = model.predict("1ubq.pdb", chains=["A"], show_progress=True)

# View first few rows
print(results.head(10))
```

Output:

```
   frustration_pred  position wildtype mutation   pdb chain
0          0.334012         0        M        A  1UBQ     A
1          1.410234         0        M        C  1UBQ     A
2         -0.892341         0        M        D  1UBQ     A
3         -0.123456         0        M        E  1UBQ     A
4          0.567890         0        M        F  1UBQ     A
...
```

## Step 4: Understand the results

Each row represents a prediction for one amino acid at one position:

- `frustration_pred`: The predicted frustration index
- `position`: 0-indexed residue position
- `wildtype`: The original amino acid
- `mutation`: The amino acid being evaluated
- `pdb`: PDB identifier
- `chain`: Chain identifier

### Filter native residues

The native (wild-type) residue is where `wildtype == mutation`:

```python
native = results[results['wildtype'] == results['mutation']]
print(f"Number of positions: {len(native)}")
print(native.head())
```

### Categorize frustration

```python
def categorize(frustration):
    if frustration <= -1.0:
        return 'highly'
    elif frustration >= 0.58:
        return 'minimally'
    return 'neutral'

native['category'] = native['frustration_pred'].apply(categorize)
print(native['category'].value_counts())
```

Output:

```
neutral       45
minimally     22
highly         9
Name: category, dtype: int64
```

## Step 5: Summary statistics

```python
print("Frustration Summary for Ubiquitin:")
print(f"  Total residues: {len(native)}")
print(f"  Mean frustration: {native['frustration_pred'].mean():.3f}")
print(f"  Std frustration: {native['frustration_pred'].std():.3f}")
print(f"  Min frustration: {native['frustration_pred'].min():.3f}")
print(f"  Max frustration: {native['frustration_pred'].max():.3f}")
print(f"  Highly frustrated: {(native['frustration_pred'] <= -1.0).sum()}")
print(f"  Minimally frustrated: {(native['frustration_pred'] >= 0.58).sum()}")
```

## Step 6: Find interesting positions

### Most frustrated positions

```python
most_frustrated = native.nsmallest(5, 'frustration_pred')
print("\nMost frustrated positions:")
for _, row in most_frustrated.iterrows():
    print(f"  Position {row['position']} ({row['wildtype']}): {row['frustration_pred']:.3f}")
```

### Most stable positions

```python
most_stable = native.nlargest(5, 'frustration_pred')
print("\nMost stable (minimally frustrated) positions:")
for _, row in most_stable.iterrows():
    print(f"  Position {row['position']} ({row['wildtype']}): {row['frustration_pred']:.3f}")
```

## Step 7: Visualize results

### Heatmap

```python
from frustrampnn.visualization import plot_frustration_heatmap

fig = plot_frustration_heatmap(results, chain="A")
fig.savefig("1ubq_heatmap.png", dpi=300, bbox_inches='tight')
print("Saved 1ubq_heatmap.png")
```

### Single-residue plot

Plot frustration for all 20 amino acids at a specific position:

```python
from frustrampnn.visualization import plot_single_residue

# Plot the most frustrated position
most_frustrated_pos = native.nsmallest(1, 'frustration_pred')['position'].iloc[0]

fig = plot_single_residue(
    results,
    position=most_frustrated_pos,
    chain="A",
    title=f"Position {most_frustrated_pos + 1}"
)
fig.savefig(f"1ubq_position_{most_frustrated_pos}.png", dpi=300, bbox_inches='tight')
print(f"Saved 1ubq_position_{most_frustrated_pos}.png")
```

### Interactive plots

```python
from frustrampnn.visualization import plot_frustration_heatmap_plotly

fig = plot_frustration_heatmap_plotly(results, chain="A")
fig.write_html("1ubq_heatmap.html")
print("Saved 1ubq_heatmap.html (interactive)")
```

## Step 8: Save results

```python
# Save full results
results.to_csv("1ubq_frustration.csv", index=False)

# Save native residue summary
native.to_csv("1ubq_native_frustration.csv", index=False)

print("Results saved!")
```

## Complete script

```python
"""Complete frustration analysis of ubiquitin."""

from urllib.request import urlretrieve
from frustrampnn import FrustraMPNN
from frustrampnn.visualization import (
    plot_frustration_heatmap,
    plot_single_residue,
)

# Download structure
urlretrieve("https://files.rcsb.org/download/1UBQ.pdb", "1ubq.pdb")

# Load model and predict
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
results = model.predict("1ubq.pdb", chains=["A"], show_progress=True)

# Get native residues
native = results[results['wildtype'] == results['mutation']]

# Summary
print(f"Ubiquitin Frustration Analysis")
print(f"=" * 40)
print(f"Residues: {len(native)}")
print(f"Mean frustration: {native['frustration_pred'].mean():.3f}")
print(f"Highly frustrated: {(native['frustration_pred'] <= -1.0).sum()}")
print(f"Minimally frustrated: {(native['frustration_pred'] >= 0.58).sum()}")

# Most frustrated
print(f"\nMost frustrated positions:")
for _, row in native.nsmallest(5, 'frustration_pred').iterrows():
    print(f"  {row['position']+1} ({row['wildtype']}): {row['frustration_pred']:.3f}")

# Visualize
fig = plot_frustration_heatmap(results, chain="A")
fig.savefig("1ubq_heatmap.png", dpi=300, bbox_inches='tight')

# Save
results.to_csv("1ubq_frustration.csv", index=False)
print(f"\nResults saved to 1ubq_frustration.csv")
```

## Next steps

- [Visualization Tutorial](visualization-tutorial.md) - Create publication-quality figures
- [Enzyme Analysis](enzyme-analysis.md) - Analyze active sites
- [Validation Tutorial](validation-tutorial.md) - Compare with physics-based methods

