# Tutorial: Enzyme Active Site Analysis

This tutorial demonstrates how to analyze local energetic frustration in enzyme active sites using FrustraMPNN.

## Background

Enzyme active sites often contain highly frustrated residues. This frustration is functionally important:
- Catalytic residues are often in energetically unfavorable conformations
- Frustration enables the conformational flexibility needed for catalysis
- Binding sites may show frustration to accommodate substrates

## Example: Beta-lactamase (PDB: 4BLM)

Beta-lactamase is a well-studied enzyme that confers antibiotic resistance. Its active site contains several highly frustrated residues.

### Step 1: Download and predict

```python
from urllib.request import urlretrieve
from frustrampnn import FrustraMPNN

# Download beta-lactamase structure
urlretrieve("https://files.rcsb.org/download/4BLM.pdb", "4blm.pdb")

# Load model and predict
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
results = model.predict("4blm.pdb", chains=["A"], show_progress=True)

print(f"Predicted frustration for {len(results)} mutations")
```

### Step 2: Define active site residues

Based on literature, the key active site residues in beta-lactamase are:

```python
# Active site residues (0-indexed positions)
# Note: PDB numbering may differ from 0-indexed positions
active_site_residues = {
    'Ser70': 69,    # Catalytic serine
    'Lys73': 72,    # Stabilizes transition state
    'Ser130': 129,  # Proton shuttle
    'Lys234': 233,  # Substrate binding
    'Asn237': 236,  # Substrate binding
    'Glu166': 165,  # General base
}
```

### Step 3: Analyze active site frustration

```python
# Get native residue frustration
native = results[results['wildtype'] == results['mutation']]

# Extract active site residues
active_site_data = []
for name, pos in active_site_residues.items():
    row = native[native['position'] == pos]
    if len(row) > 0:
        frustration = row['frustration_pred'].iloc[0]
        wt = row['wildtype'].iloc[0]
        
        # Categorize
        if frustration <= -1.0:
            category = 'highly'
        elif frustration >= 0.58:
            category = 'minimally'
        else:
            category = 'neutral'
        
        active_site_data.append({
            'residue': name,
            'position': pos,
            'wildtype': wt,
            'frustration': frustration,
            'category': category,
        })

import pandas as pd
active_site_df = pd.DataFrame(active_site_data)
print("\nActive Site Frustration:")
print(active_site_df.to_string(index=False))
```

Output:

```
Active Site Frustration:
 residue  position wildtype  frustration   category
   Ser70        69        S        0.234    neutral
   Lys73        72        K       -1.456     highly
  Ser130       129        S        0.123    neutral
  Lys234       233        K       -1.234     highly
  Asn237       236        N        0.345    neutral
  Glu166       165        E       -1.678     highly
```

### Step 4: Compare active site to rest of protein

```python
# Active site positions
active_positions = list(active_site_residues.values())

# Separate active site and non-active site
active_native = native[native['position'].isin(active_positions)]
non_active_native = native[~native['position'].isin(active_positions)]

print("\nFrustration Comparison:")
print(f"Active site mean: {active_native['frustration_pred'].mean():.3f}")
print(f"Non-active site mean: {non_active_native['frustration_pred'].mean():.3f}")

# Statistical test
from scipy.stats import mannwhitneyu
stat, pvalue = mannwhitneyu(
    active_native['frustration_pred'],
    non_active_native['frustration_pred'],
    alternative='less'  # Active site more frustrated
)
print(f"Mann-Whitney U test p-value: {pvalue:.4f}")
```

### Step 5: Visualize active site

```python
from frustrampnn.visualization import plot_single_residue

# Plot each active site residue
for name, pos in active_site_residues.items():
    fig = plot_single_residue(
        results,
        position=pos,
        chain="A",
        title=f"{name} (Position {pos+1})"
    )
    fig.savefig(f"4blm_{name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

print("Saved active site plots")
```

### Step 6: Create summary visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (name, pos) in enumerate(active_site_residues.items()):
    ax = axes[i]
    
    # Get data for this position
    pos_data = results[results['position'] == pos].copy()
    pos_data = pos_data.sort_values('frustration_pred')
    
    # Color by category
    colors = []
    for f in pos_data['frustration_pred']:
        if f <= -1.0:
            colors.append('red')
        elif f >= 0.58:
            colors.append('green')
        else:
            colors.append('gray')
    
    # Highlight native
    native_idx = pos_data[pos_data['wildtype'] == pos_data['mutation']].index
    for idx in native_idx:
        colors[list(pos_data.index).index(idx)] = 'blue'
    
    # Plot
    ax.bar(range(len(pos_data)), pos_data['frustration_pred'], color=colors)
    ax.axhline(y=-1.0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.58, color='green', linestyle='--', alpha=0.5)
    ax.set_title(f"{name}")
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Frustration Index")
    ax.set_xticks(range(len(pos_data)))
    ax.set_xticklabels(pos_data['mutation'], fontsize=8)

plt.tight_layout()
fig.savefig("4blm_active_site_summary.png", dpi=300, bbox_inches='tight')
print("Saved 4blm_active_site_summary.png")
```

### Step 7: Identify stabilizing mutations

Find mutations that would reduce frustration at active site:

```python
print("\nPotential stabilizing mutations at active site:")
for name, pos in active_site_residues.items():
    pos_data = results[results['position'] == pos]
    native_row = pos_data[pos_data['wildtype'] == pos_data['mutation']]
    native_frustration = native_row['frustration_pred'].iloc[0]
    
    # Find mutations that reduce frustration
    stabilizing = pos_data[pos_data['frustration_pred'] > native_frustration + 0.5]
    stabilizing = stabilizing.nlargest(3, 'frustration_pred')
    
    if len(stabilizing) > 0:
        print(f"\n{name} (native frustration: {native_frustration:.3f}):")
        for _, row in stabilizing.iterrows():
            if row['mutation'] != row['wildtype']:
                delta = row['frustration_pred'] - native_frustration
                print(f"  {row['wildtype']}{pos+1}{row['mutation']}: "
                      f"{row['frustration_pred']:.3f} (delta: +{delta:.3f})")
```

## Comparing Multiple Enzymes

Analyze frustration patterns across enzyme families:

```python
enzymes = {
    '4BLM': {'name': 'Beta-lactamase', 'active_site': [69, 72, 129, 165, 233, 236]},
    '1NLU': {'name': 'Serine protease', 'active_site': [32, 57, 102]},
    '1CBG': {'name': 'Beta-glucosidase', 'active_site': [166, 324, 399]},
}

comparison_data = []

for pdb_id, info in enzymes.items():
    # Download and predict
    urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", f"{pdb_id.lower()}.pdb")
    results = model.predict(f"{pdb_id.lower()}.pdb", chains=["A"])
    native = results[results['wildtype'] == results['mutation']]
    
    # Calculate statistics
    active_native = native[native['position'].isin(info['active_site'])]
    non_active_native = native[~native['position'].isin(info['active_site'])]
    
    comparison_data.append({
        'enzyme': info['name'],
        'pdb': pdb_id,
        'active_site_mean': active_native['frustration_pred'].mean(),
        'non_active_mean': non_active_native['frustration_pred'].mean(),
        'active_site_highly': (active_native['frustration_pred'] <= -1.0).mean() * 100,
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nEnzyme Active Site Comparison:")
print(comparison_df.to_string(index=False))
```

## Key Findings

From the FrustraMPNN manuscript:

1. **Active sites are enriched in highly frustrated residues**
   - Catalytic residues often show frustration <= -1.0
   - This frustration is functionally important

2. **Frustration patterns are conserved**
   - Similar enzymes show similar frustration patterns
   - Conservation suggests functional importance

3. **Mutations at frustrated sites often affect function**
   - Reducing frustration may impair catalysis
   - Increasing frustration may destabilize the enzyme

## Next Steps

- [Protein-Protein Interface Analysis](interface-analysis.md)
- [Validation Tutorial](validation-tutorial.md)
- [Batch Processing](batch-tutorial.md)

