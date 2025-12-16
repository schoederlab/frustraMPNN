# Visualization Guide

FrustraMPNN provides publication-quality visualization tools for frustration analysis.

## Installation

Install visualization dependencies:

```bash
pip install frustrampnn[viz]
```

This installs:
- matplotlib (static plots)
- seaborn (enhanced styling)
- plotly (interactive plots)
- kaleido (plotly export)
- py3Dmol (3D structure viewing)

## Color Scheme

FrustraMPNN uses a consistent color scheme matching frustrapy:

| Category | Frustration Index | Color | RGB |
|----------|-------------------|-------|-----|
| Highly frustrated | <= -1.0 | Red | #FF0000 |
| Neutral | -1.0 to 0.58 | Gray | #808080 |
| Minimally frustrated | >= 0.58 | Green | #00FF00 |
| Native residue | (highlighted) | Blue | #0000FF |

## Single-Residue Plots

Show frustration for all 20 amino acids at a specific position.

### Matplotlib (static)

```python
from frustrampnn.visualization import plot_single_residue

# Basic plot
fig = plot_single_residue(results, position=72, chain="A")
fig.savefig("position_73.png", dpi=300)

# Customized plot
fig = plot_single_residue(
    results,
    position=72,
    chain="A",
    title="Leucine 73 - Active Site",
    figsize=(12, 6),
    show_thresholds=True,    # Show category threshold lines
    highlight_native=True,   # Highlight native residue
)
fig.savefig("position_73_custom.png", dpi=300, bbox_inches='tight')
```

### Plotly (interactive)

```python
from frustrampnn.visualization import plot_single_residue_plotly

# Basic interactive plot
fig = plot_single_residue_plotly(results, position=72, chain="A")
fig.show()

# Save as HTML
fig.write_html("position_73.html")

# Save as static image
fig.write_image("position_73.png", scale=2)

# Customized
fig = plot_single_residue_plotly(
    results,
    position=72,
    chain="A",
    title="Leucine 73 - Active Site",
    width=800,
    height=500,
)
```

### Interpreting single-residue plots

The single-residue plot shows:
- X-axis: 20 amino acids (sorted by frustration)
- Y-axis: Frustration index
- Horizontal lines: Category thresholds (-1.0 and 0.58)
- Blue bar: Native (wild-type) residue
- Colors: Frustration category

Use these plots to:
- Identify which mutations would increase/decrease frustration
- Find stabilizing mutations (toward green)
- Find destabilizing mutations (toward red)

## Heatmaps

Show frustration across all positions and mutations.

### Matplotlib (static)

```python
from frustrampnn.visualization import plot_frustration_heatmap

# Basic heatmap
fig = plot_frustration_heatmap(results, chain="A")
fig.savefig("heatmap.png", dpi=300)

# Customized heatmap
fig = plot_frustration_heatmap(
    results,
    chain="A",
    figsize=(20, 10),
    cmap="RdYlGn_r",           # Red-Yellow-Green reversed
    vmin=-2.0,                  # Min value for color scale
    vmax=2.0,                   # Max value for color scale
    show_colorbar=True,
    title="Frustration Landscape",
)
fig.savefig("heatmap_custom.png", dpi=300, bbox_inches='tight')
```

### Plotly (interactive)

```python
from frustrampnn.visualization import plot_frustration_heatmap_plotly

# Basic interactive heatmap
fig = plot_frustration_heatmap_plotly(results, chain="A")
fig.show()

# Save as HTML (interactive)
fig.write_html("heatmap.html")

# Customized
fig = plot_frustration_heatmap_plotly(
    results,
    chain="A",
    title="Frustration Landscape",
    width=1200,
    height=600,
    colorscale="RdYlGn_r",
)
```

### Interpreting heatmaps

The heatmap shows:
- X-axis: Residue positions
- Y-axis: 20 amino acids
- Color: Frustration index

Use heatmaps to:
- Identify globally frustrated regions
- Find positions tolerant to mutations
- Visualize the mutational landscape

## Sequence Maps

Show frustration along the sequence with secondary structure.

```python
from frustrampnn.visualization import plot_sequence_map

fig = plot_sequence_map(
    results,
    chain="A",
    secondary_structure=ss_string,  # Optional: DSSP string
    show_categories=True,
)
fig.savefig("sequence_map.png", dpi=300)
```

## Sankey Diagrams

Show flow between calculated and predicted frustration categories.

```python
from frustrampnn.visualization import plot_sankey

fig = plot_sankey(
    calculated=calculated_results,
    predicted=predicted_results,
    title="Calculated vs Predicted",
)
fig.show()
```

## 3D Structure Visualization

Visualize frustration on the 3D structure using py3Dmol.

```python
from frustrampnn.visualization import view_structure_3d

# In Jupyter notebook
view = view_structure_3d(
    pdb_path="protein.pdb",
    results=results,
    chain="A",
    color_by="frustration",  # Color by frustration index
)
view.show()

# Save as HTML
view.write_html("structure_3d.html")
```

## Comparison Plots

Compare FrustraMPNN predictions with frustrapy calculations.

```python
from frustrampnn.validation import compare_with_frustrapy

comparison = compare_with_frustrapy(
    pdb_path="protein.pdb",
    chain="A",
    frustrampnn_results=results,
)

# Scatter plot
fig = comparison.plot()
fig.show()

# Correlation plot with regression line
fig = comparison.plot(
    show_regression=True,
    show_identity=True,
    title="FrustraMPNN vs FrustratometeR",
)
fig.savefig("comparison.png", dpi=300)
```

## Batch Visualization

Generate plots for multiple positions or structures.

```python
from frustrampnn.visualization import plot_single_residue
import os

# Create output directory
os.makedirs("plots", exist_ok=True)

# Get native residue data
native = results[results['wildtype'] == results['mutation']]

# Find highly frustrated positions
highly_frustrated = native[native['frustration_pred'] <= -1.0]

# Generate plots for each
for _, row in highly_frustrated.iterrows():
    pos = row['position']
    wt = row['wildtype']
    
    fig = plot_single_residue(
        results,
        position=pos,
        chain="A",
        title=f"Position {pos+1} ({wt})",
    )
    fig.savefig(f"plots/position_{pos}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Free memory
```

## Customizing Plots

### Matplotlib style

```python
import matplotlib.pyplot as plt

# Set global style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom colors
custom_colors = {
    'highly': '#D32F2F',
    'neutral': '#757575',
    'minimally': '#388E3C',
    'native': '#1976D2',
}

# Apply to plot
fig = plot_single_residue(
    results,
    position=72,
    chain="A",
    colors=custom_colors,
)
```

### Plotly templates

```python
import plotly.io as pio

# Set default template
pio.templates.default = "plotly_white"

# Or per-plot
fig = plot_single_residue_plotly(
    results,
    position=72,
    chain="A",
)
fig.update_layout(template="plotly_white")
```

## Exporting Figures

### High-resolution PNG

```python
fig.savefig("figure.png", dpi=300, bbox_inches='tight')
```

### Vector formats (PDF, SVG)

```python
fig.savefig("figure.pdf", bbox_inches='tight')
fig.savefig("figure.svg", bbox_inches='tight')
```

### Plotly export

```python
# HTML (interactive)
fig.write_html("figure.html")

# PNG (requires kaleido)
fig.write_image("figure.png", scale=2)

# PDF
fig.write_image("figure.pdf")

# SVG
fig.write_image("figure.svg")
```

## Publication-Ready Figures

Example workflow for publication figures:

```python
import matplotlib.pyplot as plt
from frustrampnn.visualization import (
    plot_single_residue,
    plot_frustration_heatmap,
)

# Set publication style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Heatmap
ax1 = axes[0]
plot_frustration_heatmap(results, chain="A", ax=ax1)
ax1.set_title("A", loc='left', fontweight='bold')

# Panel B: Single residue
ax2 = axes[1]
plot_single_residue(results, position=72, chain="A", ax=ax2)
ax2.set_title("B", loc='left', fontweight='bold')

plt.tight_layout()
fig.savefig("figure_1.pdf", bbox_inches='tight')
fig.savefig("figure_1.png", dpi=300, bbox_inches='tight')
```

## Troubleshooting

### "No module named 'matplotlib'"

Install visualization dependencies:

```bash
pip install frustrampnn[viz]
```

### Plotly figures not showing

In Jupyter, ensure plotly is configured:

```python
import plotly.io as pio
pio.renderers.default = "notebook"
```

### Slow rendering for large proteins

For proteins > 500 residues, consider:
- Using static matplotlib instead of interactive plotly
- Reducing figure resolution
- Plotting specific regions instead of full protein

## See Also

- [Python API](api/python-api.md) - API documentation
- [Validation](validation.md) - Comparison with frustrapy
- [Output Format](output-format.md) - Understanding results

