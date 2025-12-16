# FrustraMPNN Documentation

FrustraMPNN is a message-passing neural network for ultra-fast prediction of single-residue local energetic frustration in proteins. It achieves 1,000-4,500x speedup compared to physics-based methods while maintaining high accuracy.

## What is Local Energetic Frustration?

Local energetic frustration describes regions in proteins where amino acid interactions are not optimally satisfied. These frustrated regions are often functionally important:

- **Minimally frustrated** (green): Optimized interactions, stable regions
- **Neutral** (gray): Average interactions
- **Highly frustrated** (red): Conflicting interactions, often at active sites or binding interfaces

FrustraMPNN predicts the frustration index for all 20 possible amino acids at each position in a protein structure, enabling rapid analysis of mutational landscapes.

## Documentation Sections

### Getting Started

- [Installation Guide](installation.md) - How to install FrustraMPNN
- [Quick Start](quickstart.md) - Get running in 5 minutes
- [Tutorials](tutorials/index.md) - Step-by-step guides

### User Guides

- [Python API](api/python-api.md) - Using FrustraMPNN in Python
- [Command Line Interface](api/cli.md) - Using the CLI
- [Visualization](visualization.md) - Creating plots and figures
- [Validation](validation.md) - Comparing with physics-based methods

### Advanced Topics

- [Training Guide](training/README.md) - Training custom models
- [Batch Processing](batch-processing.md) - Processing multiple structures
- [Docker and Singularity](containers.md) - Using containers

### Reference

- [API Reference](api/reference.md) - Complete API documentation
- [Configuration](configuration.md) - Configuration options
- [Output Format](output-format.md) - Understanding results
- [Migration Guide](MIGRATION.md) - Migrating from old scripts

## Key Features

| Feature | Description |
|---------|-------------|
| Speed | 1,000-4,500x faster than FrustratometeR |
| Accuracy | Spearman correlation 0.80 on external validation |
| Ease of use | Simple Python API and CLI |
| Visualization | Publication-quality plots |
| Validation | Built-in comparison with frustrapy |

## Performance

| Protein Size | FrustraMPNN (GPU) | FrustratometeR (CPU) | Speedup |
|--------------|-------------------|----------------------|---------|
| 100 residues | ~20 ms | ~3.4 min | 1,200x |
| 300 residues | ~30 s | ~19 h | 2,300x |
| 500 residues | ~30 s | ~35 h | 4,500x |

## Citation

If you use FrustraMPNN in your research, please cite:

```bibtex
@article{beining2024frustrampnn,
  title={FrustraMPNN: Ultra-fast deep learning prediction of single-residue 
         local energetic frustration},
  author={Beining, Max and Engelberger, Felipe and Schoeder, Clara T. and 
          Ram{\'\i}rez-Sarmiento, C{\'e}sar A. and Meiler, Jens},
  journal={bioRxiv},
  year={2024}
}
```

## License

FrustraMPNN is released under the MIT License.

## Related Projects

- [frustrapy](https://github.com/engelberger/frustrapy) - Python wrapper for FrustratometeR
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Base architecture
- [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) - Transfer learning approach

