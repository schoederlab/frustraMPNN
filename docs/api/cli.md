# Command Line Interface

FrustraMPNN provides a command-line interface for frustration prediction.

## Installation

The CLI is automatically installed with the package:

```bash
pip install frustrampnn
```

Verify installation:

```bash
frustrampnn --help
```

## Commands

### frustrampnn predict

Predict frustration for a protein structure.

```bash
frustrampnn predict --pdb PROTEIN.pdb --checkpoint MODEL.ckpt [OPTIONS]
```

#### Required arguments

| Argument | Description |
|----------|-------------|
| `--pdb`, `-p` | Path to PDB file |
| `--checkpoint`, `-c` | Path to model checkpoint |

#### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output`, `-o` | stdout | Output CSV file path |
| `--chains` | all | Chains to analyze (space-separated) |
| `--device` | auto | Device to use (cuda, cpu, auto) |
| `--config` | none | Config file for old checkpoints |
| `--quiet`, `-q` | false | Suppress progress output |

#### Examples

Basic prediction:

```bash
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt
```

Save to file:

```bash
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt --output results.csv
```

Specific chains:

```bash
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt --chains A B
```

Force CPU:

```bash
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt --device cpu
```

Old checkpoint format:

```bash
frustrampnn predict --pdb protein.pdb --checkpoint old_model.ckpt --config config.yaml
```

### frustrampnn batch

Process multiple PDB files.

```bash
frustrampnn batch INPUT_DIR --checkpoint MODEL.ckpt [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_DIR` | Directory containing PDB files |
| `--checkpoint`, `-c` | Path to model checkpoint |

#### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir`, `-o` | ./results | Output directory |
| `--pattern` | *.pdb | Glob pattern for PDB files |
| `--chains` | all | Chains to analyze |
| `--device` | auto | Device to use |
| `--quiet`, `-q` | false | Suppress progress output |

#### Examples

Process all PDB files in a directory:

```bash
frustrampnn batch ./structures/ --checkpoint model.ckpt --output-dir ./results/
```

Process specific pattern:

```bash
frustrampnn batch ./structures/ --checkpoint model.ckpt --pattern "chain_A_*.pdb"
```

### frustrampnn train

Train a new model (requires training dependencies).

```bash
frustrampnn train --config CONFIG.yaml [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--config`, `-c` | Path to training config YAML |

#### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs`, `-e` | from config | Override number of epochs |
| `--seed`, `-s` | from config | Random seed |
| `--resume`, `-r` | none | Resume from checkpoint |
| `--quiet`, `-q` | false | Suppress verbose output |

#### Examples

Start training:

```bash
frustrampnn train --config config.yaml
```

Override epochs:

```bash
frustrampnn train --config config.yaml --epochs 50
```

Resume training:

```bash
frustrampnn train --config config.yaml --resume checkpoint.ckpt
```

### frustrampnn evaluate

Evaluate a trained model.

```bash
frustrampnn evaluate --checkpoint MODEL.ckpt [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint`, `-c` | Path to model checkpoint |

#### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | from checkpoint | Config YAML file |
| `--split` | test | Data split (train, val, test) |
| `--output`, `-o` | stdout | Output JSON file |
| `--quiet`, `-q` | false | Suppress verbose output |

#### Examples

Evaluate on test set:

```bash
frustrampnn evaluate --checkpoint model.ckpt
```

Evaluate on validation set:

```bash
frustrampnn evaluate --checkpoint model.ckpt --split val
```

Save results:

```bash
frustrampnn evaluate --checkpoint model.ckpt --output metrics.json
```

### frustrampnn info

Display package information.

```bash
frustrampnn info
```

Output:

```
FrustraMPNN v1.0.0
Python: 3.10.12
PyTorch: 2.1.0
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3090
```

### frustrampnn --help

Display help for any command:

```bash
frustrampnn --help
frustrampnn predict --help
frustrampnn train --help
```

## Output Format

The CLI outputs CSV format with the following columns:

```csv
frustration_pred,position,wildtype,mutation,pdb,chain
0.334,0,M,A,1UBQ,A
1.410,0,M,C,1UBQ,A
-0.892,0,M,D,1UBQ,A
...
```

### Column descriptions

| Column | Type | Description |
|--------|------|-------------|
| `frustration_pred` | float | Predicted frustration index |
| `position` | int | 0-indexed residue position |
| `wildtype` | str | Wild-type amino acid |
| `mutation` | str | Mutant amino acid |
| `pdb` | str | PDB identifier |
| `chain` | str | Chain identifier |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Model loading error |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control GPU visibility |
| `FRUSTRAMPNN_CACHE_DIR` | Cache directory for parsed PDBs |

Example:

```bash
CUDA_VISIBLE_DEVICES=0 frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt
```

## Shell Completion

Generate shell completion scripts:

```bash
# Bash
frustrampnn --install-completion bash

# Zsh
frustrampnn --install-completion zsh

# Fish
frustrampnn --install-completion fish
```

## Piping and Scripting

The CLI supports piping and scripting:

```bash
# Pipe output to another command
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt | grep "highly"

# Process multiple files in a loop
for pdb in structures/*.pdb; do
    name=$(basename "$pdb" .pdb)
    frustrampnn predict --pdb "$pdb" --checkpoint model.ckpt --output "results/${name}.csv"
done

# Combine with other tools
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt | \
    awk -F',' '$1 <= -1.0 {print $2, $3, $4}' | \
    sort -n
```

## Troubleshooting

### "Command not found"

Ensure the package is installed and the virtual environment is activated:

```bash
pip install frustrampnn
# or
source .venv/bin/activate
```

### "CUDA out of memory"

Use CPU or reduce batch size:

```bash
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt --device cpu
```

### "Checkpoint not found"

Check the path to your checkpoint file:

```bash
ls -la checkpoint.ckpt
```

### "Invalid PDB file"

Ensure the PDB file is valid:

```bash
head -20 protein.pdb
```

## See Also

- [Python API](python-api.md) - Python interface documentation
- [Configuration](../configuration.md) - Configuration options
- [Training Guide](../training/README.md) - Training custom models

