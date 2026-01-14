# Output Format

This document describes the output format of FrustraMPNN predictions.

## CSV Output

FrustraMPNN outputs predictions in CSV format with the following columns:

```csv
frustration_pred,position,wildtype,mutation,pdb,chain
0.334012,0,M,A,1UBQ,A
1.410234,0,M,C,1UBQ,A
-0.892341,0,M,D,1UBQ,A
...
```

## Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `frustration_pred` | float | Predicted frustration index |
| `position` | int | 0-indexed residue position |
| `wildtype` | str | Wild-type amino acid (1-letter code) |
| `mutation` | str | Mutant amino acid (1-letter code) |
| `pdb` | str | PDB identifier (filename without extension) |
| `chain` | str | Chain identifier |

## Frustration Index

The frustration index is a continuous value indicating the degree of energetic frustration:

| Range | Category | Interpretation |
|-------|----------|----------------|
| <= -1.0 | Highly frustrated | Conflicting interactions, often functional |
| -1.0 to 0.58 | Neutral | Average interactions |
| >= 0.58 | Minimally frustrated | Optimized interactions, stable |

### Distribution

Typical frustration index distribution:
- Mean: ~0.0
- Standard deviation: ~1.0
- Range: approximately -3.0 to +3.0

## Position Indexing

Positions are **0-indexed** in the output:

| Output Position | PDB Residue Number | Notes |
|-----------------|-------------------|-------|
| 0 | First residue | May not be residue 1 in PDB |
| N-1 | Last residue | N = sequence length |

### Converting to PDB numbering

PDB files may have non-standard numbering (gaps, insertion codes). To map positions:

```python
from Bio.PDB import PDBParser

# Parse PDB
parser = PDBParser(QUIET=True)
structure = parser.get_structure('protein', 'protein.pdb')

# Get residue mapping
chain = structure[0]['A']
residue_map = {}
for i, residue in enumerate(chain.get_residues()):
    if residue.id[0] == ' ':  # Standard residue
        residue_map[i] = residue.id[1]  # PDB residue number

# Convert position to PDB number
def to_pdb_number(position):
    return residue_map.get(position, None)
```

## Amino Acid Codes

Standard 1-letter amino acid codes:

| Code | Amino Acid | Code | Amino Acid |
|------|------------|------|------------|
| A | Alanine | M | Methionine |
| C | Cysteine | N | Asparagine |
| D | Aspartic acid | P | Proline |
| E | Glutamic acid | Q | Glutamine |
| F | Phenylalanine | R | Arginine |
| G | Glycine | S | Serine |
| H | Histidine | T | Threonine |
| I | Isoleucine | V | Valine |
| K | Lysine | W | Tryptophan |
| L | Leucine | Y | Tyrosine |

Note: X represents unknown/non-standard amino acids.

## Output Size

For a protein with N residues:
- Total rows: N x 20 (all positions x all amino acids)
- Native residues: N rows (where wildtype == mutation)
- Mutations: N x 19 rows (where wildtype != mutation)

Example for ubiquitin (76 residues):
- Total rows: 76 x 20 = 1,520
- Native residues: 76
- Mutations: 76 x 19 = 1,444

## Working with Results

### Load results

```python
import pandas as pd

results = pd.read_csv("results.csv")
```

### Filter native residues

```python
native = results[results['wildtype'] == results['mutation']]
```

### Filter mutations only

```python
mutations = results[results['wildtype'] != results['mutation']]
```

### Filter by chain

```python
chain_a = results[results['chain'] == 'A']
```

### Filter by frustration category

```python
highly_frustrated = results[results['frustration_pred'] <= -1.0]
neutral = results[(results['frustration_pred'] > -1.0) & 
                  (results['frustration_pred'] < 0.58)]
minimally_frustrated = results[results['frustration_pred'] >= 0.58]
```

### Pivot to matrix

```python
# Create position x mutation matrix
matrix = results.pivot_table(
    index='position',
    columns='mutation',
    values='frustration_pred'
)
```

### Aggregate by position

```python
# Mean frustration per position
mean_per_position = results.groupby('position')['frustration_pred'].mean()

# Native frustration per position
native_per_position = native.set_index('position')['frustration_pred']
```

## Batch Output

When processing multiple structures, results are concatenated:

```csv
frustration_pred,position,wildtype,mutation,pdb,chain
0.334,0,M,A,protein1,A
1.410,0,M,C,protein1,A
...
0.567,0,G,A,protein2,A
0.891,0,G,C,protein2,A
...
```

### Separate by PDB

```python
# Group by PDB
for pdb_id, group in results.groupby('pdb'):
    group.to_csv(f"{pdb_id}_results.csv", index=False)
```

## JSON Output (CLI)

The CLI can output JSON format for programmatic use:

```bash
frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt --format json
```

```json
{
  "pdb": "protein",
  "chain": "A",
  "predictions": [
    {
      "position": 0,
      "wildtype": "M",
      "mutation": "A",
      "frustration_pred": 0.334
    },
    ...
  ],
  "metadata": {
    "model": "checkpoint.ckpt",
    "timestamp": "2024-12-16T10:30:00",
    "version": "1.0.0"
  }
}
```

## Compatibility

The output format is compatible with:
- pandas DataFrames
- Excel (via pandas)
- R data frames
- Database import (SQL)
- Visualization tools

### Export to Excel

```python
results.to_excel("results.xlsx", index=False)
```

### Export to R

```python
# Save as CSV, then in R:
# results <- read.csv("results.csv")
```

### Export to database

```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///frustration.db')
results.to_sql('predictions', engine, index=False, if_exists='replace')
```

## Backward Compatibility

The output format is identical to the original FrustraMPNN scripts for backward compatibility. Column names and data types are preserved.

## See Also

- [Python API](api/python-api.md) - Working with results in Python
- [Visualization](visualization.md) - Plotting results
- [Validation](validation.md) - Comparing with physics-based methods

