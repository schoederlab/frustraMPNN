# Batch Processing Guide

This guide covers processing multiple protein structures with FrustraMPNN.

## Overview

FrustraMPNN supports batch processing for:
- Multiple PDB files
- Multiple chains within a structure
- Large-scale proteome analysis

## Command Line Batch Processing

### Process a directory

```bash
frustrampnn batch ./structures/ --checkpoint model.ckpt --output-dir ./results/
```

### With options

```bash
frustrampnn batch ./structures/ \
    --checkpoint model.ckpt \
    --output-dir ./results/ \
    --pattern "*.pdb" \
    --chains A \
    --device cuda
```

### Process specific files

```bash
frustrampnn batch ./structures/ \
    --checkpoint model.ckpt \
    --pattern "chain_A_*.pdb"
```

## Python Batch Processing

### Using predict_batch

```python
from frustrampnn import FrustraMPNN
from pathlib import Path

# Load model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# Get PDB files
pdb_files = list(Path("structures/").glob("*.pdb"))

# Batch predict
results = model.predict_batch(
    [str(f) for f in pdb_files],
    chains=["A"],
    show_progress=True
)

# Save combined results
results.to_csv("all_results.csv", index=False)
```

### Processing individually

For more control, process files individually:

```python
from frustrampnn import FrustraMPNN
from pathlib import Path
import pandas as pd

model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

all_results = []
failed = []

for pdb_file in Path("structures/").glob("*.pdb"):
    try:
        print(f"Processing {pdb_file.name}...")
        results = model.predict(str(pdb_file), chains=["A"])
        all_results.append(results)
    except Exception as e:
        print(f"  Failed: {e}")
        failed.append(pdb_file.name)

# Combine results
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv("all_results.csv", index=False)

# Report failures
if failed:
    print(f"\nFailed files: {failed}")
```

### Parallel processing

Use multiprocessing for CPU-bound operations:

```python
from frustrampnn import FrustraMPNN
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def process_pdb(pdb_path):
    """Process a single PDB file."""
    model = FrustraMPNN.from_pretrained("checkpoint.ckpt")
    try:
        results = model.predict(str(pdb_path), chains=["A"])
        return results
    except Exception as e:
        print(f"Failed {pdb_path}: {e}")
        return None

# Get PDB files
pdb_files = list(Path("structures/").glob("*.pdb"))

# Process in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_pdb, pdb_files))

# Combine non-None results
combined = pd.concat([r for r in results if r is not None], ignore_index=True)
```

## Memory Management

### For large datasets

When processing many structures, manage memory:

```python
import torch
import gc

model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

for i, pdb_file in enumerate(pdb_files):
    results = model.predict(str(pdb_file))
    results.to_csv(f"results/{pdb_file.stem}.csv", index=False)
    
    # Clear memory periodically
    if i % 100 == 0:
        torch.cuda.empty_cache()
        gc.collect()
```

### Streaming results

For very large datasets, stream results to disk:

```python
import csv

with open("results.csv", "w", newline="") as f:
    writer = None
    
    for pdb_file in pdb_files:
        results = model.predict(str(pdb_file))
        
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=results.columns)
            writer.writeheader()
        
        for _, row in results.iterrows():
            writer.writerow(row.to_dict())
```

## Proteome-Scale Analysis

### E. coli proteome example

FrustraMPNN can analyze entire proteomes:

```python
from frustrampnn import FrustraMPNN
from pathlib import Path
import pandas as pd
import time

# Load model
model = FrustraMPNN.from_pretrained("checkpoint.ckpt")

# Get all proteome structures
proteome_dir = Path("ecoli_proteome/")
pdb_files = list(proteome_dir.glob("*.pdb"))
print(f"Processing {len(pdb_files)} structures...")

# Track progress
start_time = time.time()
results_dir = Path("ecoli_results/")
results_dir.mkdir(exist_ok=True)

for i, pdb_file in enumerate(pdb_files):
    # Progress
    if i % 100 == 0:
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        remaining = (len(pdb_files) - i) / rate if rate > 0 else 0
        print(f"Progress: {i}/{len(pdb_files)} ({rate:.1f}/s, ~{remaining/3600:.1f}h remaining)")
    
    # Process
    try:
        results = model.predict(str(pdb_file))
        results.to_csv(results_dir / f"{pdb_file.stem}.csv", index=False)
    except Exception as e:
        print(f"Failed {pdb_file.name}: {e}")

total_time = time.time() - start_time
print(f"\nCompleted in {total_time/3600:.2f} hours")
```

### Performance estimates

| Proteome | Proteins | GPU Time | CPU Time |
|----------|----------|----------|----------|
| E. coli | ~4,300 | ~12 hours | ~1.7 years |
| Yeast | ~6,000 | ~17 hours | ~2.4 years |
| Human | ~20,000 | ~56 hours | ~8 years |

## Output Organization

### Separate files per structure

```python
from pathlib import Path

output_dir = Path("results/")
output_dir.mkdir(exist_ok=True)

for pdb_file in pdb_files:
    results = model.predict(str(pdb_file))
    output_path = output_dir / f"{pdb_file.stem}_frustration.csv"
    results.to_csv(output_path, index=False)
```

### Combined file with metadata

```python
all_results = []

for pdb_file in pdb_files:
    results = model.predict(str(pdb_file))
    results['source_file'] = pdb_file.name
    results['processed_at'] = pd.Timestamp.now()
    all_results.append(results)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv("all_frustration.csv", index=False)
```

### Database storage

For large-scale analysis, use a database:

```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///frustration.db')

for pdb_file in pdb_files:
    results = model.predict(str(pdb_file))
    results.to_sql('predictions', engine, if_exists='append', index=False)
```

## Error Handling

### Robust batch processing

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_batch(pdb_files, model, output_dir):
    """Process batch with error handling."""
    successful = []
    failed = []
    
    for pdb_file in pdb_files:
        try:
            results = model.predict(str(pdb_file))
            output_path = output_dir / f"{pdb_file.stem}.csv"
            results.to_csv(output_path, index=False)
            successful.append(pdb_file.name)
            logger.info(f"Processed {pdb_file.name}")
        except FileNotFoundError:
            logger.error(f"File not found: {pdb_file}")
            failed.append((pdb_file.name, "File not found"))
        except RuntimeError as e:
            logger.error(f"Model error for {pdb_file}: {e}")
            failed.append((pdb_file.name, str(e)))
        except Exception as e:
            logger.error(f"Unexpected error for {pdb_file}: {e}")
            failed.append((pdb_file.name, str(e)))
    
    return successful, failed

# Run
successful, failed = process_batch(pdb_files, model, output_dir)

# Report
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")
if failed:
    for name, error in failed:
        print(f"  {name}: {error}")
```

### Retry failed files

```python
def process_with_retry(pdb_file, model, max_retries=3):
    """Process with retry logic."""
    for attempt in range(max_retries):
        try:
            return model.predict(str(pdb_file))
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1} for {pdb_file.name}")
                time.sleep(1)
            else:
                raise e
```

## Aggregating Results

### Summary statistics per structure

```python
def summarize_structure(results):
    """Compute summary statistics for a structure."""
    native = results[results['wildtype'] == results['mutation']]
    
    return {
        'pdb': results['pdb'].iloc[0],
        'chain': results['chain'].iloc[0],
        'n_residues': len(native),
        'mean_frustration': native['frustration_pred'].mean(),
        'std_frustration': native['frustration_pred'].std(),
        'n_highly_frustrated': (native['frustration_pred'] <= -1.0).sum(),
        'n_minimally_frustrated': (native['frustration_pred'] >= 0.58).sum(),
        'pct_highly_frustrated': (native['frustration_pred'] <= -1.0).mean() * 100,
    }

# Summarize all structures
summaries = []
for pdb_file in pdb_files:
    results = model.predict(str(pdb_file))
    summaries.append(summarize_structure(results))

summary_df = pd.DataFrame(summaries)
summary_df.to_csv("proteome_summary.csv", index=False)
```

### Find most frustrated proteins

```python
# Sort by fraction of highly frustrated residues
most_frustrated = summary_df.nlargest(10, 'pct_highly_frustrated')
print("Most frustrated proteins:")
print(most_frustrated[['pdb', 'n_residues', 'pct_highly_frustrated']])
```

## See Also

- [Python API](api/python-api.md) - API documentation
- [CLI Reference](api/cli.md) - Command line usage
- [Output Format](output-format.md) - Understanding results

