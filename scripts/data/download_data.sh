#!/bin/bash
# Download FrustraMPNN training data
#
# This script downloads the training datasets from the FrustraMPNN data repository.
# The data includes:
#   - data_csv_ddg.zip: DDG (stability change) data
#   - data_csv_frustration.zip: Frustration data
#   - pdb_files.zip: PDB structure files
#
# Usage:
#   cd /path/to/frustraMPNN
#   bash scripts/data/download_data.sh
#
# The data will be extracted to the current directory.

set -e

# Data URLs
DATA_DDG="https://git.iwe-lab.de/mabsi/frustrampnn_data/-/raw/main/data_csv_ddg.zip?inline=false"
DATA_FRUSTRA="https://git.iwe-lab.de/mabsi/frustrampnn_data/-/raw/main/data_csv_frustration.zip?inline=false"
DATA_PDB="https://git.iwe-lab.de/mabsi/frustrampnn_data/-/raw/main/pdb_files.zip?inline=false"

# Output file names
ZIP_FILES=("data_csv_ddg.zip" "data_csv_frustration.zip" "pdb_files.zip")
URLS=("$DATA_DDG" "$DATA_FRUSTRA" "$DATA_PDB")

echo "FrustraMPNN Data Downloader"
echo "==========================="
echo ""

# Download and extract each file
for i in {0..2}; do
    OUTPUT_FILE="${ZIP_FILES[i]}"
    URL="${URLS[i]}"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[SKIP] $OUTPUT_FILE already exists. Skipping download."
        continue
    fi
    
    echo "[DOWNLOAD] Downloading $OUTPUT_FILE..."
    if ! curl -L -o "$OUTPUT_FILE" "$URL"; then
        echo "[ERROR] Failed to download $OUTPUT_FILE"
        continue
    fi
    
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "[ERROR] Download failed for $OUTPUT_FILE"
        continue
    fi
    
    echo "[EXTRACT] Unzipping $OUTPUT_FILE..."
    unzip -q "$OUTPUT_FILE"
    
    echo "[CLEANUP] Removing $OUTPUT_FILE..."
    rm "$OUTPUT_FILE"
    
    echo "[DONE] $OUTPUT_FILE processed successfully"
    echo ""
done

echo "==========================="
echo "Data download complete!"
echo ""
echo "Downloaded directories:"
ls -d */ 2>/dev/null || echo "  (no directories found)"
