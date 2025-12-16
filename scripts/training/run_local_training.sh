#!/bin/bash
# Local training script for FrustraMPNN
#
# This script demonstrates how to train a FrustraMPNN model locally.
# Adjust the paths and parameters according to your setup.
#
# Usage:
#   cd /path/to/frustraMPNN
#   bash scripts/training/run_local_training.sh
#
# Prerequisites:
#   - Download training data: bash scripts/data/download_data.sh
#   - Install package: pip install -e ".[all]"

set -e

# Configuration
PROJECT_NAME="frustraMPNN_fireprot"
CONFIG_FILE="configs/fireprot_training.yaml"  # Create this config file

# Data paths (adjust to your setup)
CSV_FILE="../data_csv_frustration/fireprot/fireprot_full_data_default_frustration_raw.csv"
SPLIT_FILE="../data_csv_frustration/fireprot/splits_default/split_seed_35.pkl"
PDB_DIR="../pdb_files/fireprot/"

# Model settings
ADD_LIGHTATTENTION="True"
ADD_ESM="False"
EPOCHS=10
SEED=0

# Output directories
WEIGHTS_DIR="./local_training_weights"
LOG_DIR="./local_training_log"

# Create output directories
mkdir -p "$WEIGHTS_DIR" "$LOG_DIR"

echo "Starting FrustraMPNN training..."
echo "Project: $PROJECT_NAME"
echo "Config: $CONFIG_FILE"
echo "Epochs: $EPOCHS"
echo ""

# Run training using the CLI
# Note: You need to create a config YAML file with your training settings
# See docs/training/README.md for configuration options

frustrampnn train \
    --config "$CONFIG_FILE" \
    --epochs "$EPOCHS" \
    --seed "$SEED"

echo ""
echo "Training complete!"
echo "Weights saved to: $WEIGHTS_DIR"
echo "Logs saved to: $LOG_DIR"
