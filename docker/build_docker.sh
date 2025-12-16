#!/bin/bash
# Build FrustraMPNN Docker image
# 
# Usage: cd docker && bash build_docker.sh
#
# This builds the Docker image from the parent directory context
# to include the full package source.

set -e

# Build from parent directory to include full package
cd "$(dirname "$0")/.."

echo "Building FrustraMPNN Docker image..."
docker build -f docker/Dockerfile -t frustrampnn:v2 .

echo ""
echo "Build complete!"
echo ""
echo "Usage examples:"
echo "  # Show help"
echo "  docker run --rm frustrampnn:v2 --help"
echo ""
echo "  # Predict frustration (with GPU)"
echo "  docker run --rm --gpus all -v \$(pwd):/data frustrampnn:v2 predict \\"
echo "      --pdb /data/protein.pdb --checkpoint /data/model.ckpt --output /data/results.csv"
echo ""
echo "  # Show package info"
echo "  docker run --rm frustrampnn:v2 info"
