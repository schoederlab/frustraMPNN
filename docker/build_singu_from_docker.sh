#!/bin/bash
# Build FrustraMPNN Singularity image from Docker
#
# Usage: bash build_singu_from_docker.sh
#
# Prerequisites:
#   - Docker image must be built first: bash build_docker.sh
#   - Singularity must be installed

set -e

echo "Building Singularity image from Docker image..."
echo "Make sure you have built the Docker image first: bash build_docker.sh"
echo ""

singularity build frustraMPNN.sif docker-daemon:frustrampnn:v2

echo ""
echo "Build complete!"
echo ""
echo "Usage examples:"
echo "  # Show help"
echo "  singularity run frustraMPNN.sif --help"
echo ""
echo "  # Predict frustration (with GPU)"
echo "  singularity exec --nv frustraMPNN.sif frustrampnn predict \\"
echo "      --pdb protein.pdb --checkpoint model.ckpt --output results.csv"
echo ""
echo "  # Show package info"
echo "  singularity exec frustraMPNN.sif frustrampnn info"
