#!/bin/bash
# ============================================================
# FrustraMPNN Singularity Runner
# ============================================================
#
# This script provides a convenient interface for running FrustraMPNN
# predictions using Singularity on HPC systems.
#
# Usage:
#   ./run_singularity.sh predict --pdb protein.pdb --checkpoint model.ckpt
#   ./run_singularity.sh batch *.pdb --checkpoint model.ckpt
#   ./run_singularity.sh info
#
# ============================================================

set -e

# Default configuration
CONTAINER_SIF="${FRUSTRAMPNN_SIF:-frustraMPNN.sif}"
USE_GPU=true
VERBOSE=false

# ============================================================
# Helper functions
# ============================================================

show_help() {
    echo "FrustraMPNN Singularity Runner"
    echo ""
    echo "Usage: $0 [OPTIONS] COMMAND [ARGS...]"
    echo ""
    echo "Options:"
    echo "  --sif FILE             Singularity image file (default: $CONTAINER_SIF)"
    echo "  --no-gpu               Disable GPU support (omit --nv flag)"
    echo "  --verbose              Show verbose output"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Commands:"
    echo "  predict                Predict frustration for a PDB file"
    echo "  batch                  Batch prediction for multiple PDB files"
    echo "  train                  Train a FrustraMPNN model"
    echo "  evaluate               Evaluate a trained model"
    echo "  info                   Show package information"
    echo "  python                 Run arbitrary Python command"
    echo ""
    echo "Examples:"
    echo "  # Predict frustration"
    echo "  $0 predict --pdb protein.pdb --checkpoint model.ckpt"
    echo ""
    echo "  # Predict with specific container"
    echo "  $0 --sif /path/to/frustraMPNN.sif predict --pdb protein.pdb --checkpoint model.ckpt"
    echo ""
    echo "  # Batch prediction"
    echo "  $0 batch *.pdb --checkpoint model.ckpt --output-dir results/"
    echo ""
    echo "  # Show package info"
    echo "  $0 info"
    echo ""
    echo "Environment Variables:"
    echo "  FRUSTRAMPNN_SIF   Default Singularity image (default: frustraMPNN.sif)"
}

log() {
    local level="$1"
    local msg="$2"
    
    if [[ "$VERBOSE" == true ]] || [[ "$level" != "INFO" ]]; then
        echo "[${level}] $msg"
    fi
}

# ============================================================
# Parse global options
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --sif)
            CONTAINER_SIF="$2"
            shift 2
            ;;
        --no-gpu)
            USE_GPU=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            # Unknown option - might be for the command
            break
            ;;
        *)
            # Command found
            break
            ;;
    esac
done

# Check if command is provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

COMMAND="$1"
shift

# ============================================================
# Check Singularity
# ============================================================

if ! command -v singularity &> /dev/null; then
    echo "ERROR: Singularity is not installed or not in PATH"
    exit 1
fi

# Check if container exists
if [ ! -f "$CONTAINER_SIF" ]; then
    echo "ERROR: Singularity image '$CONTAINER_SIF' not found"
    echo "Build it first: bash docker/build_singu_from_docker.sh"
    exit 1
fi

log "INFO" "Using Singularity image: $CONTAINER_SIF"

# ============================================================
# Build Singularity command
# ============================================================

# Build bind mounts for current directory
BINDS="-B $(pwd):/data"

# Build GPU flag
GPU_FLAG=""
if [ "$USE_GPU" = true ]; then
    GPU_FLAG="--nv"
fi

# ============================================================
# Run command
# ============================================================

log "INFO" "Running: frustrampnn $COMMAND $*"

# Use singularity run to invoke the entrypoint
singularity run $GPU_FLAG $BINDS "$CONTAINER_SIF" "$COMMAND" "$@"

exit $?
