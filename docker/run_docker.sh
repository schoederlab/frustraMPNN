#!/bin/bash
# ============================================================
# FrustraMPNN Docker Runner
# ============================================================
#
# This script provides a convenient interface for running FrustraMPNN
# predictions using Docker.
#
# Usage:
#   ./run_docker.sh predict --pdb protein.pdb --checkpoint model.ckpt
#   ./run_docker.sh batch *.pdb --checkpoint model.ckpt
#   ./run_docker.sh info
#
# ============================================================

set -e

# Default configuration
DOCKER_IMAGE="${FRUSTRAMPNN_DOCKER_IMAGE:-frustrampnn:v2}"
USE_GPU=true
VERBOSE=false

# ============================================================
# Helper functions
# ============================================================

show_help() {
    echo "FrustraMPNN Docker Runner"
    echo ""
    echo "Usage: $0 [OPTIONS] COMMAND [ARGS...]"
    echo ""
    echo "Options:"
    echo "  --docker-image IMAGE   Docker image to use (default: $DOCKER_IMAGE)"
    echo "  --no-gpu               Disable GPU support"
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
    echo "  # Batch prediction"
    echo "  $0 batch *.pdb --checkpoint model.ckpt --output-dir results/"
    echo ""
    echo "  # Show package info"
    echo "  $0 info"
    echo ""
    echo "Environment Variables:"
    echo "  FRUSTRAMPNN_DOCKER_IMAGE   Default Docker image (default: frustrampnn:v2)"
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
        --docker-image)
            DOCKER_IMAGE="$2"
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
# Check Docker
# ============================================================

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running"
    exit 1
fi

# Check if image exists
if ! docker image inspect "$DOCKER_IMAGE" > /dev/null 2>&1; then
    echo "ERROR: Docker image '$DOCKER_IMAGE' not found"
    echo "Build it first: cd docker && bash build_docker.sh"
    exit 1
fi

log "INFO" "Using Docker image: $DOCKER_IMAGE"

# ============================================================
# Build Docker command
# ============================================================

# Build volume mounts for current directory
VOLUMES="-v $(pwd):/data -w /data"

# Build GPU flag
GPU_FLAG=""
if [ "$USE_GPU" = true ]; then
    GPU_FLAG="--gpus all"
fi

# ============================================================
# Run command
# ============================================================

log "INFO" "Running: frustrampnn $COMMAND $*"
docker run --rm $GPU_FLAG $VOLUMES "$DOCKER_IMAGE" "$COMMAND" "$@"

exit $?
