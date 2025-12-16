# FrustraMPNN Docker Image
# 
# This Dockerfile creates a container with FrustraMPNN and all dependencies
# for running frustration predictions.
#
# Build: cd docker && bash build_docker.sh
# Run:   docker run --rm --gpus all frustrampnn:v2 predict --help

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl wget git bzip2 ca-certificates libglib2.0-0 libxext6 \
    libsm6 libxrender1 libssl-dev libffi-dev vim unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge (lightweight Conda + Mamba)
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

RUN curl -L -o mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/22.9.0-1/Mambaforge-Linux-x86_64.sh && \
    chmod +x mambaforge.sh && \
    bash mambaforge.sh -b -p $MAMBA_ROOT_PREFIX && \
    rm mambaforge.sh

# Create app directory
RUN mkdir -p /app /app/results

# Install Python and PyTorch with CUDA support
RUN mamba install -y -n base python=3.10 \
    numpy=1.26 pytorch=2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \
    && mamba install -y -c conda-forge -c defaults \
        biopython=1.85 pandas scipy scikit-learn tqdm matplotlib seaborn \
        pytorch-lightning omegaconf wandb plotly kaleido click \
    && mamba clean -a -y

# Install ESM (for ESM embeddings support)
RUN pip install fair-esm==2.0.0 absl-py grpcio markdown tensorboard werkzeug

# Install PyRosetta (optional, for validation with frustrapy)
RUN pip install pyrosetta-installer && \
    python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()' && \
    rm -rf ~/.cache/pip

# Install frustrapy (for validation)
RUN git clone https://github.com/Bloeci/frustrapy.git /app/frustrapy && \
    cd /app/frustrapy && \
    pip install -r requirements.txt && \
    pip install -e .

# Copy the frustrampnn package source
COPY . /app/frustrampnn-src

# Install frustrampnn package
WORKDIR /app/frustrampnn-src
RUN pip install -e ".[all]" && \
    rm -rf ~/.cache/pip

# Copy analysis scripts for validation pipelines
COPY scripts/analysis /app/scripts/analysis

# Set working directory
WORKDIR /app

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# All commands go through the frustrampnn CLI
case "$1" in
    # Direct CLI commands
    predict|train|evaluate|info|batch)
        frustrampnn "$@"
        ;;
    # Python passthrough for custom scripts
    python)
        shift
        python -Wignore "$@"
        ;;
    # Help
    --help|-h|help)
        frustrampnn --help
        ;;
    # Default: pass through to frustrampnn CLI
    *)
        frustrampnn "$@"
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh

# Use entrypoint to auto-activate env
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command shows help
CMD ["--help"]
