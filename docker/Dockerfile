# Use official CUDA base with conda pre-installed
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and Mamba
RUN apt-get update && apt-get install -y \
    curl wget git bzip2 ca-certificates libglib2.0-0 libxext6 \
    libsm6 libxrender1 libssl-dev libffi-dev vim \
    && rm -rf /var/lib/apt/lists/*


# Install Mambaforge (lightweight Conda + Mamba)
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# Fetch a specific version of Mambaforge
RUN curl -L -o mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/22.9.0-1/Mambaforge-Linux-x86_64.sh && \
    chmod +x mambaforge.sh && \
    bash mambaforge.sh -b -p $MAMBA_ROOT_PREFIX && \
    rm mambaforge.sh


# Copy environment setup (replace this with your own layer-by-layer install to avoid .yml)
RUN mkdir -p /app
COPY ../inference /app/inference
COPY scripts /app/scripts
RUN mkdir -p /app/results

# Create env and install everything with mamba
RUN mamba install -y -n base python=3.10 \
    numpy=1.26 pytorch=2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \
    && pip install fair-esm==2.0.0 absl-py grpcio markdown tensorboard werkzeug \
    && mamba install -y -c conda-forge -c defaults \
        biopython=1.85 pandas scipy scikit-learn tqdm matplotlib seaborn \
        pytorch-lightning omegaconf wandb \
    && mamba clean -a -y

# Install PyRosetta
RUN pip install pyrosetta-installer && \
    python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()' && \
    rm -rf ~/.cache/pip
    
# Install frustrapy dependencies
RUN git clone https://github.com/Bloeci/frustrapy.git /app/frustrapy && \
    cd /app/frustrapy && \
    pip install -r requirements.txt && \
    pip install -e .


# ===============================================

# Create entrypoint script for inference and single residue frustration
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /app/entrypoint.sh && \
    echo 'conda activate base' >> /app/entrypoint.sh && \
    echo 'case "$1" in' >> /app/entrypoint.sh && \
    echo '  inference)' >> /app/entrypoint.sh && \
    echo '    shift' >> /app/entrypoint.sh && \
    echo '    python -Wignore /app/inference/custom_inference_refac.py "$@"' >> /app/entrypoint.sh && \
    echo '    ;;' >> /app/entrypoint.sh && \
    echo '  singleresidue)' >> /app/entrypoint.sh && \
    echo '    shift' >> /app/entrypoint.sh && \
    echo '    python -Wignore /app/scripts/frustration_singleresidue.py "$@"' >> /app/entrypoint.sh && \
    echo '    ;;' >> /app/entrypoint.sh && \
    echo '  python)' >> /app/entrypoint.sh && \
    echo '    shift' >> /app/entrypoint.sh && \
    echo '    python -Wignore "$@"' >> /app/entrypoint.sh && \
    echo '    ;;' >> /app/entrypoint.sh && \
    echo '  *)' >> /app/entrypoint.sh && \
    echo '    echo "Usage: $0 {inference|singleresidue|python} [args...]"' >> /app/entrypoint.sh && \
    echo '    exit 1' >> /app/entrypoint.sh && \
    echo '    ;;' >> /app/entrypoint.sh && \
    echo 'esac' >> /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh
# Use entrypoint to auto-activate env
ENTRYPOINT ["/app/entrypoint.sh"]
