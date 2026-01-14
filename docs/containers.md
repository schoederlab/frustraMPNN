# Docker and Singularity Guide

This guide covers using FrustraMPNN with containers for reproducible environments.

## Overview

FrustraMPNN provides container support for:
- **Docker**: Local development and deployment
- **Singularity**: HPC cluster environments

Containers ensure reproducible results across different systems.

## Docker

### Prerequisites

- Docker installed ([installation guide](https://docs.docker.com/get-docker/))
- NVIDIA Container Toolkit for GPU support ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### Building the Docker image

```bash
cd docker
bash build_docker.sh
```

This creates an image named `frustrampnn:latest`.

### Running with Docker

#### Basic prediction

```bash
bash docker/run_docker.sh predict \
    --pdb /data/protein.pdb \
    --checkpoint /data/model.ckpt \
    --output /data/results.csv
```

#### With GPU support

```bash
docker run --gpus all -v $(pwd):/data frustrampnn:latest \
    frustrampnn predict \
    --pdb /data/protein.pdb \
    --checkpoint /data/model.ckpt
```

#### Interactive session

```bash
docker run -it --gpus all -v $(pwd):/data frustrampnn:latest bash
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  frustrampnn:
    image: frustrampnn:latest
    volumes:
      - ./data:/data
      - ./results:/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      frustrampnn predict
      --pdb /data/protein.pdb
      --checkpoint /data/model.ckpt
      --output /results/predictions.csv
```

Run with:

```bash
docker-compose up
```

### Dockerfile details

The Dockerfile includes:
- CUDA 12.1 base image
- Python 3.10
- PyTorch with CUDA support
- FrustraMPNN and all dependencies
- frustrapy for validation

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install FrustraMPNN
COPY . /app
WORKDIR /app
RUN pip install -e ".[all]"

# Set entrypoint
ENTRYPOINT ["frustrampnn"]
```

## Singularity

### Prerequisites

- Singularity installed ([installation guide](https://sylabs.io/guides/latest/user-guide/quick_start.html))
- Access to Docker image or definition file

### Building from Docker

```bash
bash docker/build_singu_from_docker.sh
```

This creates `frustrampnn.sif`.

### Building from definition file

```bash
singularity build frustrampnn.sif docker/Singularity
```

### Running with Singularity

#### Basic prediction

```bash
singularity exec frustrampnn.sif frustrampnn predict \
    --pdb protein.pdb \
    --checkpoint model.ckpt \
    --output results.csv
```

#### With GPU support

```bash
singularity exec --nv frustrampnn.sif frustrampnn predict \
    --pdb protein.pdb \
    --checkpoint model.ckpt
```

#### Binding directories

```bash
singularity exec --nv \
    --bind /path/to/data:/data \
    --bind /path/to/results:/results \
    frustrampnn.sif frustrampnn predict \
    --pdb /data/protein.pdb \
    --checkpoint /data/model.ckpt \
    --output /results/predictions.csv
```

### HPC Job Scripts

#### SLURM example

```bash
#!/bin/bash
#SBATCH --job-name=frustrampnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=frustrampnn_%j.out

module load singularity

singularity exec --nv \
    --bind $PWD:/data \
    /path/to/frustrampnn.sif \
    frustrampnn predict \
    --pdb /data/protein.pdb \
    --checkpoint /data/model.ckpt \
    --output /data/results.csv
```

Submit with:

```bash
sbatch job.sh
```

#### PBS example

```bash
#!/bin/bash
#PBS -N frustrampnn
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1
#PBS -l walltime=4:00:00

module load singularity

cd $PBS_O_WORKDIR

singularity exec --nv \
    --bind $PWD:/data \
    /path/to/frustrampnn.sif \
    frustrampnn predict \
    --pdb /data/protein.pdb \
    --checkpoint /data/model.ckpt \
    --output /data/results.csv
```

### Array jobs for batch processing

#### SLURM array job

```bash
#!/bin/bash
#SBATCH --job-name=frustrampnn_batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --array=1-100
#SBATCH --output=logs/frustrampnn_%A_%a.out

module load singularity

# Get PDB file for this array task
PDB_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_list.txt)
OUTPUT_FILE="results/$(basename $PDB_FILE .pdb)_frustration.csv"

singularity exec --nv \
    --bind $PWD:/data \
    /path/to/frustrampnn.sif \
    frustrampnn predict \
    --pdb /data/$PDB_FILE \
    --checkpoint /data/model.ckpt \
    --output /data/$OUTPUT_FILE
```

Create `pdb_list.txt`:

```bash
ls structures/*.pdb > pdb_list.txt
```

Submit:

```bash
sbatch array_job.sh
```

## Environment Variables

### Inside containers

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control GPU visibility |
| `FRUSTRAMPNN_CACHE_DIR` | Cache directory |
| `OMP_NUM_THREADS` | OpenMP threads |

Example:

```bash
singularity exec --nv \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env OMP_NUM_THREADS=4 \
    frustrampnn.sif frustrampnn predict ...
```

## Troubleshooting

### GPU not detected in Docker

1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Check NVIDIA Container Toolkit:
   ```bash
   docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

3. Ensure `--gpus all` flag is used.

### GPU not detected in Singularity

1. Check `--nv` flag is used.

2. Check CUDA libraries are available:
   ```bash
   singularity exec --nv frustrampnn.sif nvidia-smi
   ```

3. On HPC, ensure GPU module is loaded:
   ```bash
   module load cuda
   ```

### Permission denied errors

For Singularity, bind the necessary directories:

```bash
singularity exec --nv \
    --bind /home/$USER:/home/$USER \
    --bind /scratch:/scratch \
    frustrampnn.sif ...
```

### Out of memory

1. Use CPU instead:
   ```bash
   singularity exec frustrampnn.sif frustrampnn predict --device cpu ...
   ```

2. Request more memory in job script:
   ```bash
   #SBATCH --mem=32G
   ```

### Slow performance

1. Ensure GPU is being used:
   ```bash
   singularity exec --nv frustrampnn.sif python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Check for I/O bottlenecks - use local scratch:
   ```bash
   cp protein.pdb $TMPDIR/
   singularity exec --nv --bind $TMPDIR:/data frustrampnn.sif ...
   ```

## Best Practices

### For reproducibility

1. Use specific image tags:
   ```bash
   docker pull frustrampnn:v1.0.0
   ```

2. Record container version in analysis:
   ```bash
   singularity exec frustrampnn.sif frustrampnn info > container_info.txt
   ```

### For performance

1. Use local storage for I/O-intensive operations.

2. Match GPU architecture to container CUDA version.

3. Use appropriate batch sizes for available memory.

### For HPC

1. Build Singularity images on a build node, not login nodes.

2. Store images in shared filesystem accessible from compute nodes.

3. Use array jobs for batch processing.

## See Also

- [Installation Guide](installation.md) - Other installation methods
- [Batch Processing](batch-processing.md) - Processing multiple structures
- [CLI Reference](api/cli.md) - Command line usage

