#!/bin/bash
################## frustraMPNN installation script
################## specify conda/mamba folder, and installation folder for git repositories, and whether to use mamba or conda
# Default value for pkg_manager
pkg_manager='conda'
cuda=''
reinstall=false

# Define the short and long options
OPTIONS=p:c:r
LONGOPTIONS=pkg_manager:,cuda:,reinstall

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    -r|--reinstall)
      reinstall=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

# Example usage of the parsed variables
echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"
echo -e "Reinstall: $reinstall"

############################################################################################################
############################################################################################################
################## CUDA confirmation
if [ -z "$cuda" ]; then
    echo -e "\n[!]  INFO: No CUDA version specified."
    echo -e "The package manager will automatically detect and install the best PyTorch version for your system."
    echo -e "This may include CUDA support if available, or fall back to CPU-only if no compatible CUDA is found."
    echo -e "For explicit CUDA control, use: --cuda <version> (e.g., --cuda 12.1)"
    echo -e ""
fi

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths needed for installation and check for conda installation
install_dir=$(pwd)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

# Check if frustraMPNN environment already exists
if conda env list | grep -w 'frustraMPNN' >/dev/null 2>&1; then
    if [ "$reinstall" = true ]; then
        echo -e "[!]  frustraMPNN environment already exists, but reinstall requested."
        echo -e "Removing existing environment..."
        conda env remove -n frustraMPNN -y || { echo -e "Error: Failed to remove existing frustraMPNN environment"; exit 1; }
        echo -e "[o] Existing environment removed successfully"
    else
        echo -e "[!]  frustraMPNN environment already exists!"
        echo -e "Skipping installation and using existing environment."
        echo -e "To reinstall, use: $0 --reinstall"
        echo -e ""
        echo -e "Testing existing environment..."
        
        # Test the existing environment with full test suite
        source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/frustraMPNN || { echo -e "Error: Failed to activate existing frustraMPNN environment."; exit 1; }
        
        # Run the same comprehensive tests as for fresh installation
        echo -e "Testing package imports..."
        python -c "
import sys
import torch
import numpy as np
import pandas as pd
import Bio
import scipy
import sklearn
import tqdm
import matplotlib
import seaborn
import pytorch_lightning
import omegaconf
import wandb
import pyrosetta
import frustrapy
print('[o] All packages imported successfully')
" || { echo -e "Error: Package import test failed"; exit 1; }

        # Test PyTorch CUDA availability
        echo -e "Testing PyTorch CUDA functionality..."
        python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    # Test tensor operations on GPU
    device = torch.device('cuda')
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.mm(x, y)
    print('[o] GPU tensor operations working')
else:
    print('[!] CUDA not available - CPU-only installation')
    # Test CPU operations
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    z = torch.mm(x, y)
    print('[o] CPU tensor operations working')
"

        # Test ESM model loading
        echo -e "Testing ESM model import..."
        python -c "
import esm
print('[o] ESM model import successful')
" || { echo -e "Warning: ESM model test failed - this may affect some functionality"; }

        # Test PyRosetta initialization
        echo -e "Testing PyRosetta initialization..."
        python -c "
import pyrosetta
pyrosetta.init('-mute all')
print('[o] PyRosetta initialization successful')
" || { echo -e "Warning: PyRosetta initialization failed - this may affect some functionality"; }

        # Test frustrapy functionality
        echo -e "Testing frustrapy functionality..."
        python -c "
import frustrapy
print('[o] frustrapy functionality test successful')
" || { echo -e "Warning: frustrapy test failed - this may affect some functionality"; }

        # Test Bio.PDB import
        echo -e "Testing Bio.PDB.PDBParser import..."
        python -c "
from Bio.PDB import PDBParser
print('[o] Bio.PDB.PDBParser import successful')
" || { echo -e "Warning: Bio.PDB.PDBParser test failed - this may affect some functionality"; }

        echo -e "\n=== Environment Test Summary ==="
        python -c "
import torch
if torch.cuda.is_available():
    print('GPU support: [o] AVAILABLE')
    print(f'CUDA devices detected: {torch.cuda.device_count()}')
else:
    print('GPU support: [!] NOT AVAILABLE (CPU-only)')
"
        
        conda deactivate
        echo -e "[o] Existing frustraMPNN environment is functional"
        echo -e "Activate environment using command: \"$pkg_manager activate frustraMPNN\""
        exit 0
    fi
fi

### frustraMPNN install begin, create base environment
echo -e "Installing frustraMPNN environment\n"
$pkg_manager create --name frustraMPNN python=3.10 -y || { echo -e "Error: Failed to create frustraMPNN conda environment"; exit 1; }
conda env list | grep -w 'frustraMPNN' >/dev/null 2>&1 || { echo -e "Error: Conda environment 'frustraMPNN' does not exist after creation."; exit 1; }

# Load newly created frustraMPNN environment
echo -e "Loading frustraMPNN environment\n"
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/frustraMPNN || { echo -e "Error: Failed to activate the frustraMPNN environment."; exit 1; }
[ "$CONDA_DEFAULT_ENV" = "frustraMPNN" ] || { echo -e "Error: The frustraMPNN environment is not active."; exit 1; }
echo -e "frustraMPNN environment activated at ${CONDA_BASE}/envs/frustraMPNN"

# install required conda packages
echo -e "Installing conda requirements\n"
if [ -n "$cuda" ]; then
    echo -e "Installing PyTorch with CUDA $cuda support..."
    CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install -y numpy=1.26 pytorch=2.2.2 torchvision torchaudio pytorch-cuda=$cuda -c pytorch -c nvidia || { echo -e "Error: Failed to install PyTorch with CUDA."; exit 1; }
else
    echo -e "Installing PyTorch with automatic hardware detection..."
    $pkg_manager install -y numpy=1.26 pytorch=2.2.2 torchvision torchaudio pytorch-cuda -c pytorch -c nvidia || { echo -e "Error: Failed to install PyTorch."; exit 1; }
fi

# Install additional conda packages
$pkg_manager install -y -c conda-forge -c defaults \
    biopython=1.85 pandas scipy scikit-learn tqdm matplotlib seaborn \
    pytorch-lightning omegaconf wandb || { echo -e "Error: Failed to install conda packages."; exit 1; }

# Install pip packages
echo -e "Installing pip requirements\n"
pip install fair-esm==2.0.0 absl-py grpcio markdown tensorboard werkzeug || { echo -e "Error: Failed to install pip packages."; exit 1; }

# make sure all required packages were installed
required_packages=(numpy pytorch torchvision torchaudio biopython pandas scipy scikit-learn tqdm matplotlib seaborn pytorch-lightning omegaconf wandb)
missing_packages=()

# Check each package
for pkg in "${required_packages[@]}"; do
    conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

# Install PyRosetta
echo -e "Installing PyRosetta\n"
pip install pyrosetta-installer || { echo -e "Error: Failed to install pyrosetta-installer"; exit 1; }
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()' || { echo -e "Error: Failed to install PyRosetta"; exit 1; }
python -c "import pyrosetta" >/dev/null 2>&1 || { echo -e "Error: pyrosetta module not found after installation"; exit 1; }

# Install frustrapy dependencies
echo -e "Installing frustrapy\n"
frustrapy_dir="${install_dir}/frustrapy"
if [ -d "$frustrapy_dir" ]; then
    echo -e "Removing existing frustrapy directory"
    rm -rf "$frustrapy_dir"
fi

git clone https://github.com/Bloeci/frustrapy.git "$frustrapy_dir" || { echo -e "Error: Failed to clone frustrapy repository"; exit 1; }
cd "$frustrapy_dir" || { echo -e "Error: Failed to change to frustrapy directory"; exit 1; }
pip install -r requirements.txt || { echo -e "Error: Failed to install frustrapy requirements"; exit 1; }
pip install -e . || { echo -e "Error: Failed to install frustrapy"; exit 1; }
cd "$install_dir" || { echo -e "Error: Failed to return to install directory"; exit 1; }
python -c "import frustrapy" >/dev/null 2>&1 || { echo -e "Error: frustrapy module not found after installation"; exit 1; }

############################################################################################################
############################################################################################################
################## testing installation
echo -e "Testing frustraMPNN installation\n"

# Reactivate environment for testing
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/frustraMPNN || { echo -e "Error: Failed to reactivate the frustraMPNN environment for testing."; exit 1; }

# Test Python packages
echo -e "Testing package imports..."
python -c "
import sys
import torch
import numpy as np
import pandas as pd
import Bio
import scipy
import sklearn
import tqdm
import matplotlib
import seaborn
import pytorch_lightning
import omegaconf
import wandb
import pyrosetta
import frustrapy
print('[o] All packages imported successfully')
" || { echo -e "Error: Package import test failed"; exit 1; }

# Test PyTorch CUDA availability
echo -e "Testing PyTorch CUDA functionality..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    # Test tensor operations on GPU
    device = torch.device('cuda')
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.mm(x, y)
    print('[o] GPU tensor operations working')
else:
    print('[!] CUDA not available - CPU-only installation')
    # Test CPU operations
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    z = torch.mm(x, y)
    print('[o] CPU tensor operations working')
"

# Test ESM model loading
echo -e "Testing ESM model import..."
python -c "
import esm
print('[o] ESM model import successful')
" || { echo -e "Warning: ESM model test failed - this may affect some functionality"; }

# Test PyRosetta initialization
echo -e "Testing PyRosetta initialization..."
python -c "
import pyrosetta
pyrosetta.init('-mute all')
print('[o] PyRosetta initialization successful')
" || { echo -e "Warning: PyRosetta initialization failed - this may affect some functionality"; }

# Test frustrapy functionality
echo -e "Testing frustrapy functionality..."
python -c "
import frustrapy
print('[o] frustrapy functionality test successful')
" || { echo -e "Warning: frustrapy test failed - this may affect some functionality"; }

# Test Bio.PDB import
echo -e "Testing Bio.PDB.PDBParser import..."
python -c "
from Bio.PDB import PDBParser
print('[o] Bio.PDB.PDBParser import successful')
" || { echo -e "Warning: Bio.PDB.PDBParser test failed - this may affect some functionality"; }

echo -e "\n=== Installation Test Summary ==="
if [ -n "$cuda" ]; then
    echo -e "CUDA installation requested: YES (version: $cuda)"
else
    echo -e "CUDA installation requested: NO"
fi

python -c "
import torch
if torch.cuda.is_available():
    print('GPU support: [o] AVAILABLE')
    print(f'CUDA devices detected: {torch.cuda.device_count()}')
else:
    print('GPU support: [!] NOT AVAILABLE (CPU-only)')
"

# finish
conda deactivate
echo -e "frustraMPNN environment set up and tested\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
echo -e "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS 
echo -e "Successfully finished frustraMPNN installation and testing!\n"
echo -e "Activate environment using command: \"$pkg_manager activate frustraMPNN\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
