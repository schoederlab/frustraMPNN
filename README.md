# TODO

- [ ] Find better structure for building docker
  - [ ] need to copy the correct files
- [ ] Add details about training and inference
  - [ ] test scripts for training
  - [ ] example command docker / singularity / local training
- [ ] 


# frustraMPNN

## Installation

First you need to clone this repository. Replace [install_folder] with the path where you want to install it.

```bash
git clone https://github.com/schoederlab/frustraMPNN [install_folder]
```

Then navigate into your install folder using cd and run the installation code. frustraMPNN requires a CUDA-compatible Nvidia graphics card to run. In the cuda setting, please specify the CUDA version compatible with your graphics card, for example '12.1'. If unsure, leave blank but it's possible that the installation might select the wrong version, which will lead to errors. In pkg_manager specify whether you are using 'mamba' or 'conda', if left blank it will use 'conda' by default.

Note: This install script will install PyRosetta, which requires a license for commercial purposes. The installation requires about 2 Gb of storage space for the environment and dependencies.

```bash
bash install_frustraMPNN.sh --cuda '12.1' --pkg_manager 'conda'
```