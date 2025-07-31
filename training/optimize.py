#!/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datasets import FireProtDataset
from transfer_model import TransferModel
from train_thermompnn_refac import TransferModelPL, get_datasets

import optuna
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# GLOBAL
HYDRA_CONFIG_PATH='.'
HYDRA_CONFIG_NAME='fireprot_hydra_config_optuna.yaml'

# Main function
@hydra.main(version_base=None, config_name=HYDRA_CONFIG_NAME, config_path=HYDRA_CONFIG_PATH)
def main(conf: HydraConfig) -> None:
    logger = CSVLogger(conf.data_loc.log_dir, name=conf.name)
    
    
    
    
    
    

if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()