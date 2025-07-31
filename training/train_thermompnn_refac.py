#!/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

from transfer_model import TransferModel
from datasets import FireProtDataset, MegaScaleDataset, ComboDataset
from argparse import ArgumentParser

import esm

def get_esm_model(esm_model_name):
    if esm_model_name == "esm2_t48_15B_UR50D":
        return esm.pretrained.esm2_t48_15B_UR50D()
    elif esm_model_name == "esm2_t36_3B_UR50D":
        return esm.pretrained.esm2_t36_3B_UR50D()
    elif esm_model_name == "esm2_t33_650M_UR50D":
        return esm.pretrained.esm2_t33_650M_UR50D()
    elif esm_model_name == "esm2_t30_150M_UR50D":
        return esm.pretrained.esm2_t30_150M_UR50D()
    elif esm_model_name == "esm2_t12_35M_UR50D":
        return esm.pretrained.esm2_t12_35M_UR50D()
    elif esm_model_name == "esm2_t6_8M_UR50D":
        return esm.pretrained.esm2_t6_8M_UR50D()
    elif esm_model_name == "esm1_t34_670M_UR50S":
        return esm.pretrained.esm1_t34_670M_UR50S()
    elif esm_model_name == "esm1_t34_670M_UR50D":
        return esm.pretrained.esm1_t34_670M_UR50D()
    elif esm_model_name == "esm1_t34_670M_UR100":
        return esm.pretrained.esm1_t34_670M_UR100()
    elif esm_model_name == "esm1_t12_85M_UR50S":
        return esm.pretrained.esm1_t12_85M_UR50S()
    elif esm_model_name == "esm1_t6_43M_UR50S":
        return esm.pretrained.esm1_t6_43M_UR50S()
    else:
        raise ValueError(f"Unknown ESM model name: {esm_model_name}")
            
    
def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


class TransferModelPL(pl.LightningModule):
    stage : int = 1
    
    """Class managing training loop with pytorch lightning"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TransferModel(cfg)

        self.learn_rate = cfg.training.learn_rate
        self.mpnn_learn_rate = cfg.training.mpnn_learn_rate if 'mpnn_learn_rate' in cfg.training else None
        self.lr_schedule = cfg.training.lr_schedule if 'lr_schedule' in cfg.training else False

        # set up metrics dictionary
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics", "test_metrics"):
            self.metrics[split] = nn.ModuleDict()
            out = "frustration"
            self.metrics[split][out] = nn.ModuleDict()
            for name, metric in get_metrics().items():
                self.metrics[split][out][name] = metric
        
        self.reweighting_loss = cfg.training.reweighting
        self.weight_method = cfg.training.weight_method
        # self.loss = cfg.training.loss

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):

        assert len(batch) == 1
        mut_pdb, mutations = batch[0]
        pred, _ = self(mut_pdb, mutations)

        frustration_mses = []
        if self.reweighting_loss:
            weight_sum = sum([mut.weight for mut in mutations])
            # print(f'Weight sum: {weight_sum}')
        
        for mut, out in zip(mutations, pred):
            if mut.frustration is not None:
                # Reweight loss if specified
                if self.reweighting_loss:
                    weight = mut.weight
                    loss = F.mse_loss(out["frustration"], mut.frustration)
                    loss = loss * (weight / weight_sum)
                else:
                    loss = F.mse_loss(out["frustration"], mut.frustration)
                
                frustration_mses.append(loss)
                for metric in self.metrics[f"{prefix}_metrics"]["frustration"].values():
                    metric.update(out["frustration"], mut.frustration)  
        

        loss = 0.0 if len(frustration_mses) == 0 else torch.stack(frustration_mses).mean()
        on_step = False
        on_epoch = not on_step

        output = "frustration"
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                        batch_size=len(batch))
        if loss == 0.0:
            return None
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        if self.stage == 2: # for second stage, drop LR by factor of 10
            self.learn_rate /= 10.
            print('New second-stage learning rate: ', self.learn_rate)

        if not self.cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            param_list = [
                {"params": self.model.prot_mpnn.parameters(), 
                 "lr": self.mpnn_learn_rate}
            ]
        else: # fully frozen MPNN
            param_list = []

        if self.model.lightattn:  # adding light attention parameters
            if self.stage == 2:
                param_list.append({"params": self.model.light_attention.parameters(), "lr": 0.})
            else:
                param_list.append({"params": self.model.light_attention_mpnn.parameters()})
                if self.cfg.training.add_esm_embeddings:
                    param_list.append({"params": self.model.light_attention_esm.parameters()})


        mlp_params = [
            {"params": self.model.both_out.parameters()},
            {"params": self.model.frustration_out.parameters()}
        ]

        param_list = param_list + mlp_params
        # print(param_list)
        
        
        opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

        if self.lr_schedule: # enable additional lr scheduler conditioned on val frustration mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, verbose=True, mode='min', factor=0.5)
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': 'val_frustration_mse'
            }
        else:
            return opt


    def on_save_checkpoint(self, checkpoint):
        # Override the default checkpoint behavior
        # Store only trainable parameters
        checkpoint['state_dict'] = {
            k: v for k, v in self.state_dict().items()
            if k in dict(self.named_parameters()) and self.get_parameter(k).requires_grad
        }
        checkpoint['cfg'] = self.cfg

    def on_load_checkpoint(self, checkpoint):
        # Load only the trainable parameters
        # The non-trainable ones will keep their initialization from the pretrained model
        # Get checkpoint and model keys
        state_dict_checkpoint = set(checkpoint['state_dict'].keys())
        state_dict_model = set(self.state_dict().keys())
        
        # Find missing keys
        missing_keys = state_dict_model - state_dict_checkpoint
        
        # Silently fill in missing keys without verbose output
        if missing_keys:
            # Optional: Add a single log line instead of listing all keys
            # print(f"Adding {len(missing_keys)} missing keys from model to checkpoint")
            
            # Add missing keys from current model state to checkpoint
            for key in missing_keys:
                checkpoint['state_dict'][key] = self.state_dict()[key]

def get_datasets(cfg):
    """Load datasets from config file"""
    if cfg.datasets == 'fireprot':
        train = FireProtDataset(cfg, split='train')
        val = FireProtDataset(cfg, split='val')
        test = FireProtDataset(cfg, split='test')
    elif cfg.datasets == 'megascale':
        train = MegaScaleDataset(cfg, split='train')
        val = MegaScaleDataset(cfg, split='val')
        test = MegaScaleDataset(cfg, split='test')
    elif cfg.datasets == 'combo':
        train = ComboDataset(cfg, split='train')
        val = ComboDataset(cfg, split='val')
        test = ComboDataset(cfg, split='test')
    else:
        raise ValueError(f"Unknown dataset: {cfg.datasets}")
        exit(1)
    
    return train, val, test


def main(cfg, logger):
    print(f'[INFO] Starting training with config: {cfg}')
    
    torch.cuda.empty_cache()
    # ==========================
    # Setup number of workers for dataloader
    num_workers = cfg.training.num_workers
    train_workers = max(2, int(num_workers * 0.75)) if num_workers else 10
    val_workers = max(1, int(num_workers * 0.25)) if num_workers else 10
    print(f"[INFO] Using {train_workers} workers for training and {val_workers} workers for validation.")
    
    # Setup multiple GPUs if available
    if cfg.training.ddp:
        all_gpus = torch.cuda.device_count()
        strategy = 'ddp'
        print(f'[INFO] Using shared data distribution with n={all_gpus} GPUs.')
    else:
        all_gpus = 1
        strategy = 'auto'
        print(f'[INFO] Using single GPU training.')
    # ==========================
    
    # Get datasets for current run
    train_dataset, val_dataset, test_dataset = get_datasets(cfg)
    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
    val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)
    
    # Setup model
    if cfg.training.add_esm_embeddings:
        hub = cfg.data_loc.torch_hub
        if os.path.exists(hub):
            torch.hub.set_dir(hub)
        else:
            print(f'{hub} dont exits -> Creating torch hub directory. Downloading ESM model if needed.')
            os.makedirs(hub, exist_ok=True)
            torch.hub.set_dir(hub)
            get_esm_model(cfg.model.esm_model_name)
        print(f'[INFO] Using ESM embeddings from {hub} with model: {cfg.training.esm_model}')
    else:
        print('[INFO] No ESM embeddings specified.')

    # Check if weighting method is specified and available
    if cfg.training.reweighting:
        weighing_methods = ['weight_bin_inverse', 'weight_lds_inverse', 
                            'weight_bin_inverse_sqrt', 'weight_lds_inverse_sqrt']
        if cfg.training.weight_method not in weighing_methods:
            raise ValueError(f"Unknown reweighting method: {cfg.training.reweighting}")
        print(f'[INFO] Using reweighting method: {cfg.training.weight_method}')
    else:
        print('[INFO] No reweighting method specified.')

    # --------------------------
    # Setup model and trainer

    model = TransferModelPL(cfg)
    filename = cfg.name + '_{epoch:02d}_{val_frustration_spearman:.02}'
    monitor = 'val_frustration_spearman'
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor, 
        mode='max', 
        dirpath=cfg.data_loc.weights_dir, 
        filename=filename
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback], 
        logger=logger, 
        log_every_n_steps=10, 
        max_epochs=cfg.training.epochs,
        accelerator=cfg.platform.accel, 
        devices=all_gpus, 
        strategy=strategy
    )
    try:
        trainer.fit(model, train_loader, val_loader)
    except KeyboardInterrupt:
        print('\n\nTraining interrupted')


    state_dict = torch.load(checkpoint_callback.best_model_path)


    if cfg.training.testing:
        print('+--------------------------------------+\n'*2)
        print('Enter testing mode')
        test_loader = DataLoader(test_dataset, collate_fn=lambda x: x, num_workers=val_workers)
        trainer.test(model, test_loader)
        print('Testing complete')
        exit()


if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument('--config', type=str, required=True, 
    #                     help='Path to config file for the datasets and model parameters')
    # parser.add_argument('--checkpoint', type=str, default=None)
    # args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')

    config = sys.argv[1]
    # Load and merge config files
    cfg = OmegaConf.load(config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    # Seeded runs
    if cfg.training.seed is not None:
        print(f'[INFO] Running training with seed: {cfg.training.seed}')
        import random
        import numpy as np
        torch.manual_seed(cfg.training.seed)
        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
    else:
        print('[INFO] Running training without seed.')
    
    if cfg.platform.use_tpu:
        torch.set_float32_matmul_precision('medium')

    # Set up logging
    if cfg.logger == 'wandb':
        wandb.init(project=cfg.project, name=cfg.name)
        logger = WandbLogger(
            project=cfg.project, 
            name=cfg.name, 
            log_model=False, # log_model == True, checkpoints are logged at the end of training
        )
    elif cfg.logger == 'csv':
        print("[INFO] Using csv logger")
        logger = CSVLogger(cfg.data_loc.log_dir, name=cfg.name)
    else:
        logger = None
        print("No logger specified. Logging to stdout.")


    main(cfg, logger)
