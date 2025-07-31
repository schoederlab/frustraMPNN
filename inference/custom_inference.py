import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from utils import Mutation
from protein_mpnn_utils import alt_parse_PDB
from utils import ALPHABET
from utils import get_chains, get_ssm_mutations, get_trained_model
from utils import TransferModelPL, get_esm_model


def main(cfg, args):

    # define config for model loading
    config = {
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
            'epochs': 100,
            'lr_schedule': True,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': False,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'lr_schedule': True,
        },
        'platform': {
            'accel': 'gpu',
            'cache_dir': 'cache',
            'thermompnn_dir': os.path.dirname(os.path.abspath(__file__))
        },
        'data_loc': {
            'weights_dir': os.path.dirname(os.path.abspath(__file__)),
            #'torch_hub': os.path.dirname(os.path.abspath(__file__))
        }
    }

    cfg = OmegaConf.merge(config, cfg)
    # print(cfg)

    # load the chosen model and dataset
    models = {
        "ThermoMPNN": get_trained_model(model_name=args.model_path,
                                        config=cfg, override_custom=True)
    }
    
    # TESTING ESM loading
    # if cfg.training.add_esm_embeddings:    
    #     thermo_model = TransferModelPL.load_from_checkpoint(args.model_path, cfg=cfg, strict=True)
    #     # esm_model = get_esm_model(cfg.training.esm_model)
        
    #     exit()
    
    # ==============
    
    
    exit()

    input_pdb = args.pdb
    pdb_id = os.path.basename(input_pdb).rstrip('.pdb')

    datasets = {
        pdb_id: args.pdb
    }

    raw_pred_df = pd.DataFrame(columns=['Model', 'Dataset', 'frustration_pred', 'position', 'wildtype', 'mutation',])
    row = 0
    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            
            if len(args.chains) < 1:  # if unspecified, take first chain
                chains = get_chains(input_pdb)
            else:
                chains = args.chains.split(',')
            
            print(f"Chains to use: {chains}")
            for chain in chains:
                mut_pdb = alt_parse_PDB(input_pdb, chains)
                mutation_list = get_ssm_mutations(mut_pdb[0])
                final_mutation_list = []

                # build into list of Mutation objects
                for n, m in enumerate(mutation_list):
                    if m is None:
                        final_mutation_list.append(None)
                        continue
                    m = m.strip()  # clear whitespace
                    wtAA, position, mutAA = str(m[0]), int(str(m[1:-1])), str(m[-1])

                    assert wtAA in ALPHABET, f"Wild type residue {wtAA} invalid, please try again with one of the following options: {ALPHABET}"
                    assert mutAA in ALPHABET, f"Wild type residue {mutAA} invalid, please try again with one of the following options: {ALPHABET}"
                    mutation_obj = Mutation(position=position, wildtype=wtAA, mutation=mutAA,
                                            ddG=None, pdb=mut_pdb[0]['name'])
                    final_mutation_list.append(mutation_obj)

                pred, _ = model(mut_pdb, final_mutation_list)

                for mut, out in zip(final_mutation_list, pred):
                    if mut is not None:
                        col_list = ['frustration_pred', 'position', 'wildtype', 'mutation', 'pdb', 'chain']
                        val_list = [out["frustration"].cpu().item(), mut.position, mut.wildtype,
                                    mut.mutation, mut.pdb.strip('.pdb'), chain]
                        for col, val in zip(col_list, val_list):
                            raw_pred_df.loc[row, col] = val

                        raw_pred_df.loc[row, 'Model'] = name
                        raw_pred_df.loc[row, 'Dataset'] = dataset_name
                        row += 1

    print(raw_pred_df)
    if not args.output:
        raw_pred_df.to_csv("ThermoMPNN_inference_%s.csv" % pdb_id)
    else:
        raw_pred_df.to_csv(args.output)        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default='', help='Input PDB to use for custom inference')
    parser.add_argument('--chains', type=str, default='', help='Chains in input PDB to use. If not specified, will use all chains. Format: A,B,C')
    parser.add_argument('--model_path', type=str, default='', help='filepath to model to use for inference')
    parser.add_argument('--output', type=str, default='', help='Output file for inference results')

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cfg = OmegaConf.load(os.path.join(script_dir, './config.yaml'))
    # print(cfg)
    with torch.no_grad():
        # main(cfg, args)
        main(args)
    
