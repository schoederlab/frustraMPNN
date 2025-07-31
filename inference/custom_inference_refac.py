import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from utils import Mutation
from protein_mpnn_utils import alt_parse_PDB
from utils import ALPHABET
from utils import get_chains, get_ssm_mutations
from transfer_model_pl import TransferModelPL
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*")


def main(args):
    # load the chosen model and dataset
    checkpoint = args.model_path
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file {checkpoint} does not exist.")

    if args.old_model:
        cfg = OmegaConf.load(args.config)
        cfg.platform.thermompnn_dir = os.path.dirname(os.path.abspath(__file__))
        if args.config_args:
            config_args = args.config_args.split(';')
            for arg in config_args:
                key, value = arg.split('=')
                if value in ['false', '0', 'False']:
                    value = False
                elif value in ['true', '1', 'True']:
                    value = True 
                OmegaConf.update(cfg, key, value)
            
        print("Loading old model format...")
        models = {
            'model': TransferModelPL.load_from_checkpoint(checkpoint, cfg=cfg, strict=False).model
        } 
        
    else:
        print("Loading model from checkpoint...")
        model_dict = torch.load(checkpoint, weights_only=False)
        print(model_dict.keys())
        cfg = OmegaConf.create(model_dict['cfg'])
        cfg.platform.thermompnn_dir = os.path.dirname(os.path.abspath(__file__))
        # print(cfg)

        models = {
            'model': TransferModelPL.load_from_checkpoint(checkpoint, cfg=cfg, strict=False).model
        }


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
    parser.add_argument('--old_model', action='store_true', help='Use old model format (deprecated)')
    parser.add_argument('--config', type=str, default='', help='Path to config file (deprecated). Only used for old model format.')
    parser.add_argument('--config_args', type=str, default='', help='Config args separated by commas. Only used for old model format.')

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # cfg = OmegaConf.load(os.path.join(script_dir, './config.yaml'))
    # print(cfg)
    with torch.no_grad():
        # main(cfg, args)
        main(args)
    
