import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import esm
from protein_mpnn_utils import tied_featurize
from Bio.PDB import PDBParser
from torchmetrics import R2Score, MeanSquaredError, SpearmanCorrCoef


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21


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


def get_chains(pdb):
    """Get chains from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb)
    chains = [c.id for c in structure.get_chains()]
    return chains


def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


def parse_pdb(pdb_file, chain_id=None):
    """Parse a PDB file and return all residues of the specified chain.
    
    Args:
        pdb_file: Path to PDB file
        chain_id: Chain ID to extract (if None, use first chain)
        
    Returns:
        List containing dictionary with sequence and coordinate information
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_file)
    
    # Three-letter to one-letter amino acid code mapping
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'MSE': 'M'  # Selenomethionine is treated as methionine
    }
    
    for chain in structure.get_chains():
        if chain_id is None or chain.id == chain_id:
            # Process this chain
            residue_list = []
            seq = ""
            coords = []
            resn_list = []
            
            for residue in chain:
                # Filter out hetero-atoms and water
                if residue.id[0] == " " and residue.get_resname() != "HOH":
                    try:
                        # Get amino acid code and add to sequence
                        res_name = residue.get_resname()
                        aa = three_to_one.get(res_name)
                        if aa:
                            seq += aa
                            resn_list.append(str(residue.id[1]))  # Residue number
                            
                            # Get atom coordinates
                            residue_coords = []
                            for atom in residue:
                                residue_coords.append(atom.get_coord().tolist())
                            coords.append(residue_coords)
                    except Exception as e:
                        print(f"Error processing residue {residue.id}: {e}")
                        continue
            
            if len(seq) > 0:
                # Create the output dictionary
                dict_out = {}
                dict_out['seq'] = [seq]
                dict_out[f'seq_chain_{chain.id}'] = seq
                dict_out[f'coords_chain_{chain.id}'] = coords
                dict_out['resn_list'] = resn_list
                dict_out['name'] = os.path.basename(pdb_file).split('.')[0]
                dict_out['num_of_chains'] = 1
                
                return [dict_out]
    
    # If no valid chains were found
    return []


def get_ssm_mutations(pdb):
        # make mutation list for SSM run
    mutation_list = []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            # add each mutation option
            for mutAA in ALPHABET[:-1]:
                mutation_list.append(wtAA + str(seq_pos) + mutAA)
        else:
            mutation_list.append(None)

    return mutation_list


class Mutation:
    """Mutation class to store information about a specific mutation."""
    def __init__(self, position, wildtype, mutation, ddG=None, pdb=None):
        self.position = position
        self.wildtype = wildtype
        self.mutation = mutation
        self.ddG = ddG
        self.pdb = pdb
    
    def __str__(self):
        return f"{self.wildtype}{self.position}{self.mutation}"


def get_protein_mpnn(cfg, version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    from protein_mpnn_utils import ProteinMPNN
    
    hidden_dim = 128
    num_layers = 3 

    model_weight_dir = os.path.join(cfg.platform.thermompnn_dir, 'vanilla_model_weights')
    checkpoint_path = os.path.join(model_weight_dir, version)
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, k_neighbors=checkpoint['num_edges'], 
                        augment_eps=0.0)
    
    if cfg.model.load_pretrained:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if cfg.model.freeze_weights:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model
