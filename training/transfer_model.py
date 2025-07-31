import torch
import torch.nn as nn
from protein_mpnn_utils import ProteinMPNN, tied_featurize
from model_utils import featurize
import os
import numpy as np
import esm


HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

MLP = True
SUBTRACT_MUT = True

def myprint(x, name):
    print(50 * '-')
    print(name)
    print(x)
    print(x.shape)
    print(50 * '-')


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


def get_protein_mpnn(cfg, version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    hidden_dim = 128
    num_layers = 3 

    model_weight_dir = os.path.join(cfg.platform.thermompnn_dir, 'vanilla_model_weights')
    checkpoint_path = os.path.join(model_weight_dir, version)
    # checkpoint_path = "vanilla_model_weights/v_48_020.pt"
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


alpha_1 = list("ACDEFGHIKLMNPQRSTVWYX")
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x);
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]


class TransferModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        self.num_final_layers = cfg.model.num_final_layers
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'
        
        self.prot_mpnn = get_protein_mpnn(cfg)
        EMBED_DIM = 128
        HIDDEN_DIM = 128

        
        # If esm embeddings should be used, load it
        self.esm_attention = cfg.training.add_esm_embeddings
        print('ESM ATTENTION:', self.esm_attention)
        self.esm_model = None
        self.batch_converter = None
        self.embed_dim = None
        if cfg.training.add_esm_embeddings:
            print(f'Loading ESM model: {cfg.training.esm_model}')
            esm_model, alphabet = get_esm_model(cfg.training.esm_model)
            batch_converter = alphabet.get_batch_converter()
            esm_model.cuda()
            esm_model.eval()
            for param in esm_model.parameters():
                param.requires_grad = False
                
            self.esm_model = esm_model
            self.esm_layers = len(esm_model.layers)
            # print('Loaded ESM model on device:', next(esm_model.parameters()).device)
            self.batch_converter = batch_converter
            
            self.embed_dim = esm_model.embed_dim
        
        if cfg.training.add_esm_embeddings:
            hid_sizes = [ HIDDEN_DIM * self.num_final_layers + EMBED_DIM + self.embed_dim ]
        else:
            hid_sizes = [ HIDDEN_DIM * self.num_final_layers + EMBED_DIM ]
        
            
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]
        

        if self.lightattn:
            print('Enabled LightAttention')
            self.light_attention_mpnn = LightAttention(
                embeddings_dim = HIDDEN_DIM * self.num_final_layers + EMBED_DIM
            )
            
        if cfg.training.add_esm_embeddings:
            self.light_attention_esm = LightAttention(
                embeddings_dim = self.embed_dim
            )

        self.both_out = nn.Sequential()
        print('MLP HIDDEN SIZES:', hid_sizes)
        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            # print(sz1, sz2)
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.frustration_out = nn.Linear(1, 1)
    
    
        alpha_1 = list("ACDEFGHIKLMNPQRSTVWYX")
        self.aa_N_1 = {n:a for n,a in enumerate(alpha_1)}


    def _N_to_AA(self, x):
        # [[0,1,2,3]] -> ["ARND"]
        if x.ndim == 1: x = x[None]
        return ["".join([self.aa_N_1.get(a,"-") for a in y]) for y in x]
    
    def get_esm_embeddings(self, S):
        data = [
            (f'seq_{i}', self._N_to_AA(s)[0]) for i, s in enumerate(S)
        ]
        _, _, toks = self.batch_converter(data)
        toks = toks.to(next(self.parameters()).device)
        
        with torch.no_grad():
            results = self.esm_model(toks, repr_layers=[self.esm_layers], return_contacts=False)
        
        # print(type(results))
        return results['representations'][self.esm_layers]
    

    def forward(self, pdb, mutations, tied_feat=True):        
        device = next(self.parameters()).device

        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        # getting ProteinMPNN structure embeddings
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
        # print('MPNN EMBED SHAPE:', mpnn_embed[0])
        # print('MPNN HID SHAPE:', all_mpnn_hid[0])
        exit()
        
        if self.num_final_layers > 0:
            mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)

        # Get full sequence esm embeddings
        if self.esm_attention:
            esm_embed = self.get_esm_embeddings(S)

        out = []
        for mut in mutations:
            inputs = []
            if mut is None:
                out.append(None)
                continue

            aa_index = ALPHABET.index(mut.mutation)
            wt_aa_index = ALPHABET.index(mut.wildtype)


            # print(pdb, mut.position, mut.wildtype, mut.mutation)
            if self.num_final_layers > 0:
                hid = mpnn_hid[0][mut.position]  # MPNN hidden embeddings at mut position
                inputs.append(hid)

            embed = mpnn_embed[0][mut.position]  # MPNN seq embeddings at mut position
            # print('MPNN EMBED:', embed.shape)
            inputs.append(embed)

            # concatenating hidden layers and embeddings
            lin_input = torch.cat(inputs, -1)

            # passing vector through lightattn
            if self.lightattn:
                lin_input = torch.unsqueeze(torch.unsqueeze(lin_input, -1), 0)
                # myprint(lin_input, 'LIN INPUT')
                lin_input = self.light_attention_mpnn(lin_input, mask)

            if self.esm_attention:
                esm_pos_embed = esm_embed[0][mut.position + 1] # +1 for <start> token ?
                esm_pos_embed = torch.unsqueeze(torch.unsqueeze(esm_pos_embed, -1), 0)
                esm_pos_embed = self.light_attention_esm(esm_pos_embed, mask)
                lin_input = torch.cat([lin_input, esm_pos_embed], -1)
            
            # myprint(lin_input, 'LIN INPUT')

            both_input = torch.unsqueeze(self.both_out(lin_input), -1)
            # print('BOTH INPUT:', both_input)
            frustration_out = self.frustration_out(both_input)
            # print('FRUSTRATION OUT:', frustration_out)


            if self.subtract_mut:
                frustration = frustration_out[aa_index][0] - frustration_out[wt_aa_index][0]
            else:
                frustration = frustration_out[aa_index][0]

            out.append({
                "frustration": torch.unsqueeze(frustration, 0),
            })
            
            
        return out, None


class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)
