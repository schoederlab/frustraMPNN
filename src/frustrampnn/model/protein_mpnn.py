"""
ProteinMPNN model architecture.

This module contains the main ProteinMPNN message-passing neural network
for protein sequence design and structure-conditioned sequence modeling.

Original source: ProteinMPNN by Dauparas et al. (2022)
https://github.com/dauparas/ProteinMPNN
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from frustrampnn.model.features import CA_ProteinFeatures, ProteinFeatures
from frustrampnn.model.layers import DecLayer, EncLayer, cat_neighbors_nodes, gather_nodes

__all__ = [
    "ProteinMPNN",
    "loss_nll",
    "loss_smoothed",
    "_scores",
    "_S_to_seq",
]


def _scores(S, log_probs, mask):
    """Compute negative log probability scores.

    Args:
        S: Sequence tensor
        log_probs: Log probabilities
        mask: Mask tensor

    Returns:
        Scores tensor
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def _S_to_seq(S, mask):
    """Convert sequence tensor to string.

    Args:
        S: Sequence tensor
        mask: Mask tensor

    Returns:
        Sequence string
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
    return seq


def loss_nll(S, log_probs, mask):
    """Compute negative log likelihood loss.

    Args:
        S: Sequence tensor
        log_probs: Log probabilities
        mask: Mask tensor

    Returns:
        tuple: (loss tensor, average loss)
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """Compute label-smoothed negative log likelihood loss.

    Args:
        S: Sequence tensor
        log_probs: Log probabilities
        mask: Mask tensor
        weight: Smoothing weight

    Returns:
        tuple: (loss tensor, average loss)
    """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


class ProteinMPNN(nn.Module):
    """ProteinMPNN message-passing neural network for protein sequence design."""

    def __init__(self, num_letters, node_features, edge_features,
                 hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False):
        """Initialize ProteinMPNN.

        Args:
            num_letters: Number of output letters (amino acids)
            node_features: Dimension of node features
            edge_features: Dimension of edge features
            hidden_dim: Hidden dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            vocab: Vocabulary size
            k_neighbors: Number of nearest neighbors
            augment_eps: Noise augmentation epsilon
            dropout: Dropout rate
            ca_only: If True, use CA-only features
        """
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        if ca_only:
            self.features = CA_ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn,
                use_input_decoding_order=False, decoding_order=None):
        """Forward pass for graph-conditioned sequence model.

        Args:
            X: Coordinates tensor [B, L, 4, 3]
            S: Sequence tensor [B, L]
            mask: Mask tensor [B, L]
            chain_M: Chain mask [B, L]
            residue_idx: Residue indices [B, L]
            chain_encoding_all: Chain encodings [B, L]
            randn: Random tensor for decoding order
            use_input_decoding_order: If True, use provided decoding order
            decoding_order: Optional decoding order tensor

        Returns:
            tuple: (hidden states list, sequence embeddings, log probabilities)
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions

        if not use_input_decoding_order:
            # decode left-to-right with all residues visible
            decoding_order = torch.tensor([list(range(X.size(1)))], device=device)

        mask_size = E_idx.shape[1]
        # one hot encode decoding order
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse, permutation_matrix_reverse
        )
        # set all residues to be visible
        order_mask_backward = torch.ones_like(order_mask_backward)

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        all_hidden = []
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
            all_hidden.append(h_V)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return list(reversed(all_hidden)), h_S, log_probs

    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None,
               temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None,
               omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None,
               pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None,
               bias_by_res=None):
        """Sample sequences from the model.

        Args:
            X: Coordinates tensor
            randn: Random tensor for decoding order
            S_true: True sequence tensor
            chain_mask: Chain mask
            chain_encoding_all: Chain encodings
            residue_idx: Residue indices
            mask: Optional mask
            temperature: Sampling temperature
            omit_AAs_np: Amino acids to omit
            bias_AAs_np: Amino acid biases
            chain_M_pos: Position mask
            omit_AA_mask: Omit AA mask
            pssm_coef: PSSM coefficients
            pssm_bias: PSSM bias
            pssm_multi: PSSM multiplier
            pssm_log_odds_flag: PSSM log odds flag
            pssm_log_odds_mask: PSSM log odds mask
            pssm_bias_flag: PSSM bias flag
            bias_by_res: Residue-specific bias

        Returns:
            dict: Output dictionary with sampled sequences and probabilities
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask * chain_M_pos * mask
        decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse, permutation_matrix_reverse
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask is not None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for t_ in range(N_nodes):
            t = decoding_order[:, t_]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])
            mask_gathered = torch.gather(mask, 1, t[:, None])
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]

            if (mask_gathered == 0).all():
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw, 1,
                    t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1])
                )
                mask_t = torch.gather(mask, 1, t[:, None])

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(
                        mask_bw, 1,
                        t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1])
                    ) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l + 1].scatter_(
                        1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t)
                    )

                h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(
                    logits - constant[None, :] * 1e8 + constant_bias[None, :] / temperature + bias_by_res_gathered / temperature,
                    dim=-1
                )

                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:, None, None].repeat(1, 1, pssm_bias.shape[-1]))[:, 0]
                    probs = (1 - pssm_multi * pssm_coef_gathered[:, None]) * probs + pssm_multi * pssm_coef_gathered[:, None] * pssm_bias_gathered

                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(
                        pssm_log_odds_mask, 1,
                        t[:, None, None].repeat(1, 1, pssm_log_odds_mask.shape[-1])
                    )[:, 0]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)

                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(
                        omit_AA_mask, 1,
                        t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1])
                    )[:, 0]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)

                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(
                    1, t[:, None, None].repeat(1, 1, 21),
                    (chain_mask_gathered[:, :, None] * probs[:, None, :]).float()
                )

            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (S_t * chain_mask_gathered + S_true_gathered * (1.0 - chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)

        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict

    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """Compute unconditional probabilities.

        Args:
            X: Coordinates tensor
            mask: Mask tensor
            residue_idx: Residue indices
            chain_encoding_all: Chain encodings

        Returns:
            Log probabilities tensor
        """
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


