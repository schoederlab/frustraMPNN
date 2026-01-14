"""
Transfer learning model for frustration prediction.

This module contains the TransferModel architecture that builds on ProteinMPNN
to predict single-residue local energetic frustration values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef  # noqa: F401

    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

from frustrampnn.constants import ALPHABET, VOCAB_DIM
from frustrampnn.model.featurization import tied_featurize
from frustrampnn.model.model_utils import get_esm_model, get_metrics, get_protein_mpnn

__all__ = [
    "TransferModel",
    "TransferModelPL",
    "LightAttention",
]


class LightAttention(nn.Module):
    """Light attention mechanism for feature aggregation.

    Source: Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """

    def __init__(
        self,
        embeddings_dim=1024,
        output_dim=11,
        dropout=0.25,
        kernel_size=9,
        conv_dropout: float = 0.25,
    ):
        """Initialize LightAttention.

        Args:
            embeddings_dim: Dimension of input embeddings
            output_dim: Output dimension (unused, kept for compatibility)
            dropout: Dropout rate (unused, kept for compatibility)
            kernel_size: Convolution kernel size
            conv_dropout: Convolution dropout rate
        """
        super().__init__()

        self.feature_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor
            mask: [batch_size, sequence_length] mask tensor

        Returns:
            Output tensor
        """
        o = self.feature_convolution(x)
        o = self.dropout(o)

        attention = self.attention_convolution(x)

        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)


class TransferModel(nn.Module):
    """Transfer learning model for frustration prediction."""

    def __init__(self, cfg):
        """Initialize TransferModel.

        Args:
            cfg: Configuration object
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        self.num_final_layers = cfg.model.num_final_layers
        self.lightattn = cfg.model.lightattn if "lightattn" in cfg.model else False

        if "decoding_order" not in self.cfg:
            self.cfg.decoding_order = "left-to-right"

        self.prot_mpnn = get_protein_mpnn(cfg)
        EMBED_DIM = 128
        HIDDEN_DIM = 128

        # If esm embeddings should be used, load it
        self.esm_attention = cfg.training.add_esm_embeddings
        self.esm_model = None
        self.batch_converter = None
        self.embed_dim = None

        if cfg.training.add_esm_embeddings:
            esm_model, alphabet = get_esm_model(cfg.training.esm_model)
            batch_converter = alphabet.get_batch_converter()
            esm_model.cuda()
            esm_model.eval()
            for param in esm_model.parameters():
                param.requires_grad = False

            self.esm_model = esm_model
            self.esm_layers = len(esm_model.layers)
            self.batch_converter = batch_converter
            self.embed_dim = esm_model.embed_dim

        if cfg.training.add_esm_embeddings:
            hid_sizes = [HIDDEN_DIM * self.num_final_layers + EMBED_DIM + self.embed_dim]
        else:
            hid_sizes = [HIDDEN_DIM * self.num_final_layers + EMBED_DIM]

        hid_sizes += self.hidden_dims
        hid_sizes += [VOCAB_DIM]

        if self.lightattn:
            self.light_attention_mpnn = LightAttention(
                embeddings_dim=HIDDEN_DIM * self.num_final_layers + EMBED_DIM
            )

        if cfg.training.add_esm_embeddings:
            self.light_attention_esm = LightAttention(embeddings_dim=self.embed_dim)

        self.both_out = nn.Sequential()
        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:], strict=False):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.frustration_out = nn.Linear(1, 1)

        alpha_1 = list("ACDEFGHIKLMNPQRSTVWYX")
        self.aa_N_1 = dict(enumerate(alpha_1))

    def _N_to_AA(self, x):
        """Convert numeric sequence to amino acid string."""
        if x.ndim == 1:
            x = x[None]
        return ["".join([self.aa_N_1.get(a, "-") for a in y]) for y in x]

    def get_esm_embeddings(self, S):
        """Get ESM embeddings for sequences.

        Args:
            S: Sequence tensor

        Returns:
            ESM embeddings tensor
        """
        data = [(f"seq_{i}", self._N_to_AA(s)[0]) for i, s in enumerate(S)]
        _, _, toks = self.batch_converter(data)
        toks = toks.to(next(self.parameters()).device)

        with torch.no_grad():
            results = self.esm_model(toks, repr_layers=[self.esm_layers], return_contacts=False)

        return results["representations"][self.esm_layers]

    def forward(self, pdb, mutations, tied_feat=True):
        """Forward pass.

        Args:
            pdb: PDB data tuple
            mutations: List of mutation objects
            tied_feat: Whether to use tied featurization

        Returns:
            tuple: (output list, None)
        """
        device = next(self.parameters()).device

        (
            X,
            S,
            mask,
            lengths,
            chain_M,
            chain_encoding_all,
            chain_list_list,
            visible_list_list,
            masked_list_list,
            masked_chain_length_list_list,
            chain_M_pos,
            omit_AA_mask,
            residue_idx,
            dihedral_mask,
            tied_pos_list_of_lists_list,
            pssm_coef,
            pssm_bias,
            pssm_log_odds_all,
            bias_by_res_all,
            tied_beta,
        ) = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        # getting ProteinMPNN structure embeddings
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding_all, None
        )
        if self.num_final_layers > 0:
            mpnn_hid = torch.cat(all_mpnn_hid[: self.num_final_layers], -1)

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

            if self.num_final_layers > 0:
                hid = mpnn_hid[0][mut.position]
                inputs.append(hid)

            embed = mpnn_embed[0][mut.position]
            inputs.append(embed)

            # concatenating hidden layers and embeddings
            lin_input = torch.cat(inputs, -1)

            # passing vector through lightattn
            if self.lightattn:
                lin_input = torch.unsqueeze(torch.unsqueeze(lin_input, -1), 0)
                lin_input = self.light_attention_mpnn(lin_input, mask)

            if self.esm_attention:
                esm_pos_embed = esm_embed[0][mut.position + 1]  # +1 for <start> token
                esm_pos_embed = torch.unsqueeze(torch.unsqueeze(esm_pos_embed, -1), 0)
                esm_pos_embed = self.light_attention_esm(esm_pos_embed, mask)
                lin_input = torch.cat([lin_input, esm_pos_embed], -1)

            both_input = torch.unsqueeze(self.both_out(lin_input), -1)
            frustration_out = self.frustration_out(both_input)

            if self.subtract_mut:
                frustration = frustration_out[aa_index][0] - frustration_out[wt_aa_index][0]
            else:
                frustration = frustration_out[aa_index][0]

            out.append(
                {
                    "frustration": torch.unsqueeze(frustration, 0),
                }
            )

        return out, None


# Only define TransferModelPL if pytorch_lightning is available
if HAS_LIGHTNING:

    class TransferModelPL(pl.LightningModule):
        """PyTorch Lightning wrapper for TransferModel."""

        stage: int = 1

        def __init__(self, cfg):
            """Initialize TransferModelPL.

            Args:
                cfg: Configuration object
            """
            super().__init__()
            self.cfg = cfg
            self.model = TransferModel(cfg)

            self.learn_rate = cfg.training.learn_rate
            self.mpnn_learn_rate = (
                cfg.training.mpnn_learn_rate if "mpnn_learn_rate" in cfg.training else None
            )
            self.lr_schedule = cfg.training.lr_schedule if "lr_schedule" in cfg.training else False

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

        def forward(self, *args):
            return self.model(*args)

        def shared_eval(self, batch, batch_idx, prefix):
            assert len(batch) == 1
            mut_pdb, mutations = batch[0]
            pred, _ = self(mut_pdb, mutations)

            frustration_mses = []
            if self.reweighting_loss:
                weight_sum = sum([mut.weight for mut in mutations])

            for mut, out in zip(mutations, pred, strict=False):
                if mut.frustration is not None:
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
                self.log(
                    f"{prefix}_{output}_{name}",
                    metric,
                    prog_bar=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=len(batch),
                )

            if loss == 0.0:
                return None

            return loss

        def training_step(self, batch, batch_idx):
            return self.shared_eval(batch, batch_idx, "train")

        def validation_step(self, batch, batch_idx):
            return self.shared_eval(batch, batch_idx, "val")

        def test_step(self, batch, batch_idx):
            return self.shared_eval(batch, batch_idx, "test")

        def configure_optimizers(self):
            if self.stage == 2:
                self.learn_rate /= 10.0

            if not self.cfg.model.freeze_weights:
                param_list = [
                    {"params": self.model.prot_mpnn.parameters(), "lr": self.mpnn_learn_rate}
                ]
            else:
                param_list = []

            if self.model.lightattn:
                if self.stage == 2:
                    param_list.append(
                        {"params": self.model.light_attention.parameters(), "lr": 0.0}
                    )
                else:
                    param_list.append({"params": self.model.light_attention_mpnn.parameters()})
                    if self.cfg.training.add_esm_embeddings:
                        param_list.append({"params": self.model.light_attention_esm.parameters()})

            mlp_params = [
                {"params": self.model.both_out.parameters()},
                {"params": self.model.frustration_out.parameters()},
            ]

            param_list = param_list + mlp_params
            opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

            if self.lr_schedule:
                lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=opt, verbose=True, mode="min", factor=0.5
                )
                return {
                    "optimizer": opt,
                    "lr_scheduler": lr_sched,
                    "monitor": "val_frustration_mse",
                }
            else:
                return opt

        def on_save_checkpoint(self, checkpoint):
            checkpoint["state_dict"] = {
                k: v
                for k, v in self.state_dict().items()
                if k in dict(self.named_parameters()) and self.get_parameter(k).requires_grad
            }

        def on_load_checkpoint(self, checkpoint):
            state_dict_checkpoint = set(checkpoint["state_dict"].keys())
            state_dict_model = set(self.state_dict().keys())
            missing_keys = state_dict_model - state_dict_checkpoint

            if missing_keys:
                for key in missing_keys:
                    checkpoint["state_dict"][key] = self.state_dict()[key]
else:
    # Placeholder if pytorch_lightning is not available
    class TransferModelPL:
        """Placeholder for TransferModelPL when pytorch_lightning is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TransferModelPL requires pytorch_lightning. "
                "Install with: pip install pytorch-lightning"
            )
