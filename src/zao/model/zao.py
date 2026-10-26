"""Inference-only ZAO model: conformer encoder + conformer-set aggregation."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ZAO_CONFIG
from .conformer_encoder import SingleConformerEncoder
from .ensemble_encoder import ConformerEnsembleEncoder

EMBEDDING_MODULES = ("encoder", "ensemble_encoder")


class ZAOModel(nn.Module):
    """Maps a batch of conformer sets to 2,048-dimensional molecular embeddings.

    Args:
        config: the architecture to build; defaults to ``ZAO_CONFIG``. Every key
            the network reads is required.
    """

    def __init__(self, config: dict[str, Any] = ZAO_CONFIG):
        super().__init__()
        general = config["general"]
        enc = config["encoder"]
        ens = config["ensemble"]
        dropout = general["dropout"]

        self.use_boltzmann_aggregation = ens["use_boltzmann_aggregation"]
        self.boltzmann_temperature = ens["boltzmann_temperature"]

        self.encoder = SingleConformerEncoder(
            atom_input_dim=enc["input_dim"],
            hidden_dim=enc["hidden_dim"],
            depth=enc["depth"],
            num_nearest_neighbors=enc["num_nearest_neighbors"],
            norm_coors=enc["norm_coors"],
            coor_weights_clamp_value=enc["coor_weights_clamp_value"],
            dropout=dropout,
            num_queries=enc["num_queries"],
            substructure_embed_dim=enc["substructure_embed_dim"],
            attention_temperature_init=enc["attention_temperature_init"],
            edge_dim=enc["edge_dim"],
            mol_graph_num_layers=enc["mol_graph_num_layers"],
            conformer_embed_dim=general["conformer_embed_dim"],
            conformer_pooling_config=enc["conformer_pooling"],
        )
        self.ensemble_encoder = ConformerEnsembleEncoder(
            embed_dim=general["mol_embed_dim"],
            input_dim=general["conformer_embed_dim"],
            nhead=ens["nhead"],
            dim_feedforward=general["mol_embed_dim"] * ens["dim_feedforward_ratio"],
            num_layers=ens["num_layers"],
            dropout=dropout,
        )

    def forward(
        self,
        feats: torch.Tensor,
        coors: torch.Tensor,
        mask: torch.Tensor,
        bond_types: torch.Tensor,
        energy: torch.Tensor,
        num_conformers: int,
    ) -> torch.Tensor:
        """
        Args:
            feats: (B*K, N, 7) atom features, K conformers per molecule, molecule-major.
            coors: (B*K, N, 3) coordinates in angstrom.
            mask: (B*K, N) atom mask.
            bond_types: (B*K, N, N) bond type codes 0-4.
            energy: (B*K, 1) MMFF94 energy of each conformer (kcal/mol).
            num_conformers: K.

        Returns:
            (B, mol_embed_dim) molecular embeddings.
        """
        conf_embeds = self.encoder(feats, coors, mask, bond_types, num_conformers)
        batch_size = conf_embeds.shape[0] // num_conformers
        conf_embeds = conf_embeds.view(batch_size, num_conformers, -1)

        if self.use_boltzmann_aggregation:
            energy = energy.view(batch_size, num_conformers)
            relative_energy = energy - energy.min(dim=1, keepdim=True)[0]
            weights = F.softmax(-relative_energy / self.boltzmann_temperature, dim=1)
            conf_embeds = conf_embeds * weights.unsqueeze(-1) * num_conformers

        return self.ensemble_encoder(conf_embeds)

    def load_pretrained_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the embedding-path weights from a full pretrained state dict."""
        state = {k: v for k, v in state_dict.items() if k.split(".", 1)[0] in EMBEDDING_MODULES}
        self.load_state_dict(state, strict=True)
