"""Encoder for one conformer: coordinates and atom features to one conformer embedding."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from egnn_pytorch import EGNN_Network

from .ensemble_encoder import DistanceAttentionBias, TransformerEncoderLayerWithBias
from .mol_graph_encoder import MolGraphEncoder

GEOM_FEAT_DIM = 9


def _k_nearest_neighbor_coords(coors: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Coordinates of the k nearest neighbors of each atom, (B, N, k, 3)."""
    B, N, _ = coors.shape
    dist_sq = torch.cdist(coors, coors).pow(2)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    dist_sq = dist_sq.masked_fill(~pair_mask, float("inf"))
    eye = torch.eye(N, device=coors.device, dtype=torch.bool).unsqueeze(0)
    dist_sq = dist_sq.masked_fill(eye, float("inf"))

    _, indices = torch.topk(dist_sq, k=k, dim=-1, largest=False)  # (B, N, k)
    neighbors = torch.gather(coors, 1, indices.reshape(B, -1).unsqueeze(-1).expand(-1, -1, 3))
    return neighbors.reshape(B, N, k, 3)


class SingleConformerEncoder(nn.Module):
    """Encodes one conformer into an embedding."""

    def __init__(
        self,
        atom_input_dim: int,
        hidden_dim: int,
        depth: int,
        num_nearest_neighbors: int,
        norm_coors: bool,
        coor_weights_clamp_value: float,
        dropout: float,
        num_queries: int,
        substructure_embed_dim: int,
        attention_temperature_init: float,
        edge_dim: int,
        mol_graph_num_layers: int,
        conformer_embed_dim: int,
        conformer_pooling_config: dict,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.input_proj = nn.Linear(atom_input_dim + GEOM_FEAT_DIM, hidden_dim)

        self.egnn = EGNN_Network(
            dim=hidden_dim,
            depth=depth,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=norm_coors,
            coor_weights_clamp_value=coor_weights_clamp_value,
            edge_dim=edge_dim,
            dropout=0,
        )

        self.mol_graph_encoder = MolGraphEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=mol_graph_num_layers,
            dropout=dropout,
        )
        self.gin_input_proj = nn.Linear(atom_input_dim, hidden_dim)
        self.act = nn.SiLU()

        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.orthogonal_(self.queries)
        self.query_output_proj = nn.Linear(hidden_dim * 2, substructure_embed_dim)
        self.log_attention_temperature = nn.Parameter(
            torch.tensor(math.log(attention_temperature_init))
        )

        cp = conformer_pooling_config
        self.pooling_input_proj = (
            nn.Linear(substructure_embed_dim, conformer_embed_dim)
            if substructure_embed_dim != conformer_embed_dim
            else nn.Identity()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, conformer_embed_dim))
        self.pooling_distance_bias = DistanceAttentionBias(
            num_rbf=cp["num_rbf"], num_heads=cp["nhead"], max_distance=cp["rbf_max_distance"]
        )
        self.pooling_transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithBias(
                    d_model=conformer_embed_dim,
                    nhead=cp["nhead"],
                    dim_feedforward=cp["dim_feedforward"],
                    dropout=dropout,
                )
                for _ in range(cp["layers"])
            ]
        )
        self.pooling_norm = nn.LayerNorm(conformer_embed_dim)

    def _geometric_features(self, coors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-atom geometry from its 4 nearest neighbors, (B, N, GEOM_FEAT_DIM)."""
        k = 4
        neighbors = _k_nearest_neighbor_coords(coors, mask, k)  # (B, N, k, 3)

        vecs_chiral = neighbors[:, :, :3, :] - coors.unsqueeze(2)
        cross_prod = torch.cross(vecs_chiral[:, :, 1], vecs_chiral[:, :, 2], dim=-1)
        chirality = (vecs_chiral[:, :, 0] * cross_prod).sum(dim=-1, keepdim=True)
        chirality = chirality.masked_fill(~mask.unsqueeze(-1), 0.0)

        vecs = neighbors - coors.unsqueeze(2)
        vecs_norm = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)
        cos_angles = torch.einsum("bnid,bnjd->bnij", vecs_norm, vecs_norm)
        triu = torch.triu(torch.ones(k, k, device=coors.device, dtype=torch.bool), diagonal=1)
        angles = torch.acos(cos_angles[:, :, triu].clamp(-1 + 1e-7, 1 - 1e-7))
        angle_stats = torch.cat(
            [
                angles.mean(dim=-1, keepdim=True),
                angles.std(dim=-1, keepdim=True),
                angles.min(dim=-1, keepdim=True)[0],
                angles.max(dim=-1, keepdim=True)[0],
            ],
            dim=-1,
        )

        n0, n1, n2 = neighbors[:, :, 0, :], neighbors[:, :, 1, :], neighbors[:, :, 2, :]
        b1 = n0 - coors

        def dihedral(b2, b3):
            p1 = torch.cross(b1, b2, dim=-1)
            p2 = torch.cross(b2, b3, dim=-1)
            p1 = p1 / (p1.norm(dim=-1, keepdim=True) + 1e-8)
            p2 = p2 / (p2.norm(dim=-1, keepdim=True) + 1e-8)
            cos = (p1 * p2).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
            return torch.acos(cos)

        dihedrals = torch.cat([dihedral(n1 - n0, n2 - n1), dihedral(n2 - n0, n1 - n2)], dim=-1)
        dihedral_stats = torch.cat(
            [
                dihedrals.mean(dim=-1, keepdim=True),
                dihedrals.std(dim=-1, keepdim=True),
                dihedrals.min(dim=-1, keepdim=True)[0],
                dihedrals.max(dim=-1, keepdim=True)[0],
            ],
            dim=-1,
        )

        result = torch.cat([chirality, angle_stats, dihedral_stats], dim=-1)
        return result.masked_fill(~mask.unsqueeze(-1), 0.0)

    def _conformer_pooling(self, sub_embeds: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """K substructure tokens plus their distances -> (B, conformer_embed_dim)."""
        B = sub_embeds.shape[0]
        tokens = self.pooling_input_proj(sub_embeds)
        x = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)

        bias = self.pooling_distance_bias(distances)  # (B, nhead, K, K)
        nhead, size = bias.shape[1], x.shape[1]
        full_bias = torch.zeros(B, nhead, size, size, device=x.device, dtype=x.dtype)
        full_bias[:, :, 1:, 1:] = bias
        attn_mask = full_bias.reshape(B * nhead, size, size)

        for layer in self.pooling_transformer_layers:
            x = layer(x, attn_mask=attn_mask)
        return self.pooling_norm(x[:, 0, :])

    def forward(
        self,
        feats: torch.Tensor,
        coors: torch.Tensor,
        mask: torch.Tensor,
        bond_types: torch.Tensor,
        num_conformers: int,
    ) -> torch.Tensor:
        """
        Args:
            feats: (B*K, N, atom_input_dim) atom features.
            coors: (B*K, N, 3) coordinates.
            mask: (B*K, N) atom mask.
            bond_types: (B*K, N, N) bond type codes.
            num_conformers: K, with the conformers of a molecule adjacent.

        Returns:
            (B*K, conformer_embed_dim) conformer embeddings.
        """
        geom_feats = self._geometric_features(coors, mask)
        h = self.act(self.input_proj(torch.cat([feats, geom_feats], dim=-1)))
        edges = F.one_hot(bond_types, num_classes=self.edge_dim).float()
        h_out, coors_out = self.egnn(h, coors, edges=edges, mask=mask)

        h_2d = self.act(self.gin_input_proj(feats))
        BC, N, D = h_2d.shape
        g_mol = self.mol_graph_encoder(
            h_2d[::num_conformers], bond_types[::num_conformers], mask[::num_conformers]
        )
        h_gin = g_mol.unsqueeze(1).expand(-1, num_conformers, -1, -1).reshape(BC, N, D)

        B = h_out.shape[0]
        temperature = self.log_attention_temperature.exp()
        attn_logits = (
            torch.matmul(self.queries.unsqueeze(0).expand(B, -1, -1), h_out.transpose(1, 2))
            / temperature
        )
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=2)  # (B, K, N)

        sub_3d = torch.bmm(attn_weights, h_out)
        sub_2d = torch.bmm(attn_weights, h_gin)
        substructure_embeds = self.query_output_proj(torch.cat([sub_3d, sub_2d], dim=-1))

        centroids = torch.bmm(attn_weights, coors_out)  # (B, K, 3)
        distances = torch.norm(centroids.unsqueeze(2) - centroids.unsqueeze(1), dim=-1)

        return self._conformer_pooling(substructure_embeds, distances)
