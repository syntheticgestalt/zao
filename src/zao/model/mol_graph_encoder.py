"""GIN encoder for 2D molecular graphs."""

import torch
import torch.nn as nn

NUM_BOND_TYPES = 5


class GINLayer(nn.Module):
    """Single GIN layer with bond-type-aware aggregation."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.bond_embedding = nn.Embedding(NUM_BOND_TYPES, hidden_dim)
        nn.init.zeros_(self.bond_embedding.weight[0])
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, h: torch.Tensor, bond_types: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: (B, N, D) node features; padding atoms must be zero.
            bond_types: (B, N, N) bond type codes.
            mask: (B, N) atom mask.

        Returns:
            (B, N, D) updated node features.
        """
        B, N, D = h.shape

        bond_embed = self.bond_embedding(bond_types)  # (B, N, N, D)
        bond_mask = (bond_types > 0).unsqueeze(-1).float()  # (B, N, N, 1)

        h_neighbors = h.unsqueeze(1).expand(B, N, N, D)
        messages = (h_neighbors + bond_embed) * bond_mask
        neighbor_sum = messages.sum(dim=2)  # (B, N, D)

        h_updated = (1 + self.eps) * h + neighbor_sum
        h_updated = self.mlp(h_updated) + h
        return h_updated * mask.unsqueeze(-1).float()


class MolGraphEncoder(nn.Module):
    """Multi-layer GIN encoder for molecular graphs."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GINLayer(hidden_dim, dropout) for _ in range(num_layers)])

    def forward(
        self, atom_feats: torch.Tensor, bond_types: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            atom_feats: (B, N, input_dim) atom features.
            bond_types: (B, N, N) bond type codes.
            mask: (B, N) atom mask.

        Returns:
            (B, N, hidden_dim) per-atom embeddings.
        """
        h = self.input_proj(atom_feats) * mask.unsqueeze(-1).float()
        for layer in self.layers:
            h = layer(h, bond_types, mask)
        return h
