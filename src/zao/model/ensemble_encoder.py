"""Conformer-set aggregation, and the transformer pieces the conformer encoder shares."""

import torch
import torch.nn as nn


class DistanceAttentionBias(nn.Module):
    """Turns inter-substructure distances into an additive attention bias per head."""

    def __init__(self, num_rbf: int, num_heads: int, max_distance: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, max_distance, num_rbf))
        self.gamma = nn.Parameter(torch.ones(num_rbf) * 0.5)
        self.proj = nn.Linear(num_rbf, num_heads, bias=False)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """(B, S, S) distances -> (B, num_heads, S, S) bias."""
        rbf = torch.exp(-self.gamma.abs() * (distances.unsqueeze(-1) - self.centers) ** 2)
        return self.proj(rbf).permute(0, 3, 1, 2)


class TransformerEncoderLayerWithBias(nn.Module):
    """Post-norm transformer encoder layer that takes an additive attention bias."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, S, D) input and a (B*nhead, S, S) additive bias -> (B, S, D)."""
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout3(ff_out))


class ConformerEnsembleEncoder(nn.Module):
    """Aggregates the per-conformer embeddings of one molecule into one vector."""

    def __init__(
        self,
        embed_dim: int,
        input_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithBias(embed_dim, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, conformer_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, K, input_dim) conformer embeddings -> (B, embed_dim) molecular embedding."""
        x = self.input_proj(conformer_embeddings)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.norm(x[:, 0, :])
