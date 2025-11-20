"""Attention utilities for GNOSIS forecasting models."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Multi-head attention block with residual connection."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.attention(x, x, x, need_weights=True)
        attn_output = self.layer_norm(attn_output + x)
        return attn_output, attn_weights


class TemporalAttention(nn.Module):


    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.context_vector(torch.tanh(self.projection(x))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return context, weights
