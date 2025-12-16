"""
Attention mechanisms for LSTM/GRU time series models.

Includes multi-head attention along with temporal and market regime-aware variants
specialized for trading data.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Multi-head attention mechanism for sequence models."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional mask tensor
        Returns:
            output: Attention output
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, hidden_dim = x.size()

        # Generate Q, K, V
        q_values = self.query(x)
        k_values = self.key(x)
        v_values = self.value(x)

        # Reshape for multi-head attention
        q_values = q_values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_values = k_values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_values = v_values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q_values, k_values.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v_values)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        # Residual connection and layer norm
        output = self.layer_norm(x + attention_output)

        return output, attention_weights.mean(dim=1)  # Average across heads


class TemporalAttention(nn.Module):
    """Temporal attention for trading time series."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            context_vector: Weighted sum of inputs
            attention_weights: Attention weights
        """
        # Calculate attention weights
        attention_weights = self.attention_net(x)

        # Apply attention weights
        context_vector = torch.sum(x * attention_weights, dim=1)

        return context_vector, attention_weights.squeeze(-1)


class MarketRegimeAttention(nn.Module):
    """Market regime-aware attention mechanism."""

    def __init__(self, input_dim: int, regime_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.regime_dim = regime_dim

        # Regime detection network
        self.regime_detector = nn.Sequential(
            nn.Linear(input_dim, regime_dim),
            nn.ReLU(),
            nn.Linear(regime_dim, 4),  # 4 market regimes
            nn.Softmax(dim=-1),
        )

        # Regime-specific attention
        self.regime_attentions = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(4)])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            context_vector: Regime-aware context
            regime_probs: Market regime probabilities
            attention_weights: Combined attention weights
        """
        batch_size, seq_len, _ = x.size()

        # Detect market regimes for each timestep
        regime_probs = self.regime_detector(x)  # (batch_size, seq_len, 4)

        # Calculate regime-specific attention weights
        regime_attentions = []
        for attention_layer in self.regime_attentions:
            regime_attention = torch.sigmoid(attention_layer(x))  # (batch_size, seq_len, 1)
            regime_attentions.append(regime_attention)

        regime_attentions_tensor = torch.cat(regime_attentions, dim=-1)  # (batch_size, seq_len, 4)

        # Combine regime probabilities with attention weights
        attention_weights = torch.sum(
            regime_probs.unsqueeze(-1) * regime_attentions_tensor.unsqueeze(-1),
            dim=-2,
        ).squeeze(-1)

        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention to input
        context_vector = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)

        return context_vector, regime_probs, attention_weights
