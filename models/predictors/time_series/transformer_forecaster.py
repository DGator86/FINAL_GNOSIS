"""
Transformer-based forecaster for time series prediction
Uses positional encoding and multi-head self-attention
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class TransformerForecaster(nn.Module):
    """
    Transformer-based time series forecaster

    Architecture:
    1. Input projection to d_model dimensions
    2. Positional encoding
    3. Stack of transformer blocks
    4. Multi-horizon prediction heads
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
        forecast_horizons: Optional[List[int]] = None,
        max_seq_len: int = 1000,
    ):
        super().__init__()

        if forecast_horizons is None:
            forecast_horizons = [1, 5, 60, 1440]

        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizons = forecast_horizons

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.output_projection = nn.Linear(d_model, d_model // 2)

        self.forecast_heads = nn.ModuleDict()
        for horizon in forecast_horizons:
            self.forecast_heads[f"horizon_{horizon}"] = nn.Sequential(
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1),
            )

        self.volatility_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Softplus(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn = transformer_block(x)
            if return_attention:
                attention_weights.append(attn)

        x_final = x[:, -1, :]
        x_final = self.output_projection(x_final)

        predictions: Dict[str, torch.Tensor] = {}
        for horizon in self.forecast_horizons:
            head_name = f"horizon_{horizon}"
            pred = self.forecast_heads[head_name](x_final)
            predictions[head_name] = pred.squeeze(-1)

        volatility = self.volatility_head(x_final).squeeze(-1)
        predictions["volatility"] = volatility

        if return_attention:
            return predictions, attention_weights
        return predictions


class TransformerForecastModel:
    """Production wrapper for Transformer Forecaster"""

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[TransformerForecaster] = None
        self.config: Optional[Dict[str, Union[int, float, List[int]]]] = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location="cpu")

        self.config = checkpoint["config"]
        self.model = TransformerForecaster(**self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loaded Transformer model from {model_path}")

    def predict(self, features: np.ndarray, with_attention: bool = False) -> Dict:
        if self.model is None:
            raise ValueError("Model not loaded")

        if features.ndim == 2:
            features = features[np.newaxis, ...]

        x = torch.FloatTensor(features)

        with torch.no_grad():
            if with_attention:
                predictions, attention_weights = self.model(x, return_attention=True)

                avg_attention = torch.stack(attention_weights).mean(dim=0).mean(dim=1)
                attention_scores = avg_attention.squeeze(0).cpu().numpy()

                result = {"predictions": {}, "attention": attention_scores}
            else:
                predictions = self.model(x)
                result = {"predictions": {}}

            for horizon in self.model.forecast_horizons:
                key = f"price_forecast_{horizon}min"
                pred_key = f"horizon_{horizon}"
                if pred_key in predictions:
                    result["predictions"][key] = float(predictions[pred_key].item())

            if "volatility" in predictions:
                result["predictions"]["volatility_forecast"] = float(predictions["volatility"].item())

        return result
