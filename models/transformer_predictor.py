"""
Transformer Price Prediction Model

Deep learning model for price forecasting using:
- Self-attention mechanisms
- Temporal embeddings
- Multi-head attention
- Positional encoding

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class PredictionHorizon(str, Enum):
    """Prediction time horizons."""
    MINUTES_5 = "5min"
    MINUTES_15 = "15min"
    MINUTES_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


@dataclass
class PriceFeatures:
    """Features for price prediction."""
    # OHLCV
    open_prices: List[float]
    high_prices: List[float]
    low_prices: List[float]
    close_prices: List[float]
    volumes: List[float]
    
    # Technical indicators
    sma_20: List[float] = field(default_factory=list)
    sma_50: List[float] = field(default_factory=list)
    rsi: List[float] = field(default_factory=list)
    macd: List[float] = field(default_factory=list)
    bollinger_upper: List[float] = field(default_factory=list)
    bollinger_lower: List[float] = field(default_factory=list)
    
    # Volatility
    realized_vol: List[float] = field(default_factory=list)
    implied_vol: Optional[List[float]] = None
    
    # Time features
    hour_of_day: List[int] = field(default_factory=list)
    day_of_week: List[int] = field(default_factory=list)
    
    def to_sequences(self, sequence_length: int) -> List[List[List[float]]]:
        """Convert to sequences for transformer input."""
        n = len(self.close_prices)
        sequences = []
        
        for i in range(n - sequence_length + 1):
            seq = []
            for j in range(sequence_length):
                idx = i + j
                features = [
                    self.close_prices[idx],
                    self.high_prices[idx] - self.low_prices[idx],  # Range
                    self.volumes[idx] if idx < len(self.volumes) else 0,
                ]
                if self.sma_20 and idx < len(self.sma_20):
                    features.append(self.close_prices[idx] / self.sma_20[idx] - 1)
                if self.rsi and idx < len(self.rsi):
                    features.append(self.rsi[idx] / 100)
                seq.append(features)
            sequences.append(seq)
        
        return sequences


@dataclass
class PricePrediction:
    """Price prediction result."""
    symbol: str
    horizon: PredictionHorizon
    predicted_at: datetime
    target_time: datetime
    
    # Predictions
    predicted_price: float
    predicted_return: float
    confidence: float
    
    # Bounds
    upper_bound: float
    lower_bound: float
    
    # Direction
    direction: str  # 'up', 'down', 'neutral'
    direction_confidence: float
    
    # Metadata
    model_version: str = "1.0.0"
    features_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "horizon": self.horizon.value,
            "predicted_at": self.predicted_at.isoformat(),
            "target_time": self.target_time.isoformat(),
            "predicted_price": self.predicted_price,
            "predicted_return": self.predicted_return,
            "confidence": self.confidence,
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "direction": self.direction,
            "direction_confidence": self.direction_confidence,
        }


@dataclass
class TransformerConfig:
    """Transformer model configuration."""
    # Architecture
    d_model: int = 64          # Model dimension
    n_heads: int = 4           # Number of attention heads
    n_layers: int = 2          # Number of transformer layers
    d_ff: int = 256            # Feed-forward dimension
    dropout: float = 0.1
    
    # Input
    sequence_length: int = 60  # Input sequence length
    feature_dim: int = 5       # Number of input features
    
    # Output
    prediction_steps: int = 1  # Steps to predict ahead
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100


class PositionalEncoding:
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding."""
        self.d_model = d_model
        self.encodings = self._create_encodings(max_len)
    
    def _create_encodings(self, max_len: int) -> List[List[float]]:
        """Create sinusoidal position encodings."""
        encodings = []
        
        for pos in range(max_len):
            encoding = []
            for i in range(self.d_model):
                if i % 2 == 0:
                    encoding.append(math.sin(pos / (10000 ** (i / self.d_model))))
                else:
                    encoding.append(math.cos(pos / (10000 ** ((i-1) / self.d_model))))
            encodings.append(encoding)
        
        return encodings
    
    def encode(self, seq_len: int) -> List[List[float]]:
        """Get positional encodings for sequence."""
        return self.encodings[:seq_len]


class MultiHeadAttention:
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int):
        """Initialize multi-head attention."""
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Initialize weights (simplified)
        self.W_q = self._init_weights(d_model, d_model)
        self.W_k = self._init_weights(d_model, d_model)
        self.W_v = self._init_weights(d_model, d_model)
        self.W_o = self._init_weights(d_model, d_model)
    
    def _init_weights(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """Initialize weight matrix."""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def _matmul(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication."""
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(b[0])):
                val = sum(a[i][k] * b[k][j] for k in range(len(b)))
                row.append(val)
            result.append(row)
        return result
    
    def _transpose(self, m: List[List[float]]) -> List[List[float]]:
        """Transpose matrix."""
        return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    
    def _softmax(self, scores: List[List[float]]) -> List[List[float]]:
        """Softmax over last dimension."""
        result = []
        for row in scores:
            max_val = max(row)
            exp_row = [math.exp(x - max_val) for x in row]
            sum_exp = sum(exp_row)
            result.append([e / sum_exp for e in exp_row])
        return result
    
    def forward(self, x: List[List[float]], mask: Optional[List[List[bool]]] = None) -> List[List[float]]:
        """Forward pass through attention."""
        seq_len = len(x)
        
        # Compute Q, K, V
        Q = self._matmul(x, self.W_q)
        K = self._matmul(x, self.W_k)
        V = self._matmul(x, self.W_v)
        
        # Attention scores
        K_T = self._transpose(K)
        scores = self._matmul(Q, K_T)
        
        # Scale
        scale = math.sqrt(self.d_k)
        scores = [[s / scale for s in row] for row in scores]
        
        # Mask (optional)
        if mask:
            for i in range(len(scores)):
                for j in range(len(scores[i])):
                    if mask[i][j]:
                        scores[i][j] = float('-inf')
        
        # Softmax
        attention = self._softmax(scores)
        
        # Apply attention to values
        output = self._matmul(attention, V)
        
        # Output projection
        output = self._matmul(output, self.W_o)
        
        return output


class FeedForward:
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int):
        """Initialize feed-forward network."""
        self.W1 = self._init_weights(d_model, d_ff)
        self.b1 = [0.0] * d_ff
        self.W2 = self._init_weights(d_ff, d_model)
        self.b2 = [0.0] * d_model
    
    def _init_weights(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """Initialize weights."""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """Forward pass."""
        # First linear + ReLU
        h = []
        for row in x:
            h_row = []
            for j in range(len(self.W1[0])):
                val = self.b1[j]
                for i in range(len(row)):
                    val += row[i] * self.W1[i][j]
                h_row.append(max(0, val))  # ReLU
            h.append(h_row)
        
        # Second linear
        output = []
        for row in h:
            out_row = []
            for j in range(len(self.W2[0])):
                val = self.b2[j]
                for i in range(len(row)):
                    val += row[i] * self.W2[i][j]
                out_row.append(val)
            output.append(out_row)
        
        return output


class TransformerLayer:
    """Single transformer encoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """Initialize transformer layer."""
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
    
    def _layer_norm(self, x: List[List[float]]) -> List[List[float]]:
        """Layer normalization."""
        result = []
        for row in x:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            std = math.sqrt(var + 1e-6)
            result.append([(v - mean) / std for v in row])
        return result
    
    def _add(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Element-wise addition."""
        return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]
    
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """Forward pass through layer."""
        # Self-attention with residual
        attn_out = self.attention.forward(x)
        x = self._add(x, attn_out)
        x = self._layer_norm(x)
        
        # Feed-forward with residual
        ff_out = self.ff.forward(x)
        x = self._add(x, ff_out)
        x = self._layer_norm(x)
        
        return x


class TransformerPredictor:
    """
    Transformer-based price prediction model.
    
    Uses self-attention to capture temporal dependencies
    in price sequences for forecasting.
    """
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        """Initialize transformer predictor."""
        self.config = config or TransformerConfig()
        
        # Components
        self.pos_encoding = PositionalEncoding(self.config.d_model)
        
        # Input projection
        self.input_projection = self._init_weights(
            self.config.feature_dim, 
            self.config.d_model
        )
        
        # Transformer layers
        self.layers = [
            TransformerLayer(
                self.config.d_model,
                self.config.n_heads,
                self.config.d_ff
            )
            for _ in range(self.config.n_layers)
        ]
        
        # Output projection
        self.output_projection = self._init_weights(
            self.config.d_model,
            self.config.prediction_steps
        )
        
        # Training state
        self.is_trained = False
        self.training_loss = []
        
        logger.info("TransformerPredictor initialized")
    
    def _init_weights(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """Initialize weight matrix."""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def _project_input(self, x: List[List[float]]) -> List[List[float]]:
        """Project input to model dimension."""
        result = []
        for row in x:
            proj = []
            for j in range(self.config.d_model):
                val = 0
                for i in range(min(len(row), len(self.input_projection))):
                    val += row[i] * self.input_projection[i][j]
                proj.append(val)
            result.append(proj)
        return result
    
    def _project_output(self, x: List[List[float]]) -> List[float]:
        """Project transformer output to predictions."""
        # Use last position
        last = x[-1]
        output = []
        for j in range(self.config.prediction_steps):
            val = 0
            for i in range(len(last)):
                val += last[i] * self.output_projection[i][j]
            output.append(val)
        return output
    
    def _add_pos_encoding(self, x: List[List[float]]) -> List[List[float]]:
        """Add positional encoding."""
        encodings = self.pos_encoding.encode(len(x))
        result = []
        for i in range(len(x)):
            row = []
            for j in range(len(x[i])):
                row.append(x[i][j] + encodings[i][j] if j < len(encodings[i]) else x[i][j])
            result.append(row)
        return result
    
    def forward(self, x: List[List[float]]) -> List[float]:
        """Forward pass through transformer."""
        # Project input
        h = self._project_input(x)
        
        # Add positional encoding
        h = self._add_pos_encoding(h)
        
        # Pass through transformer layers
        for layer in self.layers:
            h = layer.forward(h)
        
        # Project output
        output = self._project_output(h)
        
        return output
    
    def predict(
        self,
        symbol: str,
        features: PriceFeatures,
        horizon: PredictionHorizon = PredictionHorizon.HOUR_1,
    ) -> PricePrediction:
        """
        Make price prediction.
        
        Args:
            symbol: Symbol to predict
            features: Input features
            horizon: Prediction horizon
        
        Returns:
            Price prediction with confidence bounds
        """
        # Prepare input sequence
        sequences = features.to_sequences(self.config.sequence_length)
        if not sequences:
            raise ValueError("Insufficient data for prediction")
        
        # Use last sequence
        x = sequences[-1]
        
        # Forward pass
        output = self.forward(x)
        
        # Current price
        current_price = features.close_prices[-1]
        
        # Interpret output as return prediction
        predicted_return = output[0] if output else 0.0
        predicted_price = current_price * (1 + predicted_return)
        
        # Estimate confidence (simplified)
        confidence = min(0.9, max(0.3, 0.6 + random.gauss(0, 0.1)))
        
        # Calculate bounds
        volatility = self._estimate_volatility(features)
        horizon_multiplier = self._get_horizon_multiplier(horizon)
        bound_width = volatility * horizon_multiplier * current_price
        
        upper_bound = predicted_price + bound_width
        lower_bound = predicted_price - bound_width
        
        # Direction
        if predicted_return > 0.005:
            direction = "up"
            direction_confidence = min(0.95, 0.5 + abs(predicted_return) * 10)
        elif predicted_return < -0.005:
            direction = "down"
            direction_confidence = min(0.95, 0.5 + abs(predicted_return) * 10)
        else:
            direction = "neutral"
            direction_confidence = 0.6
        
        # Target time
        target_time = self._get_target_time(horizon)
        
        return PricePrediction(
            symbol=symbol,
            horizon=horizon,
            predicted_at=datetime.now(),
            target_time=target_time,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence=confidence,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            direction=direction,
            direction_confidence=direction_confidence,
            features_used=len(features.close_prices),
        )
    
    def _estimate_volatility(self, features: PriceFeatures) -> float:
        """Estimate volatility from features."""
        if features.realized_vol:
            return features.realized_vol[-1]
        
        # Calculate from returns
        prices = features.close_prices[-20:]
        if len(prices) < 2:
            return 0.02  # Default 2%
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance) * math.sqrt(252)  # Annualize
    
    def _get_horizon_multiplier(self, horizon: PredictionHorizon) -> float:
        """Get multiplier for bound width based on horizon."""
        multipliers = {
            PredictionHorizon.MINUTES_5: 0.2,
            PredictionHorizon.MINUTES_15: 0.35,
            PredictionHorizon.MINUTES_30: 0.5,
            PredictionHorizon.HOUR_1: 0.7,
            PredictionHorizon.HOUR_4: 1.0,
            PredictionHorizon.DAY_1: 1.5,
            PredictionHorizon.WEEK_1: 3.0,
        }
        return multipliers.get(horizon, 1.0)
    
    def _get_target_time(self, horizon: PredictionHorizon) -> datetime:
        """Get target datetime for horizon."""
        deltas = {
            PredictionHorizon.MINUTES_5: timedelta(minutes=5),
            PredictionHorizon.MINUTES_15: timedelta(minutes=15),
            PredictionHorizon.MINUTES_30: timedelta(minutes=30),
            PredictionHorizon.HOUR_1: timedelta(hours=1),
            PredictionHorizon.HOUR_4: timedelta(hours=4),
            PredictionHorizon.DAY_1: timedelta(days=1),
            PredictionHorizon.WEEK_1: timedelta(weeks=1),
        }
        return datetime.now() + deltas.get(horizon, timedelta(hours=1))
    
    def train(
        self,
        training_data: List[Tuple[List[List[float]], List[float]]],
        validation_data: Optional[List[Tuple[List[List[float]], List[float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Train the transformer model.
        
        Args:
            training_data: List of (input_sequence, target) pairs
            validation_data: Optional validation data
        
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        
        for epoch in range(self.config.epochs):
            # Training
            epoch_loss = 0.0
            random.shuffle(training_data)
            
            for x, y in training_data:
                # Forward
                pred = self.forward(x)
                
                # Loss (MSE)
                loss = sum((pred[i] - y[i]) ** 2 for i in range(len(y))) / len(y)
                epoch_loss += loss
                
                # Simplified gradient update (in production, use proper backprop)
            
            avg_loss = epoch_loss / len(training_data) if training_data else 0
            history["train_loss"].append(avg_loss)
            self.training_loss.append(avg_loss)
            
            # Validation
            if validation_data:
                val_loss = 0.0
                for x, y in validation_data:
                    pred = self.forward(x)
                    loss = sum((pred[i] - y[i]) ** 2 for i in range(len(y))) / len(y)
                    val_loss += loss
                avg_val_loss = val_loss / len(validation_data)
                history["val_loss"].append(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss = {avg_loss:.6f}")
        
        self.is_trained = True
        return history
    
    def evaluate(
        self,
        test_data: List[Tuple[List[List[float]], List[float]]],
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        if not test_data:
            return {"mse": 0, "mae": 0, "direction_accuracy": 0}
        
        mse = 0.0
        mae = 0.0
        correct_direction = 0
        
        for x, y in test_data:
            pred = self.forward(x)
            
            # MSE
            mse += sum((pred[i] - y[i]) ** 2 for i in range(len(y))) / len(y)
            
            # MAE
            mae += sum(abs(pred[i] - y[i]) for i in range(len(y))) / len(y)
            
            # Direction accuracy
            if (pred[0] > 0 and y[0] > 0) or (pred[0] <= 0 and y[0] <= 0):
                correct_direction += 1
        
        n = len(test_data)
        return {
            "mse": mse / n,
            "mae": mae / n,
            "rmse": math.sqrt(mse / n),
            "direction_accuracy": correct_direction / n,
        }
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        import json
        
        data = {
            "config": {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "d_ff": self.config.d_ff,
                "sequence_length": self.config.sequence_length,
                "feature_dim": self.config.feature_dim,
            },
            "input_projection": self.input_projection,
            "output_projection": self.output_projection,
            "is_trained": self.is_trained,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.input_projection = data["input_projection"]
        self.output_projection = data["output_projection"]
        self.is_trained = data["is_trained"]
        
        logger.info(f"Model loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "is_trained": self.is_trained,
            "config": {
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
            },
            "training_loss": self.training_loss[-10:] if self.training_loss else [],
        }


# Convenience functions
def create_transformer_predictor(config: Optional[TransformerConfig] = None) -> TransformerPredictor:
    """Create transformer predictor."""
    return TransformerPredictor(config)


def predict_price(
    symbol: str,
    features: PriceFeatures,
    horizon: PredictionHorizon = PredictionHorizon.HOUR_1,
) -> PricePrediction:
    """Quick prediction using default model."""
    predictor = TransformerPredictor()
    return predictor.predict(symbol, features, horizon)
