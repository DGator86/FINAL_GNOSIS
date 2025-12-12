"""
LSTM Lookahead Model for Short-Term Price Predictions
Optimized for real-time trading with bidirectional processing and uncertainty quantification
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class LookaheadConfig:
    """Configuration for LSTM Lookahead Model"""

    input_dim: int = 150  # Number of features
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    forecast_horizons: List[int] = None  # [1, 5, 15, 60] minutes
    sequence_length: int = 60  # lookback window
    bidirectional: bool = True

    # Training params
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.forecast_horizons is None:
            self.forecast_horizons = [1, 5, 15, 60]  # 1min, 5min, 15min, 1hr


class BidirectionalLSTMLookahead(nn.Module):
    """Bidirectional LSTM for multi-horizon price prediction"""

    def __init__(self, config: LookaheadConfig):
        super().__init__()
        self.config = config

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        # Calculate LSTM output dimension
        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim

        # Attention mechanism for weighted temporal aggregation
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1),
        )

        # Prediction heads for each horizon
        self.prediction_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(lstm_output_dim // 2, 1),
            )
            for h in config.forecast_horizons
        })

        # Uncertainty estimation heads (predict log variance)
        self.uncertainty_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 4),
                nn.ReLU(),
                nn.Linear(lstm_output_dim // 4, 1),
                nn.Softplus(),  # Ensure positive variance
            )
            for h in config.forecast_horizons
        })

        # Direction classifier (up/down/neutral)
        self.direction_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_dim // 2, 3),  # up, down, neutral
        )

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-horizon predictions

        Args:
            x: [batch_size, seq_len, input_dim]

        Returns:
            predictions: Dict of {horizon: [batch_size, 1]} price changes
            uncertainties: Dict of {horizon: [batch_size, 1]} prediction uncertainties
            directions: [batch_size, 3] direction probabilities
            attention_weights: [batch_size, seq_len, 1] attention weights
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch_size, seq_len, lstm_output_dim]

        # Attention mechanism - focus on important timesteps
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Context vector - weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, lstm_output_dim]

        # Multi-horizon predictions
        predictions = {}
        uncertainties = {}
        for horizon in self.config.forecast_horizons:
            key = f"horizon_{horizon}"
            predictions[key] = self.prediction_heads[key](context)
            uncertainties[key] = self.uncertainty_heads[key](context)

        # Direction classification
        directions = torch.softmax(self.direction_classifier(context), dim=-1)

        return predictions, uncertainties, directions, attention_weights


class LSTMLookaheadPredictor:
    """High-level interface for LSTM lookahead predictions"""

    def __init__(self, config: Optional[LookaheadConfig] = None, model_path: Optional[str] = None):
        self.config = config or LookaheadConfig()
        self.model = BidirectionalLSTMLookahead(self.config).to(self.config.device)
        self.scaler = StandardScaler()
        self.is_fitted = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def prepare_data(self, features: pd.DataFrame, target_col: str = "close") -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for training

        Args:
            features: DataFrame with features and target column
            target_col: Name of the target column (price)

        Returns:
            X: Feature sequences [n_samples, seq_len, n_features]
            y: Target values (future returns at each horizon)
        """
        if target_col not in features.columns:
            raise ValueError(f"Target column '{target_col}' not found in features")

        # Calculate returns for each horizon
        targets = {}
        for horizon in self.config.forecast_horizons:
            # Future return at this horizon
            targets[f"return_{horizon}"] = (
                features[target_col].shift(-horizon) / features[target_col] - 1
            ) * 100  # Convert to percentage

        # Drop target column from features
        feature_cols = [col for col in features.columns if col != target_col]
        X_df = features[feature_cols].copy()

        # Handle missing values
        X_df = X_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Normalize features
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X_df)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X_df)

        # Create sequences
        X_sequences = []
        y_sequences = []

        max_horizon = max(self.config.forecast_horizons)
        for i in range(self.config.sequence_length, len(X_scaled) - max_horizon):
            # Input sequence
            X_sequences.append(X_scaled[i - self.config.sequence_length : i])

            # Target vector (returns for all horizons)
            y_vec = []
            for horizon in self.config.forecast_horizons:
                target_val = targets[f"return_{horizon}"].iloc[i]
                y_vec.append(target_val if not pd.isna(target_val) else 0.0)
            y_sequences.append(y_vec)

        return np.array(X_sequences), np.array(y_sequences)

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the LSTM model

        Args:
            X: Feature sequences [n_samples, seq_len, n_features]
            y: Targets [n_samples, n_horizons]
            validation_split: Fraction of data for validation

        Returns:
            history: Training history with losses
        """
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.config.device)
        y_train_t = torch.FloatTensor(y_train).to(self.config.device)
        X_val_t = torch.FloatTensor(X_val).to(self.config.device)
        y_val_t = torch.FloatTensor(y_val).to(self.config.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"Training LSTM Lookahead model for {self.config.max_epochs} epochs")

        for epoch in range(self.config.max_epochs):
            # Training
            self.model.train()
            train_loss = self._train_epoch(X_train_t, y_train_t, optimizer)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = self._validate(X_val_t, y_val_t)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return history

    def _train_epoch(self, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        batch_size = self.config.batch_size
        n_batches = len(X) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            optimizer.zero_grad()

            # Forward pass
            predictions, uncertainties, directions, _ = self.model(X_batch)

            # Multi-horizon loss with uncertainty weighting
            loss = 0.0
            for idx, horizon in enumerate(self.config.forecast_horizons):
                key = f"horizon_{horizon}"
                pred = predictions[key].squeeze()
                target = y_batch[:, idx]
                uncertainty = uncertainties[key].squeeze()

                # Heteroscedastic loss (uncertainty-weighted MSE)
                mse = (pred - target) ** 2
                loss += torch.mean(mse / (2 * uncertainty) + 0.5 * torch.log(uncertainty))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / n_batches

    def _validate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Validation loss"""
        predictions, uncertainties, _, _ = self.model(X)

        loss = 0.0
        for idx, horizon in enumerate(self.config.forecast_horizons):
            key = f"horizon_{horizon}"
            pred = predictions[key].squeeze()
            target = y[:, idx]
            uncertainty = uncertainties[key].squeeze()

            mse = (pred - target) ** 2
            loss += torch.mean(mse / (2 * uncertainty) + 0.5 * torch.log(uncertainty))

        return loss.item()

    def predict(self, features: np.ndarray) -> Dict[str, any]:
        """
        Make predictions on new data

        Args:
            features: Recent feature sequence [seq_len, n_features] or [batch, seq_len, n_features]

        Returns:
            Dictionary with predictions, uncertainties, and direction probabilities
        """
        self.model.eval()

        # Handle single sequence
        if features.ndim == 2:
            features = features[np.newaxis, :]

        # Ensure correct sequence length
        if features.shape[1] != self.config.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.config.sequence_length}, got {features.shape[1]}"
            )

        # Normalize
        original_shape = features.shape
        features_flat = features.reshape(-1, features.shape[-1])
        features_scaled = self.scaler.transform(features_flat)
        features_scaled = features_scaled.reshape(original_shape)

        # Convert to tensor
        X = torch.FloatTensor(features_scaled).to(self.config.device)

        with torch.no_grad():
            predictions, uncertainties, directions, attention_weights = self.model(X)

        # Convert to dict format
        result = {
            "predictions": {},
            "uncertainties": {},
            "direction_probs": {
                "up": directions[0, 0].item(),
                "neutral": directions[0, 1].item(),
                "down": directions[0, 2].item(),
            },
            "direction": ["up", "neutral", "down"][torch.argmax(directions[0]).item()],
            "attention_weights": attention_weights[0].cpu().numpy(),
        }

        for horizon in self.config.forecast_horizons:
            key = f"horizon_{horizon}"
            result["predictions"][horizon] = predictions[key][0, 0].item()
            result["uncertainties"][horizon] = uncertainties[key][0, 0].item()

        return result

    def save(self, path: str):
        """Save model and scaler"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "config": self.config,
            "is_fitted": self.is_fitted,
        }

        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and scaler"""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.config = checkpoint["config"]
        self.model = BidirectionalLSTMLookahead(self.config).to(self.config.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler = checkpoint["scaler"]
        self.is_fitted = checkpoint["is_fitted"]

        self.model.eval()
        logger.info(f"Model loaded from {path}")
