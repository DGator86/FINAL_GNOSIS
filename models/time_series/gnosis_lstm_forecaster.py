"""
LSTM Forecaster with Attention for GNOSIS Trading System
Multi-horizon time series prediction with uncertainty quantification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from .attention_mechanism import AttentionLayer
from .base_model import BaseGnosisModel


class LSTMForecaster(nn.Module):
    """LSTM with attention for multi-horizon forecasting."""
warnings.filterwarnings("ignore")


class AttentionLSTMBackbone(nn.Module):
    """Backbone LSTM with optional attention for multi-horizon forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        if use_attention:
            self.attention = AttentionLayer(hidden_dim, num_heads=8)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the LSTM forecaster."""
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the backbone forecaster."""

        lstm_out, _ = self.lstm(x)

        attention_weights = None
        if self.use_attention:
            attended_out, attention_weights = self.attention(lstm_out)
            final_output = attended_out[:, -1, :]
        else:
            final_output = lstm_out[:, -1, :]

        predictions = self.fc_layers(final_output)
        uncertainty = self.uncertainty_head(final_output)

        return predictions, uncertainty, attention_weights


class GnosisLSTMForecaster(BaseGnosisModel):
    """GNOSIS LSTM Forecaster with production features."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "lstm_forecaster")

        self.sequence_length = config.get("sequence_length", 60)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.uncertainty_weight = config.get("uncertainty_weight", 0.1)
        self.horizons: List[int] = config.get("horizons", [1, 5, 15, 30])
        self.use_attention = config.get("use_attention", True)
        self.uncertainty_loss_weight = config.get("uncertainty_loss_weight", 0.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.scaler = StandardScaler()
        self.models: Dict[int, Dict[str, Any]] = {}

    def _create_sequences(
        self, features: np.ndarray, targets: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised learning sequences for a given horizon."""

        X, y = [], []

        for i in range(self.sequence_length, len(features) - horizon + 1):
            X.append(features[i - self.sequence_length : i])
            y.append(targets[i + horizon - 1])

        return np.array(X), np.array(y)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train LSTM models for multiple horizons."""

        validation_split = kwargs.get("validation_split", 0.2)
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 32)
        early_stopping_patience = kwargs.get("early_stopping_patience", 10)

        self.logger.info("Starting LSTM training for multiple horizons")

        # Ensure X is at least 3D: [samples, sequence_length, features]
        X_input = np.atleast_2d(X)
        if X_input.ndim == 2:
            # If 2D [samples, features], reshape to [samples, 1, features]
            X_input = X_input.reshape(X_input.shape[0], 1, X_input.shape[1])
        
        X_scaled = self.scaler.fit_transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)
        # Validate and handle input shape
        if X.ndim == 1:
            raise ValueError(
                f"Expected X to be 2D [timesteps, n_features], but got 1D shape {X.shape}"
            )
        elif X.ndim == 2:
            # X is [timesteps, n_features] - expected format
            self.logger.info(f"Input shape: {X.shape} [timesteps, n_features]")
            X_scaled = self.scaler.fit_transform(X)
        elif X.ndim == 3:
            # X is [n_samples, sequence_length, n_features] - need to flatten for scaling
            self.logger.info(
                f"Input shape: {X.shape} [n_samples, sequence_length, n_features] - "
                "flattening for scaling"
            )
            original_shape = X.shape
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(original_shape)
        else:
            raise ValueError(
                f"Expected X to be 2D or 3D, but got {X.ndim}D with shape {X.shape}"
            )
        # Defensive shape handling: ensure X is at least 2D
        X_input = np.atleast_2d(X)
        X_reshaped = X_input.reshape(-1, X_input.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped).reshape(X_input.shape)
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        training_results: Dict[int, Dict[str, Any]] = {}

        for horizon in self.horizons:
            self.logger.info(f"Training model for horizon {horizon}")

            X_seq_train, y_seq_train = self._create_sequences(X_train, y_train, horizon)
            X_seq_val, y_seq_val = self._create_sequences(X_val, y_val, horizon)

            if len(X_seq_train) == 0 or len(X_seq_val) == 0:
                self.logger.warning(
                    "Insufficient data to create sequences for horizon %s; skipping.", horizon
                )
                continue

            input_dim = X_seq_train.shape[-1]
            model = LSTMForecaster(
            model = AttentionLSTMBackbone(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=1,
                dropout=self.dropout,
                use_attention=self.use_attention,
            ).to(self.device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            )

            best_val_loss = float("inf")
            patience_counter = 0
            train_losses: List[float] = []
            val_losses: List[float] = []

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_batches = 0


                for i in range(0, len(X_seq_train), batch_size):
                    batch_X = torch.FloatTensor(X_seq_train[i : i + batch_size]).to(self.device)
                    batch_y = torch.FloatTensor(y_seq_train[i : i + batch_size]).to(self.device)

                    optimizer.zero_grad()
                    predictions, uncertainty, _ = model(batch_X)

                    mse_loss = criterion(predictions.squeeze(), batch_y)
                    uncertainty_loss = torch.mean(uncertainty)
                    loss = mse_loss + self.uncertainty_weight * uncertainty_loss
                    loss = mse_loss + self.uncertainty_loss_weight * uncertainty_loss
                    loss = mse_loss + 0.1 * uncertainty_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_batches += 1

                model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for i in range(0, len(X_seq_val), batch_size):
                        batch_X = torch.FloatTensor(X_seq_val[i : i + batch_size]).to(
                            self.device
                        )
                        batch_y = torch.FloatTensor(y_seq_val[i : i + batch_size]).to(
                            self.device
                        )

                        predictions, _, _ = model(batch_X)
                        loss = criterion(predictions.squeeze(), batch_y)
                        val_loss += loss.item()
                        val_batches += 1

                train_loss /= max(1, train_batches)
                val_loss /= max(1, val_batches)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.models[horizon] = {
                        "model": model.state_dict(),
                        "scaler": self.scaler,
                        "config": {
                            "input_dim": input_dim,
                            "hidden_dim": self.hidden_dim,
                            "num_layers": self.num_layers,
                            "output_dim": 1,
                            "dropout": self.dropout,
                            "use_attention": self.use_attention,
                        },
                    }
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    self.logger.info(
                        "Early stopping at epoch %s for horizon %s", epoch, horizon
                    )
                    break

                if epoch % 10 == 0:
                    self.logger.info(
                        "Horizon %s, Epoch %s: Train Loss: %.6f, Val Loss: %.6f",
                        horizon,
                        epoch,
                        train_loss,
                        val_loss,
                    )

            training_results[horizon] = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "final_epoch": len(train_losses),
            }

        self.is_trained = bool(self.models)
        self.training_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "results": training_results,
                "config": self.config,
            }
        )

        self.logger.info("LSTM training completed for all horizons")
        return training_results

    def predict(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate multi-horizon predictions with optional uncertainty."""

        if not self.is_trained or not self.models:
            raise ValueError("Model must be trained before making predictions")

        return_uncertainty = kwargs.get("return_uncertainty", True)

        # Ensure X is at least 3D: [samples, sequence_length, features]
        X_input = np.atleast_2d(X)
        if X_input.ndim == 2:
            # If 2D [samples, features], reshape to [samples, 1, features]
            X_input = X_input.reshape(X_input.shape[0], 1, X_input.shape[1])
        
        X_scaled = self.scaler.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)
        # Defensive shape handling: ensure X is at least 2D
        X_input = np.atleast_2d(X)
        X_reshaped = X_input.reshape(-1, X_input.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped).reshape(X_input.shape)
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        predictions: Dict[str, np.ndarray] = {}
        uncertainties: Dict[str, np.ndarray] = {}
        attention_weights: Dict[str, np.ndarray] = {}

        for horizon in self.horizons:
            if horizon not in self.models:
                continue

            model_config = self.models[horizon]["config"]
            model = LSTMForecaster(**model_config).to(self.device)
            model = AttentionLSTMBackbone(**model_config).to(self.device)
            model.load_state_dict(self.models[horizon]["model"])
            model.eval()

            X_seq: List[np.ndarray] = []
            for i in range(self.sequence_length, len(X_scaled) + 1):
                X_seq.append(X_scaled[i - self.sequence_length : i])

            if not X_seq:
                self.logger.warning(
                    "Insufficient data to create prediction sequences for horizon %s", horizon
                )
                continue

            X_seq_arr = np.array(X_seq)

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq_arr).to(self.device)
                preds, uncs, att_weights = model(X_tensor)

                predictions[f"horizon_{horizon}"] = preds.cpu().numpy().flatten()

                if return_uncertainty:
                    uncertainties[f"horizon_{horizon}"] = uncs.cpu().numpy().flatten()

                if att_weights is not None:
                    attention_weights[f"horizon_{horizon}"] = att_weights.cpu().numpy()

        result: Dict[str, Any] = {"predictions": predictions}

        if return_uncertainty:
            result["uncertainties"] = uncertainties
            result["attention_weights"] = attention_weights

        return result
