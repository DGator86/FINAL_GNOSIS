"""
LSTM Forecaster with Multi-Head Attention for price/volatility prediction
Supports multi-horizon forecasting with uncertainty quantification
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM Forecaster"""

    input_dim: int = 150  # Number of features
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    forecast_horizons: Optional[List[int]] = None  # [1, 5, 60, 1440] minutes
    sequence_length: int = 60  # lookback window

    # Training params
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.forecast_horizons is None:
            self.forecast_horizons = [1, 5, 60, 1440]  # 1min, 5min, 1hr, 1day


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting"""

    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

        # Create sequences
        self.X, self.y = self._create_sequences()

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for supervised learning"""
        X, y = [], []

        for i in range(self.sequence_length, len(self.features)):
            X.append(self.features[i - self.sequence_length : i])
            y.append(self.targets[i])

        return np.array(X), np.array(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx]))


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for time series"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        residual = x

        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        attention, attention_weights = self._scaled_dot_product_attention(Q, K, V)

        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(attention)
        output = self.layer_norm(output + residual)

        return output, attention_weights

    def _scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention = torch.matmul(attention_weights, V)
        return attention, attention_weights


class LSTMForecaster(nn.Module):
    """
    LSTM-based forecaster with multi-head attention for time series prediction

    Architecture:
    1. LSTM layers for temporal modeling
    2. Multi-head attention for important time step identification
    3. Multiple forecast heads for different horizons
    4. Uncertainty estimation via Monte Carlo dropout
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )

        self.attention = MultiHeadAttention(d_model=config.hidden_dim, num_heads=config.num_heads, dropout=config.dropout)

        self.forecast_heads = nn.ModuleDict()
        for horizon in config.forecast_horizons:
            self.forecast_heads[f"horizon_{horizon}"] = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1),
            )

        self.volatility_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus(),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, mc_samples: int = 1
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        if mc_samples > 1:
            return self._mc_forward(x, mc_samples, return_attention)

        lstm_out, _ = self.lstm(x)

        attended_out, attention_weights = self.attention(lstm_out)

        final_hidden = attended_out[:, -1, :]
        final_hidden = self.dropout(final_hidden)

        predictions: Dict[str, torch.Tensor] = {}
        for horizon in self.config.forecast_horizons:
            head_name = f"horizon_{horizon}"
            pred = self.forecast_heads[head_name](final_hidden)
            predictions[head_name] = pred.squeeze(-1)

        volatility = self.volatility_head(final_hidden).squeeze(-1)
        predictions["volatility"] = volatility

        if return_attention:
            return predictions, attention_weights
        return predictions

    def _mc_forward(self, x: torch.Tensor, mc_samples: int, return_attention: bool = False):
        self.train()

        all_predictions = []
        all_attention_weights = []

        with torch.no_grad():
            for _ in range(mc_samples):
                if return_attention:
                    pred, attn = self.forward(x, return_attention=True, mc_samples=1)
                    all_attention_weights.append(attn)
                else:
                    pred = self.forward(x, return_attention=False, mc_samples=1)

                all_predictions.append(pred)

        self.eval()

        mc_predictions: Dict[str, torch.Tensor] = {}

        stacked_preds: Dict[str, torch.Tensor] = {}
        for key in all_predictions[0].keys():
            stacked_preds[key] = torch.stack([p[key] for p in all_predictions])

        for key, stacked in stacked_preds.items():
            mc_predictions[f"{key}_mean"] = stacked.mean(dim=0)
            mc_predictions[f"{key}_std"] = stacked.std(dim=0)
            mc_predictions[f"{key}_samples"] = stacked

        if return_attention and all_attention_weights:
            avg_attention = torch.stack(all_attention_weights).mean(dim=0)
            return mc_predictions, avg_attention

        return mc_predictions

    def predict_with_uncertainty(self, x: torch.Tensor, mc_samples: int = 50) -> Dict[str, Dict[str, torch.Tensor]]:
        mc_preds = self._mc_forward(x, mc_samples)

        result: Dict[str, Dict[str, torch.Tensor]] = {}
        for horizon in self.config.forecast_horizons:
            key = f"horizon_{horizon}"
            if f"{key}_samples" in mc_preds:
                samples = mc_preds[f"{key}_samples"]

                result[key] = {
                    "mean": mc_preds[f"{key}_mean"],
                    "std": mc_preds[f"{key}_std"],
                    "lower_ci": torch.quantile(samples, 0.05, dim=0),
                    "upper_ci": torch.quantile(samples, 0.95, dim=0),
                    "samples": samples,
                }

        return result


class LSTMTrainer:
    """Trainer for LSTM Forecaster"""

    def __init__(self, model: LSTMForecaster, config: LSTMConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.criterion = nn.MSELoss()

        self.best_loss = float("inf")
        self.patience_counter = 0

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def prepare_data(
        self, features: pd.DataFrame, targets: pd.DataFrame, train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        X = features.values.astype(np.float32)

        y_list = []
        for horizon in self.config.forecast_horizons:
            if f"return_{horizon}min" in targets.columns:
                y_list.append(targets[f"return_{horizon}min"].values)
            else:
                returns = features.iloc[:, 0].pct_change(horizon).shift(-horizon)
                y_list.append(returns.fillna(0).values)

        y = np.column_stack(y_list).astype(np.float32)

        split_idx = int(len(X) * train_split)

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_dataset = TimeSeriesDataset(X_train, y_train, self.config.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.config.sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(x)

            total_batch_loss = 0.0
            for i, horizon in enumerate(self.config.forecast_horizons):
                pred_key = f"horizon_{horizon}"
                if pred_key in predictions:
                    horizon_loss = self.criterion(predictions[pred_key], y[:, i])
                    total_batch_loss += horizon_loss

            total_batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += total_batch_loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x)

                total_batch_loss = 0.0
                for i, horizon in enumerate(self.config.forecast_horizons):
                    pred_key = f"horizon_{horizon}"
                    if pred_key in predictions:
                        horizon_loss = self.criterion(predictions[pred_key], y[:, i])
                        total_batch_loss += horizon_loss

                total_loss += total_batch_loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        logger.info(f"Starting training for {self.config.max_epochs} epochs")

        for epoch in range(self.config.max_epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
            else:
                self.patience_counter += 1

            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return {"train_losses": self.train_losses, "val_losses": self.val_losses}

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_loss": self.best_loss,
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        logger.info(f"Checkpoint loaded from {path}")


class LSTMForecastModel:
    """
    Production wrapper for LSTM Forecaster
    Integrates with GNOSIS pipeline
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[LSTMForecaster] = None
        self.config: Optional[LSTMConfig] = None
        self.scaler = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location="cpu")

        self.config = checkpoint["config"]
        self.model = LSTMForecaster(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loaded LSTM model from {model_path}")

    def predict(
        self, features: np.ndarray, with_uncertainty: bool = True, mc_samples: int = 30
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if features.ndim == 2:
            features = features[np.newaxis, ...]

        x = torch.FloatTensor(features)

        if with_uncertainty:
            predictions = self.model.predict_with_uncertainty(x, mc_samples)

            result: Dict[str, Union[float, Dict[str, float]]] = {}
            for horizon in self.config.forecast_horizons:
                key = f"price_forecast_{horizon}min"
                if f"horizon_{horizon}" in predictions:
                    pred_data = predictions[f"horizon_{horizon}"]
                    result[key] = {
                        "value": float(pred_data["mean"].item()),
                        "uncertainty": float(pred_data["std"].item()),
                        "confidence_interval": {
                            "lower": float(pred_data["lower_ci"].item()),
                            "upper": float(pred_data["upper_ci"].item()),
                        },
                    }
        else:
            predictions = self.model(x)
            result = {}
            for horizon in self.config.forecast_horizons:
                key = f"price_forecast_{horizon}min"
                pred_key = f"horizon_{horizon}"
                if pred_key in predictions:
                    result[key] = float(predictions[pred_key].item())

        return result

    def get_attention_weights(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded")

        if features.ndim == 2:
            features = features[np.newaxis, ...]

        x = torch.FloatTensor(features)

        with torch.no_grad():
            _, attention_weights = self.model(x, return_attention=True)

        return attention_weights.mean(dim=1).squeeze(0).cpu().numpy()


if __name__ == "__main__":
    config = LSTMConfig(
        input_dim=150, hidden_dim=256, num_layers=3, forecast_horizons=[1, 5, 60, 1440], max_epochs=50
    )

    model = LSTMForecaster(config)
    trainer = LSTMTrainer(model, config)

    np.random.seed(42)
    n_samples = 10000
    n_features = 150

    features = pd.DataFrame(np.random.randn(n_samples, n_features))

    base_returns = np.random.randn(n_samples) * 0.01
    targets = pd.DataFrame()
    for horizon in config.forecast_horizons:
        targets[f"return_{horizon}min"] = np.roll(base_returns, -horizon)

    train_loader, val_loader = trainer.prepare_data(features, targets)

    history = trainer.train(train_loader, val_loader)

    trainer.save_checkpoint("lstm_model.pt")

    prod_model = LSTMForecastModel("lstm_model.pt")

    test_features = np.random.randn(60, 150)
    predictions = prod_model.predict(test_features, with_uncertainty=True)

    print("Predictions:")
    print(json.dumps(predictions, indent=2))
