"""LSTM-based lookahead model with time-series cross validation and Bayesian tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from models import ml_utils
from schemas.core_schemas import DirectionEnum, LSTMLookaheadSnapshot


class LookaheadLSTM(nn.Module):
    """Simple two-layer LSTM forecaster."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


@dataclass
class LookaheadConfig:
    horizon: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    epochs: int = 20
    lr: float = 1e-3
    max_trials: int = 10
    sequence_length: int = 32


class LookaheadModel:
    """Production-grade LSTM lookahead with Optuna tuning and vectorized pipeline."""

    def __init__(self, config: Optional[LookaheadConfig] = None) -> None:
        self.config = config or LookaheadConfig()
        self.model: Optional[LookaheadLSTM] = None
        self.scaler: Optional[ml_utils.SeriesScaler] = None

    def _build_datasets(self, features: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        if target_column not in features.columns:
            raise ValueError(f"Missing target column {target_column}")

        df = features.dropna(subset=[target_column]).copy()
        y = df[target_column].pct_change().fillna(0.0).to_numpy()
        X = df.drop(columns=[target_column])

        self.scaler = ml_utils.SeriesScaler()
        X_scaled = self.scaler.fit_transform(X.to_numpy())

        sequences, labels = ml_utils.build_sequences(
            X_scaled, y, sequence_length=self.config.sequence_length
        )
        return sequences, labels

    def _objective(self, trial: optuna.Trial, sequences: np.ndarray, labels: np.ndarray) -> float:
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        model = LookaheadLSTM(input_size=sequences.shape[-1], hidden_size=hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        tscv = TimeSeriesSplit(n_splits=5)

        losses: list[float] = []
        for train_idx, val_idx in tscv.split(sequences):
            train_x = torch.tensor(sequences[train_idx], dtype=torch.float32)
            val_x = torch.tensor(sequences[val_idx], dtype=torch.float32)
            train_y = torch.tensor(labels[train_idx], dtype=torch.float32).unsqueeze(-1)
            val_y = torch.tensor(labels[val_idx], dtype=torch.float32).unsqueeze(-1)

            for _ in range(5):
                optimizer.zero_grad()
                preds = model(train_x)
                loss = criterion(preds, train_y)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                val_loss = criterion(model(val_x), val_y).item()
            losses.append(val_loss)

        return float(np.mean(losses))

    def train(self, features: pd.DataFrame, target_column: str = "target_regime") -> Tuple[float, float]:
        sequences, labels = self._build_datasets(features, target_column)

        def optuna_objective(trial: optuna.Trial) -> float:
            return self._objective(trial, sequences, labels)

        study = optuna.create_study(direction="minimize")
        study.optimize(optuna_objective, n_trials=self.config.max_trials)
        best_params = study.best_trial.params

        self.model = LookaheadLSTM(
            input_size=sequences.shape[-1],
            hidden_size=best_params.get("hidden_size", self.config.hidden_size),
            num_layers=self.config.num_layers,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=best_params.get("lr", self.config.lr))
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(sequences, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

        # Simple train/val split for diagnostics
        midpoint = int(0.8 * len(labels))
        with torch.no_grad():
            train_loss = criterion(self.model(X_tensor[:midpoint]), y_tensor[:midpoint]).item()
            val_loss = criterion(self.model(X_tensor[midpoint:]), y_tensor[midpoint:]).item()
        logger.info("Lookahead LSTM train_loss=%.4f val_loss=%.4f", train_loss, val_loss)
        return train_loss, val_loss

    def predict(self, features: pd.DataFrame) -> LSTMLookaheadSnapshot:
        if not self.model or not self.scaler:
            raise RuntimeError("Model not trained")

        X_scaled = self.scaler.transform(features.to_numpy())
        sequence = torch.tensor([X_scaled[-self.config.sequence_length :]], dtype=torch.float32)
        with torch.no_grad():
            pred_pct = float(self.model(sequence).squeeze().item())

        direction = (
            DirectionEnum.LONG.value
            if pred_pct > 0
            else DirectionEnum.SHORT.value
            if pred_pct < 0
            else DirectionEnum.NEUTRAL.value
        )
        direction_probs = {
            DirectionEnum.LONG.value: max(0.0, pred_pct),
            DirectionEnum.SHORT.value: max(0.0, -pred_pct),
            DirectionEnum.NEUTRAL.value: max(0.0, 1 - abs(pred_pct)),
        }
        return LSTMLookaheadSnapshot(
            timestamp=features.index[-1],
            symbol=str(features.index.name or "unknown"),
            horizons=[self.config.horizon],
            predictions={self.config.horizon: pred_pct},
            uncertainties={self.config.horizon: np.std(X_scaled)},
            direction=direction,
            direction_probs=direction_probs,
            confidence=float(min(1.0, abs(pred_pct))),
            model_version="lstm_lookahead_v2",
            forecast_returns=[pred_pct],
        )

    def save(self, path: Path) -> None:
        if not self.model or not self.scaler:
            raise RuntimeError("Cannot save an untrained model")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"state_dict": self.model.state_dict(), "scaler": self.scaler}, path)
        logger.info("Saved lookahead model to %s", path)

    def load(self, path: Path, input_size: Optional[int] = None) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = joblib.load(path)
        state_dict = payload["state_dict"]
        scaler = payload.get("scaler")
        self.scaler = scaler
        if input_size is None:
            input_size = next(iter(state_dict.values())).shape[-1]
        self.model = LookaheadLSTM(input_size=input_size, hidden_size=self.config.hidden_size)
        self.model.load_state_dict(state_dict)
        logger.info("Loaded lookahead model from %s", path)
