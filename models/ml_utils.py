"""Shared ML utilities for lookahead models and hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class SeriesScaler:
    """Wrapper around StandardScaler that preserves fit stats for reuse."""

    scaler: StandardScaler = StandardScaler()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)


def build_sequences(data: np.ndarray, labels: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized sequence builder for LSTM inputs.

    Args:
        data: 2D numpy array (timesteps x features)
        labels: 1D label array aligned with data
        sequence_length: length of each window
    """

    if len(data) <= sequence_length:
        raise ValueError("Not enough data to build sequences")

    windows = np.lib.stride_tricks.sliding_window_view(data, (sequence_length, data.shape[1]))
    windows = windows.reshape(-1, sequence_length, data.shape[1])
    target = labels[sequence_length - 1 : len(windows) + sequence_length - 1]
    return windows, target


def cross_validate_time_series(model_builder, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """Generic rolling-origin cross validation returning mean score."""

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model = model_builder()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))
    return float(np.mean(scores))


# Test: assert build_sequences(np.ones((40, 2)), np.ones(40), 16)[0].shape[1] == 16
