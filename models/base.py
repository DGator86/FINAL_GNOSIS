"""
Base Model Class for GNOSIS ML Models
Provides common functionality for all ML models in the trading system.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


class BaseGnosisModel(ABC):
    """Base class for all GNOSIS ML models."""

    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.model: Any = None
        self.scaler: Any = None
        self.is_trained = False
        self.training_history: List[Dict[str, Any]] = []
        self.feature_importance: Dict[str, float] = {}
        self.last_updated: Optional[str] = None
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup model-specific logger."""
        logger = logging.getLogger(f"gnosis.ml.{self.model_name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""

    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
            "is_trained": self.is_trained,
            "timestamp": datetime.now().isoformat(),
        }

        path = Path(filepath)
        if path.parent and path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_data, path)
        self.last_updated = model_data["timestamp"]
        self.logger.info("Model saved to %s", path)

    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data: Dict[str, Any] = joblib.load(path)
        self.model = model_data.get("model")
        self.scaler = model_data.get("scaler")
        self.config = model_data.get("config", self.config)
        self.feature_importance = model_data.get("feature_importance", {})
        self.training_history = model_data.get("training_history", [])
        self.is_trained = model_data.get("is_trained", False)
        self.last_updated = model_data.get("timestamp")

        self.logger.info("Model loaded from %s", path)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance

    def validate_input(self, X: np.ndarray | pd.DataFrame | List[Any]) -> np.ndarray:
        """Validate and preprocess input data."""
        if X is None or len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X_array = np.asarray(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        return X_array.astype(np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "config": self.config,
            "training_epochs": len(self.training_history),
            "feature_count": len(self.feature_importance),
            "last_updated": self.last_updated,
        }

    def export_config(self) -> str:
        """Export the current model configuration as a JSON string."""
        return json.dumps(self.config, indent=2)

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration values for the model and return the new config."""
        self.config.update(updates)
        self.last_updated = datetime.now().isoformat()
        return self.config
