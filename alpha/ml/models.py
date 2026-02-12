"""
Gnosis Alpha - ML Models

Lightweight models optimized for short-term directional prediction.
Uses scikit-learn compatible interface for easy deployment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        RandomForestRegressor,
        GradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.info("LightGBM not available, using sklearn alternatives")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost not available, using sklearn alternatives")


class ModelType(str, Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    LOGISTIC = "logistic"
    ENSEMBLE = "ensemble"


class PredictionType(str, Enum):
    """Type of prediction."""
    DIRECTION = "direction"  # BUY/SELL/HOLD
    PROBABILITY = "probability"  # Probability of each direction
    RETURN = "return"  # Expected return


@dataclass
class ModelConfig:
    """Configuration for Alpha ML models."""
    model_type: ModelType = ModelType.GRADIENT_BOOSTING
    prediction_type: PredictionType = PredictionType.DIRECTION
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_samples_leaf: int = 20
    
    # Feature scaling
    scale_features: bool = True
    
    # Classification thresholds
    buy_threshold: float = 0.02   # 2% expected return = BUY
    sell_threshold: float = -0.02  # -2% expected return = SELL
    
    # Confidence calibration
    calibrate_probabilities: bool = True
    
    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type.value,
            "prediction_type": self.prediction_type.value,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_leaf": self.min_samples_leaf,
            "scale_features": self.scale_features,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }


@dataclass
class Prediction:
    """Model prediction result."""
    direction: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-1
    expected_return: float  # Expected % return
    probabilities: Dict[str, float] = field(default_factory=dict)  # Per-class probabilities
    
    # Metadata
    model_version: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "probabilities": self.probabilities,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
        }


class AlphaModel(ABC):
    """Base class for Alpha ML models."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.version = "0.0.0"
        self.created_at: Optional[datetime] = None
        self.metrics: Dict[str, float] = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AlphaModel":
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Prediction:
        """Generate prediction."""
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config.to_dict(),
            "feature_names": self.feature_names,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metrics": self.metrics,
            "is_fitted": self.is_fitted,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "AlphaModel":
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Determine model class based on config
        config_dict = model_data.get("config", {})
        config = ModelConfig(
            model_type=ModelType(config_dict.get("model_type", "gradient_boosting")),
            prediction_type=PredictionType(config_dict.get("prediction_type", "direction")),
            n_estimators=config_dict.get("n_estimators", 100),
            max_depth=config_dict.get("max_depth", 6),
            learning_rate=config_dict.get("learning_rate", 0.1),
            min_samples_leaf=config_dict.get("min_samples_leaf", 20),
            scale_features=config_dict.get("scale_features", True),
            buy_threshold=config_dict.get("buy_threshold", 0.02),
            sell_threshold=config_dict.get("sell_threshold", -0.02),
        )
        
        # Create appropriate model instance
        if config.prediction_type == PredictionType.DIRECTION:
            instance = DirectionalClassifier(config)
        else:
            instance = ReturnPredictor(config)
        
        instance.model = model_data.get("model")
        instance.scaler = model_data.get("scaler")
        instance.feature_names = model_data.get("feature_names", [])
        instance.version = model_data.get("version", "0.0.0")
        instance.is_fitted = model_data.get("is_fitted", False)
        instance.metrics = model_data.get("metrics", {})
        
        created_at_str = model_data.get("created_at")
        if created_at_str:
            instance.created_at = datetime.fromisoformat(created_at_str)
        
        logger.info(f"Model loaded from {path} (version: {instance.version})")
        return instance


class DirectionalClassifier(AlphaModel):
    """
    Classifies market direction as BUY/SELL/HOLD.
    
    Uses a two-stage approach:
    1. Train a regressor to predict returns
    2. Classify based on return thresholds
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self.return_predictor = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "DirectionalClassifier":
        """
        Train the directional classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target returns (continuous)
            feature_names: Names of features for interpretability
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for training")
        
        if len(X) == 0:
            raise ValueError("No training data provided")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Scale features
        if self.config.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Create and train the underlying model
        if self.config.model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_samples=self.config.min_samples_leaf,
                verbose=-1,
            )
        elif self.config.model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=self.config.min_samples_leaf,
                verbosity=0,
            )
        elif self.config.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                n_jobs=-1,
            )
        else:
            # Default to Gradient Boosting
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
            )
        
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        self.created_at = datetime.now(timezone.utc)
        
        logger.info(f"DirectionalClassifier trained with {len(X)} samples")
        
        return self
    
    def predict(self, X: np.ndarray) -> Prediction:
        """
        Predict direction for input features.
        
        Args:
            X: Feature vector or matrix
            
        Returns:
            Prediction with direction, confidence, and expected return
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained")
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict return
        expected_return = float(self.model.predict(X_scaled)[0])
        
        # Classify direction based on thresholds
        if expected_return >= self.config.buy_threshold:
            direction = "BUY"
        elif expected_return <= self.config.sell_threshold:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        # Calculate confidence based on distance from threshold
        threshold_range = self.config.buy_threshold - self.config.sell_threshold
        
        if direction == "BUY":
            # Confidence increases as return exceeds buy threshold
            excess = expected_return - self.config.buy_threshold
            confidence = min(0.5 + (excess / threshold_range) * 0.5, 1.0)
        elif direction == "SELL":
            # Confidence increases as return falls below sell threshold
            excess = self.config.sell_threshold - expected_return
            confidence = min(0.5 + (excess / threshold_range) * 0.5, 1.0)
        else:
            # HOLD confidence based on how close to neutral
            mid_point = (self.config.buy_threshold + self.config.sell_threshold) / 2
            distance_from_mid = abs(expected_return - mid_point)
            confidence = max(0.5 - (distance_from_mid / threshold_range), 0.3)
        
        # Build probability estimates
        # Map expected return to approximate probabilities
        prob_buy = self._return_to_probability(expected_return, "BUY")
        prob_sell = self._return_to_probability(expected_return, "SELL")
        prob_hold = 1.0 - prob_buy - prob_sell
        
        return Prediction(
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            probabilities={
                "BUY": prob_buy,
                "SELL": prob_sell,
                "HOLD": max(0, prob_hold),
            },
            model_version=self.version,
        )
    
    def _return_to_probability(self, expected_return: float, direction: str) -> float:
        """Convert expected return to probability estimate for a direction."""
        if direction == "BUY":
            if expected_return >= self.config.buy_threshold:
                return min(0.5 + (expected_return - self.config.buy_threshold) * 5, 0.9)
            return max(0.1, 0.3 + expected_return * 5)
        elif direction == "SELL":
            if expected_return <= self.config.sell_threshold:
                return min(0.5 + (self.config.sell_threshold - expected_return) * 5, 0.9)
            return max(0.1, 0.3 - expected_return * 5)
        return 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        
        return {}


class ReturnPredictor(AlphaModel):
    """
    Predicts expected return magnitude.
    
    Useful for position sizing and risk assessment.
    """
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "ReturnPredictor":
        """Train the return predictor."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for training")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Scale features
        if self.config.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Create model
        if self.config.model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_samples=self.config.min_samples_leaf,
                verbose=-1,
            )
        elif self.config.model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                verbosity=0,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
            )
        
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        self.created_at = datetime.now(timezone.utc)
        
        return self
    
    def predict(self, X: np.ndarray) -> Prediction:
        """Predict expected return."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        expected_return = float(self.model.predict(X_scaled)[0])
        
        # Determine direction from return
        if expected_return >= self.config.buy_threshold:
            direction = "BUY"
            confidence = min(0.5 + abs(expected_return) * 5, 0.95)
        elif expected_return <= self.config.sell_threshold:
            direction = "SELL"
            confidence = min(0.5 + abs(expected_return) * 5, 0.95)
        else:
            direction = "HOLD"
            confidence = 0.5
        
        return Prediction(
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            model_version=self.version,
        )


class EnsembleModel(AlphaModel):
    """
    Ensemble of multiple models for robust predictions.
    
    Combines predictions from multiple model types.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self.models: List[AlphaModel] = []
        self.weights: List[float] = []
    
    def add_model(self, model: AlphaModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)
        self.weights.append(weight)
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "EnsembleModel":
        """Train all models in the ensemble."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create default ensemble if no models added
        if not self.models:
            # Add three different model types
            for model_type in [ModelType.GRADIENT_BOOSTING, ModelType.RANDOM_FOREST]:
                config = ModelConfig(model_type=model_type)
                model = DirectionalClassifier(config)
                self.add_model(model, weight=1.0)
        
        # Train each model
        for model in self.models:
            model.fit(X, y, feature_names)
        
        self.is_fitted = True
        self.created_at = datetime.now(timezone.utc)
        
        return self
    
    def predict(self, X: np.ndarray) -> Prediction:
        """Generate ensemble prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained")
        
        # Get predictions from all models
        predictions = [model.predict(X) for model in self.models]
        
        # Weighted average of expected returns
        total_weight = sum(self.weights)
        expected_return = sum(
            p.expected_return * w for p, w in zip(predictions, self.weights)
        ) / total_weight
        
        # Vote on direction (weighted)
        direction_votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        for pred, weight in zip(predictions, self.weights):
            direction_votes[pred.direction] += weight * pred.confidence
        
        direction = max(direction_votes, key=direction_votes.get)
        
        # Confidence from vote strength
        total_votes = sum(direction_votes.values())
        confidence = direction_votes[direction] / total_votes if total_votes > 0 else 0.5
        
        # Average probabilities
        prob_buy = sum(p.probabilities.get("BUY", 0) * w for p, w in zip(predictions, self.weights)) / total_weight
        prob_sell = sum(p.probabilities.get("SELL", 0) * w for p, w in zip(predictions, self.weights)) / total_weight
        prob_hold = 1.0 - prob_buy - prob_sell
        
        return Prediction(
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            probabilities={
                "BUY": prob_buy,
                "SELL": prob_sell,
                "HOLD": max(0, prob_hold),
            },
            model_version=self.version,
        )


def create_model(config: Optional[ModelConfig] = None) -> AlphaModel:
    """Factory function to create appropriate model."""
    config = config or ModelConfig()
    
    if config.model_type == ModelType.ENSEMBLE:
        return EnsembleModel(config)
    elif config.prediction_type == PredictionType.RETURN:
        return ReturnPredictor(config)
    else:
        return DirectionalClassifier(config)
