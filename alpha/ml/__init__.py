"""
Gnosis Alpha - Machine Learning Framework

Lightweight ML optimized for short-term directional trading:
- 0-7 day prediction horizon
- Fast inference for real-time signals
- Walk-forward validation for realistic backtesting
- Feature engineering from technical indicators

Models:
- DirectionalClassifier: Predicts BUY/SELL/HOLD
- ReturnRegressor: Predicts expected return magnitude
- ConfidenceCalibrator: Calibrates signal confidence

Usage:
    from alpha.ml import (
        AlphaFeatureEngine,
        AlphaModel,
        AlphaBacktester,
        AlphaTrainer,
    )
    
    # Train a model
    trainer = AlphaTrainer()
    model = trainer.train(symbols=["AAPL", "TSLA"], lookback_days=365)
    
    # Generate ML-enhanced signals
    features = AlphaFeatureEngine().extract("AAPL")
    prediction = model.predict(features)
"""

from alpha.ml.features import AlphaFeatureEngine, FeatureSet
from alpha.ml.models import (
    AlphaModel,
    DirectionalClassifier,
    ReturnPredictor,
    ModelConfig,
)
from alpha.ml.backtest import AlphaBacktester, BacktestResult
from alpha.ml.trainer import AlphaTrainer, TrainingConfig

__all__ = [
    # Features
    "AlphaFeatureEngine",
    "FeatureSet",
    # Models
    "AlphaModel",
    "DirectionalClassifier",
    "ReturnPredictor",
    "ModelConfig",
    # Backtesting
    "AlphaBacktester",
    "BacktestResult",
    # Training
    "AlphaTrainer",
    "TrainingConfig",
]

__version__ = "1.0.0"
