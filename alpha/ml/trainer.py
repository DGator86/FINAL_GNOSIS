"""
Gnosis Alpha - Model Training Pipeline

End-to-end training with cross-validation and hyperparameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

import numpy as np

logger = logging.getLogger(__name__)

from alpha.ml.features import AlphaFeatureEngine
from alpha.ml.models import (
    AlphaModel,
    DirectionalClassifier,
    ReturnPredictor,
    EnsembleModel,
    ModelConfig,
    ModelType,
)
from alpha.ml.backtest import AlphaBacktester, BacktestConfig, BacktestResult


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Training data
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
        "META", "TSLA", "AMD", "CRM", "NFLX"
    ])
    train_start_days_ago: int = 365  # 1 year of training data
    validation_split: float = 0.2    # 20% for validation
    
    # Prediction target
    forward_days: int = 5  # Predict 5-day return
    
    # Model selection
    model_type: ModelType = ModelType.GRADIENT_BOOSTING
    use_ensemble: bool = False
    
    # Hyperparameter search
    tune_hyperparameters: bool = True
    n_trials: int = 20
    
    # Output
    model_dir: str = "alpha/models"
    model_name: str = "alpha_directional"


@dataclass
class TrainingResult:
    """Results from model training."""
    model: AlphaModel
    config: TrainingConfig
    
    # Training metrics
    train_samples: int = 0
    val_samples: int = 0
    train_rmse: float = 0.0
    val_rmse: float = 0.0
    train_direction_accuracy: float = 0.0
    val_direction_accuracy: float = 0.0
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Best hyperparameters (if tuned)
    best_params: Dict[str, Any] = field(default_factory=dict)
    
    # Model path
    model_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "config": {
                "symbols": self.config.symbols,
                "forward_days": self.config.forward_days,
                "model_type": self.config.model_type.value,
            },
            "metrics": {
                "train_samples": self.train_samples,
                "val_samples": self.val_samples,
                "train_rmse": self.train_rmse,
                "val_rmse": self.val_rmse,
                "train_direction_accuracy": self.train_direction_accuracy,
                "val_direction_accuracy": self.val_direction_accuracy,
            },
            "best_params": self.best_params,
            "model_path": self.model_path,
            "top_features": dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
        }
    
    def print_summary(self) -> None:
        """Print training summary."""
        print("\n" + "="*60)
        print("  GNOSIS ALPHA - Training Results")
        print("="*60)
        
        print(f"\n📊 Data:")
        print(f"  Training samples: {self.train_samples:,}")
        print(f"  Validation samples: {self.val_samples:,}")
        print(f"  Symbols: {', '.join(self.config.symbols[:5])}...")
        
        print(f"\n📈 Performance:")
        print(f"  Train RMSE: {self.train_rmse:.4f}")
        print(f"  Val RMSE: {self.val_rmse:.4f}")
        print(f"  Train Direction Accuracy: {self.train_direction_accuracy*100:.1f}%")
        print(f"  Val Direction Accuracy: {self.val_direction_accuracy*100:.1f}%")
        
        if self.best_params:
            print(f"\n⚙️ Best Hyperparameters:")
            for k, v in self.best_params.items():
                print(f"  {k}: {v}")
        
        if self.feature_importance:
            print(f"\n🔑 Top Features:")
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for name, importance in sorted_features:
                print(f"  {name}: {importance:.4f}")
        
        if self.model_path:
            print(f"\n💾 Model saved: {self.model_path}")
        
        print("="*60 + "\n")


class AlphaTrainer:
    """
    End-to-end model training pipeline.
    
    Features:
    - Multi-symbol training data aggregation
    - Train/validation split with time-based separation
    - Optional hyperparameter tuning
    - Walk-forward cross-validation
    - Model persistence
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.feature_engine = AlphaFeatureEngine()
    
    def train(
        self,
        symbols: Optional[List[str]] = None,
        save_model: bool = True,
    ) -> TrainingResult:
        """
        Train a model on historical data.
        
        Args:
            symbols: Override default symbols
            save_model: Whether to save model to disk
            
        Returns:
            TrainingResult with trained model and metrics
        """
        if symbols:
            self.config.symbols = symbols
        
        logger.info(f"Training Alpha model on {len(self.config.symbols)} symbols")
        
        # Build training dataset
        X_train, y_train, X_val, y_val = self._build_dataset()
        
        if len(X_train) == 0:
            raise ValueError("No training data available")
        
        logger.info(f"Dataset: {len(X_train)} train, {len(X_val)} validation samples")
        
        # Tune hyperparameters if enabled
        if self.config.tune_hyperparameters:
            best_params = self._tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            best_params = {}
        
        # Create and train model
        model_config = ModelConfig(
            model_type=self.config.model_type,
            n_estimators=best_params.get("n_estimators", 100),
            max_depth=best_params.get("max_depth", 6),
            learning_rate=best_params.get("learning_rate", 0.1),
            min_samples_leaf=best_params.get("min_samples_leaf", 20),
        )
        
        if self.config.use_ensemble:
            model = EnsembleModel(model_config)
        else:
            model = DirectionalClassifier(model_config)
        
        model.fit(X_train, y_train, feature_names=self.feature_engine.FEATURE_NAMES)
        
        # Evaluate
        train_metrics = self._evaluate(model, X_train, y_train)
        val_metrics = self._evaluate(model, X_val, y_val)
        
        # Get feature importance
        feature_importance = {}
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
        
        # Create result
        result = TrainingResult(
            model=model,
            config=self.config,
            train_samples=len(X_train),
            val_samples=len(X_val),
            train_rmse=train_metrics["rmse"],
            val_rmse=val_metrics["rmse"],
            train_direction_accuracy=train_metrics["direction_accuracy"],
            val_direction_accuracy=val_metrics["direction_accuracy"],
            feature_importance=feature_importance,
            best_params=best_params,
        )
        
        # Save model
        if save_model:
            model_path = self._save_model(model)
            result.model_path = model_path
        
        logger.info(f"Training complete. Val accuracy: {val_metrics['direction_accuracy']*100:.1f}%")
        
        return result
    
    def _build_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build training and validation datasets."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.train_start_days_ago)
        
        # Split point for validation
        val_start = end_date - timedelta(
            days=int(self.config.train_start_days_ago * self.config.validation_split)
        )
        
        all_train_X = []
        all_train_y = []
        all_val_X = []
        all_val_y = []
        
        for symbol in self.config.symbols:
            try:
                # Get full dataset
                X, y, dates = self.feature_engine.build_training_dataset(
                    symbol,
                    start_date,
                    end_date,
                    forward_days=self.config.forward_days,
                )
                
                if len(X) == 0:
                    continue
                
                # Split by time
                for i, date in enumerate(dates):
                    # Convert to datetime if needed
                    if hasattr(date, 'to_pydatetime'):
                        date = date.to_pydatetime()
                    
                    if date < val_start:
                        all_train_X.append(X[i])
                        all_train_y.append(y[i])
                    else:
                        all_val_X.append(X[i])
                        all_val_y.append(y[i])
                
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
        
        return (
            np.array(all_train_X) if all_train_X else np.array([]),
            np.array(all_train_y) if all_train_y else np.array([]),
            np.array(all_val_X) if all_val_X else np.array([]),
            np.array(all_val_y) if all_val_y else np.array([]),
        )
    
    def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Simple grid search for hyperparameters."""
        logger.info("Tuning hyperparameters...")
        
        best_params = {}
        best_score = -float('inf')
        
        # Parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "min_samples_leaf": [10, 20, 50],
        }
        
        # Simple grid search (limited combinations for speed)
        for n_est in param_grid["n_estimators"]:
            for depth in param_grid["max_depth"]:
                for lr in param_grid["learning_rate"]:
                    config = ModelConfig(
                        model_type=self.config.model_type,
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        min_samples_leaf=20,
                    )
                    
                    model = DirectionalClassifier(config)
                    
                    try:
                        model.fit(X_train, y_train)
                        metrics = self._evaluate(model, X_val, y_val)
                        
                        # Score based on direction accuracy
                        score = metrics["direction_accuracy"]
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "n_estimators": n_est,
                                "max_depth": depth,
                                "learning_rate": lr,
                                "min_samples_leaf": 20,
                            }
                    except Exception as e:
                        logger.debug(f"Trial failed: {e}")
        
        logger.info(f"Best params: {best_params}, score: {best_score:.4f}")
        return best_params
    
    def _evaluate(
        self,
        model: AlphaModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        if len(X) == 0:
            return {"rmse": 0.0, "direction_accuracy": 0.0}
        
        predictions = []
        for i in range(len(X)):
            try:
                pred = model.predict(X[i])
                predictions.append(pred.expected_return)
            except Exception:
                predictions.append(0.0)
        
        predictions = np.array(predictions)
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - y) ** 2))
        
        # Direction accuracy
        pred_directions = np.sign(predictions)
        actual_directions = np.sign(y)
        direction_accuracy = np.mean(pred_directions == actual_directions)
        
        return {
            "rmse": rmse,
            "direction_accuracy": direction_accuracy,
        }
    
    def _save_model(self, model: AlphaModel) -> str:
        """Save model to disk."""
        # Create model directory
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.version = f"v{timestamp}"
        
        # Save model
        model_path = model_dir / f"{self.config.model_name}_{timestamp}.pkl"
        model.save(model_path)
        
        # Save latest symlink/reference
        latest_path = model_dir / f"{self.config.model_name}_latest.pkl"
        model.save(latest_path)
        
        # Save metadata
        metadata = {
            "version": model.version,
            "created_at": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "model_type": self.config.model_type.value,
        }
        
        metadata_path = model_dir / f"{self.config.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(model_path)
    
    def cross_validate(
        self,
        n_folds: int = 5,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Uses expanding window approach for proper temporal ordering.
        """
        if symbols:
            self.config.symbols = symbols
        
        # Build full dataset
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.train_start_days_ago)
        
        all_X = []
        all_y = []
        all_dates = []
        
        for symbol in self.config.symbols:
            try:
                X, y, dates = self.feature_engine.build_training_dataset(
                    symbol, start_date, end_date, 
                    forward_days=self.config.forward_days
                )
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    all_dates.extend(dates)
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
        
        if not all_X:
            return {"direction_accuracy": [], "rmse": []}
        
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        # Time-based folds
        fold_size = len(X) // (n_folds + 1)
        
        results = {
            "direction_accuracy": [],
            "rmse": [],
        }
        
        for fold in range(n_folds):
            # Expanding window: train on all data up to fold point
            train_end = (fold + 1) * fold_size
            val_start = train_end
            val_end = val_start + fold_size
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            
            if len(X_train) < 100 or len(X_val) < 20:
                continue
            
            # Train and evaluate
            model = DirectionalClassifier(ModelConfig(model_type=self.config.model_type))
            model.fit(X_train, y_train)
            metrics = self._evaluate(model, X_val, y_val)
            
            results["direction_accuracy"].append(metrics["direction_accuracy"])
            results["rmse"].append(metrics["rmse"])
            
            logger.info(f"Fold {fold+1}: accuracy={metrics['direction_accuracy']:.3f}")
        
        # Print summary
        print("\nCross-Validation Results:")
        print(f"  Direction Accuracy: {np.mean(results['direction_accuracy'])*100:.1f}% "
              f"(+/- {np.std(results['direction_accuracy'])*100:.1f}%)")
        print(f"  RMSE: {np.mean(results['rmse']):.4f} "
              f"(+/- {np.std(results['rmse']):.4f})")
        
        return results


def quick_train(
    symbols: Optional[List[str]] = None,
    days: int = 180,
) -> TrainingResult:
    """
    Quick training function for easy usage.
    
    Args:
        symbols: Symbols to train on (default: top 10)
        days: Days of historical data
        
    Returns:
        TrainingResult with trained model
    """
    config = TrainingConfig(
        train_start_days_ago=days,
        tune_hyperparameters=False,  # Skip for speed
    )
    
    if symbols:
        config.symbols = symbols
    
    trainer = AlphaTrainer(config)
    return trainer.train()
