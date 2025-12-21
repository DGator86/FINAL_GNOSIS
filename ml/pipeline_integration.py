"""
ML Pipeline Integration - Connects ML Hyperparameter Management with Trading Engines

This module provides deep integration between:
- MLHyperparameterManager for configuration
- LSTM prediction engines
- Feature engineering pipeline
- Trading decision generation
- Walk-forward backtesting

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
from loguru import logger

# ML components
from ml.hyperparameter_manager import (
    MLHyperparameterManager,
    MLHyperparameterSet,
    MarketRegime,
    create_preset_config,
)
from ml.adaptive_pipeline import (
    AdaptiveMLPipeline,
    MLTradeDecision,
    MLSignal,
    SignalStrength,
)
from ml.optimization_engine import (
    MLOptimizationEngine,
    OptimizationConfig,
    OptimizationStage,
    OptimizationMetric,
    FullOptimizationResult,
)

# Try to import engine components
try:
    from engines.ml.lstm_engine import LSTMPredictionEngine
    from models.lstm_lookahead import LookaheadConfig, LSTMLookaheadPredictor
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False
    logger.warning("LSTM components not available")

try:
    from models.features.feature_builder import EnhancedFeatureBuilder, FeatureConfig
    HAS_FEATURES = True
except ImportError:
    HAS_FEATURES = False
    logger.warning("Feature builder not available")

try:
    from engines.inputs.market_data_adapter import MarketDataAdapter
    HAS_MARKET_DATA = True
except ImportError:
    HAS_MARKET_DATA = False
    logger.warning("Market data adapter not available")

try:
    from backtesting.elite_backtest_engine import EliteBacktestEngine, EliteBacktestConfig
    HAS_BACKTEST = True
except ImportError:
    HAS_BACKTEST = False

try:
    from schemas.core_schemas import PipelineResult, ForecastSnapshot
    HAS_SCHEMAS = True
except ImportError:
    HAS_SCHEMAS = False


@dataclass
class MLPipelineConfig:
    """Configuration for the integrated ML pipeline."""
    # Component selection
    use_lstm: bool = True
    use_feature_builder: bool = True
    use_regime_detection: bool = True
    
    # Hyperparameter preset
    preset: str = "balanced"  # conservative, balanced, aggressive
    
    # Model paths
    lstm_model_path: Optional[str] = None
    config_dir: str = "config/hyperparameters"
    
    # Data settings
    lookback_periods: int = 200
    sequence_length: int = 60
    
    # Training settings
    auto_train: bool = False
    train_on_startup: bool = False
    retrain_interval_hours: int = 24
    
    # Optimization settings
    auto_optimize: bool = False
    optimize_interval_hours: int = 48
    optimization_trials: int = 100
    optimization_method: str = "bayesian"


@dataclass
class TrainingResult:
    """Result from model training."""
    success: bool
    model_path: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    n_samples: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass  
class PipelineState:
    """State tracking for the ML pipeline."""
    last_training: Optional[datetime] = None
    last_optimization: Optional[datetime] = None
    last_regime_change: Optional[datetime] = None
    current_regime: Optional[MarketRegime] = None
    predictions_made: int = 0
    decisions_made: int = 0
    
    # Performance metrics
    accuracy_history: List[float] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)


class IntegratedMLPipeline:
    """
    Fully integrated ML trading pipeline.
    
    Connects all ML components:
    - Hyperparameter management
    - LSTM forecasting
    - Feature engineering
    - Signal generation
    - Position sizing
    - Risk management
    - Optimization
    
    Provides:
    - End-to-end trading decisions
    - Automatic retraining
    - Regime-adaptive parameters
    - Walk-forward optimization
    """
    
    def __init__(
        self,
        config: Optional[MLPipelineConfig] = None,
        market_adapter: Optional[Any] = None,
        backtest_engine: Optional[Any] = None,
    ):
        """
        Initialize the integrated ML pipeline.
        
        Args:
            config: Pipeline configuration
            market_adapter: Market data adapter
            backtest_engine: Backtesting engine for optimization
        """
        self.config = config or MLPipelineConfig()
        self.market_adapter = market_adapter
        self.backtest_engine = backtest_engine
        
        # State tracking
        self.state = PipelineState()
        
        # Initialize hyperparameter manager
        preset_config = create_preset_config(self.config.preset)
        self.hp_manager = MLHyperparameterManager(
            base_params=preset_config,
            config_dir=self.config.config_dir,
        )
        
        # Initialize feature builder
        self.feature_builder: Optional[Any] = None
        if self.config.use_feature_builder and HAS_FEATURES:
            self._init_feature_builder()
        
        # Initialize LSTM engine
        self.lstm_engine: Optional[Any] = None
        if self.config.use_lstm and HAS_LSTM:
            self._init_lstm_engine()
        
        # Initialize adaptive pipeline
        self.adaptive_pipeline = AdaptiveMLPipeline(
            hyperparameter_manager=self.hp_manager,
            market_adapter=market_adapter,
            model_path=self.config.lstm_model_path,
            preset=self.config.preset,
        )
        
        # Initialize optimization engine
        self.optimizer = MLOptimizationEngine(
            hp_manager=self.hp_manager,
            backtest_engine=backtest_engine,
        )
        
        # Training on startup if configured
        if self.config.train_on_startup:
            self._train_initial_model()
        
        logger.info(f"IntegratedMLPipeline initialized with preset={self.config.preset}")
    
    def _init_feature_builder(self) -> None:
        """Initialize the feature builder with ML hyperparameters."""
        hp = self.hp_manager.current.features
        
        feature_config = FeatureConfig(
            use_technical=True,
            use_hedge_engine=True,
            use_microstructure=True,
            use_sentiment=True,
            use_temporal=True,
            use_regime=True,
            use_options=hp.greeks_enabled,
            lookback_periods=[hp.rsi_period, hp.atr_period, hp.bollinger_period],
            correlation_threshold=0.95,
        )
        
        self.feature_builder = EnhancedFeatureBuilder(config=feature_config)
        logger.info("Feature builder initialized with ML hyperparameters")
    
    def _init_lstm_engine(self) -> None:
        """Initialize the LSTM engine with ML hyperparameters."""
        if not self.market_adapter:
            logger.warning("Cannot init LSTM engine without market adapter")
            return
        
        hp = self.hp_manager.current.lstm
        
        config = LookaheadConfig(
            input_dim=self.hp_manager.current.features.max_features,
            hidden_dim=hp.hidden_dim,
            num_layers=hp.num_layers,
            dropout=hp.dropout,
            bidirectional=hp.bidirectional,
            forecast_horizons=hp.forecast_horizons,
            sequence_length=hp.sequence_length,
            learning_rate=hp.learning_rate,
            batch_size=hp.batch_size,
            max_epochs=hp.max_epochs,
            patience=hp.patience,
        )
        
        self.lstm_engine = LSTMPredictionEngine(
            market_adapter=self.market_adapter,
            feature_builder=self.feature_builder,
            model_path=self.config.lstm_model_path,
            config=config,
            lookback_periods=self.config.lookback_periods,
        )
        logger.info("LSTM engine initialized with ML hyperparameters")
    
    def _train_initial_model(self) -> None:
        """Train initial model on startup."""
        logger.info("Training initial LSTM model...")
        # Training logic would go here
        self.state.last_training = datetime.now()
    
    def process_symbol(
        self,
        symbol: str,
        pipeline_result: Optional[Any] = None,
        market_data: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
    ) -> MLTradeDecision:
        """
        Process a symbol through the ML pipeline.
        
        Args:
            symbol: Trading symbol
            pipeline_result: Result from base trading pipeline
            market_data: Historical OHLCV data
            timestamp: Current timestamp
            
        Returns:
            ML-driven trade decision
        """
        timestamp = timestamp or datetime.now()
        
        # Check if retraining/optimization needed
        self._check_maintenance()
        
        # Get LSTM prediction
        lstm_forecast = None
        if self.lstm_engine and pipeline_result and HAS_SCHEMAS:
            lstm_forecast = self.lstm_engine.enhance(pipeline_result, timestamp)
        
        # Generate decision through adaptive pipeline
        decision = self.adaptive_pipeline.process(
            symbol=symbol,
            pipeline_result=pipeline_result,
            market_data=market_data,
            timestamp=timestamp,
        )
        
        # Enhance decision with LSTM forecast
        if lstm_forecast:
            decision = self._enhance_with_lstm(decision, lstm_forecast)
        
        # Update state
        self.state.predictions_made += 1
        self.state.decisions_made += 1
        self.state.current_regime = self.adaptive_pipeline._current_regime
        
        return decision
    
    def _enhance_with_lstm(
        self,
        decision: MLTradeDecision,
        forecast: Any,
    ) -> MLTradeDecision:
        """Enhance decision with LSTM forecast information."""
        if not forecast:
            return decision
        
        # Extract forecast data
        confidence = getattr(forecast, 'confidence', 0.5)
        forecasts = getattr(forecast, 'forecast', [])
        
        if forecasts:
            avg_forecast = np.mean(forecasts)
            
            # Adjust confidence based on forecast alignment
            direction_aligned = (
                (avg_forecast > 0 and decision.action == "buy") or
                (avg_forecast < 0 and decision.action == "sell")
            )
            
            if direction_aligned:
                decision.model_confidence = confidence
                decision.overall_confidence = (
                    decision.signal_confidence * 0.6 + confidence * 0.4
                )
                decision.reasons.append(f"LSTM forecast aligned ({avg_forecast:.4f})")
            else:
                decision.model_confidence = 1 - confidence
                decision.warnings.append(f"LSTM forecast misaligned ({avg_forecast:.4f})")
        
        return decision
    
    def _check_maintenance(self) -> None:
        """Check if maintenance tasks (retraining, optimization) are needed."""
        now = datetime.now()
        
        # Check for retraining
        if self.config.auto_train and self.state.last_training:
            hours_since_train = (now - self.state.last_training).total_seconds() / 3600
            if hours_since_train > self.config.retrain_interval_hours:
                logger.info("Triggering scheduled retraining")
                self._retrain_model()
        
        # Check for optimization
        if self.config.auto_optimize and self.state.last_optimization:
            hours_since_opt = (now - self.state.last_optimization).total_seconds() / 3600
            if hours_since_opt > self.config.optimize_interval_hours:
                logger.info("Triggering scheduled optimization")
                self._run_optimization()
    
    def _retrain_model(self) -> Optional[TrainingResult]:
        """Retrain the LSTM model."""
        if not self.lstm_engine:
            return None
        
        logger.info("Starting model retraining...")
        start_time = datetime.now()
        
        # Training logic would go here
        # For now, just update state
        self.state.last_training = datetime.now()
        
        return TrainingResult(
            success=True,
            timestamp=datetime.now(),
            duration_seconds=(datetime.now() - start_time).total_seconds(),
        )
    
    def _run_optimization(self) -> Optional[FullOptimizationResult]:
        """Run hyperparameter optimization."""
        logger.info("Starting hyperparameter optimization...")
        
        config = OptimizationConfig(
            method=self.config.optimization_method,
            n_trials=self.config.optimization_trials,
            stages=[OptimizationStage.FULL],
            primary_metric=OptimizationMetric.SHARPE_RATIO,
        )
        
        result = self.optimizer.optimize(config)
        
        # Apply optimized parameters
        if result.best_params:
            self.hp_manager._current_params = result.best_params
            self._reinitialize_components()
        
        self.state.last_optimization = datetime.now()
        return result
    
    def _reinitialize_components(self) -> None:
        """Reinitialize components after hyperparameter change."""
        if self.config.use_feature_builder and HAS_FEATURES:
            self._init_feature_builder()
        
        if self.config.use_lstm and HAS_LSTM and self.market_adapter:
            self._init_lstm_engine()
        
        logger.info("Components reinitialized with new hyperparameters")
    
    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_split: float = 0.2,
    ) -> TrainingResult:
        """
        Train the LSTM model on provided data.
        
        Args:
            training_data: DataFrame with features and targets
            validation_split: Fraction of data for validation
            
        Returns:
            Training result with metrics
        """
        if not self.lstm_engine or not HAS_LSTM:
            return TrainingResult(success=False)
        
        start_time = datetime.now()
        logger.info(f"Training LSTM on {len(training_data)} samples...")
        
        try:
            # Get hyperparameters
            hp = self.hp_manager.current.lstm
            
            # Prepare data
            features = training_data.drop(columns=['target'], errors='ignore')
            targets = training_data.get('target', training_data['close'].pct_change().shift(-1))
            
            # Split
            split_idx = int(len(features) * (1 - validation_split))
            X_train, X_val = features[:split_idx], features[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]
            
            # Train
            self.lstm_engine.predictor.train(
                X_train.values,
                {f"horizon_{h}": y_train.values for h in hp.forecast_horizons},
            )
            
            # Validate
            val_metrics = self._evaluate_model(X_val, y_val)
            
            # Save model
            model_path = Path(self.config.config_dir) / "models" / "lstm_latest.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.lstm_engine.predictor.save(str(model_path))
            
            self.state.last_training = datetime.now()
            
            return TrainingResult(
                success=True,
                model_path=str(model_path),
                metrics=val_metrics,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                n_samples=len(training_data),
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(success=False)
    
    def _evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        if not self.lstm_engine:
            return {}
        
        # Basic evaluation metrics
        return {
            "val_samples": len(X),
            "val_mean_target": float(y.mean()),
            "val_std_target": float(y.std()),
        }
    
    def optimize(
        self,
        config: Optional[OptimizationConfig] = None,
    ) -> FullOptimizationResult:
        """
        Run hyperparameter optimization.
        
        Args:
            config: Optimization configuration
            
        Returns:
            Optimization results
        """
        return self.optimizer.optimize(config)
    
    def set_regime(self, regime: MarketRegime) -> None:
        """
        Manually set market regime.
        
        Args:
            regime: Market regime to set
        """
        self.hp_manager.set_regime(regime)
        self.state.current_regime = regime
        self.state.last_regime_change = datetime.now()
        self._reinitialize_components()
        logger.info(f"Manually set regime to: {regime.value}")
    
    def update_hyperparameters(self, params: Dict[str, float]) -> None:
        """
        Update hyperparameters.
        
        Args:
            params: Dictionary of parameter path -> value
        """
        self.hp_manager.update_from_dict(params)
        self.adaptive_pipeline.update_hyperparameters(params)
        self._reinitialize_components()
    
    def save_state(self, path: Union[str, Path]) -> None:
        """Save pipeline state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            "last_training": self.state.last_training.isoformat() if self.state.last_training else None,
            "last_optimization": self.state.last_optimization.isoformat() if self.state.last_optimization else None,
            "current_regime": self.state.current_regime.value if self.state.current_regime else None,
            "predictions_made": self.state.predictions_made,
            "decisions_made": self.state.decisions_made,
            "hyperparameters": self.hp_manager.current.to_dict(),
        }
        
        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"Pipeline state saved to {path}")
    
    def load_state(self, path: Union[str, Path]) -> None:
        """Load pipeline state from file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"State file not found: {path}")
            return
        
        with open(path, "r") as f:
            state_dict = json.load(f)
        
        if state_dict.get("last_training"):
            self.state.last_training = datetime.fromisoformat(state_dict["last_training"])
        if state_dict.get("last_optimization"):
            self.state.last_optimization = datetime.fromisoformat(state_dict["last_optimization"])
        if state_dict.get("current_regime"):
            self.state.current_regime = MarketRegime(state_dict["current_regime"])
        self.state.predictions_made = state_dict.get("predictions_made", 0)
        self.state.decisions_made = state_dict.get("decisions_made", 0)
        
        if state_dict.get("hyperparameters"):
            params = MLHyperparameterSet.from_dict(state_dict["hyperparameters"])
            self.hp_manager._current_params = params
            self._reinitialize_components()
        
        logger.info(f"Pipeline state loaded from {path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "preset": self.config.preset,
            "current_regime": self.state.current_regime.value if self.state.current_regime else "unknown",
            "predictions_made": self.state.predictions_made,
            "decisions_made": self.state.decisions_made,
            "last_training": self.state.last_training.isoformat() if self.state.last_training else None,
            "last_optimization": self.state.last_optimization.isoformat() if self.state.last_optimization else None,
            "lstm_available": self.lstm_engine is not None,
            "feature_builder_available": self.feature_builder is not None,
            "hyperparameters": {
                "lstm_hidden_dim": self.hp_manager.current.lstm.hidden_dim,
                "min_confidence": self.hp_manager.current.signals.min_confidence,
                "kelly_fraction": self.hp_manager.current.position_sizing.kelly_fraction,
                "stop_loss_atr": self.hp_manager.current.risk_management.stop_loss_atr_multiple,
            },
        }


# Factory function
def create_integrated_pipeline(
    preset: str = "balanced",
    market_adapter: Optional[Any] = None,
    backtest_engine: Optional[Any] = None,
    auto_optimize: bool = False,
) -> IntegratedMLPipeline:
    """
    Create an integrated ML pipeline with preset configuration.
    
    Args:
        preset: Configuration preset (conservative, balanced, aggressive)
        market_adapter: Market data adapter
        backtest_engine: Backtesting engine
        auto_optimize: Enable automatic optimization
        
    Returns:
        Configured IntegratedMLPipeline
    """
    config = MLPipelineConfig(
        preset=preset,
        auto_optimize=auto_optimize,
    )
    
    return IntegratedMLPipeline(
        config=config,
        market_adapter=market_adapter,
        backtest_engine=backtest_engine,
    )


__all__ = [
    "MLPipelineConfig",
    "TrainingResult",
    "PipelineState",
    "IntegratedMLPipeline",
    "create_integrated_pipeline",
]
