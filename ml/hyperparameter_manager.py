"""
ML Hyperparameter Manager - Central Configuration for All ML Components

This module provides unified hyperparameter management across:
- LSTM Forecasting Models
- Feature Engineering
- Signal Generation
- Position Sizing
- Risk Management
- Strategy Selection

Features:
- Regime-aware parameter adjustment
- Automatic parameter validation
- Parameter persistence and loading
- Cross-component parameter coordination
- Walk-forward compatible parameter sets

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy

import numpy as np
from loguru import logger


class MarketRegime(str, Enum):
    """Market regime classification for adaptive parameters."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class ParameterScope(str, Enum):
    """Scope of parameter application."""
    GLOBAL = "global"  # Applied everywhere
    MODEL = "model"  # Model-specific
    FEATURE = "feature"  # Feature engineering
    SIGNAL = "signal"  # Signal generation
    POSITION = "position"  # Position sizing
    RISK = "risk"  # Risk management
    STRATEGY = "strategy"  # Strategy selection


@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""
    name: str
    scope: ParameterScope
    default_value: float
    min_value: float
    max_value: float
    description: str = ""
    log_scale: bool = False  # Use log scale for optimization
    integer: bool = False  # Constrain to integers
    
    # Regime-specific overrides (regime -> value)
    regime_overrides: Dict[str, float] = field(default_factory=dict)
    
    # Dependencies on other parameters
    depends_on: Optional[str] = None
    dependency_multiplier: float = 1.0
    
    def validate(self, value: float) -> float:
        """Validate and constrain value to bounds."""
        value = max(self.min_value, min(self.max_value, value))
        if self.integer:
            value = int(round(value))
        return value
    
    def get_value(self, regime: Optional[MarketRegime] = None) -> float:
        """Get parameter value, optionally adjusted for regime."""
        if regime and regime.value in self.regime_overrides:
            return self.regime_overrides[regime.value]
        return self.default_value


@dataclass
class LSTMHyperparameters:
    """Hyperparameters for LSTM forecasting models."""
    # Architecture
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    attention_heads: int = 4
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Sequence
    sequence_length: int = 60
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 5, 15, 60])
    
    # Uncertainty
    uncertainty_weight: float = 0.1
    direction_weight: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LSTMHyperparameters":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FeatureHyperparameters:
    """Hyperparameters for feature engineering."""
    # Technical indicators
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 55])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Volume features
    volume_ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    vwap_enabled: bool = True
    
    # Volatility features
    realized_vol_window: int = 20
    iv_percentile_window: int = 252
    
    # Momentum features
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Options-specific
    put_call_ratio_enabled: bool = True
    greeks_enabled: bool = True
    
    # Feature selection
    max_features: int = 50
    feature_importance_threshold: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureHyperparameters":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SignalHyperparameters:
    """Hyperparameters for signal generation."""
    # Confidence thresholds
    min_confidence: float = 0.6
    high_confidence: float = 0.8
    
    # Direction thresholds
    bullish_threshold: float = 0.3
    bearish_threshold: float = -0.3
    neutral_zone: float = 0.1
    
    # Signal combination weights
    hedge_weight: float = 0.25
    sentiment_weight: float = 0.20
    liquidity_weight: float = 0.15
    elasticity_weight: float = 0.15
    ml_forecast_weight: float = 0.25
    
    # Multi-timeframe
    timeframe_weights: Dict[str, float] = field(default_factory=lambda: {
        "1min": 0.1,
        "5min": 0.2,
        "15min": 0.3,
        "1hour": 0.25,
        "4hour": 0.15,
    })
    
    # Confirmation requirements
    min_confirming_signals: int = 3
    require_volume_confirmation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalHyperparameters":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PositionSizingHyperparameters:
    """Hyperparameters for position sizing."""
    # Kelly criterion
    kelly_fraction: float = 0.25  # Use 25% Kelly
    max_kelly_bet: float = 0.10  # Cap at 10% even if Kelly says more
    
    # Position limits
    max_position_pct: float = 0.04  # 4% max per position
    max_portfolio_heat: float = 0.20  # 20% total risk
    max_correlated_positions: int = 3
    
    # Scaling
    confidence_scaling: bool = True  # Scale size by confidence
    volatility_scaling: bool = True  # Scale size by volatility
    regime_scaling: bool = True  # Adjust for market regime
    
    # Risk multipliers by regime
    regime_size_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "trending_bull": 1.2,
        "trending_bear": 0.8,
        "range_bound": 1.0,
        "high_volatility": 0.6,
        "low_volatility": 1.1,
        "crisis": 0.3,
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionSizingHyperparameters":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RiskManagementHyperparameters:
    """Hyperparameters for risk management."""
    # Stop loss
    stop_loss_atr_multiple: float = 2.0
    trailing_stop_enabled: bool = True
    trailing_stop_activation: float = 0.5  # Activate at 50% of target
    trailing_stop_distance: float = 0.3  # Trail at 30% of gains
    
    # Take profit
    take_profit_atr_multiple: float = 3.0
    partial_profit_levels: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    partial_profit_amounts: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
    # Risk/reward
    min_reward_risk: float = 1.5
    max_loss_per_trade_pct: float = 0.02  # 2% max loss per trade
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    
    # Greeks limits (for options)
    max_portfolio_delta: float = 0.5
    max_portfolio_gamma: float = 0.1
    max_portfolio_vega: float = 0.3
    max_portfolio_theta: float = -0.1
    
    # Time-based
    max_holding_period_days: int = 30
    force_exit_before_expiry_days: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskManagementHyperparameters":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategySelectionHyperparameters:
    """Hyperparameters for strategy selection."""
    # IV thresholds
    high_iv_threshold: float = 50.0  # IV rank above this = sell premium
    low_iv_threshold: float = 30.0  # IV rank below this = buy premium
    
    # DTE preferences by timeframe
    scalp_dte_range: Tuple[int, int] = (0, 3)
    intraday_dte_range: Tuple[int, int] = (0, 7)
    swing_dte_range: Tuple[int, int] = (7, 30)
    position_dte_range: Tuple[int, int] = (30, 60)
    
    # Strategy preferences
    prefer_spreads_over_naked: bool = True
    max_spread_width_pct: float = 0.10  # 10% of underlying
    min_premium_collected: float = 0.30  # $0.30 minimum premium
    
    # Liquidity requirements
    min_open_interest: int = 100
    min_daily_volume: int = 10
    max_bid_ask_spread_pct: float = 0.10  # 10% of mid
    
    # Delta preferences
    otm_delta_range: Tuple[float, float] = (0.15, 0.35)
    atm_delta_tolerance: float = 0.05  # +/- 5% from 0.50
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategySelectionHyperparameters":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MLHyperparameterSet:
    """Complete set of ML hyperparameters for the trading system."""
    # Component-specific parameters
    lstm: LSTMHyperparameters = field(default_factory=LSTMHyperparameters)
    features: FeatureHyperparameters = field(default_factory=FeatureHyperparameters)
    signals: SignalHyperparameters = field(default_factory=SignalHyperparameters)
    position_sizing: PositionSizingHyperparameters = field(default_factory=PositionSizingHyperparameters)
    risk_management: RiskManagementHyperparameters = field(default_factory=RiskManagementHyperparameters)
    strategy_selection: StrategySelectionHyperparameters = field(default_factory=StrategySelectionHyperparameters)
    
    # Metadata
    name: str = "default"
    description: str = ""
    created_at: str = ""
    version: str = "1.0.0"
    
    # Optimization metadata
    optimized_for_regime: Optional[str] = None
    optimization_score: float = 0.0
    validation_sharpe: float = 0.0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lstm": self.lstm.to_dict(),
            "features": self.features.to_dict(),
            "signals": self.signals.to_dict(),
            "position_sizing": self.position_sizing.to_dict(),
            "risk_management": self.risk_management.to_dict(),
            "strategy_selection": self.strategy_selection.to_dict(),
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "version": self.version,
            "optimized_for_regime": self.optimized_for_regime,
            "optimization_score": self.optimization_score,
            "validation_sharpe": self.validation_sharpe,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLHyperparameterSet":
        """Create from dictionary."""
        return cls(
            lstm=LSTMHyperparameters.from_dict(data.get("lstm", {})),
            features=FeatureHyperparameters.from_dict(data.get("features", {})),
            signals=SignalHyperparameters.from_dict(data.get("signals", {})),
            position_sizing=PositionSizingHyperparameters.from_dict(data.get("position_sizing", {})),
            risk_management=RiskManagementHyperparameters.from_dict(data.get("risk_management", {})),
            strategy_selection=StrategySelectionHyperparameters.from_dict(data.get("strategy_selection", {})),
            name=data.get("name", "default"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            version=data.get("version", "1.0.0"),
            optimized_for_regime=data.get("optimized_for_regime"),
            optimization_score=data.get("optimization_score", 0.0),
            validation_sharpe=data.get("validation_sharpe", 0.0),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save hyperparameters to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved hyperparameters to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MLHyperparameterSet":
        """Load hyperparameters from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def copy(self) -> "MLHyperparameterSet":
        """Create a deep copy."""
        return MLHyperparameterSet.from_dict(self.to_dict())


class MLHyperparameterManager:
    """
    Central manager for all ML hyperparameters.
    
    Features:
    - Regime-aware parameter adjustment
    - Parameter set management (save/load/compare)
    - Validation and constraint enforcement
    - Parameter exploration for optimization
    """
    
    def __init__(
        self,
        base_params: Optional[MLHyperparameterSet] = None,
        config_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the hyperparameter manager.
        
        Args:
            base_params: Base hyperparameter set (uses defaults if None)
            config_dir: Directory for saving/loading configurations
        """
        self.base_params = base_params or MLHyperparameterSet()
        self.config_dir = Path(config_dir) if config_dir else Path("config/hyperparameters")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Regime-specific parameter sets
        self._regime_params: Dict[MarketRegime, MLHyperparameterSet] = {}
        
        # Current active parameters
        self._current_params = self.base_params.copy()
        self._current_regime: Optional[MarketRegime] = None
        
        # Parameter history for optimization
        self._param_history: List[Dict[str, Any]] = []
        
        logger.info(f"MLHyperparameterManager initialized with config_dir={self.config_dir}")
    
    @property
    def current(self) -> MLHyperparameterSet:
        """Get current active parameters."""
        return self._current_params
    
    def set_regime(self, regime: MarketRegime) -> MLHyperparameterSet:
        """
        Adjust parameters for a specific market regime.
        
        Args:
            regime: Market regime to adjust for
            
        Returns:
            Adjusted parameter set
        """
        if regime in self._regime_params:
            self._current_params = self._regime_params[regime].copy()
        else:
            # Create regime-adjusted parameters from base
            self._current_params = self._create_regime_params(regime)
        
        self._current_regime = regime
        logger.info(f"Parameters adjusted for regime: {regime.value}")
        return self._current_params
    
    def _create_regime_params(self, regime: MarketRegime) -> MLHyperparameterSet:
        """Create regime-specific parameters by adjusting base params."""
        params = self.base_params.copy()
        
        # Adjust position sizing
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
            params.position_sizing.kelly_fraction *= 0.5
            params.position_sizing.max_position_pct *= 0.5
            params.risk_management.stop_loss_atr_multiple = 3.0
        elif regime == MarketRegime.LOW_VOLATILITY:
            params.position_sizing.kelly_fraction *= 1.2
            params.risk_management.stop_loss_atr_multiple = 1.5
        
        # Adjust signal thresholds
        if regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
            params.signals.min_confidence = 0.55  # Lower threshold for trends
            params.signals.min_confirming_signals = 2
        elif regime == MarketRegime.RANGE_BOUND:
            params.signals.min_confidence = 0.70  # Higher threshold for ranges
            params.signals.neutral_zone = 0.2
        
        # Adjust strategy selection
        if regime == MarketRegime.HIGH_VOLATILITY:
            params.strategy_selection.high_iv_threshold = 40.0  # Lower threshold
            params.strategy_selection.prefer_spreads_over_naked = True
        elif regime == MarketRegime.LOW_VOLATILITY:
            params.strategy_selection.low_iv_threshold = 25.0
        
        params.optimized_for_regime = regime.value
        return params
    
    def get_optimization_space(
        self,
        scopes: Optional[List[ParameterScope]] = None,
    ) -> List[ParameterSpec]:
        """
        Get parameter specifications for optimization.
        
        Args:
            scopes: List of scopes to include (all if None)
            
        Returns:
            List of parameter specifications
        """
        specs: List[ParameterSpec] = []
        
        # LSTM parameters
        if scopes is None or ParameterScope.MODEL in scopes:
            specs.extend([
                ParameterSpec(
                    name="lstm.hidden_dim",
                    scope=ParameterScope.MODEL,
                    default_value=128,
                    min_value=32,
                    max_value=512,
                    description="LSTM hidden dimension",
                    integer=True,
                ),
                ParameterSpec(
                    name="lstm.num_layers",
                    scope=ParameterScope.MODEL,
                    default_value=2,
                    min_value=1,
                    max_value=4,
                    description="Number of LSTM layers",
                    integer=True,
                ),
                ParameterSpec(
                    name="lstm.dropout",
                    scope=ParameterScope.MODEL,
                    default_value=0.2,
                    min_value=0.0,
                    max_value=0.5,
                    description="Dropout rate",
                ),
                ParameterSpec(
                    name="lstm.learning_rate",
                    scope=ParameterScope.MODEL,
                    default_value=0.001,
                    min_value=1e-5,
                    max_value=0.01,
                    description="Learning rate",
                    log_scale=True,
                ),
                ParameterSpec(
                    name="lstm.sequence_length",
                    scope=ParameterScope.MODEL,
                    default_value=60,
                    min_value=20,
                    max_value=200,
                    description="Sequence length for LSTM input",
                    integer=True,
                ),
            ])
        
        # Signal parameters
        if scopes is None or ParameterScope.SIGNAL in scopes:
            specs.extend([
                ParameterSpec(
                    name="signals.min_confidence",
                    scope=ParameterScope.SIGNAL,
                    default_value=0.6,
                    min_value=0.3,
                    max_value=0.9,
                    description="Minimum confidence for trade signal",
                ),
                ParameterSpec(
                    name="signals.ml_forecast_weight",
                    scope=ParameterScope.SIGNAL,
                    default_value=0.25,
                    min_value=0.0,
                    max_value=0.5,
                    description="Weight of ML forecast in signal",
                ),
                ParameterSpec(
                    name="signals.hedge_weight",
                    scope=ParameterScope.SIGNAL,
                    default_value=0.25,
                    min_value=0.1,
                    max_value=0.4,
                    description="Weight of hedge engine signal",
                ),
            ])
        
        # Position sizing parameters
        if scopes is None or ParameterScope.POSITION in scopes:
            specs.extend([
                ParameterSpec(
                    name="position_sizing.kelly_fraction",
                    scope=ParameterScope.POSITION,
                    default_value=0.25,
                    min_value=0.05,
                    max_value=0.50,
                    description="Fraction of Kelly criterion to use",
                ),
                ParameterSpec(
                    name="position_sizing.max_position_pct",
                    scope=ParameterScope.POSITION,
                    default_value=0.04,
                    min_value=0.01,
                    max_value=0.10,
                    description="Maximum position size as % of portfolio",
                ),
                ParameterSpec(
                    name="position_sizing.max_portfolio_heat",
                    scope=ParameterScope.POSITION,
                    default_value=0.20,
                    min_value=0.05,
                    max_value=0.40,
                    description="Maximum total portfolio risk exposure",
                ),
            ])
        
        # Risk management parameters
        if scopes is None or ParameterScope.RISK in scopes:
            specs.extend([
                ParameterSpec(
                    name="risk_management.stop_loss_atr_multiple",
                    scope=ParameterScope.RISK,
                    default_value=2.0,
                    min_value=1.0,
                    max_value=4.0,
                    description="Stop loss as multiple of ATR",
                ),
                ParameterSpec(
                    name="risk_management.take_profit_atr_multiple",
                    scope=ParameterScope.RISK,
                    default_value=3.0,
                    min_value=1.5,
                    max_value=6.0,
                    description="Take profit as multiple of ATR",
                ),
                ParameterSpec(
                    name="risk_management.min_reward_risk",
                    scope=ParameterScope.RISK,
                    default_value=1.5,
                    min_value=1.0,
                    max_value=3.0,
                    description="Minimum reward:risk ratio",
                ),
            ])
        
        # Strategy selection parameters
        if scopes is None or ParameterScope.STRATEGY in scopes:
            specs.extend([
                ParameterSpec(
                    name="strategy_selection.high_iv_threshold",
                    scope=ParameterScope.STRATEGY,
                    default_value=50.0,
                    min_value=30.0,
                    max_value=70.0,
                    description="IV rank threshold for premium selling",
                ),
                ParameterSpec(
                    name="strategy_selection.low_iv_threshold",
                    scope=ParameterScope.STRATEGY,
                    default_value=30.0,
                    min_value=15.0,
                    max_value=45.0,
                    description="IV rank threshold for premium buying",
                ),
            ])
        
        return specs
    
    def update_parameter(self, param_path: str, value: float) -> None:
        """
        Update a single parameter by its path.
        
        Args:
            param_path: Dot-separated parameter path (e.g., "lstm.hidden_dim")
            value: New value
        """
        parts = param_path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid parameter path: {param_path}")
        
        component, param_name = parts
        
        component_map = {
            "lstm": self._current_params.lstm,
            "features": self._current_params.features,
            "signals": self._current_params.signals,
            "position_sizing": self._current_params.position_sizing,
            "risk_management": self._current_params.risk_management,
            "strategy_selection": self._current_params.strategy_selection,
        }
        
        if component not in component_map:
            raise ValueError(f"Unknown component: {component}")
        
        component_obj = component_map[component]
        if not hasattr(component_obj, param_name):
            raise ValueError(f"Unknown parameter: {param_name} in {component}")
        
        setattr(component_obj, param_name, value)
        logger.debug(f"Updated {param_path} = {value}")
    
    def update_from_dict(self, params: Dict[str, float]) -> None:
        """
        Update multiple parameters from a dictionary.
        
        Args:
            params: Dictionary of param_path -> value
        """
        for path, value in params.items():
            self.update_parameter(path, value)
    
    def get_parameter(self, param_path: str) -> Any:
        """
        Get a parameter value by its path.
        
        Args:
            param_path: Dot-separated parameter path
            
        Returns:
            Parameter value
        """
        parts = param_path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid parameter path: {param_path}")
        
        component, param_name = parts
        
        component_map = {
            "lstm": self._current_params.lstm,
            "features": self._current_params.features,
            "signals": self._current_params.signals,
            "position_sizing": self._current_params.position_sizing,
            "risk_management": self._current_params.risk_management,
            "strategy_selection": self._current_params.strategy_selection,
        }
        
        if component not in component_map:
            raise ValueError(f"Unknown component: {component}")
        
        return getattr(component_map[component], param_name)
    
    def save_current(self, name: Optional[str] = None) -> Path:
        """
        Save current parameters to file.
        
        Args:
            name: Name for the configuration (uses timestamp if None)
            
        Returns:
            Path to saved file
        """
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._current_params.name = name
        path = self.config_dir / f"{name}.json"
        self._current_params.save(path)
        return path
    
    def load_config(self, name: str) -> MLHyperparameterSet:
        """
        Load a named configuration.
        
        Args:
            name: Name of the configuration
            
        Returns:
            Loaded parameter set
        """
        path = self.config_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Configuration not found: {path}")
        
        self._current_params = MLHyperparameterSet.load(path)
        logger.info(f"Loaded configuration: {name}")
        return self._current_params
    
    def list_configs(self) -> List[str]:
        """List available configurations."""
        return [p.stem for p in self.config_dir.glob("*.json")]
    
    def compare_configs(
        self,
        name1: str,
        name2: str,
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        Compare two configurations.
        
        Args:
            name1: First configuration name
            name2: Second configuration name
            
        Returns:
            Dictionary of differing parameters with (value1, value2) tuples
        """
        config1 = MLHyperparameterSet.load(self.config_dir / f"{name1}.json")
        config2 = MLHyperparameterSet.load(self.config_dir / f"{name2}.json")
        
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        differences = {}
        
        def compare_dicts(d1: Dict, d2: Dict, prefix: str = ""):
            for key in set(d1.keys()) | set(d2.keys()):
                path = f"{prefix}.{key}" if prefix else key
                v1 = d1.get(key)
                v2 = d2.get(key)
                
                if isinstance(v1, dict) and isinstance(v2, dict):
                    compare_dicts(v1, v2, path)
                elif v1 != v2:
                    differences[path] = (v1, v2)
        
        compare_dicts(dict1, dict2)
        return differences
    
    def record_evaluation(
        self,
        params: Dict[str, float],
        score: float,
        metrics: Dict[str, float],
    ) -> None:
        """
        Record a parameter evaluation for optimization history.
        
        Args:
            params: Parameters that were evaluated
            score: Objective score
            metrics: Additional metrics
        """
        self._param_history.append({
            "params": params.copy(),
            "score": score,
            "metrics": metrics.copy(),
            "timestamp": datetime.now().isoformat(),
            "regime": self._current_regime.value if self._current_regime else None,
        })
    
    def get_best_params(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get the best parameter sets from history.
        
        Args:
            n: Number of best sets to return
            
        Returns:
            List of best parameter records
        """
        if not self._param_history:
            return []
        
        sorted_history = sorted(
            self._param_history,
            key=lambda x: x["score"],
            reverse=True,
        )
        return sorted_history[:n]


# Factory function for creating preset configurations
def create_preset_config(preset: str) -> MLHyperparameterSet:
    """
    Create a preset hyperparameter configuration.
    
    Args:
        preset: Preset name ("conservative", "aggressive", "balanced")
        
    Returns:
        Preset parameter set
    """
    params = MLHyperparameterSet()
    
    if preset == "conservative":
        params.position_sizing.kelly_fraction = 0.15
        params.position_sizing.max_position_pct = 0.02
        params.position_sizing.max_portfolio_heat = 0.10
        params.signals.min_confidence = 0.75
        params.risk_management.stop_loss_atr_multiple = 1.5
        params.risk_management.min_reward_risk = 2.0
        params.name = "conservative"
        params.description = "Conservative settings for capital preservation"
        
    elif preset == "aggressive":
        params.position_sizing.kelly_fraction = 0.40
        params.position_sizing.max_position_pct = 0.06
        params.position_sizing.max_portfolio_heat = 0.30
        params.signals.min_confidence = 0.50
        params.risk_management.stop_loss_atr_multiple = 2.5
        params.risk_management.min_reward_risk = 1.2
        params.name = "aggressive"
        params.description = "Aggressive settings for maximum returns"
        
    elif preset == "balanced":
        # Default values are already balanced
        params.name = "balanced"
        params.description = "Balanced settings for steady growth"
        
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    return params


__all__ = [
    "MarketRegime",
    "ParameterScope",
    "ParameterSpec",
    "LSTMHyperparameters",
    "FeatureHyperparameters",
    "SignalHyperparameters",
    "PositionSizingHyperparameters",
    "RiskManagementHyperparameters",
    "StrategySelectionHyperparameters",
    "MLHyperparameterSet",
    "MLHyperparameterManager",
    "create_preset_config",
]
