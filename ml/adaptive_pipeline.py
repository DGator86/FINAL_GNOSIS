"""
Adaptive ML Pipeline - ML-Integrated Trading Decision Pipeline

This module integrates ML predictions throughout the trading process:
- Feature Engineering with ML-optimized parameters
- LSTM-based price forecasting
- ML-enhanced signal generation
- Adaptive position sizing based on model confidence
- Dynamic risk management with uncertainty quantification

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from ml.hyperparameter_manager import (
    MLHyperparameterManager,
    MLHyperparameterSet,
    MarketRegime,
    create_preset_config,
)

# Import engine components
try:
    from engines.inputs.market_data_adapter import MarketDataAdapter
    HAS_MARKET_ADAPTER = True
except ImportError:
    HAS_MARKET_ADAPTER = False

try:
    from engines.ml.lstm_engine import LSTMPredictionEngine
    from models.lstm_lookahead import LookaheadConfig, LSTMLookaheadPredictor
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False

try:
    from models.features.feature_builder import EnhancedFeatureBuilder
    HAS_FEATURE_BUILDER = True
except ImportError:
    HAS_FEATURE_BUILDER = False

try:
    from schemas.core_schemas import (
        PipelineResult,
        HedgeSnapshot,
        SentimentSnapshot,
        LiquiditySnapshot,
        ElasticitySnapshot,
        ForecastSnapshot,
        TradeIdea,
        DirectionEnum,
    )
    HAS_SCHEMAS = True
except ImportError:
    HAS_SCHEMAS = False


class SignalStrength(str, Enum):
    """Signal strength classification."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class MLSignal:
    """ML-enhanced trading signal."""
    direction: str  # "bullish", "bearish", "neutral"
    strength: SignalStrength
    confidence: float  # 0-1
    
    # Component scores
    ml_forecast_score: float = 0.0
    hedge_score: float = 0.0
    sentiment_score: float = 0.0
    liquidity_score: float = 0.0
    elasticity_score: float = 0.0
    
    # ML-specific
    forecast_horizons: Dict[int, float] = field(default_factory=dict)  # horizon -> predicted return
    uncertainty: float = 0.0
    regime: str = ""
    
    # Metadata
    timestamp: datetime = None
    symbol: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MLPositionSize:
    """ML-optimized position sizing recommendation."""
    base_size: float  # As fraction of portfolio
    adjusted_size: float  # After ML adjustments
    
    # Adjustments applied
    confidence_adjustment: float = 1.0
    volatility_adjustment: float = 1.0
    regime_adjustment: float = 1.0
    uncertainty_adjustment: float = 1.0
    
    # Limits
    max_allowed: float = 0.0
    reason: str = ""
    
    @property
    def final_size(self) -> float:
        """Get final position size."""
        return min(self.adjusted_size, self.max_allowed)


@dataclass
class MLTradeDecision:
    """Complete ML-driven trade decision."""
    # Decision
    should_trade: bool
    action: str  # "buy", "sell", "hold"
    
    # Signal
    signal: MLSignal
    
    # Position
    position_size: MLPositionSize
    
    # Risk parameters
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: Optional[float] = None
    
    # Timing
    urgency: str = "normal"  # "immediate", "normal", "patient"
    valid_until: Optional[datetime] = None
    
    # Confidence breakdown
    overall_confidence: float = 0.0
    model_confidence: float = 0.0
    signal_confidence: float = 0.0
    
    # Reasoning
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RegimeDetector:
    """Detect market regime from pipeline data."""
    
    def __init__(self, params: Optional[MLHyperparameterSet] = None):
        self.params = params or MLHyperparameterSet()
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []
    
    def detect(
        self,
        pipeline_result: Optional[Any] = None,
        market_data: Optional[pd.DataFrame] = None,
    ) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            pipeline_result: Pipeline result with engine snapshots
            market_data: Historical market data (OHLCV)
            
        Returns:
            Detected market regime
        """
        # Collect evidence for each regime
        regime_scores = {regime: 0.0 for regime in MarketRegime}
        
        # From pipeline result
        if pipeline_result and hasattr(pipeline_result, 'hedge_snapshot'):
            hedge = pipeline_result.hedge_snapshot
            if hedge:
                # High movement energy suggests trending
                energy = getattr(hedge, 'movement_energy', 50)
                if energy > 70:
                    regime_scores[MarketRegime.TRENDING_BULL] += 0.3
                    regime_scores[MarketRegime.TRENDING_BEAR] += 0.3
                elif energy < 30:
                    regime_scores[MarketRegime.RANGE_BOUND] += 0.4
                
                # Elasticity indicates regime
                elasticity = getattr(hedge, 'elasticity', 0.5)
                if elasticity > 0.7:
                    regime_scores[MarketRegime.HIGH_VOLATILITY] += 0.3
                elif elasticity < 0.3:
                    regime_scores[MarketRegime.LOW_VOLATILITY] += 0.3
        
        if pipeline_result and hasattr(pipeline_result, 'elasticity_snapshot'):
            elasticity_snap = pipeline_result.elasticity_snapshot
            if elasticity_snap:
                vol_regime = getattr(elasticity_snap, 'volatility_regime', 'moderate')
                if vol_regime == 'high':
                    regime_scores[MarketRegime.HIGH_VOLATILITY] += 0.4
                elif vol_regime == 'low':
                    regime_scores[MarketRegime.LOW_VOLATILITY] += 0.4
        
        # From market data
        if market_data is not None and len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            
            # Volatility
            vol = returns.std() * np.sqrt(252)
            if vol > 0.30:  # >30% annualized vol
                regime_scores[MarketRegime.HIGH_VOLATILITY] += 0.3
            elif vol < 0.15:
                regime_scores[MarketRegime.LOW_VOLATILITY] += 0.3
            
            # Trend
            sma_20 = market_data['close'].rolling(20).mean()
            sma_50 = market_data['close'].rolling(50).mean() if len(market_data) >= 50 else sma_20
            
            current_price = market_data['close'].iloc[-1]
            if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                regime_scores[MarketRegime.TRENDING_BULL] += 0.4
            elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
                regime_scores[MarketRegime.TRENDING_BEAR] += 0.4
            else:
                regime_scores[MarketRegime.RANGE_BOUND] += 0.2
        
        # Select highest scoring regime
        detected_regime = max(regime_scores, key=regime_scores.get)
        
        # Record history
        self._regime_history.append((datetime.now(), detected_regime))
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]
        
        return detected_regime
    
    def get_regime_stability(self, lookback: int = 10) -> float:
        """
        Get stability of regime detection.
        
        Returns:
            Stability score 0-1 (1 = very stable)
        """
        if len(self._regime_history) < 2:
            return 1.0
        
        recent = [r for _, r in self._regime_history[-lookback:]]
        most_common = max(set(recent), key=recent.count)
        stability = recent.count(most_common) / len(recent)
        return stability


class AdaptiveMLPipeline:
    """
    ML-integrated trading pipeline with adaptive hyperparameters.
    
    This pipeline:
    1. Detects market regime
    2. Adjusts hyperparameters for regime
    3. Generates ML forecasts
    4. Combines signals with ML weights
    5. Sizes positions based on confidence
    6. Sets risk parameters with uncertainty
    """
    
    def __init__(
        self,
        hyperparameter_manager: Optional[MLHyperparameterManager] = None,
        market_adapter: Optional[Any] = None,
        model_path: Optional[str] = None,
        preset: str = "balanced",
    ):
        """
        Initialize the adaptive ML pipeline.
        
        Args:
            hyperparameter_manager: Hyperparameter manager (creates default if None)
            market_adapter: Market data adapter
            model_path: Path to pre-trained LSTM model
            preset: Preset configuration ("conservative", "balanced", "aggressive")
        """
        # Initialize hyperparameter manager
        if hyperparameter_manager:
            self.hp_manager = hyperparameter_manager
        else:
            preset_config = create_preset_config(preset)
            self.hp_manager = MLHyperparameterManager(base_params=preset_config)
        
        # Initialize regime detector
        self.regime_detector = RegimeDetector(self.hp_manager.current)
        self._current_regime: Optional[MarketRegime] = None
        
        # Initialize LSTM if available
        self.lstm_engine: Optional[Any] = None
        self._model_path = model_path
        self._market_adapter = market_adapter
        
        if HAS_LSTM and market_adapter:
            self._init_lstm_engine()
        
        # Signal history for analysis
        self._signal_history: List[MLSignal] = []
        self._decision_history: List[MLTradeDecision] = []
        
        logger.info(f"AdaptiveMLPipeline initialized with preset={preset}")
    
    def _init_lstm_engine(self) -> None:
        """Initialize LSTM prediction engine with current hyperparameters."""
        if not HAS_LSTM:
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
            market_adapter=self._market_adapter,
            config=config,
            model_path=self._model_path,
        )
    
    def update_hyperparameters(self, params: Dict[str, float]) -> None:
        """
        Update hyperparameters and reinitialize components.
        
        Args:
            params: Dictionary of parameter path -> value
        """
        self.hp_manager.update_from_dict(params)
        
        # Reinitialize LSTM if model parameters changed
        model_params = [k for k in params if k.startswith("lstm.")]
        if model_params and self._market_adapter:
            self._init_lstm_engine()
        
        logger.info(f"Updated {len(params)} hyperparameters")
    
    def process(
        self,
        symbol: str,
        pipeline_result: Optional[Any] = None,
        market_data: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
    ) -> MLTradeDecision:
        """
        Process symbol through ML pipeline to generate trade decision.
        
        Args:
            symbol: Trading symbol
            pipeline_result: Result from base pipeline with engine snapshots
            market_data: Historical OHLCV data
            timestamp: Current timestamp
            
        Returns:
            ML-driven trade decision
        """
        timestamp = timestamp or datetime.now()
        
        # Step 1: Detect regime and adjust parameters
        regime = self.regime_detector.detect(pipeline_result, market_data)
        if regime != self._current_regime:
            self.hp_manager.set_regime(regime)
            self._current_regime = regime
            logger.info(f"Regime changed to {regime.value}, parameters adjusted")
        
        # Step 2: Generate ML forecast
        forecast = self._generate_forecast(symbol, pipeline_result, timestamp)
        
        # Step 3: Generate combined signal
        signal = self._generate_signal(symbol, pipeline_result, forecast, timestamp)
        
        # Step 4: Calculate position size
        position_size = self._calculate_position_size(signal, regime)
        
        # Step 5: Generate trade decision
        decision = self._generate_decision(signal, position_size, regime)
        
        # Record history
        self._signal_history.append(signal)
        self._decision_history.append(decision)
        
        # Trim history
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-1000:]
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-1000:]
        
        return decision
    
    def _generate_forecast(
        self,
        symbol: str,
        pipeline_result: Optional[Any],
        timestamp: datetime,
    ) -> Optional[ForecastSnapshot]:
        """Generate LSTM forecast."""
        if not self.lstm_engine:
            return None
        
        try:
            if HAS_SCHEMAS and pipeline_result:
                forecast = self.lstm_engine.enhance(pipeline_result, timestamp)
                return forecast
        except Exception as e:
            logger.warning(f"LSTM forecast failed for {symbol}: {e}")
        
        return None
    
    def _generate_signal(
        self,
        symbol: str,
        pipeline_result: Optional[Any],
        forecast: Optional[Any],
        timestamp: datetime,
    ) -> MLSignal:
        """Generate ML-enhanced trading signal."""
        hp = self.hp_manager.current.signals
        
        # Initialize scores
        scores = {
            "ml_forecast": 0.0,
            "hedge": 0.0,
            "sentiment": 0.0,
            "liquidity": 0.0,
            "elasticity": 0.0,
        }
        
        # Weight map
        weights = {
            "ml_forecast": hp.ml_forecast_weight,
            "hedge": hp.hedge_weight,
            "sentiment": hp.sentiment_weight,
            "liquidity": hp.liquidity_weight,
            "elasticity": hp.elasticity_weight,
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Score from ML forecast
        forecast_horizons = {}
        if forecast:
            forecasts = getattr(forecast, 'forecast', [])
            if forecasts:
                avg_forecast = np.mean(forecasts)
                scores["ml_forecast"] = np.clip(avg_forecast * 10, -1, 1)  # Scale to -1 to 1
            
            # Extract horizon-specific forecasts
            metadata = getattr(forecast, 'metadata', {})
            if 'predictions_pct' in metadata:
                forecast_horizons = metadata['predictions_pct']
        
        # Score from pipeline engines
        if pipeline_result:
            # Hedge engine
            hedge = getattr(pipeline_result, 'hedge_snapshot', None)
            if hedge:
                pressure = getattr(hedge, 'pressure_net', 0)
                scores["hedge"] = np.clip(pressure / 100, -1, 1)
            
            # Sentiment engine
            sentiment = getattr(pipeline_result, 'sentiment_snapshot', None)
            if sentiment:
                sent_score = getattr(sentiment, 'sentiment_score', 0)
                scores["sentiment"] = np.clip(sent_score, -1, 1)
            
            # Liquidity engine
            liquidity = getattr(pipeline_result, 'liquidity_snapshot', None)
            if liquidity:
                liq_score = getattr(liquidity, 'liquidity_score', 0.5)
                # High liquidity is neutral, low liquidity is slightly negative
                scores["liquidity"] = 0 if liq_score > 0.5 else -0.2
            
            # Elasticity engine
            elasticity = getattr(pipeline_result, 'elasticity_snapshot', None)
            if elasticity:
                trend = getattr(elasticity, 'trend_strength', 0)
                scores["elasticity"] = np.clip(trend, -1, 1)
        
        # Calculate weighted score
        weighted_score = sum(scores[k] * weights[k] for k in scores)
        
        # Determine direction and strength
        if weighted_score > hp.bullish_threshold:
            direction = "bullish"
            if weighted_score > hp.bullish_threshold + 0.3:
                strength = SignalStrength.STRONG_BUY
            elif weighted_score > hp.bullish_threshold + 0.15:
                strength = SignalStrength.BUY
            else:
                strength = SignalStrength.WEAK_BUY
        elif weighted_score < hp.bearish_threshold:
            direction = "bearish"
            if weighted_score < hp.bearish_threshold - 0.3:
                strength = SignalStrength.STRONG_SELL
            elif weighted_score < hp.bearish_threshold - 0.15:
                strength = SignalStrength.SELL
            else:
                strength = SignalStrength.WEAK_SELL
        else:
            direction = "neutral"
            strength = SignalStrength.NEUTRAL
        
        # Calculate confidence
        confirming = sum(1 for s in scores.values() if (s > 0) == (weighted_score > 0))
        base_confidence = confirming / len(scores)
        
        # Adjust confidence by signal strength
        strength_multiplier = abs(weighted_score) / max(abs(hp.bullish_threshold), abs(hp.bearish_threshold))
        confidence = min(base_confidence * (0.5 + 0.5 * strength_multiplier), 1.0)
        
        # Get uncertainty from forecast
        uncertainty = 0.5  # Default
        if forecast:
            uncertainty = 1.0 - getattr(forecast, 'confidence', 0.5)
        
        return MLSignal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            ml_forecast_score=scores["ml_forecast"],
            hedge_score=scores["hedge"],
            sentiment_score=scores["sentiment"],
            liquidity_score=scores["liquidity"],
            elasticity_score=scores["elasticity"],
            forecast_horizons=forecast_horizons,
            uncertainty=uncertainty,
            regime=self._current_regime.value if self._current_regime else "",
            timestamp=timestamp,
            symbol=symbol,
        )
    
    def _calculate_position_size(
        self,
        signal: MLSignal,
        regime: MarketRegime,
    ) -> MLPositionSize:
        """Calculate ML-optimized position size."""
        hp = self.hp_manager.current.position_sizing
        
        # Base size from Kelly
        if signal.confidence > 0 and signal.direction != "neutral":
            # Simplified Kelly: f* = edge / odds
            # Here we use confidence as edge proxy
            edge = signal.confidence - 0.5  # Excess confidence over random
            base_size = hp.kelly_fraction * edge * 2  # Scale to reasonable range
            base_size = max(0, base_size)
        else:
            base_size = 0.0
        
        # Apply adjustments
        adjustments = {
            "confidence": 1.0,
            "volatility": 1.0,
            "regime": 1.0,
            "uncertainty": 1.0,
        }
        
        # Confidence adjustment
        if hp.confidence_scaling:
            adjustments["confidence"] = 0.5 + signal.confidence
        
        # Volatility adjustment (reduce size in high vol)
        if hp.volatility_scaling:
            if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
                adjustments["volatility"] = 0.6
            elif regime == MarketRegime.LOW_VOLATILITY:
                adjustments["volatility"] = 1.2
        
        # Regime adjustment
        if hp.regime_scaling:
            regime_mult = hp.regime_size_multipliers.get(regime.value, 1.0)
            adjustments["regime"] = regime_mult
        
        # Uncertainty adjustment (reduce size when uncertain)
        adjustments["uncertainty"] = 1.0 - (signal.uncertainty * 0.5)
        
        # Calculate adjusted size
        total_adjustment = np.prod(list(adjustments.values()))
        adjusted_size = base_size * total_adjustment
        
        # Apply limits
        max_allowed = hp.max_position_pct
        final_size = min(adjusted_size, max_allowed)
        
        return MLPositionSize(
            base_size=base_size,
            adjusted_size=adjusted_size,
            confidence_adjustment=adjustments["confidence"],
            volatility_adjustment=adjustments["volatility"],
            regime_adjustment=adjustments["regime"],
            uncertainty_adjustment=adjustments["uncertainty"],
            max_allowed=max_allowed,
            reason=f"Kelly base={base_size:.4f}, adj={total_adjustment:.2f}",
        )
    
    def _generate_decision(
        self,
        signal: MLSignal,
        position_size: MLPositionSize,
        regime: MarketRegime,
    ) -> MLTradeDecision:
        """Generate final trade decision."""
        hp_signals = self.hp_manager.current.signals
        hp_risk = self.hp_manager.current.risk_management
        
        reasons = []
        warnings = []
        
        # Determine if we should trade
        should_trade = (
            signal.direction != "neutral"
            and signal.confidence >= hp_signals.min_confidence
            and position_size.final_size > 0.001  # Minimum meaningful size
        )
        
        # Action
        if should_trade:
            action = "buy" if signal.direction == "bullish" else "sell"
            reasons.append(f"Signal: {signal.strength.value} ({signal.confidence:.1%} confidence)")
            reasons.append(f"Direction: {signal.direction} in {regime.value} regime")
        else:
            action = "hold"
            if signal.confidence < hp_signals.min_confidence:
                reasons.append(f"Confidence {signal.confidence:.1%} below threshold {hp_signals.min_confidence:.1%}")
            if position_size.final_size <= 0.001:
                reasons.append("Position size too small")
        
        # Risk parameters
        stop_loss_pct = hp_risk.stop_loss_atr_multiple * 0.01  # Simplified ATR proxy
        take_profit_pct = hp_risk.take_profit_atr_multiple * 0.01
        
        # Adjust for uncertainty
        if signal.uncertainty > 0.5:
            stop_loss_pct *= 1.5  # Widen stop when uncertain
            warnings.append(f"High uncertainty ({signal.uncertainty:.1%}), widened stop")
        
        # Check reward/risk
        if take_profit_pct / stop_loss_pct < hp_risk.min_reward_risk:
            should_trade = False
            reasons.append(f"R:R ratio below minimum {hp_risk.min_reward_risk}")
        
        # Urgency based on signal strength
        if signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]:
            urgency = "immediate"
        elif signal.strength in [SignalStrength.WEAK_BUY, SignalStrength.WEAK_SELL]:
            urgency = "patient"
        else:
            urgency = "normal"
        
        return MLTradeDecision(
            should_trade=should_trade,
            action=action,
            signal=signal,
            position_size=position_size,
            stop_loss=stop_loss_pct,
            take_profit=take_profit_pct,
            trailing_stop=stop_loss_pct * 0.5 if hp_risk.trailing_stop_enabled else None,
            urgency=urgency,
            valid_until=signal.timestamp + timedelta(minutes=15),
            overall_confidence=signal.confidence * position_size.confidence_adjustment,
            model_confidence=1.0 - signal.uncertainty,
            signal_confidence=signal.confidence,
            reasons=reasons,
            warnings=warnings,
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline performance."""
        if not self._decision_history:
            return {"error": "No decisions recorded"}
        
        decisions = self._decision_history
        trades = [d for d in decisions if d.should_trade]
        
        return {
            "total_decisions": len(decisions),
            "trade_signals": len(trades),
            "hold_signals": len(decisions) - len(trades),
            "avg_confidence": np.mean([d.overall_confidence for d in decisions]),
            "avg_position_size": np.mean([d.position_size.final_size for d in trades]) if trades else 0,
            "regime_distribution": self._get_regime_distribution(),
            "signal_distribution": self._get_signal_distribution(),
        }
    
    def _get_regime_distribution(self) -> Dict[str, int]:
        """Get distribution of regimes in signal history."""
        regimes = [s.regime for s in self._signal_history if s.regime]
        return {r: regimes.count(r) for r in set(regimes)}
    
    def _get_signal_distribution(self) -> Dict[str, int]:
        """Get distribution of signal strengths."""
        strengths = [s.strength.value for s in self._signal_history]
        return {s: strengths.count(s) for s in set(strengths)}


__all__ = [
    "SignalStrength",
    "MLSignal",
    "MLPositionSize",
    "MLTradeDecision",
    "RegimeDetector",
    "AdaptiveMLPipeline",
]
