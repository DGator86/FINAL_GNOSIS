"""
ML-Enabled Backtest Engine - Full Gnosis System Backtesting with Real Historical Data

This engine connects all 4 trading engines (Hedge, Liquidity, Sentiment, Elasticity),
the agent voting system, and ML components to real historical data from Alpaca.

Key Features:
- Real historical data from Alpaca (up to 5+ years for stocks)
- All 4 engines running on each bar
- Composer consensus for trade signals
- Optional LSTM lookahead predictions
- Position management with realistic costs
- Comprehensive performance metrics
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Engine imports
from schemas.core_schemas import (
    AgentSuggestion,
    DirectionEnum,
    HedgeSnapshot,
    LiquiditySnapshot,
    SentimentSnapshot,
    ElasticitySnapshot,
)


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    entry_date: datetime
    exit_date: Optional[datetime] = None
    symbol: str = ""
    direction: str = "long"
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0

    # Costs
    entry_cost_bps: float = 0.0
    exit_cost_bps: float = 0.0

    # Signals at entry
    hedge_signal: float = 0.0
    liquidity_signal: float = 0.0
    sentiment_signal: float = 0.0
    composite_signal: float = 0.0
    confidence: float = 0.0

    # ML predictions at entry (optional)
    lstm_prediction: float = 0.0
    lstm_confidence: float = 0.0

    # Outcome
    is_winner: bool = False
    exit_reason: str = ""  # "stop_loss", "take_profit", "signal_reversal", "end_of_test"


@dataclass
class MLBacktestConfig:
    """Configuration for ML-enabled backtesting."""

    # Data settings
    symbol: str = "SPY"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-01"
    timeframe: str = "1Day"  # 1Min, 5Min, 15Min, 1Hour, 1Day

    # Capital settings
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10  # 10% of capital per trade
    max_positions: int = 1  # Max concurrent positions

    # Cost modeling
    slippage_bps: float = 5.0  # 5 basis points
    impact_bps: float = 5.0  # 5 basis points
    commission_per_trade: float = 0.0  # Alpaca is commission-free

    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit

    # Signal thresholds
    entry_threshold: float = 0.3  # Minimum consensus for entry
    exit_threshold: float = 0.1  # Exit when consensus drops below

    # Engine weights for composer
    hedge_weight: float = 0.4
    liquidity_weight: float = 0.2
    sentiment_weight: float = 0.4

    # ML settings
    use_lstm: bool = False  # Enable LSTM predictions
    lstm_model_path: Optional[str] = None
    lstm_weight: float = 0.3  # Weight of LSTM in final signal

    # Output settings
    save_trades: bool = True
    save_equity_curve: bool = True
    output_dir: str = "runs/backtests"

    # Metadata
    tag: str = ""
    notes: str = ""


@dataclass
class MLBacktestResults:
    """Results from ML-enabled backtest."""

    # Dates
    start_date: datetime = None
    end_date: datetime = None
    total_bars: int = 0

    # Returns
    initial_capital: float = 100_000.0
    final_capital: float = 100_000.0
    total_return: float = 0.0
    total_return_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    # Execution
    total_costs: float = 0.0
    avg_cost_per_trade: float = 0.0

    # ML metrics
    lstm_accuracy: float = 0.0
    lstm_used: bool = False

    # Data
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    # Attribution
    hedge_contribution: float = 0.0
    liquidity_contribution: float = 0.0
    sentiment_contribution: float = 0.0

    # Config reference
    config: Optional[MLBacktestConfig] = None


class HistoricalDataEngine:
    """
    Simulates engine outputs for historical bars.

    Since we can't get real options data historically, we derive
    synthetic but realistic engine signals from price/volume data.
    """

    def __init__(self, config: MLBacktestConfig):
        self.config = config

    def compute_hedge_snapshot(
        self,
        bar: Dict[str, Any],
        history: pd.DataFrame,
        timestamp: datetime
    ) -> HedgeSnapshot:
        """
        Compute hedge engine metrics from price action.

        Uses volatility and price momentum as proxy for dealer positioning.
        """
        symbol = self.config.symbol

        if len(history) < 20:
            return HedgeSnapshot(timestamp=timestamp, symbol=symbol)

        # Calculate realized volatility (proxy for gamma exposure)
        returns = history['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2

        # Calculate momentum (proxy for directional flow)
        sma_5 = history['close'].tail(5).mean()
        sma_20 = history['close'].tail(20).mean()
        momentum = (sma_5 - sma_20) / sma_20 if sma_20 > 0 else 0

        # Estimate pressures from volume and price action
        recent_volume = history['volume'].tail(5).mean()
        avg_volume = history['volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Directional volume (up vs down days)
        recent = history.tail(5)
        up_volume = recent[recent['close'] > recent['open']]['volume'].sum()
        down_volume = recent[recent['close'] < recent['open']]['volume'].sum()

        total_vol = up_volume + down_volume
        if total_vol > 0:
            pressure_up = (up_volume / total_vol) * volume_ratio
            pressure_down = (down_volume / total_vol) * volume_ratio
        else:
            pressure_up = pressure_down = 0.5

        pressure_net = pressure_up - pressure_down

        # Elasticity (inverse of volatility - volatile = low elasticity)
        elasticity = max(0.1, 1.0 - min(1.0, volatility))

        # Movement energy
        movement_energy = abs(momentum) * volume_ratio * 100

        # Energy asymmetry
        energy_asymmetry = momentum * 2  # -1 to +1 scale
        energy_asymmetry = max(-1, min(1, energy_asymmetry))

        # Greek pressures (estimated from volatility dynamics)
        vol_change = returns.tail(5).std() - returns.tail(20).std() if len(returns) >= 20 else 0
        gamma_pressure = abs(vol_change) * 100
        vanna_pressure = abs(momentum * volatility) * 50
        charm_pressure = abs(vol_change * momentum) * 25

        # Regime classification
        if abs(momentum) > 0.02 and volume_ratio > 1.2:
            regime = "trending"
        elif volatility > 0.3:
            regime = "volatile"
        elif volatility < 0.1:
            regime = "compressed"
        else:
            regime = "neutral"

        # Confidence based on data quality
        confidence = min(1.0, len(history) / 50)

        return HedgeSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            elasticity=elasticity,
            movement_energy=movement_energy,
            energy_asymmetry=energy_asymmetry,
            pressure_up=pressure_up,
            pressure_down=pressure_down,
            pressure_net=pressure_net,
            gamma_pressure=gamma_pressure,
            vanna_pressure=vanna_pressure,
            charm_pressure=charm_pressure,
            dealer_gamma_sign=momentum,
            regime=regime,
            confidence=confidence,
        )

    def compute_liquidity_snapshot(
        self,
        bar: Dict[str, Any],
        history: pd.DataFrame,
        timestamp: datetime
    ) -> LiquiditySnapshot:
        """Compute liquidity metrics from volume and price range."""
        symbol = self.config.symbol

        if len(history) < 5:
            return LiquiditySnapshot(timestamp=timestamp, symbol=symbol)

        # Average volume
        avg_volume = history['volume'].mean()
        recent_volume = history['volume'].tail(5).mean()

        # Estimate spread from high-low range
        avg_range = ((history['high'] - history['low']) / history['close']).mean()
        estimated_spread = avg_range * 0.1  # Spread is fraction of range

        # Depth proxy (volume * inverse of spread)
        depth = recent_volume / (1 + estimated_spread * 100) if estimated_spread > 0 else recent_volume

        # Impact cost
        impact_cost = estimated_spread * 50  # bps

        # Liquidity score
        volume_score = min(1.0, avg_volume / 10_000_000)
        spread_score = max(0.0, 1.0 - estimated_spread * 100)
        liquidity_score = volume_score * 0.7 + spread_score * 0.3

        return LiquiditySnapshot(
            timestamp=timestamp,
            symbol=symbol,
            liquidity_score=liquidity_score,
            bid_ask_spread=estimated_spread * 100,  # percentage
            volume=recent_volume,
            depth=depth,
            impact_cost=impact_cost,
        )

    def compute_sentiment_snapshot(
        self,
        bar: Dict[str, Any],
        history: pd.DataFrame,
        timestamp: datetime
    ) -> SentimentSnapshot:
        """Compute sentiment from price momentum and volume patterns."""
        symbol = self.config.symbol

        if len(history) < 20:
            return SentimentSnapshot(timestamp=timestamp, symbol=symbol)

        # Technical sentiment (momentum + mean reversion signals)
        returns = history['close'].pct_change().dropna()
        sma_10 = history['close'].tail(10).mean()
        sma_20 = history['close'].tail(20).mean()
        current_price = bar['close']

        # Trend following signal
        trend_signal = (sma_10 - sma_20) / sma_20 if sma_20 > 0 else 0

        # Mean reversion signal (oversold/overbought)
        distance_from_sma = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        reversion_signal = -distance_from_sma * 0.5  # Contrarian

        technical_sentiment = trend_signal * 0.7 + reversion_signal * 0.3
        technical_sentiment = max(-1, min(1, technical_sentiment * 10))  # Scale to -1 to 1

        # Flow sentiment (volume + price direction)
        recent_returns = returns.tail(5)
        positive_returns = recent_returns[recent_returns > 0]
        negative_returns = recent_returns[recent_returns < 0]

        if len(recent_returns) > 0:
            pos_ratio = len(positive_returns) / len(recent_returns)
            flow_sentiment = (pos_ratio - 0.5) * 2  # -1 to 1
        else:
            flow_sentiment = 0.0

        # News sentiment (simulated - random walk with autocorrelation)
        # In real use, this would come from actual news feeds
        news_sentiment = np.tanh(technical_sentiment * 0.5)

        # Combined sentiment
        sentiment_score = (
            technical_sentiment * 0.4 +
            flow_sentiment * 0.3 +
            news_sentiment * 0.3
        )

        # Confidence (agreement between sources)
        sources = [technical_sentiment, flow_sentiment, news_sentiment]
        avg_diff = np.mean([abs(s - sentiment_score) for s in sources])
        confidence = max(0.2, 1.0 - avg_diff)

        return SentimentSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            sentiment_score=sentiment_score,
            news_sentiment=news_sentiment,
            flow_sentiment=flow_sentiment,
            technical_sentiment=technical_sentiment,
            confidence=confidence,
        )

    def compute_elasticity_snapshot(
        self,
        bar: Dict[str, Any],
        history: pd.DataFrame,
        timestamp: datetime
    ) -> ElasticitySnapshot:
        """Compute volatility and trend metrics."""
        symbol = self.config.symbol

        if len(history) < 20:
            return ElasticitySnapshot(timestamp=timestamp, symbol=symbol)

        returns = history['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2

        # Trend strength (ADX-like)
        highs = history['high'].tail(14)
        lows = history['low'].tail(14)
        closes = history['close'].tail(14)

        if len(closes) >= 2:
            plus_dm = (highs.diff() > 0) & (highs.diff() > -lows.diff())
            minus_dm = (-lows.diff() > 0) & (-lows.diff() > highs.diff())
            trend_strength = abs(plus_dm.sum() - minus_dm.sum()) / len(closes)
        else:
            trend_strength = 0.0

        # Regime classification
        if volatility > 0.4:
            regime = "high_volatility"
        elif volatility > 0.2:
            regime = "moderate"
        elif volatility > 0.1:
            regime = "low_volatility"
        else:
            regime = "compressed"

        return ElasticitySnapshot(
            timestamp=timestamp,
            symbol=symbol,
            volatility=volatility,
            volatility_regime=regime,
            trend_strength=trend_strength,
        )


class MLBacktestEngine:
    """
    ML-Enabled Backtest Engine for Full Gnosis System.

    Integrates:
    - Real historical data from Alpaca
    - All 4 engines (Hedge, Liquidity, Sentiment, Elasticity)
    - Composer consensus voting
    - Optional LSTM predictions
    - Position management with realistic costs
    - Comprehensive metrics calculation
    """

    def __init__(self, config: MLBacktestConfig):
        self.config = config
        self.historical_engine = HistoricalDataEngine(config)

        # State
        self.current_capital = config.initial_capital
        self.current_position: Optional[BacktestTrade] = None
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict[str, Any]] = []

        # LSTM (optional)
        self.lstm_predictor = None
        if config.use_lstm and config.lstm_model_path:
            self._load_lstm()

        logger.info(f"MLBacktestEngine initialized for {config.symbol}")

    def _load_lstm(self):
        """Load LSTM model if available."""
        try:
            from models.predictors.lstm_lookahead import LSTMLookaheadPredictor, LookaheadConfig
            self.lstm_predictor = LSTMLookaheadPredictor(model_path=self.config.lstm_model_path)
            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")
            self.lstm_predictor = None

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical data from Alpaca."""
        from adapters.alpaca_market_adapter import AlpacaMarketDataAdapter

        try:
            adapter = AlpacaMarketDataAdapter()

            start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.config.end_date, "%Y-%m-%d")

            # Make timezone aware
            start = start.replace(tzinfo=timezone.utc)
            end = end.replace(tzinfo=timezone.utc)

            logger.info(f"Fetching {self.config.symbol} data from {start} to {end}")

            bars = adapter.get_bars(
                symbol=self.config.symbol,
                start=start,
                end=end,
                timeframe=self.config.timeframe,
            )

            if not bars:
                raise ValueError(f"No data returned for {self.config.symbol}")

            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                }
                for bar in bars
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Fetched {len(df)} bars for {self.config.symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

    def compute_consensus(
        self,
        hedge: HedgeSnapshot,
        liquidity: LiquiditySnapshot,
        sentiment: SentimentSnapshot,
        elasticity: ElasticitySnapshot,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Compute composer consensus from engine snapshots.

        Returns:
            consensus_value: -1 to +1 (SHORT to LONG)
            confidence: 0 to 1
            details: Dictionary with component signals
        """
        # Extract directional signals from each engine

        # Hedge signal: energy asymmetry and pressure
        hedge_signal = hedge.energy_asymmetry * 0.5 + np.sign(hedge.pressure_net) * 0.5
        hedge_signal = max(-1, min(1, hedge_signal))

        # Liquidity signal: high liquidity = safe to trade, low = caution
        # Not directional but affects confidence
        liquidity_multiplier = liquidity.liquidity_score

        # Sentiment signal: direct sentiment score
        sentiment_signal = sentiment.sentiment_score
        sentiment_signal = max(-1, min(1, sentiment_signal))

        # Elasticity signal: trend-following in trends, mean-reversion in ranges
        if elasticity.trend_strength > 0.3:
            # Strong trend - follow momentum
            elasticity_signal = hedge_signal * 0.5
        else:
            # Range-bound - slight mean reversion
            elasticity_signal = -sentiment_signal * 0.2

        # Weighted consensus
        w_hedge = self.config.hedge_weight
        w_liquidity = self.config.liquidity_weight
        w_sentiment = self.config.sentiment_weight

        # Liquidity weight goes to confidence, not direction
        effective_weights = w_hedge + w_sentiment + (w_liquidity * 0.5)

        consensus = (
            hedge_signal * w_hedge +
            sentiment_signal * w_sentiment +
            elasticity_signal * (w_liquidity * 0.5)
        ) / effective_weights if effective_weights > 0 else 0

        # Confidence based on:
        # 1. Agreement between signals
        # 2. Liquidity quality
        # 3. Individual engine confidences
        signals = [hedge_signal, sentiment_signal, elasticity_signal]
        signal_std = np.std(signals) if len(signals) > 1 else 0
        agreement = max(0, 1 - signal_std)

        confidence = (
            agreement * 0.4 +
            liquidity_multiplier * 0.3 +
            hedge.confidence * 0.15 +
            sentiment.confidence * 0.15
        )
        confidence = max(0.1, min(1.0, confidence))

        details = {
            'hedge_signal': hedge_signal,
            'liquidity_score': liquidity.liquidity_score,
            'sentiment_signal': sentiment_signal,
            'elasticity_signal': elasticity_signal,
            'agreement': agreement,
            'hedge_regime': hedge.regime,
            'volatility_regime': elasticity.volatility_regime,
        }

        return consensus, confidence, details

    def run_backtest(self) -> MLBacktestResults:
        """Run the full ML-enabled backtest."""

        # Fetch data
        df = self.fetch_historical_data()

        if len(df) < 50:
            raise ValueError(f"Insufficient data: {len(df)} bars (need at least 50)")

        # Initialize results
        results = MLBacktestResults(
            config=self.config,
            initial_capital=self.config.initial_capital,
        )

        # Reset state
        self.current_capital = self.config.initial_capital
        self.current_position = None
        self.trades = []
        self.equity_curve = []

        logger.info(f"Running backtest on {len(df)} bars...")

        # Process each bar
        for i in range(50, len(df)):  # Start after warmup period
            bar = df.iloc[i].to_dict()
            history = df.iloc[:i]
            timestamp = bar['timestamp']

            if pd.isna(timestamp):
                timestamp = datetime.now(timezone.utc)
            elif not hasattr(timestamp, 'tzinfo') or timestamp.tzinfo is None:
                timestamp = pd.to_datetime(timestamp).to_pydatetime()
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Compute engine snapshots
            hedge = self.historical_engine.compute_hedge_snapshot(bar, history, timestamp)
            liquidity = self.historical_engine.compute_liquidity_snapshot(bar, history, timestamp)
            sentiment = self.historical_engine.compute_sentiment_snapshot(bar, history, timestamp)
            elasticity = self.historical_engine.compute_elasticity_snapshot(bar, history, timestamp)

            # Compute consensus
            consensus, confidence, details = self.compute_consensus(
                hedge, liquidity, sentiment, elasticity
            )

            # Optional LSTM enhancement
            lstm_pred = 0.0
            lstm_conf = 0.0
            if self.lstm_predictor and self.config.use_lstm:
                lstm_pred, lstm_conf = self._get_lstm_prediction(history)
                if lstm_conf > 0:
                    # Blend LSTM with consensus
                    lstm_weight = self.config.lstm_weight
                    consensus = consensus * (1 - lstm_weight) + lstm_pred * lstm_weight
                    confidence = confidence * (1 - lstm_weight * 0.5) + lstm_conf * (lstm_weight * 0.5)

            # Position management
            current_price = bar['close']

            # Check exit conditions for open position
            if self.current_position is not None:
                should_exit, exit_reason = self._check_exit(
                    self.current_position, current_price, consensus
                )
                if should_exit:
                    self._close_position(current_price, timestamp, exit_reason)

            # Check entry conditions
            if self.current_position is None:
                if abs(consensus) >= self.config.entry_threshold and confidence >= 0.3:
                    direction = "long" if consensus > 0 else "short"
                    self._open_position(
                        direction=direction,
                        price=current_price,
                        timestamp=timestamp,
                        consensus=consensus,
                        confidence=confidence,
                        details=details,
                        lstm_pred=lstm_pred,
                        lstm_conf=lstm_conf,
                    )

            # Record equity curve
            position_value = 0.0
            if self.current_position:
                if self.current_position.direction == "long":
                    unrealized = (current_price - self.current_position.entry_price) * self.current_position.position_size
                else:
                    unrealized = (self.current_position.entry_price - current_price) * self.current_position.position_size
                position_value = self.current_position.entry_price * self.current_position.position_size + unrealized

            total_equity = self.current_capital + position_value

            self.equity_curve.append({
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'equity': total_equity,
                'capital': self.current_capital,
                'position_value': position_value,
                'consensus': consensus,
                'confidence': confidence,
            })

        # Close any remaining position
        if self.current_position is not None:
            final_bar = df.iloc[-1]
            final_timestamp = final_bar['timestamp']
            if pd.isna(final_timestamp):
                final_timestamp = datetime.now(timezone.utc)
            elif not hasattr(final_timestamp, 'tzinfo') or final_timestamp.tzinfo is None:
                final_timestamp = pd.to_datetime(final_timestamp).to_pydatetime()
                if final_timestamp.tzinfo is None:
                    final_timestamp = final_timestamp.replace(tzinfo=timezone.utc)
            self._close_position(final_bar['close'], final_timestamp, "end_of_test")

        # Calculate results
        results = self._calculate_results(df, results)

        # Save results
        if self.config.save_trades or self.config.save_equity_curve:
            self._save_results(results)

        return results

    def _get_lstm_prediction(self, history: pd.DataFrame) -> Tuple[float, float]:
        """Get LSTM prediction if available."""
        try:
            if self.lstm_predictor is None:
                return 0.0, 0.0

            # Prepare features (simplified)
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            features = history[feature_cols].tail(60).values

            if len(features) < 60:
                return 0.0, 0.0

            # This would use actual LSTM prediction
            # For now, return 0 as LSTM needs training
            return 0.0, 0.0

        except Exception as e:
            logger.debug(f"LSTM prediction error: {e}")
            return 0.0, 0.0

    def _check_exit(
        self,
        position: BacktestTrade,
        current_price: float,
        consensus: float
    ) -> Tuple[bool, str]:
        """Check if position should be closed."""

        # Calculate P&L
        if position.direction == "long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price

        # Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return True, "stop_loss"

        # Take profit
        if pnl_pct >= self.config.take_profit_pct:
            return True, "take_profit"

        # Signal reversal (consensus flips direction)
        if position.direction == "long" and consensus < -self.config.exit_threshold:
            return True, "signal_reversal"
        if position.direction == "short" and consensus > self.config.exit_threshold:
            return True, "signal_reversal"

        return False, ""

    def _open_position(
        self,
        direction: str,
        price: float,
        timestamp: datetime,
        consensus: float,
        confidence: float,
        details: Dict[str, Any],
        lstm_pred: float = 0.0,
        lstm_conf: float = 0.0,
    ):
        """Open a new position."""

        # Calculate position size
        position_value = self.current_capital * self.config.position_size_pct
        position_size = position_value / price

        # Apply entry costs
        entry_cost_bps = self.config.slippage_bps + self.config.impact_bps
        if direction == "long":
            adjusted_price = price * (1 + entry_cost_bps / 10000)
        else:
            adjusted_price = price * (1 - entry_cost_bps / 10000)

        # Calculate stop loss and take profit levels
        if direction == "long":
            stop_loss = adjusted_price * (1 - self.config.stop_loss_pct)
            take_profit = adjusted_price * (1 + self.config.take_profit_pct)
        else:
            stop_loss = adjusted_price * (1 + self.config.stop_loss_pct)
            take_profit = adjusted_price * (1 - self.config.take_profit_pct)

        # Create trade
        self.current_position = BacktestTrade(
            entry_date=timestamp,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=adjusted_price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_cost_bps=entry_cost_bps,
            hedge_signal=details.get('hedge_signal', 0),
            liquidity_signal=details.get('liquidity_score', 0),
            sentiment_signal=details.get('sentiment_signal', 0),
            composite_signal=consensus,
            confidence=confidence,
            lstm_prediction=lstm_pred,
            lstm_confidence=lstm_conf,
        )

        # Deduct capital for position
        self.current_capital -= position_value

        logger.debug(f"Opened {direction} position at {adjusted_price:.2f}")

    def _close_position(
        self,
        price: float,
        timestamp: datetime,
        exit_reason: str,
    ):
        """Close current position."""
        if self.current_position is None:
            return

        position = self.current_position

        # Apply exit costs
        exit_cost_bps = self.config.slippage_bps + self.config.impact_bps
        if position.direction == "long":
            adjusted_price = price * (1 - exit_cost_bps / 10000)
        else:
            adjusted_price = price * (1 + exit_cost_bps / 10000)

        # Calculate P&L
        position_value = position.entry_price * position.position_size
        if position.direction == "long":
            gross_pnl = (adjusted_price - position.entry_price) * position.position_size
        else:
            gross_pnl = (position.entry_price - adjusted_price) * position.position_size

        # Net P&L (after costs)
        total_cost = position_value * (position.entry_cost_bps + exit_cost_bps) / 10000
        net_pnl = gross_pnl - total_cost
        pnl_pct = net_pnl / position_value if position_value > 0 else 0

        # Update position
        position.exit_date = timestamp
        position.exit_price = adjusted_price
        position.exit_cost_bps = exit_cost_bps
        position.gross_pnl = gross_pnl
        position.net_pnl = net_pnl
        position.pnl_pct = pnl_pct
        position.is_winner = net_pnl > 0
        position.exit_reason = exit_reason

        # Return capital + P&L
        self.current_capital += position_value + net_pnl

        # Record trade
        self.trades.append(position)
        self.current_position = None

        logger.debug(f"Closed {position.direction} at {adjusted_price:.2f}, P&L: ${net_pnl:.2f} ({pnl_pct*100:.2f}%)")

    def _calculate_results(self, df: pd.DataFrame, results: MLBacktestResults) -> MLBacktestResults:
        """Calculate all backtest metrics."""

        # Basic info
        results.start_date = df['timestamp'].iloc[0]
        results.end_date = df['timestamp'].iloc[-1]
        results.total_bars = len(df)

        # Returns
        results.final_capital = self.current_capital
        results.total_return = results.final_capital - results.initial_capital
        results.total_return_pct = results.total_return / results.initial_capital

        # Trade statistics
        results.trades = self.trades
        results.total_trades = len(self.trades)

        if results.total_trades > 0:
            winners = [t for t in self.trades if t.is_winner]
            losers = [t for t in self.trades if not t.is_winner]

            results.winning_trades = len(winners)
            results.losing_trades = len(losers)
            results.win_rate = len(winners) / len(self.trades)

            win_pnls = [t.net_pnl for t in winners]
            loss_pnls = [abs(t.net_pnl) for t in losers]

            results.avg_win = np.mean(win_pnls) if win_pnls else 0
            results.avg_loss = np.mean(loss_pnls) if loss_pnls else 0
            results.avg_trade = np.mean([t.net_pnl for t in self.trades])
            results.largest_win = max(win_pnls) if win_pnls else 0
            results.largest_loss = max(loss_pnls) if loss_pnls else 0

            total_wins = sum(win_pnls)
            total_losses = sum(loss_pnls)
            results.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            # Costs
            results.total_costs = sum(
                t.entry_price * t.position_size * (t.entry_cost_bps + t.exit_cost_bps) / 10000
                for t in self.trades
            )
            results.avg_cost_per_trade = results.total_costs / len(self.trades)

        # Equity curve and risk metrics
        results.equity_curve = self.equity_curve

        if len(self.equity_curve) > 1:
            equities = [e['equity'] for e in self.equity_curve]

            # Daily returns
            returns = pd.Series(equities).pct_change().dropna()
            results.daily_returns = list(returns)

            # Volatility
            results.volatility = returns.std() * np.sqrt(252)

            # Sharpe ratio (assuming 0% risk-free rate)
            if results.volatility > 0:
                results.sharpe_ratio = (returns.mean() * 252) / results.volatility

            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252)
                if downside_std > 0:
                    results.sortino_ratio = (returns.mean() * 252) / downside_std

            # Max drawdown
            equity_series = pd.Series(equities)
            rolling_max = equity_series.expanding().max()
            drawdown = equity_series - rolling_max
            drawdown_pct = drawdown / rolling_max

            results.max_drawdown = abs(drawdown.min())
            results.max_drawdown_pct = abs(drawdown_pct.min())

            # Calmar ratio
            if results.max_drawdown_pct > 0:
                results.calmar_ratio = results.total_return_pct / results.max_drawdown_pct

        # Attribution (simplified)
        if results.total_trades > 0:
            hedge_trades = [t for t in self.trades if abs(t.hedge_signal) > abs(t.sentiment_signal)]
            sentiment_trades = [t for t in self.trades if abs(t.sentiment_signal) >= abs(t.hedge_signal)]

            results.hedge_contribution = sum(t.net_pnl for t in hedge_trades)
            results.sentiment_contribution = sum(t.net_pnl for t in sentiment_trades)
            results.liquidity_contribution = 0  # Liquidity affects confidence, not direction

        # LSTM metrics
        results.lstm_used = self.config.use_lstm and self.lstm_predictor is not None

        return results

    def _save_results(self, results: MLBacktestResults):
        """Save results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tag = self.config.tag or f"{self.config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save summary
        summary = {
            'tag': tag,
            'symbol': self.config.symbol,
            'start_date': str(results.start_date),
            'end_date': str(results.end_date),
            'total_bars': results.total_bars,
            'initial_capital': results.initial_capital,
            'final_capital': results.final_capital,
            'total_return': results.total_return,
            'total_return_pct': results.total_return_pct,
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'max_drawdown_pct': results.max_drawdown_pct,
            'calmar_ratio': results.calmar_ratio,
            'volatility': results.volatility,
            'total_costs': results.total_costs,
            'lstm_used': results.lstm_used,
            'config': {
                'timeframe': self.config.timeframe,
                'position_size_pct': self.config.position_size_pct,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct,
                'entry_threshold': self.config.entry_threshold,
                'hedge_weight': self.config.hedge_weight,
                'liquidity_weight': self.config.liquidity_weight,
                'sentiment_weight': self.config.sentiment_weight,
            },
        }

        summary_path = output_dir / f"{tag}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to {summary_path}")

        # Save trades
        if self.config.save_trades and results.trades:
            trades_data = []
            for t in results.trades:
                trades_data.append({
                    'entry_date': str(t.entry_date),
                    'exit_date': str(t.exit_date),
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'position_size': t.position_size,
                    'net_pnl': t.net_pnl,
                    'pnl_pct': t.pnl_pct,
                    'is_winner': t.is_winner,
                    'exit_reason': t.exit_reason,
                    'composite_signal': t.composite_signal,
                    'confidence': t.confidence,
                })

            trades_path = output_dir / f"{tag}_trades.json"
            with open(trades_path, 'w') as f:
                json.dump(trades_data, f, indent=2, default=str)

        # Save equity curve
        if self.config.save_equity_curve and results.equity_curve:
            equity_path = output_dir / f"{tag}_equity.json"
            with open(equity_path, 'w') as f:
                json.dump(results.equity_curve, f, indent=2, default=str)


def run_ml_backtest(
    symbol: str = "SPY",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-01",
    timeframe: str = "1Day",
    initial_capital: float = 100_000.0,
    tag: str = "",
    **kwargs
) -> MLBacktestResults:
    """
    Convenience function to run ML-enabled backtest.

    Args:
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
        initial_capital: Starting capital
        tag: Run identifier
        **kwargs: Additional MLBacktestConfig parameters

    Returns:
        MLBacktestResults with all metrics and trade data
    """
    config = MLBacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        initial_capital=initial_capital,
        tag=tag or f"{symbol}_{start_date}_{end_date}",
        **kwargs
    )

    engine = MLBacktestEngine(config)
    return engine.run_backtest()


def print_results_summary(results: MLBacktestResults):
    """Print formatted results summary."""
    print("\n" + "="*60)
    print("ML BACKTEST RESULTS SUMMARY")
    print("="*60)
    print(f"Symbol: {results.config.symbol if results.config else 'N/A'}")
    print(f"Period: {results.start_date} to {results.end_date}")
    print(f"Total Bars: {results.total_bars}")
    print("-"*60)
    print("RETURNS")
    print(f"  Initial Capital:  ${results.initial_capital:,.2f}")
    print(f"  Final Capital:    ${results.final_capital:,.2f}")
    print(f"  Total Return:     ${results.total_return:,.2f} ({results.total_return_pct*100:.2f}%)")
    print("-"*60)
    print("TRADES")
    print(f"  Total Trades:     {results.total_trades}")
    print(f"  Win Rate:         {results.win_rate*100:.1f}%")
    print(f"  Profit Factor:    {results.profit_factor:.2f}")
    print(f"  Avg Win:          ${results.avg_win:,.2f}")
    print(f"  Avg Loss:         ${results.avg_loss:,.2f}")
    print(f"  Largest Win:      ${results.largest_win:,.2f}")
    print(f"  Largest Loss:     ${results.largest_loss:,.2f}")
    print("-"*60)
    print("RISK METRICS")
    print(f"  Sharpe Ratio:     {results.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:    {results.sortino_ratio:.2f}")
    print(f"  Max Drawdown:     {results.max_drawdown_pct*100:.2f}%")
    print(f"  Calmar Ratio:     {results.calmar_ratio:.2f}")
    print(f"  Volatility:       {results.volatility*100:.1f}%")
    print("-"*60)
    print("COSTS")
    print(f"  Total Costs:      ${results.total_costs:,.2f}")
    print(f"  Avg Cost/Trade:   ${results.avg_cost_per_trade:,.2f}")
    print("-"*60)
    print(f"LSTM Used: {results.lstm_used}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ML-enabled Gnosis backtest")
    parser.add_argument("--symbol", type=str, default="SPY", help="Trading symbol")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-01", help="End date")
    parser.add_argument("--timeframe", type=str, default="1Day", help="Timeframe")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--tag", type=str, default="", help="Run tag")
    parser.add_argument("--use-lstm", action="store_true", help="Enable LSTM predictions")

    args = parser.parse_args()

    results = run_ml_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        initial_capital=args.capital,
        tag=args.tag,
        use_lstm=args.use_lstm,
    )

    print_results_summary(results)
