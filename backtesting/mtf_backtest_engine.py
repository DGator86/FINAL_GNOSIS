#!/usr/bin/env python3
"""
Multi-Timeframe (MTF) Backtest Engine

Backtests using multiple timeframe confirmation for higher confidence trades.
Analyzes signals across: 1W, 1D, 4H, 1H, 15min timeframes.

Key Features:
- Multi-timeframe signal alignment detection
- Timeframe-weighted confidence scoring
- Higher timeframe trend confirmation
- Lower timeframe entry timing
- Configurable alignment requirements

Author: GNOSIS Trading System
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import json

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Timeframe(Enum):
    """Supported timeframes."""
    W1 = "1week"
    D1 = "1day"
    H4 = "4hour"
    H1 = "1hour"
    M15 = "15min"


# Timeframe weights for confidence calculation (higher TF = more weight)
TIMEFRAME_WEIGHTS = {
    Timeframe.W1: 0.30,   # Weekly - strongest weight
    Timeframe.D1: 0.25,   # Daily
    Timeframe.H4: 0.20,   # 4-hour
    Timeframe.H1: 0.15,   # 1-hour
    Timeframe.M15: 0.10,  # 15-min - entry timing
}

# Minutes per timeframe for resampling
TIMEFRAME_MINUTES = {
    Timeframe.W1: 7 * 24 * 60,
    Timeframe.D1: 24 * 60,
    Timeframe.H4: 4 * 60,
    Timeframe.H1: 60,
    Timeframe.M15: 15,
}


@dataclass
class TimeframeSignal:
    """Signal for a single timeframe."""
    timeframe: Timeframe
    timestamp: datetime
    
    # Trend signals (-1 to 1)
    trend_signal: float = 0.0       # MA-based trend
    momentum_signal: float = 0.0    # ROC/momentum
    sentiment_signal: float = 0.0   # Combined sentiment
    
    # Direction
    direction: str = "neutral"  # bullish, bearish, neutral
    strength: float = 0.0       # 0-1 signal strength
    
    # Technical levels
    price: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    atr: float = 0.0
    rsi: float = 50.0
    
    # Volume
    volume_ratio: float = 1.0   # vs average


@dataclass
class MTFSignal:
    """Multi-timeframe combined signal."""
    timestamp: datetime
    symbol: str
    
    # Individual timeframe signals
    signals: Dict[Timeframe, TimeframeSignal] = field(default_factory=dict)
    
    # Alignment metrics
    alignment_score: float = 0.0      # -1 to 1 (all bearish to all bullish)
    alignment_count: int = 0          # How many TFs agree
    total_timeframes: int = 0
    
    # Weighted confidence
    weighted_confidence: float = 0.0  # 0-1
    
    # Final direction
    direction: str = "neutral"
    
    # Higher timeframe context
    htf_bias: str = "neutral"        # W1 + D1 combined
    ltf_confirmation: bool = False    # H4/H1/M15 confirm HTF
    
    # Entry quality
    entry_quality: str = "none"       # none, weak, moderate, strong, perfect


@dataclass 
class MTFTrade:
    """Trade record with MTF context."""
    trade_id: str = ""
    symbol: str = ""
    direction: str = "long"
    
    # Timing
    entry_date: datetime = None
    exit_date: datetime = None
    hold_days: float = 0.0
    
    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    
    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # MTF context at entry
    alignment_score: float = 0.0
    alignment_count: int = 0
    weighted_confidence: float = 0.0
    htf_bias: str = "neutral"
    ltf_confirmation: bool = False
    entry_quality: str = "none"
    
    # Individual TF signals at entry
    w1_signal: float = 0.0
    d1_signal: float = 0.0
    h4_signal: float = 0.0
    h1_signal: float = 0.0
    m15_signal: float = 0.0
    
    # Exit
    exit_reason: str = ""


@dataclass
class MTFBacktestConfig:
    """Configuration for MTF backtesting."""
    
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    
    # Date range
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-01"
    
    # Timeframes to analyze
    timeframes: List[Timeframe] = field(default_factory=lambda: [
        Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1
    ])
    
    # Alignment requirements
    min_alignment_count: int = 3      # Minimum TFs that must agree
    min_weighted_confidence: float = 0.50
    require_htf_confirmation: bool = True  # W1 or D1 must agree
    require_ltf_entry: bool = True    # H4/H1 must confirm for entry
    
    # Capital
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.10
    max_positions: int = 3
    
    # Risk management
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    max_loss_pct: float = 0.03
    
    # Costs
    slippage_bps: float = 5.0
    
    # Output
    output_dir: str = "runs/mtf_backtests"


class MTFDataManager:
    """Manages multi-timeframe data fetching and resampling."""
    
    def __init__(self):
        self.cache: Dict[str, Dict[Timeframe, pd.DataFrame]] = {}
    
    def fetch_base_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch base (finest granularity) data."""
        
        # Try Massive.com first for intraday
        try:
            from config.credentials import get_massive_api_keys
            from massive import RESTClient
            
            primary, secondary = get_massive_api_keys()
            api_key = primary or secondary
            
            if api_key:
                client = RESTClient(api_key=api_key)
                
                # Fetch hourly data as base
                aggs = list(client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="hour",
                    from_=start_date,
                    to=end_date,
                    adjusted=True,
                    limit=50000,
                ))
                
                if aggs:
                    data = []
                    for agg in aggs:
                        data.append({
                            'timestamp': datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc),
                            'open': float(agg.open),
                            'high': float(agg.high),
                            'low': float(agg.low),
                            'close': float(agg.close),
                            'volume': float(agg.volume),
                        })
                    
                    df = pd.DataFrame(data)
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    df['symbol'] = symbol
                    
                    logger.info(f"Fetched {len(df)} hourly bars for {symbol} from Massive.com")
                    return df
                    
        except Exception as e:
            logger.warning(f"Massive.com hourly fetch failed: {e}")
        
        # Fallback to daily from Alpaca
        return self._fetch_daily_from_alpaca(symbol, start_date, end_date)
    
    def _fetch_daily_from_alpaca(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily data from Alpaca."""
        from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
        
        adapter = AlpacaMarketDataAdapter()
        
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        bars = adapter.get_bars(
            symbol=symbol,
            start=start,
            end=end,
            timeframe="1Day",
        )
        
        if not bars:
            raise ValueError(f"No data for {symbol}")
        
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
        df['symbol'] = symbol
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} daily bars for {symbol} from Alpaca")
        return df
    
    def resample_to_timeframes(
        self,
        base_df: pd.DataFrame,
        timeframes: List[Timeframe],
    ) -> Dict[Timeframe, pd.DataFrame]:
        """Resample base data to multiple timeframes."""
        
        result = {}
        
        # Set timestamp as index for resampling
        df = base_df.copy()
        df.set_index('timestamp', inplace=True)
        
        for tf in timeframes:
            if tf == Timeframe.W1:
                rule = 'W'
            elif tf == Timeframe.D1:
                rule = 'D'
            elif tf == Timeframe.H4:
                rule = '4H'
            elif tf == Timeframe.H1:
                rule = '1H'
            elif tf == Timeframe.M15:
                rule = '15min'
            else:
                continue
            
            try:
                resampled = df.resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                }).dropna()
                
                resampled = resampled.reset_index()
                resampled['symbol'] = df['symbol'].iloc[0] if 'symbol' in df.columns else base_df['symbol'].iloc[0]
                
                result[tf] = resampled
                logger.debug(f"Resampled to {tf.value}: {len(resampled)} bars")
                
            except Exception as e:
                logger.warning(f"Failed to resample to {tf.value}: {e}")
        
        return result
    
    def get_mtf_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframes: List[Timeframe],
    ) -> Dict[Timeframe, pd.DataFrame]:
        """Get multi-timeframe data for a symbol."""
        
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Fetch base data
        base_df = self.fetch_base_data(symbol, start_date, end_date)
        
        # Resample to all timeframes
        mtf_data = self.resample_to_timeframes(base_df, timeframes)
        
        # Cache it
        self.cache[cache_key] = mtf_data
        
        return mtf_data


class MTFSignalGenerator:
    """Generates signals for each timeframe."""
    
    def __init__(self):
        pass
    
    def compute_signal(
        self,
        tf: Timeframe,
        bar: Dict,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> TimeframeSignal:
        """Compute signal for a single timeframe."""
        
        signal = TimeframeSignal(
            timeframe=tf,
            timestamp=timestamp,
            price=bar['close'],
        )
        
        if len(history) < 20:
            return signal
        
        # Moving averages
        signal.sma_20 = history['close'].tail(20).mean()
        signal.sma_50 = history['close'].tail(min(50, len(history))).mean()
        
        # Trend signal (price vs MAs)
        price = bar['close']
        
        if signal.sma_20 > 0 and signal.sma_50 > 0:
            ma_trend = 0.0
            
            # Price above both MAs = bullish
            if price > signal.sma_20 > signal.sma_50:
                ma_trend = 1.0
            # Price below both MAs = bearish
            elif price < signal.sma_20 < signal.sma_50:
                ma_trend = -1.0
            # Mixed
            elif price > signal.sma_20:
                ma_trend = 0.5
            elif price < signal.sma_20:
                ma_trend = -0.5
            
            signal.trend_signal = ma_trend
        
        # Momentum (Rate of Change)
        lookback = min(10, len(history) - 1)
        if lookback > 0:
            roc = (price - history['close'].iloc[-lookback]) / history['close'].iloc[-lookback]
            signal.momentum_signal = np.clip(roc * 10, -1, 1)  # Scale and clip
        
        # RSI
        if len(history) >= 14:
            returns = history['close'].diff()
            gains = returns.where(returns > 0, 0).tail(14).mean()
            losses = (-returns.where(returns < 0, 0)).tail(14).mean()
            
            if losses > 0:
                rs = gains / losses
                signal.rsi = 100 - (100 / (1 + rs))
            else:
                signal.rsi = 100 if gains > 0 else 50
        
        # Volume ratio
        avg_volume = history['volume'].mean()
        if avg_volume > 0:
            signal.volume_ratio = bar['volume'] / avg_volume
        
        # ATR
        if len(history) >= 14:
            df = history.tail(15).copy()
            df['prev_close'] = df['close'].shift(1)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['prev_close']),
                    abs(df['low'] - df['prev_close'])
                )
            )
            signal.atr = df['tr'].tail(14).mean()
        
        # Combined sentiment
        signal.sentiment_signal = (
            signal.trend_signal * 0.5 +
            signal.momentum_signal * 0.3 +
            (signal.rsi - 50) / 50 * 0.2  # Normalize RSI to -1 to 1
        )
        signal.sentiment_signal = np.clip(signal.sentiment_signal, -1, 1)
        
        # Determine direction
        combined = (signal.trend_signal + signal.momentum_signal + signal.sentiment_signal) / 3
        
        if combined > 0.2:
            signal.direction = "bullish"
        elif combined < -0.2:
            signal.direction = "bearish"
        else:
            signal.direction = "neutral"
        
        signal.strength = abs(combined)
        
        return signal
    
    def compute_mtf_signal(
        self,
        symbol: str,
        timestamp: datetime,
        tf_data: Dict[Timeframe, Tuple[Dict, pd.DataFrame]],
    ) -> MTFSignal:
        """Compute combined multi-timeframe signal."""
        
        mtf = MTFSignal(
            timestamp=timestamp,
            symbol=symbol,
        )
        
        # Compute signal for each timeframe
        bullish_count = 0
        bearish_count = 0
        weighted_sum = 0.0
        weight_total = 0.0
        
        for tf, (bar, history) in tf_data.items():
            signal = self.compute_signal(tf, bar, history, timestamp)
            mtf.signals[tf] = signal
            mtf.total_timeframes += 1
            
            # Count alignment
            if signal.direction == "bullish":
                bullish_count += 1
                weighted_sum += TIMEFRAME_WEIGHTS.get(tf, 0.1)
            elif signal.direction == "bearish":
                bearish_count += 1
                weighted_sum -= TIMEFRAME_WEIGHTS.get(tf, 0.1)
            
            weight_total += TIMEFRAME_WEIGHTS.get(tf, 0.1)
        
        # Alignment score (-1 to 1)
        if mtf.total_timeframes > 0:
            mtf.alignment_score = (bullish_count - bearish_count) / mtf.total_timeframes
        
        # Alignment count (how many agree with majority)
        if bullish_count > bearish_count:
            mtf.alignment_count = bullish_count
            mtf.direction = "bullish"
        elif bearish_count > bullish_count:
            mtf.alignment_count = bearish_count
            mtf.direction = "bearish"
        else:
            mtf.alignment_count = max(bullish_count, bearish_count)
            mtf.direction = "neutral"
        
        # Weighted confidence
        if weight_total > 0:
            mtf.weighted_confidence = abs(weighted_sum) / weight_total
        
        # Higher timeframe bias (W1 + D1)
        htf_bullish = 0
        htf_bearish = 0
        
        if Timeframe.W1 in mtf.signals:
            if mtf.signals[Timeframe.W1].direction == "bullish":
                htf_bullish += 2  # Weekly has more weight
            elif mtf.signals[Timeframe.W1].direction == "bearish":
                htf_bearish += 2
        
        if Timeframe.D1 in mtf.signals:
            if mtf.signals[Timeframe.D1].direction == "bullish":
                htf_bullish += 1
            elif mtf.signals[Timeframe.D1].direction == "bearish":
                htf_bearish += 1
        
        if htf_bullish > htf_bearish:
            mtf.htf_bias = "bullish"
        elif htf_bearish > htf_bullish:
            mtf.htf_bias = "bearish"
        else:
            mtf.htf_bias = "neutral"
        
        # Lower timeframe confirmation
        ltf_confirms = False
        ltf_tfs = [Timeframe.H4, Timeframe.H1, Timeframe.M15]
        ltf_agree = 0
        
        for tf in ltf_tfs:
            if tf in mtf.signals:
                if mtf.signals[tf].direction == mtf.htf_bias:
                    ltf_agree += 1
        
        mtf.ltf_confirmation = ltf_agree >= 2 or (ltf_agree >= 1 and mtf.htf_bias != "neutral")
        
        # Entry quality
        if mtf.alignment_count >= 4 and mtf.htf_bias == mtf.direction and mtf.ltf_confirmation:
            mtf.entry_quality = "perfect"
        elif mtf.alignment_count >= 3 and mtf.htf_bias == mtf.direction:
            mtf.entry_quality = "strong"
        elif mtf.alignment_count >= 3:
            mtf.entry_quality = "moderate"
        elif mtf.alignment_count >= 2:
            mtf.entry_quality = "weak"
        else:
            mtf.entry_quality = "none"
        
        return mtf


class MTFBacktestEngine:
    """
    Multi-Timeframe Backtesting Engine.
    
    Uses timeframe alignment for trade decisions:
    - Higher timeframes (W1, D1) determine trend bias
    - Lower timeframes (H4, H1, M15) provide entry timing
    - Confidence scales with alignment count
    """
    
    def __init__(self, config: MTFBacktestConfig):
        self.config = config
        self.data_manager = MTFDataManager()
        self.signal_gen = MTFSignalGenerator()
        
        # State
        self.capital = config.initial_capital
        self.positions: Dict[str, MTFTrade] = {}
        self.trades: List[MTFTrade] = []
        self.equity_curve: List[Dict] = []
        self.trade_counter = 0
        
        logger.info(
            f"MTFBacktestEngine initialized | "
            f"symbols={config.symbols} | "
            f"timeframes={[tf.value for tf in config.timeframes]} | "
            f"min_alignment={config.min_alignment_count}"
        )
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run the MTF backtest."""
        
        print("\n" + "="*70)
        print("  MULTI-TIMEFRAME (MTF) BACKTEST ENGINE")
        print("  Timeframe Confirmation Strategy")
        print("="*70)
        print(f"\nTimeframes: {[tf.value for tf in self.config.timeframes]}")
        print(f"Alignment Required: {self.config.min_alignment_count}+ timeframes")
        print(f"Min Confidence: {self.config.min_weighted_confidence:.0%}")
        print(f"HTF Confirmation: {self.config.require_htf_confirmation}")
        print()
        
        # Fetch and prepare data
        all_mtf_data: Dict[str, Dict[Timeframe, pd.DataFrame]] = {}
        
        for symbol in self.config.symbols:
            try:
                mtf_data = self.data_manager.get_mtf_data(
                    symbol,
                    self.config.start_date,
                    self.config.end_date,
                    self.config.timeframes,
                )
                if mtf_data:
                    all_mtf_data[symbol] = mtf_data
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
        
        if not all_mtf_data:
            raise ValueError("No data fetched for any symbol")
        
        # Use daily timeframe as primary iteration
        primary_tf = Timeframe.D1 if Timeframe.D1 in self.config.timeframes else self.config.timeframes[0]
        
        # Get all dates from primary timeframe
        all_dates = set()
        for symbol, mtf_data in all_mtf_data.items():
            if primary_tf in mtf_data:
                all_dates.update(mtf_data[primary_tf]['timestamp'].tolist())
        all_dates = sorted(all_dates)
        
        print(f"Data: {len(all_dates)} periods for {list(all_mtf_data.keys())}")
        print(f"Period: {all_dates[0]} to {all_dates[-1]}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print()
        
        # Reset state
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Warmup
        warmup = 50
        
        # Process each period
        for i in range(warmup, len(all_dates)):
            timestamp = all_dates[i]
            
            # Build MTF signals for each symbol
            for symbol, mtf_data in all_mtf_data.items():
                # Get current bar and history for each timeframe
                tf_current_data = {}
                
                for tf, df in mtf_data.items():
                    mask = df['timestamp'] <= timestamp
                    if mask.sum() > 0:
                        current_df = df[mask]
                        bar = current_df.iloc[-1].to_dict()
                        history = current_df
                        tf_current_data[tf] = (bar, history)
                
                if not tf_current_data:
                    continue
                
                # Compute MTF signal
                mtf_signal = self.signal_gen.compute_mtf_signal(
                    symbol, timestamp, tf_current_data
                )
                
                # Get current price from primary TF
                if primary_tf in tf_current_data:
                    current_price = tf_current_data[primary_tf][0]['close']
                    atr = self.signal_gen.compute_signal(
                        primary_tf, 
                        tf_current_data[primary_tf][0],
                        tf_current_data[primary_tf][1],
                        timestamp
                    ).atr
                else:
                    continue
                
                # Check exits
                if symbol in self.positions:
                    should_exit, reason = self._check_exit(
                        self.positions[symbol],
                        current_price,
                        atr,
                        mtf_signal,
                    )
                    
                    if should_exit:
                        self._close_position(symbol, current_price, timestamp, reason)
                
                # Check entries
                if symbol not in self.positions:
                    if self._should_enter(mtf_signal):
                        self._open_position(
                            symbol,
                            mtf_signal,
                            current_price,
                            timestamp,
                            atr,
                        )
            
            # Record equity
            self._record_equity(timestamp, all_mtf_data, primary_tf)
        
        # Close remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in all_mtf_data and primary_tf in all_mtf_data[symbol]:
                df = all_mtf_data[symbol][primary_tf]
                final_price = df.iloc[-1]['close']
                self._close_position(symbol, final_price, all_dates[-1], "end_of_test")
        
        # Calculate and display results
        results = self._calculate_results()
        self._print_results(results)
        self._save_results(results)
        
        return results
    
    def _should_enter(self, mtf_signal: MTFSignal) -> bool:
        """Check if MTF signal warrants entry."""
        
        # Direction must be clear
        if mtf_signal.direction == "neutral":
            return False
        
        # Minimum alignment count
        if mtf_signal.alignment_count < self.config.min_alignment_count:
            return False
        
        # Minimum weighted confidence
        if mtf_signal.weighted_confidence < self.config.min_weighted_confidence:
            return False
        
        # HTF confirmation required?
        if self.config.require_htf_confirmation:
            if mtf_signal.htf_bias != mtf_signal.direction:
                return False
        
        # LTF entry confirmation?
        if self.config.require_ltf_entry:
            if not mtf_signal.ltf_confirmation:
                return False
        
        # Entry quality check
        if mtf_signal.entry_quality in ["none", "weak"]:
            return False
        
        # Max positions check
        if len(self.positions) >= self.config.max_positions:
            return False
        
        return True
    
    def _check_exit(
        self,
        trade: MTFTrade,
        current_price: float,
        atr: float,
        mtf_signal: MTFSignal,
    ) -> Tuple[bool, str]:
        """Check if position should be exited."""
        
        # P&L calculation
        if trade.direction == "long":
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
        
        # ATR-based stops
        stop_distance = atr * self.config.atr_stop_mult if atr > 0 else trade.entry_price * 0.02
        target_distance = atr * self.config.atr_target_mult if atr > 0 else trade.entry_price * 0.03
        
        # Stop loss
        if trade.direction == "long":
            if current_price <= trade.entry_price - stop_distance:
                return True, "stop_loss"
        else:
            if current_price >= trade.entry_price + stop_distance:
                return True, "stop_loss"
        
        # Take profit
        if trade.direction == "long":
            if current_price >= trade.entry_price + target_distance:
                return True, "take_profit"
        else:
            if current_price <= trade.entry_price - target_distance:
                return True, "take_profit"
        
        # Max loss
        if pnl_pct < -self.config.max_loss_pct:
            return True, "max_loss"
        
        # MTF reversal (HTF bias changes)
        if mtf_signal.htf_bias != "neutral":
            if trade.direction == "long" and mtf_signal.htf_bias == "bearish":
                if mtf_signal.alignment_count >= 3:
                    return True, "mtf_reversal"
            elif trade.direction == "short" and mtf_signal.htf_bias == "bullish":
                if mtf_signal.alignment_count >= 3:
                    return True, "mtf_reversal"
        
        return False, ""
    
    def _open_position(
        self,
        symbol: str,
        mtf_signal: MTFSignal,
        current_price: float,
        timestamp: datetime,
        atr: float,
    ):
        """Open a new position."""
        
        if symbol in self.positions:
            return
        
        # Position sizing
        position_value = self.capital * self.config.max_position_pct
        
        # Slippage
        slippage = self.config.slippage_bps / 10000
        if mtf_signal.direction == "bullish":
            entry_price = current_price * (1 + slippage)
            direction = "long"
        else:
            entry_price = current_price * (1 - slippage)
            direction = "short"
        
        position_size = position_value / entry_price
        
        # Create trade
        self.trade_counter += 1
        trade = MTFTrade(
            trade_id=f"MTF{self.trade_counter:05d}",
            symbol=symbol,
            direction=direction,
            entry_date=timestamp,
            entry_price=entry_price,
            position_size=position_size,
            alignment_score=mtf_signal.alignment_score,
            alignment_count=mtf_signal.alignment_count,
            weighted_confidence=mtf_signal.weighted_confidence,
            htf_bias=mtf_signal.htf_bias,
            ltf_confirmation=mtf_signal.ltf_confirmation,
            entry_quality=mtf_signal.entry_quality,
        )
        
        # Store individual TF signals
        if Timeframe.W1 in mtf_signal.signals:
            trade.w1_signal = mtf_signal.signals[Timeframe.W1].sentiment_signal
        if Timeframe.D1 in mtf_signal.signals:
            trade.d1_signal = mtf_signal.signals[Timeframe.D1].sentiment_signal
        if Timeframe.H4 in mtf_signal.signals:
            trade.h4_signal = mtf_signal.signals[Timeframe.H4].sentiment_signal
        if Timeframe.H1 in mtf_signal.signals:
            trade.h1_signal = mtf_signal.signals[Timeframe.H1].sentiment_signal
        if Timeframe.M15 in mtf_signal.signals:
            trade.m15_signal = mtf_signal.signals[Timeframe.M15].sentiment_signal
        
        # Deduct capital
        self.capital -= position_value
        self.positions[symbol] = trade
        
        logger.debug(
            f"OPEN {direction.upper()} {symbol} @ ${entry_price:.2f} | "
            f"align={mtf_signal.alignment_count}/{mtf_signal.total_timeframes} | "
            f"conf={mtf_signal.weighted_confidence:.2f} | "
            f"quality={mtf_signal.entry_quality}"
        )
    
    def _close_position(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
        exit_reason: str,
    ):
        """Close an existing position."""
        
        if symbol not in self.positions:
            return
        
        trade = self.positions.pop(symbol)
        
        # Slippage
        slippage = self.config.slippage_bps / 10000
        if trade.direction == "long":
            exit_price = current_price * (1 - slippage)
            gross_pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            exit_price = current_price * (1 + slippage)
            gross_pnl = (trade.entry_price - exit_price) * trade.position_size
        
        # Costs
        position_value = trade.entry_price * trade.position_size
        total_slippage = position_value * slippage * 2
        net_pnl = gross_pnl - total_slippage
        pnl_pct = net_pnl / position_value if position_value > 0 else 0
        
        # Hold time
        hold_days = (timestamp - trade.entry_date).days if trade.entry_date else 0
        
        # Update trade
        trade.exit_date = timestamp
        trade.exit_price = exit_price
        trade.hold_days = hold_days
        trade.gross_pnl = gross_pnl
        trade.net_pnl = net_pnl
        trade.pnl_pct = pnl_pct
        trade.exit_reason = exit_reason
        
        # Return capital
        self.capital += position_value + net_pnl
        self.trades.append(trade)
        
        logger.debug(
            f"CLOSE {trade.direction.upper()} {symbol} @ ${exit_price:.2f} | "
            f"P&L=${net_pnl:.2f} ({pnl_pct:.1%}) | {hold_days}d | {exit_reason}"
        )
    
    def _record_equity(
        self,
        timestamp: datetime,
        all_mtf_data: Dict,
        primary_tf: Timeframe,
    ):
        """Record current equity."""
        
        position_value = 0
        for symbol, trade in self.positions.items():
            if symbol in all_mtf_data and primary_tf in all_mtf_data[symbol]:
                df = all_mtf_data[symbol][primary_tf]
                mask = df['timestamp'] <= timestamp
                if mask.sum() > 0:
                    current_price = df[mask].iloc[-1]['close']
                    if trade.direction == "long":
                        unrealized = (current_price - trade.entry_price) * trade.position_size
                    else:
                        unrealized = (trade.entry_price - current_price) * trade.position_size
                    position_value += trade.entry_price * trade.position_size + unrealized
        
        total_equity = self.capital + position_value
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'positions': len(self.positions),
        })
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive results."""
        
        results = {
            'config': {
                'symbols': self.config.symbols,
                'timeframes': [tf.value for tf in self.config.timeframes],
                'min_alignment': self.config.min_alignment_count,
                'min_confidence': self.config.min_weighted_confidence,
            },
            'capital': {
                'initial': self.config.initial_capital,
                'final': self.capital,
                'return': self.capital - self.config.initial_capital,
                'return_pct': (self.capital - self.config.initial_capital) / self.config.initial_capital,
            },
        }
        
        if self.trades:
            winners = [t for t in self.trades if t.net_pnl > 0]
            losers = [t for t in self.trades if t.net_pnl <= 0]
            
            results['trades'] = {
                'total': len(self.trades),
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': len(winners) / len(self.trades),
                'avg_win': np.mean([t.net_pnl for t in winners]) if winners else 0,
                'avg_loss': np.mean([t.net_pnl for t in losers]) if losers else 0,
                'largest_win': max([t.net_pnl for t in winners]) if winners else 0,
                'largest_loss': min([t.net_pnl for t in losers]) if losers else 0,
                'avg_hold_days': np.mean([t.hold_days for t in self.trades]),
            }
            
            # Profit factor
            gross_profit = sum(t.net_pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.net_pnl for t in losers)) if losers else 1
            results['trades']['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # MTF analysis
            results['mtf_analysis'] = {
                'avg_alignment_count': np.mean([t.alignment_count for t in self.trades]),
                'avg_confidence': np.mean([t.weighted_confidence for t in self.trades]),
                'winning_alignment': np.mean([t.alignment_count for t in winners]) if winners else 0,
                'losing_alignment': np.mean([t.alignment_count for t in losers]) if losers else 0,
                'winning_confidence': np.mean([t.weighted_confidence for t in winners]) if winners else 0,
                'losing_confidence': np.mean([t.weighted_confidence for t in losers]) if losers else 0,
            }
            
            # Entry quality breakdown
            quality_pnl = {}
            for q in ['perfect', 'strong', 'moderate', 'weak']:
                q_trades = [t for t in self.trades if t.entry_quality == q]
                if q_trades:
                    quality_pnl[q] = {
                        'count': len(q_trades),
                        'win_rate': len([t for t in q_trades if t.net_pnl > 0]) / len(q_trades),
                        'total_pnl': sum(t.net_pnl for t in q_trades),
                        'avg_pnl': np.mean([t.net_pnl for t in q_trades]),
                    }
            results['entry_quality'] = quality_pnl
            
            # Exit reasons
            exit_reasons = {}
            for t in self.trades:
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
            results['exit_reasons'] = exit_reasons
            
            # Symbol breakdown
            symbol_stats = {}
            for symbol in self.config.symbols:
                sym_trades = [t for t in self.trades if t.symbol == symbol]
                if sym_trades:
                    sym_winners = [t for t in sym_trades if t.net_pnl > 0]
                    symbol_stats[symbol] = {
                        'trades': len(sym_trades),
                        'win_rate': len(sym_winners) / len(sym_trades),
                        'total_pnl': sum(t.net_pnl for t in sym_trades),
                    }
            results['symbol_stats'] = symbol_stats
        
        # Risk metrics
        if self.equity_curve:
            equities = [e['equity'] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
            
            results['risk'] = {'max_drawdown_pct': max_dd}
            
            if len(equities) > 1:
                daily_returns = pd.Series(equities).pct_change().dropna()
                if len(daily_returns) > 0 and daily_returns.std() > 0:
                    results['risk']['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                    
                    downside = daily_returns[daily_returns < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        results['risk']['sortino_ratio'] = daily_returns.mean() / downside.std() * np.sqrt(252)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        
        print("\n" + "="*70)
        print("  MTF BACKTEST RESULTS")
        print("="*70)
        
        cap = results['capital']
        print(f"\n{'CAPITAL':=^50}")
        print(f"  Initial:        ${cap['initial']:>12,.2f}")
        print(f"  Final:          ${cap['final']:>12,.2f}")
        print(f"  Return:         ${cap['return']:>12,.2f} ({cap['return_pct']:.2%})")
        
        if 'trades' in results:
            t = results['trades']
            print(f"\n{'TRADES':=^50}")
            print(f"  Total:          {t['total']:>12}")
            print(f"  Winners:        {t['winners']:>12} ({t['win_rate']:.1%})")
            print(f"  Losers:         {t['losers']:>12}")
            print(f"  Profit Factor:  {t['profit_factor']:>12.2f}")
            print(f"  Avg Win:        ${t['avg_win']:>12,.2f}")
            print(f"  Avg Loss:       ${t['avg_loss']:>12,.2f}")
            print(f"  Avg Hold:       {t['avg_hold_days']:>12.1f} days")
        
        if 'mtf_analysis' in results:
            m = results['mtf_analysis']
            print(f"\n{'MTF ANALYSIS':=^50}")
            print(f"  Avg Alignment:  {m['avg_alignment_count']:>12.1f} TFs")
            print(f"  Avg Confidence: {m['avg_confidence']:>12.2f}")
            print(f"  Win Alignment:  {m['winning_alignment']:>12.1f} TFs")
            print(f"  Loss Alignment: {m['losing_alignment']:>12.1f} TFs")
            print(f"  Win Confidence: {m['winning_confidence']:>12.2f}")
            print(f"  Loss Confidence:{m['losing_confidence']:>12.2f}")
        
        if 'entry_quality' in results:
            print(f"\n{'ENTRY QUALITY BREAKDOWN':=^50}")
            for quality, stats in results['entry_quality'].items():
                print(f"  {quality:10s}: {stats['count']:3d} trades, "
                      f"{stats['win_rate']:.1%} win, ${stats['total_pnl']:>8,.2f} P&L")
        
        if 'risk' in results:
            r = results['risk']
            print(f"\n{'RISK METRICS':=^50}")
            print(f"  Max Drawdown:   {r['max_drawdown_pct']:>12.2%}")
            if 'sharpe_ratio' in r:
                print(f"  Sharpe Ratio:   {r['sharpe_ratio']:>12.2f}")
            if 'sortino_ratio' in r:
                print(f"  Sortino Ratio:  {r['sortino_ratio']:>12.2f}")
        
        if 'symbol_stats' in results:
            print(f"\n{'SYMBOL BREAKDOWN':=^50}")
            for sym, stats in sorted(results['symbol_stats'].items(), key=lambda x: -x[1]['total_pnl']):
                print(f"  {sym:6s}: {stats['trades']:3d} trades, "
                      f"{stats['win_rate']:.1%} win, ${stats['total_pnl']:>10,.2f}")
        
        if 'exit_reasons' in results:
            print(f"\n{'EXIT REASONS':=^50}")
            for reason, count in sorted(results['exit_reasons'].items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")
        
        print("\n" + "="*70)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mtf_backtest_{timestamp}.json"
        
        # Convert trades to serializable format
        trades_data = []
        for t in self.trades:
            trades_data.append({
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_date': t.entry_date.isoformat() if t.entry_date else None,
                'exit_date': t.exit_date.isoformat() if t.exit_date else None,
                'hold_days': t.hold_days,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'net_pnl': t.net_pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason,
                'alignment_count': t.alignment_count,
                'weighted_confidence': t.weighted_confidence,
                'htf_bias': t.htf_bias,
                'ltf_confirmation': t.ltf_confirmation,
                'entry_quality': t.entry_quality,
                'w1_signal': t.w1_signal,
                'd1_signal': t.d1_signal,
                'h4_signal': t.h4_signal,
                'h1_signal': t.h1_signal,
            })
        
        output = {
            'summary': results,
            'trades': trades_data,
        }
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")


def main():
    """Run MTF backtest."""
    
    config = MTFBacktestConfig(
        symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL"],
        start_date="2020-01-01",
        end_date="2024-12-01",
        timeframes=[Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1],
        min_alignment_count=3,
        min_weighted_confidence=0.50,
        require_htf_confirmation=True,
        require_ltf_entry=True,
        initial_capital=100_000.0,
        max_position_pct=0.08,
        max_positions=5,
    )
    
    engine = MTFBacktestEngine(config)
    results = engine.run_backtest()
    
    return results


if __name__ == "__main__":
    main()
