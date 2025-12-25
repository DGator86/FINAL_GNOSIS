#!/usr/bin/env python3
"""
Liquidity & Sentiment Pipeline Backtest

Runs a focused backtest using ONLY liquidity and sentiment signals derived
from real market data (Massive.com/Alpaca). No synthetic fallbacks.

Features:
- Liquidity scoring from volume/spread analysis
- Sentiment derived from price momentum and flow
- Real OHLCV data from Massive.com
- No options data (avoids synthetic fallback)

Author: GNOSIS Trading System
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.core_schemas import (
    DirectionEnum,
    LiquiditySnapshot,
    SentimentSnapshot,
    TradeIdea,
    StrategyType,
)


@dataclass
class LiquiditySentimentConfig:
    """Configuration for liquidity-sentiment backtest."""
    
    # Data
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-01"
    
    # Capital
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.10  # 10% per position
    max_positions: int = 3
    
    # Costs
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    
    # Signals - Liquidity weight vs Sentiment weight
    liquidity_weight: float = 0.5
    sentiment_weight: float = 0.5
    
    # Signal thresholds
    min_liquidity_score: float = 0.3  # Minimum liquidity to trade
    min_sentiment_strength: float = 0.15  # Minimum sentiment magnitude
    min_combined_confidence: float = 0.40
    
    # Risk management
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    max_loss_pct: float = 0.03
    
    # Output
    output_dir: str = "runs/liquidity_sentiment_backtests"


@dataclass
class BacktestTrade:
    """Record of a single trade."""
    trade_id: str = ""
    symbol: str = ""
    direction: str = "long"
    entry_date: datetime = None
    exit_date: datetime = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    
    # Signal values at entry
    liquidity_score: float = 0.0
    sentiment_score: float = 0.0
    combined_score: float = 0.0
    confidence: float = 0.0


class MassiveDataFetcher:
    """Fetch real market data from Massive.com."""
    
    def __init__(self, api_key: Optional[str] = None):
        from config.credentials import get_massive_api_keys
        primary, secondary = get_massive_api_keys(primary=api_key)
        self.api_key = primary or secondary
        self.client = None
        
        if self.api_key:
            try:
                from massive import RESTClient
                self.client = RESTClient(api_key=self.api_key)
                logger.info("MassiveDataFetcher initialized with API key")
            except ImportError:
                logger.warning("Massive client not installed, falling back to Alpaca")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "day",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Massive.com or Alpaca."""
        
        # Try Massive.com first
        if self.client:
            try:
                aggs = list(self.client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan=timeframe,
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
                            'vwap': float(getattr(agg, 'vwap', 0)),
                            'transactions': int(getattr(agg, 'transactions', 0)),
                        })
                    
                    df = pd.DataFrame(data)
                    df['symbol'] = symbol
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    logger.info(f"Fetched {len(df)} bars for {symbol} from Massive.com")
                    return df
                    
            except Exception as e:
                logger.warning(f"Massive.com fetch failed for {symbol}: {e}")
        
        # Fallback to Alpaca
        return self._fetch_from_alpaca(symbol, start_date, end_date)
    
    def _fetch_from_alpaca(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fallback to Alpaca for data."""
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
        
        logger.info(f"Fetched {len(df)} bars for {symbol} from Alpaca")
        return df


class LiquiditySentimentEngine:
    """
    Backtest engine focused on Liquidity and Sentiment signals only.
    
    Signal Sources:
    1. Liquidity: Volume patterns, spread estimates, market depth proxies
    2. Sentiment: Price momentum, trend analysis, flow indicators
    
    No options data or synthetic fallbacks - real market data only.
    """
    
    def __init__(self, config: LiquiditySentimentConfig):
        self.config = config
        self.fetcher = MassiveDataFetcher()
        
        # State
        self.capital = config.initial_capital
        self.positions: Dict[str, BacktestTrade] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        self.trade_counter = 0
        
        logger.info(
            f"LiquiditySentimentEngine initialized | "
            f"symbols={config.symbols} | "
            f"liquidity_weight={config.liquidity_weight} | "
            f"sentiment_weight={config.sentiment_weight}"
        )
    
    def compute_liquidity_signal(
        self,
        bar: Dict,
        history: pd.DataFrame,
        symbol: str,
    ) -> Tuple[float, LiquiditySnapshot]:
        """
        Compute liquidity signal from real market data.
        
        Metrics:
        - Volume relative to average
        - Estimated spread from high-low range
        - Depth proxy from volume/range ratio
        - Impact cost estimate
        """
        if len(history) < 20:
            return 0.0, LiquiditySnapshot(timestamp=bar['timestamp'], symbol=symbol)
        
        # Volume analysis
        avg_volume = history['volume'].mean()
        recent_volume = history['volume'].tail(5).mean()
        current_volume = bar['volume']
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        volume_trend = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Spread estimate from high-low range
        avg_range = ((history['high'] - history['low']) / history['close']).mean()
        current_range = (bar['high'] - bar['low']) / bar['close']
        spread_estimate = avg_range * 0.1  # ~10% of range is spread
        
        # Depth proxy (higher volume with tighter range = more depth)
        range_ratio = current_range / avg_range if avg_range > 0 else 1.0
        depth_proxy = volume_ratio / (1 + range_ratio) if range_ratio > 0 else volume_ratio
        
        # Impact cost (bps)
        impact_cost = spread_estimate * 50 * (1 / max(0.5, volume_ratio))
        
        # Liquidity score (0-1)
        volume_score = min(1.0, np.log1p(avg_volume) / 20)  # Log scale for volume
        spread_score = max(0.0, 1.0 - spread_estimate * 50)  # Lower spread = higher score
        depth_score = min(1.0, depth_proxy / 2)
        
        liquidity_score = (
            volume_score * 0.4 +
            spread_score * 0.3 +
            depth_score * 0.3
        )
        
        # Liquidity signal: High liquidity with increasing volume is bullish
        # Low liquidity with decreasing volume is bearish (potential breakdown)
        if volume_ratio > 1.2 and liquidity_score > 0.6:
            signal = 0.3 * (volume_ratio - 1)  # Positive signal
        elif volume_ratio < 0.7 and liquidity_score < 0.4:
            signal = -0.3 * (1 - volume_ratio)  # Negative signal
        else:
            signal = 0.0  # Neutral
        
        signal = np.clip(signal, -1, 1)
        
        snapshot = LiquiditySnapshot(
            timestamp=bar['timestamp'],
            symbol=symbol,
            liquidity_score=liquidity_score,
            bid_ask_spread=spread_estimate * 100,
            volume=current_volume,
            depth=depth_proxy * avg_volume,
            impact_cost=impact_cost,
        )
        
        return signal, snapshot
    
    def compute_sentiment_signal(
        self,
        bar: Dict,
        history: pd.DataFrame,
        symbol: str,
    ) -> Tuple[float, SentimentSnapshot]:
        """
        Compute sentiment signal from price action.
        
        Metrics:
        - Price momentum (SMAs)
        - Trend direction and strength
        - Return distribution (skewness)
        - Volume-weighted direction
        """
        if len(history) < 20:
            return 0.0, SentimentSnapshot(timestamp=bar['timestamp'], symbol=symbol)
        
        returns = history['close'].pct_change().dropna()
        current_price = bar['close']
        
        # Moving averages
        sma_5 = history['close'].tail(5).mean()
        sma_10 = history['close'].tail(10).mean()
        sma_20 = history['close'].tail(20).mean()
        sma_50 = history['close'].tail(min(50, len(history))).mean()
        
        # Trend signal (price vs moving averages)
        trend_short = (current_price - sma_5) / sma_5 if sma_5 > 0 else 0
        trend_medium = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        trend_long = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
        
        # Momentum (rate of change)
        roc_5 = (current_price - history['close'].iloc[-5]) / history['close'].iloc[-5] if len(history) >= 5 else 0
        roc_10 = (current_price - history['close'].iloc[-10]) / history['close'].iloc[-10] if len(history) >= 10 else 0
        
        # Volume-weighted direction
        recent = history.tail(10)
        up_volume = recent[recent['close'] > recent['open']]['volume'].sum()
        down_volume = recent[recent['close'] < recent['open']]['volume'].sum()
        total_vol = up_volume + down_volume
        
        if total_vol > 0:
            flow_sentiment = (up_volume - down_volume) / total_vol
        else:
            flow_sentiment = 0.0
        
        # Technical sentiment (composite)
        technical_sentiment = (
            np.tanh(trend_short * 10) * 0.3 +
            np.tanh(trend_medium * 5) * 0.3 +
            np.tanh(roc_5 * 20) * 0.2 +
            flow_sentiment * 0.2
        )
        
        # Momentum confirmation
        momentum_aligned = (
            np.sign(trend_short) == np.sign(trend_medium) == np.sign(roc_5)
        )
        
        # Final sentiment signal
        if momentum_aligned:
            sentiment_signal = technical_sentiment * 1.2
        else:
            sentiment_signal = technical_sentiment * 0.7
        
        sentiment_signal = np.clip(sentiment_signal, -1, 1)
        
        # Confidence based on signal consistency
        signals = [trend_short, trend_medium, roc_5, flow_sentiment]
        signal_std = np.std(signals)
        confidence = max(0.2, 1.0 - signal_std * 2)
        
        snapshot = SentimentSnapshot(
            timestamp=bar['timestamp'],
            symbol=symbol,
            sentiment_score=sentiment_signal,
            technical_sentiment=technical_sentiment,
            flow_sentiment=flow_sentiment,
            confidence=confidence,
        )
        
        return sentiment_signal, snapshot
    
    def compute_combined_signal(
        self,
        liquidity_signal: float,
        sentiment_signal: float,
        liquidity_snapshot: LiquiditySnapshot,
        sentiment_snapshot: SentimentSnapshot,
    ) -> Tuple[str, float, float]:
        """
        Combine liquidity and sentiment into trading decision.
        
        Returns: (direction, combined_score, confidence)
        """
        # Weighted combination
        combined = (
            liquidity_signal * self.config.liquidity_weight +
            sentiment_signal * self.config.sentiment_weight
        )
        
        # Confidence from both sources
        liq_conf = liquidity_snapshot.liquidity_score if liquidity_snapshot else 0.5
        sent_conf = sentiment_snapshot.confidence if sentiment_snapshot else 0.5
        
        confidence = liq_conf * 0.4 + sent_conf * 0.6
        
        # Only trade if both signals align or one is very strong
        signals_aligned = np.sign(liquidity_signal) == np.sign(sentiment_signal)
        strong_signal = abs(combined) > 0.3
        
        if signals_aligned or strong_signal:
            if combined > self.config.min_sentiment_strength:
                direction = "long"
            elif combined < -self.config.min_sentiment_strength:
                direction = "short"
            else:
                direction = "neutral"
        else:
            direction = "neutral"
        
        # Boost confidence if signals align
        if signals_aligned and direction != "neutral":
            confidence = min(1.0, confidence * 1.2)
        
        return direction, combined, confidence
    
    def compute_atr(self, history: pd.DataFrame) -> float:
        """Compute Average True Range."""
        if len(history) < self.config.atr_period:
            return history['close'].iloc[-1] * 0.02
        
        df = history.tail(self.config.atr_period + 1).copy()
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['prev_close']),
                abs(df['low'] - df['prev_close'])
            )
        )
        return df['tr'].tail(self.config.atr_period).mean()
    
    def open_position(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        timestamp: datetime,
        liquidity_score: float,
        sentiment_score: float,
        combined_score: float,
        confidence: float,
        atr: float,
    ) -> Optional[BacktestTrade]:
        """Open a new position."""
        
        if len(self.positions) >= self.config.max_positions:
            return None
        
        if symbol in self.positions:
            return None
        
        # Position sizing
        position_value = self.capital * self.config.max_position_pct
        position_value = min(position_value, self.capital * 0.25)
        
        # Entry with slippage
        slippage = self.config.slippage_bps / 10000
        if direction == "long":
            entry_price = current_price * (1 + slippage)
        else:
            entry_price = current_price * (1 - slippage)
        
        position_size = position_value / entry_price
        
        # Create trade
        self.trade_counter += 1
        trade = BacktestTrade(
            trade_id=f"LS{self.trade_counter:05d}",
            symbol=symbol,
            direction=direction,
            entry_date=timestamp,
            entry_price=entry_price,
            position_size=position_size,
            liquidity_score=liquidity_score,
            sentiment_score=sentiment_score,
            combined_score=combined_score,
            confidence=confidence,
        )
        
        # Deduct capital
        self.capital -= position_value
        self.positions[symbol] = trade
        
        logger.debug(
            f"OPEN {direction.upper()} {symbol} @ ${entry_price:.2f} | "
            f"liq={liquidity_score:.2f} sent={sentiment_score:.2f} conf={confidence:.2f}"
        )
        
        return trade
    
    def close_position(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
        exit_reason: str,
    ) -> Optional[BacktestTrade]:
        """Close an existing position."""
        
        if symbol not in self.positions:
            return None
        
        trade = self.positions.pop(symbol)
        
        # Exit with slippage
        slippage = self.config.slippage_bps / 10000
        if trade.direction == "long":
            exit_price = current_price * (1 - slippage)
            gross_pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            exit_price = current_price * (1 + slippage)
            gross_pnl = (trade.entry_price - exit_price) * trade.position_size
        
        # Costs
        position_value = trade.entry_price * trade.position_size
        total_slippage = position_value * slippage * 2  # Entry + exit
        net_pnl = gross_pnl - total_slippage
        pnl_pct = net_pnl / position_value if position_value > 0 else 0
        
        # Update trade
        trade.exit_date = timestamp
        trade.exit_price = exit_price
        trade.gross_pnl = gross_pnl
        trade.net_pnl = net_pnl
        trade.pnl_pct = pnl_pct
        trade.exit_reason = exit_reason
        
        # Return capital
        self.capital += position_value + net_pnl
        self.trades.append(trade)
        
        logger.debug(
            f"CLOSE {trade.direction.upper()} {symbol} @ ${exit_price:.2f} | "
            f"P&L=${net_pnl:.2f} ({pnl_pct:.1%}) | {exit_reason}"
        )
        
        return trade
    
    def check_exit(
        self,
        trade: BacktestTrade,
        current_price: float,
        atr: float,
        sentiment_signal: float,
    ) -> Tuple[bool, str]:
        """Check if position should be exited."""
        
        # P&L calculation
        if trade.direction == "long":
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
        
        # Stop loss (ATR-based)
        stop_distance = atr * self.config.atr_stop_mult
        if trade.direction == "long":
            if current_price <= trade.entry_price - stop_distance:
                return True, "stop_loss"
        else:
            if current_price >= trade.entry_price + stop_distance:
                return True, "stop_loss"
        
        # Take profit (ATR-based)
        target_distance = atr * self.config.atr_target_mult
        if trade.direction == "long":
            if current_price >= trade.entry_price + target_distance:
                return True, "take_profit"
        else:
            if current_price <= trade.entry_price - target_distance:
                return True, "take_profit"
        
        # Max loss
        if pnl_pct < -self.config.max_loss_pct:
            return True, "max_loss"
        
        # Signal reversal
        if trade.direction == "long" and sentiment_signal < -0.3:
            return True, "signal_reversal"
        elif trade.direction == "short" and sentiment_signal > 0.3:
            return True, "signal_reversal"
        
        return False, ""
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run the liquidity-sentiment backtest."""
        
        print("\n" + "="*70)
        print("  LIQUIDITY & SENTIMENT PIPELINE BACKTEST")
        print("  Real Market Data Only - No Synthetic Fallback")
        print("="*70)
        
        # Fetch data
        symbol_data: Dict[str, pd.DataFrame] = {}
        for symbol in self.config.symbols:
            try:
                df = self.fetcher.fetch_ohlcv(
                    symbol,
                    self.config.start_date,
                    self.config.end_date,
                )
                if len(df) > 50:
                    symbol_data[symbol] = df
                else:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        if not symbol_data:
            raise ValueError("No data fetched for any symbol")
        
        # Get common dates
        all_dates = set()
        for df in symbol_data.values():
            all_dates.update(df['timestamp'].tolist())
        all_dates = sorted(all_dates)
        
        print(f"\nData: {len(all_dates)} bars for {list(symbol_data.keys())}")
        print(f"Period: {all_dates[0]} to {all_dates[-1]}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"Signal Weights: Liquidity={self.config.liquidity_weight}, Sentiment={self.config.sentiment_weight}")
        print()
        
        # Reset state
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Warmup
        warmup = 50
        
        # Process bars
        for i in range(warmup, len(all_dates)):
            timestamp = all_dates[i]
            
            # Get current data
            bar_data = {}
            for symbol, df in symbol_data.items():
                mask = df['timestamp'] <= timestamp
                if mask.sum() > 0:
                    current_df = df[mask]
                    bar_data[symbol] = {
                        'bar': current_df.iloc[-1].to_dict(),
                        'history': current_df,
                    }
            
            # Check exits
            for symbol in list(self.positions.keys()):
                if symbol in bar_data:
                    trade = self.positions[symbol]
                    bar = bar_data[symbol]['bar']
                    history = bar_data[symbol]['history']
                    current_price = bar['close']
                    
                    atr = self.compute_atr(history)
                    _, sentiment_snap = self.compute_sentiment_signal(bar, history, symbol)
                    
                    should_exit, reason = self.check_exit(
                        trade, current_price, atr, sentiment_snap.sentiment_score
                    )
                    
                    if should_exit:
                        self.close_position(symbol, current_price, timestamp, reason)
            
            # Check entries
            for symbol in symbol_data.keys():
                if symbol in bar_data and symbol not in self.positions:
                    bar = bar_data[symbol]['bar']
                    history = bar_data[symbol]['history']
                    current_price = bar['close']
                    
                    # Compute signals
                    liq_signal, liq_snap = self.compute_liquidity_signal(bar, history, symbol)
                    sent_signal, sent_snap = self.compute_sentiment_signal(bar, history, symbol)
                    
                    # Check liquidity minimum
                    if liq_snap.liquidity_score < self.config.min_liquidity_score:
                        continue
                    
                    # Combine signals
                    direction, combined, confidence = self.compute_combined_signal(
                        liq_signal, sent_signal, liq_snap, sent_snap
                    )
                    
                    # Entry decision
                    if direction != "neutral" and confidence >= self.config.min_combined_confidence:
                        atr = self.compute_atr(history)
                        self.open_position(
                            symbol=symbol,
                            direction=direction,
                            current_price=current_price,
                            timestamp=timestamp,
                            liquidity_score=liq_snap.liquidity_score,
                            sentiment_score=sent_snap.sentiment_score,
                            combined_score=combined,
                            confidence=confidence,
                            atr=atr,
                        )
            
            # Record equity
            position_value = sum(
                (bar_data[s]['bar']['close'] - t.entry_price) * t.position_size
                if t.direction == "long" else
                (t.entry_price - bar_data[s]['bar']['close']) * t.position_size
                for s, t in self.positions.items() if s in bar_data
            )
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.capital + position_value + sum(
                    t.entry_price * t.position_size for t in self.positions.values()
                ),
                'positions': len(self.positions),
            })
        
        # Close remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in bar_data:
                self.close_position(
                    symbol,
                    bar_data[symbol]['bar']['close'],
                    all_dates[-1],
                    "end_of_test"
                )
        
        # Calculate results
        results = self.calculate_results()
        self.print_results(results)
        self.save_results(results)
        
        return results
    
    def calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results."""
        
        results = {
            'config': {
                'symbols': self.config.symbols,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'liquidity_weight': self.config.liquidity_weight,
                'sentiment_weight': self.config.sentiment_weight,
            },
            'capital': {
                'initial': self.config.initial_capital,
                'final': self.capital,
                'return': self.capital - self.config.initial_capital,
                'return_pct': (self.capital - self.config.initial_capital) / self.config.initial_capital,
            },
        }
        
        # Trade statistics
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
                'avg_pnl': np.mean([t.net_pnl for t in self.trades]),
                'total_pnl': sum(t.net_pnl for t in self.trades),
            }
            
            # Profit factor
            gross_profit = sum(t.net_pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.net_pnl for t in losers)) if losers else 1
            results['trades']['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Signal analysis
            results['signals'] = {
                'avg_liquidity_score': np.mean([t.liquidity_score for t in self.trades]),
                'avg_sentiment_score': np.mean([t.sentiment_score for t in self.trades]),
                'avg_confidence': np.mean([t.confidence for t in self.trades]),
                'winning_avg_confidence': np.mean([t.confidence for t in winners]) if winners else 0,
                'losing_avg_confidence': np.mean([t.confidence for t in losers]) if losers else 0,
            }
            
            # Exit reasons
            exit_reasons = {}
            for t in self.trades:
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
            results['exit_reasons'] = exit_reasons
        else:
            results['trades'] = {'total': 0, 'message': 'No trades executed'}
        
        # Drawdown analysis
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
            
            results['risk'] = {
                'max_drawdown_pct': max_dd,
                'final_equity': equities[-1],
            }
            
            # Daily returns for Sharpe
            if len(equities) > 1:
                daily_returns = pd.Series(equities).pct_change().dropna()
                if len(daily_returns) > 0 and daily_returns.std() > 0:
                    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                    results['risk']['sharpe_ratio'] = sharpe
                    
                    # Sortino (downside deviation)
                    downside = daily_returns[daily_returns < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        sortino = daily_returns.mean() / downside.std() * np.sqrt(252)
                        results['risk']['sortino_ratio'] = sortino
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        
        print("\n" + "="*70)
        print("  BACKTEST RESULTS")
        print("="*70)
        
        cap = results['capital']
        print(f"\n{'CAPITAL':=^50}")
        print(f"  Initial:        ${cap['initial']:>12,.2f}")
        print(f"  Final:          ${cap['final']:>12,.2f}")
        print(f"  Return:         ${cap['return']:>12,.2f} ({cap['return_pct']:.2%})")
        
        if 'trades' in results and results['trades'].get('total', 0) > 0:
            t = results['trades']
            print(f"\n{'TRADES':=^50}")
            print(f"  Total Trades:   {t['total']:>12}")
            print(f"  Winners:        {t['winners']:>12} ({t['win_rate']:.1%})")
            print(f"  Losers:         {t['losers']:>12}")
            print(f"  Profit Factor:  {t['profit_factor']:>12.2f}")
            print(f"  Avg Win:        ${t['avg_win']:>12,.2f}")
            print(f"  Avg Loss:       ${t['avg_loss']:>12,.2f}")
            print(f"  Largest Win:    ${t['largest_win']:>12,.2f}")
            print(f"  Largest Loss:   ${t['largest_loss']:>12,.2f}")
        
        if 'signals' in results:
            s = results['signals']
            print(f"\n{'SIGNAL ANALYSIS':=^50}")
            print(f"  Avg Liquidity:  {s['avg_liquidity_score']:>12.3f}")
            print(f"  Avg Sentiment:  {s['avg_sentiment_score']:>12.3f}")
            print(f"  Avg Confidence: {s['avg_confidence']:>12.3f}")
            print(f"  Win Confidence: {s['winning_avg_confidence']:>12.3f}")
            print(f"  Loss Confidence:{s['losing_avg_confidence']:>12.3f}")
        
        if 'risk' in results:
            r = results['risk']
            print(f"\n{'RISK METRICS':=^50}")
            print(f"  Max Drawdown:   {r['max_drawdown_pct']:>12.2%}")
            if 'sharpe_ratio' in r:
                print(f"  Sharpe Ratio:   {r['sharpe_ratio']:>12.2f}")
            if 'sortino_ratio' in r:
                print(f"  Sortino Ratio:  {r['sortino_ratio']:>12.2f}")
        
        if 'exit_reasons' in results:
            print(f"\n{'EXIT REASONS':=^50}")
            for reason, count in results['exit_reasons'].items():
                print(f"  {reason}:{count:>10}")
        
        print("\n" + "="*70)
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        import json
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"liquidity_sentiment_{timestamp}.json"
        
        # Convert trades to serializable format
        trades_data = []
        for t in self.trades:
            trades_data.append({
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_date': t.entry_date.isoformat() if t.entry_date else None,
                'exit_date': t.exit_date.isoformat() if t.exit_date else None,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'net_pnl': t.net_pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason,
                'liquidity_score': t.liquidity_score,
                'sentiment_score': t.sentiment_score,
                'confidence': t.confidence,
            })
        
        output = {
            'summary': results,
            'trades': trades_data,
            'equity_curve': [
                {'timestamp': e['timestamp'].isoformat(), 'equity': e['equity']}
                for e in self.equity_curve
            ],
        }
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")


def main():
    """Run the liquidity-sentiment backtest."""
    
    config = LiquiditySentimentConfig(
        symbols=["SPY", "QQQ", "IWM"],
        start_date="2024-01-01",
        end_date="2024-12-01",
        initial_capital=100_000.0,
        liquidity_weight=0.5,
        sentiment_weight=0.5,
        min_liquidity_score=0.3,
        min_sentiment_strength=0.15,
        min_combined_confidence=0.35,
    )
    
    engine = LiquiditySentimentEngine(config)
    results = engine.run_backtest()
    
    return results


if __name__ == "__main__":
    main()
