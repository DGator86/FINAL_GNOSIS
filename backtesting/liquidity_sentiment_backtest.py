"""
Liquidity & Sentiment Pipeline Backtest

Focused backtest using only liquidity and sentiment signals
with real market data from Massive.com (Polygon.io).

NO synthetic data fallback - uses actual historical price/volume data
to derive realistic liquidity and sentiment metrics.

Author: GNOSIS Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.credentials import get_massive_api_keys


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LiquiditySentimentConfig:
    """Configuration for liquidity-sentiment backtest."""
    
    # Data settings
    symbols: List[str] = field(default_factory=lambda: ["SPY"])
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-01"
    
    # Capital settings
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.10  # 10% max per position
    max_positions: int = 3
    
    # Signal weights (liquidity & sentiment focused)
    liquidity_weight: float = 0.50
    sentiment_weight: float = 0.50
    
    # Signal thresholds
    min_signal_strength: float = 0.20
    min_liquidity_score: float = 0.30
    
    # Cost modeling
    slippage_bps: float = 5.0
    commission_per_trade: float = 1.0
    
    # Risk management
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    max_loss_pct: float = 0.02
    
    # Output
    save_results: bool = True
    output_dir: str = "runs/liquidity_sentiment"


@dataclass
class LSTradeRecord:
    """Record of a liquidity-sentiment trade."""
    
    trade_id: int = 0
    symbol: str = ""
    direction: str = "long"
    
    entry_date: datetime = None
    exit_date: datetime = None
    hold_days: int = 0
    
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    
    # Signal values at entry
    liquidity_score: float = 0.0
    sentiment_score: float = 0.0
    combined_signal: float = 0.0
    
    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Costs
    total_costs: float = 0.0
    
    # Outcome
    is_winner: bool = False
    exit_reason: str = ""


@dataclass
class LSBacktestResults:
    """Results from liquidity-sentiment backtest."""
    
    # Configuration
    config: LiquiditySentimentConfig = None
    
    # Period
    start_date: datetime = None
    end_date: datetime = None
    total_bars: int = 0
    
    # Capital
    initial_capital: float = 100_000.0
    final_capital: float = 100_000.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Costs
    total_costs: float = 0.0
    
    # Signal analysis
    avg_liquidity_score: float = 0.0
    avg_sentiment_score: float = 0.0
    liquidity_win_correlation: float = 0.0
    sentiment_win_correlation: float = 0.0
    
    # Data
    trades: List[LSTradeRecord] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# MASSIVE.COM DATA ADAPTER
# ============================================================================

class MassiveStockDataAdapter:
    """Fetch real stock data from Massive.com (Polygon.io)."""
    
    def __init__(self, api_key: Optional[str] = None):
        primary, secondary = get_massive_api_keys(primary=api_key)
        self.api_key = primary or secondary
        
        if not self.api_key:
            raise ValueError("Massive.com API key required. Set MASSIVE_API_KEY env var.")
        
        try:
            from massive import RESTClient
            self.client = RESTClient(api_key=self.api_key)
            logger.info("MassiveStockDataAdapter initialized")
        except ImportError:
            raise ImportError("Massive client not installed. Run: pip install massive")
    
    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data from Massive.com."""
        
        try:
            aggs = list(self.client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                adjusted=True,
                limit=50000,
            ))
            
            if not aggs:
                raise ValueError(f"No data returned for {symbol}")
            
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc),
                    'open': float(agg.open),
                    'high': float(agg.high),
                    'low': float(agg.low),
                    'close': float(agg.close),
                    'volume': float(agg.volume),
                    'vwap': float(getattr(agg, 'vwap', 0)) or 0,
                    'transactions': int(getattr(agg, 'transactions', 0)) or 0,
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} bars for {symbol} from Massive.com")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Massive.com: {e}")
            raise


# ============================================================================
# SIGNAL GENERATORS
# ============================================================================

class LiquiditySignalGenerator:
    """Generate liquidity signals from price/volume data."""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def compute(
        self,
        bar: Dict[str, Any],
        history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute liquidity metrics."""
        
        if len(history) < 5:
            return {
                'liquidity_score': 0.5,
                'volume_ratio': 1.0,
                'spread_estimate': 0.001,
                'depth_score': 0.5,
                'impact_cost_bps': 5.0,
            }
        
        # Volume analysis
        avg_volume = history['volume'].tail(self.lookback).mean()
        recent_volume = history['volume'].tail(5).mean()
        current_volume = bar.get('volume', avg_volume)
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Spread estimate from high-low range (proxy)
        avg_range = ((history['high'] - history['low']) / history['close']).tail(self.lookback).mean()
        spread_estimate = avg_range * 0.1  # Spread is fraction of range
        
        # Depth score (higher volume = more depth)
        volume_percentile = (history['volume'] < current_volume).mean()
        depth_score = volume_percentile
        
        # Impact cost estimate (higher spread = higher impact)
        impact_cost_bps = spread_estimate * 100 * 50  # Convert to bps
        
        # Combined liquidity score
        volume_score = min(1.0, volume_ratio / 2.0)  # Cap at 2x average
        spread_score = max(0.0, 1.0 - spread_estimate * 50)  # Lower spread = higher score
        
        liquidity_score = (
            volume_score * 0.4 +
            spread_score * 0.3 +
            depth_score * 0.3
        )
        
        return {
            'liquidity_score': liquidity_score,
            'volume_ratio': volume_ratio,
            'spread_estimate': spread_estimate,
            'depth_score': depth_score,
            'impact_cost_bps': impact_cost_bps,
        }


class SentimentSignalGenerator:
    """Generate sentiment signals from price patterns."""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def compute(
        self,
        bar: Dict[str, Any],
        history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute sentiment metrics from price action."""
        
        if len(history) < 10:
            return {
                'sentiment_score': 0.0,
                'trend_sentiment': 0.0,
                'momentum_sentiment': 0.0,
                'volume_sentiment': 0.0,
                'mean_reversion_signal': 0.0,
            }
        
        current = bar['close']
        returns = history['close'].pct_change().dropna()
        
        # Moving averages
        sma_5 = history['close'].tail(5).mean()
        sma_10 = history['close'].tail(10).mean()
        sma_20 = history['close'].tail(min(20, len(history))).mean()
        sma_50 = history['close'].tail(min(50, len(history))).mean()
        
        # Trend sentiment (price vs MAs)
        trend_5 = (current - sma_5) / sma_5 if sma_5 > 0 else 0
        trend_20 = (current - sma_20) / sma_20 if sma_20 > 0 else 0
        trend_sentiment = np.tanh((trend_5 * 0.6 + trend_20 * 0.4) * 20)
        
        # Momentum sentiment (rate of change)
        if len(returns) >= 5:
            recent_momentum = returns.tail(5).mean()
            momentum_sentiment = np.tanh(recent_momentum * 100)
        else:
            momentum_sentiment = 0.0
        
        # Volume sentiment (buying vs selling pressure)
        recent = history.tail(5)
        up_volume = recent[recent['close'] > recent['open']]['volume'].sum()
        down_volume = recent[recent['close'] < recent['open']]['volume'].sum()
        total = up_volume + down_volume
        
        if total > 0:
            volume_sentiment = (up_volume - down_volume) / total
        else:
            volume_sentiment = 0.0
        
        # Mean reversion signal
        distance_from_sma = (current - sma_20) / sma_20 if sma_20 > 0 else 0
        mean_reversion_signal = -np.tanh(distance_from_sma * 10)
        
        # Combined sentiment score
        sentiment_score = (
            trend_sentiment * 0.35 +
            momentum_sentiment * 0.30 +
            volume_sentiment * 0.25 +
            mean_reversion_signal * 0.10
        )
        
        return {
            'sentiment_score': np.clip(sentiment_score, -1.0, 1.0),
            'trend_sentiment': trend_sentiment,
            'momentum_sentiment': momentum_sentiment,
            'volume_sentiment': volume_sentiment,
            'mean_reversion_signal': mean_reversion_signal,
        }


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class LiquiditySentimentBacktest:
    """
    Backtest engine focused on liquidity and sentiment signals.
    
    Uses real market data from Massive.com - NO synthetic fallback.
    """
    
    def __init__(self, config: LiquiditySentimentConfig):
        self.config = config
        
        # Data adapter
        self.data_adapter = MassiveStockDataAdapter()
        
        # Signal generators
        self.liquidity_gen = LiquiditySignalGenerator()
        self.sentiment_gen = SentimentSignalGenerator()
        
        # State
        self.capital = config.initial_capital
        self.positions: Dict[str, LSTradeRecord] = {}
        self.trades: List[LSTradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.trade_counter = 0
        
        logger.info(
            f"LiquiditySentimentBacktest initialized | "
            f"symbols={config.symbols} | "
            f"capital=${config.initial_capital:,.0f}"
        )
    
    def run(self) -> LSBacktestResults:
        """Run the backtest."""
        
        logger.info("="*60)
        logger.info("LIQUIDITY & SENTIMENT PIPELINE BACKTEST")
        logger.info("Using REAL market data from Massive.com")
        logger.info("="*60)
        
        # Parse dates
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.config.end_date, "%Y-%m-%d").date()
        
        # Fetch data for all symbols
        all_data = {}
        for symbol in self.config.symbols:
            try:
                df = self.data_adapter.get_daily_bars(symbol, start, end)
                all_data[symbol] = df
                logger.info(f"Loaded {len(df)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded for any symbol")
        
        # Combine and sort by date
        combined = pd.concat(all_data.values())
        dates = sorted(combined['timestamp'].dt.date.unique())
        
        logger.info(f"Running backtest from {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
        
        # Main loop
        total_bars = 0
        for i, current_date in enumerate(dates):
            for symbol, df in all_data.items():
                # Get bars up to current date
                mask = df['timestamp'].dt.date <= current_date
                history = df[mask].copy()
                
                if len(history) < 10:
                    continue
                
                current_bar = history.iloc[-1].to_dict()
                timestamp = current_bar['timestamp']
                
                # Compute signals
                liquidity = self.liquidity_gen.compute(current_bar, history.iloc[:-1])
                sentiment = self.sentiment_gen.compute(current_bar, history.iloc[:-1])
                
                # Compute combined signal
                combined_signal = (
                    liquidity['liquidity_score'] * self.config.liquidity_weight +
                    sentiment['sentiment_score'] * self.config.sentiment_weight
                )
                
                # Check existing positions
                if symbol in self.positions:
                    self._check_exit(
                        symbol, current_bar, timestamp,
                        liquidity, sentiment, combined_signal, history
                    )
                
                # Check for new entries
                if symbol not in self.positions:
                    self._check_entry(
                        symbol, current_bar, timestamp,
                        liquidity, sentiment, combined_signal, history
                    )
                
                total_bars += 1
            
            # Record equity
            position_value = sum(
                p.position_size * (current_bar['close'] / p.entry_price)
                for p in self.positions.values()
            ) if self.positions else 0
            
            self.equity_curve.append({
                'date': current_date,
                'capital': self.capital,
                'position_value': position_value,
                'total_equity': self.capital + position_value,
                'positions': len(self.positions),
            })
            
            # Progress
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(dates)} days | Capital: ${self.capital:,.2f}")
        
        # Close remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in all_data:
                last_bar = all_data[symbol].iloc[-1].to_dict()
                self._close_position(symbol, last_bar, last_bar['timestamp'], "end_of_backtest")
        
        # Calculate results
        results = self._calculate_results(total_bars, dates[0], dates[-1])
        
        # Save results
        if self.config.save_results:
            self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _compute_atr(self, history: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR."""
        if len(history) < period:
            return history['close'].iloc[-1] * 0.02
        
        df = history.tail(period + 1).copy()
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['prev_close']),
                abs(df['low'] - df['prev_close'])
            )
        )
        return df['tr'].tail(period).mean()
    
    def _check_entry(
        self,
        symbol: str,
        bar: Dict,
        timestamp: datetime,
        liquidity: Dict,
        sentiment: Dict,
        combined_signal: float,
        history: pd.DataFrame,
    ):
        """Check for entry signals."""
        
        if len(self.positions) >= self.config.max_positions:
            return
        
        # Check liquidity threshold
        if liquidity['liquidity_score'] < self.config.min_liquidity_score:
            return
        
        # Check signal strength
        if abs(sentiment['sentiment_score']) < self.config.min_signal_strength:
            return
        
        # Determine direction from sentiment
        if sentiment['sentiment_score'] > self.config.min_signal_strength:
            direction = "long"
        elif sentiment['sentiment_score'] < -self.config.min_signal_strength:
            direction = "short"
        else:
            return
        
        # Calculate position size
        risk_capital = self.capital * self.config.max_position_pct
        
        # Entry price with slippage
        slippage = self.config.slippage_bps / 10000
        if direction == "long":
            entry_price = bar['close'] * (1 + slippage)
        else:
            entry_price = bar['close'] * (1 - slippage)
        
        # Costs
        commission = self.config.commission_per_trade
        total_costs = risk_capital * slippage + commission
        
        position_size = risk_capital - total_costs
        
        if position_size <= 0:
            return
        
        # Create trade record
        self.trade_counter += 1
        trade = LSTradeRecord(
            trade_id=self.trade_counter,
            symbol=symbol,
            direction=direction,
            entry_date=timestamp,
            entry_price=entry_price,
            position_size=position_size,
            liquidity_score=liquidity['liquidity_score'],
            sentiment_score=sentiment['sentiment_score'],
            combined_signal=combined_signal,
        )
        
        self.positions[symbol] = trade
        self.capital -= position_size + total_costs
        
        logger.debug(
            f"ENTRY: {symbol} {direction} @ ${entry_price:.2f} | "
            f"Size: ${position_size:,.0f} | "
            f"Liq: {liquidity['liquidity_score']:.2f} | "
            f"Sent: {sentiment['sentiment_score']:.2f}"
        )
    
    def _check_exit(
        self,
        symbol: str,
        bar: Dict,
        timestamp: datetime,
        liquidity: Dict,
        sentiment: Dict,
        combined_signal: float,
        history: pd.DataFrame,
    ):
        """Check for exit signals."""
        
        position = self.positions[symbol]
        current_price = bar['close']
        
        # Calculate ATR for stops
        atr = self._compute_atr(history, self.config.atr_period)
        
        # Check stop loss
        if position.direction == "long":
            stop_price = position.entry_price - (atr * self.config.atr_stop_mult)
            target_price = position.entry_price + (atr * self.config.atr_target_mult)
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            if current_price <= stop_price:
                self._close_position(symbol, bar, timestamp, "stop_loss")
                return
            if current_price >= target_price:
                self._close_position(symbol, bar, timestamp, "take_profit")
                return
        else:
            stop_price = position.entry_price + (atr * self.config.atr_stop_mult)
            target_price = position.entry_price - (atr * self.config.atr_target_mult)
            pnl_pct = (position.entry_price - current_price) / position.entry_price
            
            if current_price >= stop_price:
                self._close_position(symbol, bar, timestamp, "stop_loss")
                return
            if current_price <= target_price:
                self._close_position(symbol, bar, timestamp, "take_profit")
                return
        
        # Check max loss
        if pnl_pct < -self.config.max_loss_pct:
            self._close_position(symbol, bar, timestamp, "max_loss")
            return
        
        # Check signal reversal
        if position.direction == "long" and sentiment['sentiment_score'] < -0.3:
            self._close_position(symbol, bar, timestamp, "signal_reversal")
            return
        elif position.direction == "short" and sentiment['sentiment_score'] > 0.3:
            self._close_position(symbol, bar, timestamp, "signal_reversal")
            return
    
    def _close_position(
        self,
        symbol: str,
        bar: Dict,
        timestamp: datetime,
        reason: str,
    ):
        """Close a position."""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Exit price with slippage
        slippage = self.config.slippage_bps / 10000
        if position.direction == "long":
            exit_price = bar['close'] * (1 - slippage)
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            exit_price = bar['close'] * (1 + slippage)
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        # Calculate P&L
        gross_pnl = position.position_size * pnl_pct
        commission = self.config.commission_per_trade
        slippage_cost = position.position_size * slippage
        total_costs = commission + slippage_cost
        net_pnl = gross_pnl - total_costs
        
        # Update position record
        position.exit_date = timestamp
        position.exit_price = exit_price
        position.hold_days = (timestamp - position.entry_date).days
        position.gross_pnl = gross_pnl
        position.net_pnl = net_pnl
        position.pnl_pct = pnl_pct
        position.total_costs = total_costs
        position.is_winner = net_pnl > 0
        position.exit_reason = reason
        
        # Update capital
        self.capital += position.position_size + net_pnl
        
        # Store trade
        self.trades.append(position)
        del self.positions[symbol]
        
        logger.debug(
            f"EXIT: {symbol} @ ${exit_price:.2f} | "
            f"PnL: ${net_pnl:,.2f} ({pnl_pct:.1%}) | "
            f"Reason: {reason}"
        )
    
    def _calculate_results(
        self,
        total_bars: int,
        start_date: date,
        end_date: date,
    ) -> LSBacktestResults:
        """Calculate backtest results."""
        
        results = LSBacktestResults(
            config=self.config,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            total_bars=total_bars,
            initial_capital=self.config.initial_capital,
            final_capital=self.capital,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )
        
        # Returns
        results.total_return = self.capital - self.config.initial_capital
        results.total_return_pct = results.total_return / self.config.initial_capital
        
        # CAGR
        years = (end_date - start_date).days / 365.25
        if years > 0 and self.capital > 0:
            results.cagr = (self.capital / self.config.initial_capital) ** (1 / years) - 1
        
        # Trade statistics
        results.total_trades = len(self.trades)
        if results.total_trades > 0:
            results.winning_trades = sum(1 for t in self.trades if t.is_winner)
            results.losing_trades = results.total_trades - results.winning_trades
            results.win_rate = results.winning_trades / results.total_trades
            
            wins = [t.net_pnl for t in self.trades if t.is_winner]
            losses = [abs(t.net_pnl) for t in self.trades if not t.is_winner]
            
            results.avg_win = np.mean(wins) if wins else 0
            results.avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins)
            total_losses = sum(losses)
            results.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            results.total_costs = sum(t.total_costs for t in self.trades)
            
            # Signal analysis
            results.avg_liquidity_score = np.mean([t.liquidity_score for t in self.trades])
            results.avg_sentiment_score = np.mean([t.sentiment_score for t in self.trades])
            
            # Correlation with wins
            if len(self.trades) >= 3:
                liq_scores = [t.liquidity_score for t in self.trades]
                sent_scores = [t.sentiment_score for t in self.trades]
                wins_binary = [1 if t.is_winner else 0 for t in self.trades]
                
                results.liquidity_win_correlation = np.corrcoef(liq_scores, wins_binary)[0, 1]
                results.sentiment_win_correlation = np.corrcoef(sent_scores, wins_binary)[0, 1]
        
        # Risk metrics from equity curve
        if self.equity_curve:
            equity = [e['total_equity'] for e in self.equity_curve]
            peak = equity[0]
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak
                if dd > max_dd:
                    max_dd = dd
            results.max_drawdown_pct = max_dd
            
            # Daily returns
            if len(equity) > 1:
                daily_returns = pd.Series(equity).pct_change().dropna()
                
                if len(daily_returns) > 0 and daily_returns.std() > 0:
                    results.sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
                    
                    downside = daily_returns[daily_returns < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        results.sortino_ratio = (daily_returns.mean() * 252) / (downside.std() * np.sqrt(252))
        
        return results
    
    def _save_results(self, results: LSBacktestResults):
        """Save results to disk."""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary file
        summary = {
            'config': {
                'symbols': self.config.symbols,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_capital': self.config.initial_capital,
                'liquidity_weight': self.config.liquidity_weight,
                'sentiment_weight': self.config.sentiment_weight,
            },
            'performance': {
                'final_capital': results.final_capital,
                'total_return': results.total_return,
                'total_return_pct': results.total_return_pct,
                'cagr': results.cagr,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'max_drawdown_pct': results.max_drawdown_pct,
            },
            'trades': {
                'total_trades': results.total_trades,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'total_costs': results.total_costs,
            },
            'signals': {
                'avg_liquidity_score': results.avg_liquidity_score,
                'avg_sentiment_score': results.avg_sentiment_score,
                'liquidity_win_correlation': results.liquidity_win_correlation,
                'sentiment_win_correlation': results.sentiment_win_correlation,
            },
        }
        
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ls_backtest_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir / filename}")
    
    def _print_summary(self, results: LSBacktestResults):
        """Print results summary."""
        
        print("\n" + "="*70)
        print("  LIQUIDITY & SENTIMENT BACKTEST RESULTS")
        print("  Data Source: Massive.com (Real Market Data)")
        print("="*70)
        
        print(f"\n{'CONFIGURATION':^70}")
        print("-"*70)
        print(f"  Symbols: {', '.join(self.config.symbols)}")
        print(f"  Period: {self.config.start_date} to {self.config.end_date}")
        print(f"  Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"  Liquidity Weight: {self.config.liquidity_weight:.0%}")
        print(f"  Sentiment Weight: {self.config.sentiment_weight:.0%}")
        
        print(f"\n{'PERFORMANCE':^70}")
        print("-"*70)
        print(f"  Final Capital: ${results.final_capital:,.2f}")
        print(f"  Total Return: ${results.total_return:,.2f} ({results.total_return_pct:.2%})")
        print(f"  CAGR: {results.cagr:.2%}")
        print(f"  Max Drawdown: {results.max_drawdown_pct:.2%}")
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {results.sortino_ratio:.2f}")
        
        print(f"\n{'TRADE STATISTICS':^70}")
        print("-"*70)
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Win Rate: {results.win_rate:.1%}")
        print(f"  Profit Factor: {results.profit_factor:.2f}")
        print(f"  Avg Win: ${results.avg_win:,.2f}")
        print(f"  Avg Loss: ${results.avg_loss:,.2f}")
        print(f"  Total Costs: ${results.total_costs:,.2f}")
        
        print(f"\n{'SIGNAL ANALYSIS':^70}")
        print("-"*70)
        print(f"  Avg Liquidity Score: {results.avg_liquidity_score:.3f}")
        print(f"  Avg Sentiment Score: {results.avg_sentiment_score:.3f}")
        print(f"  Liquidity-Win Correlation: {results.liquidity_win_correlation:.3f}")
        print(f"  Sentiment-Win Correlation: {results.sentiment_win_correlation:.3f}")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def run_liquidity_sentiment_backtest(
    symbols: List[str] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-01",
    initial_capital: float = 100_000.0,
    liquidity_weight: float = 0.50,
    sentiment_weight: float = 0.50,
) -> LSBacktestResults:
    """
    Run a liquidity & sentiment focused backtest.
    
    Uses REAL market data from Massive.com - no synthetic fallback.
    """
    
    if symbols is None:
        symbols = ["SPY"]
    
    config = LiquiditySentimentConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        liquidity_weight=liquidity_weight,
        sentiment_weight=sentiment_weight,
    )
    
    engine = LiquiditySentimentBacktest(config)
    return engine.run()


if __name__ == "__main__":
    # Run demo backtest
    results = run_liquidity_sentiment_backtest(
        symbols=["SPY", "QQQ"],
        start_date="2024-01-01",
        end_date="2024-11-30",
        initial_capital=100_000.0,
    )
