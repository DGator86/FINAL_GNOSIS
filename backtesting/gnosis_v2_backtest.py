#!/usr/bin/env python3
"""
GNOSIS V2 Architecture Backtest

Comprehensive backtest of the refactored GNOSIS architecture:
- ComposerAgentV4 with PENTA confluence
- AlphaTradeAgentV2 / FullGnosisTradeAgentV2
- LiquidityEngineV5 with PENTA methodology
- GnosisMonitor / AlphaMonitor

Author: GNOSIS Trading System
Version: 2.0.0
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import random

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    symbol: str
    direction: str
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    penta_confluence: Optional[str] = None
    
    # Results
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_winner: bool = False
    exit_reason: str = ""


@dataclass
class BacktestResults:
    """Results from GNOSIS V2 backtest."""
    # Configuration
    start_date: datetime = None
    end_date: datetime = None
    initial_capital: float = 100_000
    symbols: List[str] = field(default_factory=list)
    
    # Performance
    final_capital: float = 100_000
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
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # PENTA Analysis
    penta_trades: int = 0
    quad_trades: int = 0
    triple_trades: int = 0
    regular_trades: int = 0
    penta_win_rate: float = 0.0
    
    # Data
    equity_curve: List[Dict] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)
    
    def print_summary(self):
        """Print formatted backtest summary."""
        print("\n" + "="*70)
        print("  GNOSIS V2 ARCHITECTURE - Backtest Results")
        print("="*70)
        print(f"\nPeriod: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.final_capital:,.2f}")
        
        print("\n📊 Performance:")
        print(f"  Total Return: ${self.total_return:,.2f} ({self.total_return_pct*100:+.2f}%)")
        print(f"  Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {self.max_drawdown_pct*100:.2f}%")
        
        print("\n📈 Trade Statistics:")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Winners: {self.winning_trades} | Losers: {self.losing_trades}")
        print(f"  Win Rate: {self.win_rate*100:.1f}%")
        print(f"  Profit Factor: {self.profit_factor:.2f}")
        print(f"  Avg Win: {self.avg_win*100:+.2f}% | Avg Loss: {self.avg_loss*100:.2f}%")
        
        print("\n🔥 PENTA Methodology Impact:")
        print(f"  PENTA Confluence Trades: {self.penta_trades} ({self.penta_trades/max(1,self.total_trades)*100:.1f}%)")
        print(f"  QUAD Confluence Trades: {self.quad_trades}")
        print(f"  TRIPLE Confluence Trades: {self.triple_trades}")
        print(f"  Regular Trades: {self.regular_trades}")
        if self.penta_trades > 0:
            print(f"  PENTA Win Rate: {self.penta_win_rate*100:.1f}%")
        
        print("="*70 + "\n")


class GnosisV2Backtester:
    """
    Backtester for GNOSIS V2 Architecture.
    
    Uses:
    - ComposerAgentV4 for consensus building
    - AlphaTradeAgentV2 for signal generation
    - LiquidityEngineV5 (PENTA) for smart money analysis
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000,
        max_positions: int = 5,
        max_position_pct: float = 0.10,
        min_confidence: float = 0.50,
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.min_confidence = min_confidence
        
        # Initialize V2 components
        self._init_components()
    
    def _init_components(self):
        """Initialize GNOSIS V2 components."""
        try:
            from agents.composer.composer_agent_v4 import ComposerAgentV4
            from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
            from agents.monitoring import AlphaMonitor
            
            self.composer = ComposerAgentV4()
            self.trade_agent = AlphaTradeAgentV2(config={
                "min_confidence": self.min_confidence,
            })
            self.monitor = AlphaMonitor({})
            
            logger.info("GNOSIS V2 components initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import GNOSIS V2 components: {e}")
            raise
    
    def run(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> BacktestResults:
        """
        Run backtest with GNOSIS V2 architecture.
        
        Args:
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            price_data: Optional pre-loaded price data
        
        Returns:
            BacktestResults with all metrics
        """
        logger.info(f"Starting GNOSIS V2 backtest: {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {symbols}")
        
        # Load price data if not provided
        if price_data is None:
            price_data = self._load_price_data(symbols, start_date, end_date)
        
        # Initialize state
        cash = self.initial_capital
        positions: Dict[str, BacktestTrade] = {}
        all_trades: List[BacktestTrade] = []
        equity_curve: List[Dict] = []
        
        # Generate trading days
        trading_days = self._get_trading_days(start_date, end_date, price_data)
        logger.info(f"Trading days: {len(trading_days)}")
        
        # Main simulation loop
        for current_date in trading_days:
            # Calculate current equity
            position_value = self._calculate_position_value(positions, price_data, current_date)
            equity = cash + position_value
            equity_curve.append({
                "date": current_date,
                "equity": equity,
                "cash": cash,
                "positions": len(positions),
            })
            
            # Check exits for existing positions
            closed = self._check_exits(positions, price_data, current_date)
            for trade in closed:
                cash += trade.exit_price * trade.quantity
                all_trades.append(trade)
                del positions[trade.symbol]
            
            # Generate new signals
            if len(positions) < self.max_positions:
                for symbol in symbols:
                    if symbol in positions:
                        continue
                    if len(positions) >= self.max_positions:
                        break
                    
                    # Get current price
                    price = self._get_price(symbol, current_date, price_data)
                    if price is None:
                        continue
                    
                    # Generate simulated agent signals
                    signals = self._generate_simulated_signals(symbol, current_date, price_data)
                    
                    # Use ComposerAgentV4 for consensus
                    composer_output = self.composer.compose(
                        hedge_signal=signals["hedge"],
                        sentiment_signal=signals["sentiment"],
                        liquidity_signal=signals["liquidity"],
                        penta_confluence=signals.get("penta_confluence"),
                    )
                    
                    # Use AlphaTradeAgentV2 for signal
                    signal = self.trade_agent.process_composer_output(
                        composer_output,
                        symbol=symbol,
                        current_price=price,
                    )
                    
                    # Check if signal is actionable
                    if signal.direction == "HOLD":
                        continue
                    if signal.confidence < self.min_confidence:
                        continue
                    
                    # Calculate position size
                    position_value = equity * self.max_position_pct
                    quantity = int(position_value / price)
                    if quantity <= 0:
                        continue
                    
                    # Check cash availability
                    cost = price * quantity
                    if cost > cash:
                        continue
                    
                    # Open position
                    trade = BacktestTrade(
                        symbol=symbol,
                        direction=signal.direction,
                        entry_date=current_date,
                        entry_price=price,
                        quantity=quantity,
                        stop_loss=signal.stop_loss or price * 0.97,
                        take_profit=signal.take_profit or price * 1.05,
                        confidence=signal.confidence,
                        penta_confluence=getattr(composer_output, 'penta_confluence', None),
                    )
                    
                    positions[symbol] = trade
                    cash -= cost
                    
                    # Track in monitor
                    self.monitor.update(signal={
                        "symbol": signal.symbol,
                        "signal_type": signal.direction,
                        "confidence": signal.confidence,
                    })
        
        # Close remaining positions
        for symbol, trade in list(positions.items()):
            exit_price = self._get_price(symbol, end_date, price_data)
            if exit_price:
                trade.exit_date = end_date
                trade.exit_price = exit_price
                trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                trade.is_winner = trade.pnl > 0
                trade.exit_reason = "end_of_backtest"
                all_trades.append(trade)
        
        # Calculate results
        results = self._calculate_results(
            start_date, end_date, symbols,
            equity_curve, all_trades
        )
        
        logger.info(f"Backtest complete: {results.total_trades} trades, {results.win_rate*100:.1f}% win rate")
        
        return results
    
    def _load_price_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Load or generate price data."""
        logger.info("Generating synthetic price data for backtest...")
        
        price_data = {}
        np.random.seed(42)  # For reproducibility
        
        # Generate ~1.5 years of data with buffer
        buffer_start = start_date - timedelta(days=60)
        buffer_end = end_date + timedelta(days=30)
        num_days = (buffer_end - buffer_start).days
        
        dates = pd.date_range(start=buffer_start, periods=num_days, freq='D')
        
        for symbol in symbols:
            # Base prices for different symbols
            base_prices = {
                "SPY": 450.0,
                "QQQ": 380.0,
                "AAPL": 180.0,
                "MSFT": 380.0,
                "NVDA": 500.0,
                "GOOGL": 140.0,
                "AMZN": 180.0,
                "TSLA": 250.0,
            }
            base_price = base_prices.get(symbol, 100.0)
            
            # Generate realistic returns with trend
            trend = 0.0002  # Small upward bias
            volatility = 0.015  # 1.5% daily vol
            returns = np.random.normal(trend, volatility, num_days)
            
            # Add some mean reversion
            prices = [base_price]
            for r in returns[1:]:
                new_price = prices[-1] * (1 + r)
                prices.append(new_price)
            
            prices = np.array(prices)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices * (1 - np.random.uniform(0, 0.005, num_days)),
                'high': prices * (1 + np.random.uniform(0, 0.01, num_days)),
                'low': prices * (1 - np.random.uniform(0, 0.01, num_days)),
                'close': prices,
                'volume': np.random.randint(10_000_000, 100_000_000, num_days),
            })
            df.set_index('timestamp', inplace=True)
            price_data[symbol] = df
        
        return price_data
    
    def _get_trading_days(
        self,
        start_date: datetime,
        end_date: datetime,
        price_data: Dict[str, pd.DataFrame],
    ) -> List[datetime]:
        """Get list of trading days."""
        ref_symbol = list(price_data.keys())[0]
        df = price_data[ref_symbol]
        
        days = [
            d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d
            for d in df.index
            if start_date <= (d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d) <= end_date
        ]
        
        # Filter to weekdays only
        return [d for d in days if d.weekday() < 5]
    
    def _get_price(
        self,
        symbol: str,
        date: datetime,
        price_data: Dict[str, pd.DataFrame],
    ) -> Optional[float]:
        """Get price for symbol on date."""
        if symbol not in price_data:
            return None
        
        df = price_data[symbol]
        
        try:
            # Try exact match
            if date in df.index:
                return float(df.loc[date, 'close'])
            
            # Find nearest date
            idx = df.index.get_indexer([date], method='ffill')[0]
            if idx >= 0 and idx < len(df):
                return float(df['close'].iloc[idx])
        except Exception:
            pass
        
        return None
    
    def _calculate_position_value(
        self,
        positions: Dict[str, BacktestTrade],
        price_data: Dict[str, pd.DataFrame],
        date: datetime,
    ) -> float:
        """Calculate total position value."""
        total = 0.0
        for symbol, trade in positions.items():
            price = self._get_price(symbol, date, price_data)
            if price:
                total += price * trade.quantity
        return total
    
    def _generate_simulated_signals(
        self,
        symbol: str,
        date: datetime,
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Generate simulated agent signals based on price action."""
        df = price_data.get(symbol)
        if df is None or len(df) < 20:
            return self._neutral_signals()
        
        # Get recent data
        mask = df.index <= date
        recent = df[mask].tail(20)
        if len(recent) < 10:
            return self._neutral_signals()
        
        # Calculate technical indicators
        prices = recent['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # Momentum
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        # Volatility
        volatility = np.std(returns) if len(returns) > 1 else 0.01
        
        # Volume trend
        volumes = recent['volume'].values
        vol_trend = (volumes[-5:].mean() / volumes.mean()) if volumes.mean() > 0 else 1.0
        
        # Generate signals based on indicators
        signals = {}
        
        # Hedge signal (based on momentum and volatility)
        if momentum > 0.02 and vol_trend > 1.1:
            signals["hedge"] = {"direction": "bullish", "confidence": min(0.9, 0.5 + momentum * 5)}
        elif momentum < -0.02 and vol_trend > 1.1:
            signals["hedge"] = {"direction": "bearish", "confidence": min(0.9, 0.5 + abs(momentum) * 5)}
        else:
            signals["hedge"] = {"direction": "neutral", "confidence": 0.4}
        
        # Sentiment signal (slightly random, trend-following)
        sentiment_bias = 0.5 + momentum * 2 + random.uniform(-0.1, 0.1)
        if sentiment_bias > 0.55:
            signals["sentiment"] = {"direction": "bullish", "confidence": min(0.85, sentiment_bias)}
        elif sentiment_bias < 0.45:
            signals["sentiment"] = {"direction": "bearish", "confidence": min(0.85, 1 - sentiment_bias)}
        else:
            signals["sentiment"] = {"direction": "neutral", "confidence": 0.5}
        
        # Liquidity signal (based on volume)
        if vol_trend > 1.2:
            direction = "bullish" if momentum > 0 else "bearish"
            signals["liquidity"] = {"direction": direction, "confidence": min(0.8, 0.5 + vol_trend * 0.2)}
        else:
            signals["liquidity"] = {"direction": "neutral", "confidence": 0.5}
        
        # PENTA confluence (probabilistic based on strength)
        all_bullish = all(s.get("direction") == "bullish" for s in signals.values())
        all_bearish = all(s.get("direction") == "bearish" for s in signals.values())
        avg_confidence = np.mean([s.get("confidence", 0.5) for s in signals.values()])
        
        if all_bullish or all_bearish:
            if avg_confidence > 0.75:
                signals["penta_confluence"] = "PENTA" if random.random() > 0.7 else "QUAD"
            elif avg_confidence > 0.65:
                signals["penta_confluence"] = "QUAD" if random.random() > 0.6 else "TRIPLE"
            elif avg_confidence > 0.55:
                signals["penta_confluence"] = "TRIPLE" if random.random() > 0.5 else None
        
        return signals
    
    def _neutral_signals(self) -> Dict[str, Any]:
        """Return neutral signals."""
        return {
            "hedge": {"direction": "neutral", "confidence": 0.4},
            "sentiment": {"direction": "neutral", "confidence": 0.4},
            "liquidity": {"direction": "neutral", "confidence": 0.4},
            "penta_confluence": None,
        }
    
    def _check_exits(
        self,
        positions: Dict[str, BacktestTrade],
        price_data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> List[BacktestTrade]:
        """Check positions for exit conditions."""
        closed = []
        
        for symbol, trade in list(positions.items()):
            price = self._get_price(symbol, current_date, price_data)
            if price is None:
                continue
            
            should_exit = False
            exit_reason = ""
            
            # Check stop loss
            if trade.direction == "BUY" and price <= trade.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            elif trade.direction == "SELL" and price >= trade.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Check take profit
            if trade.direction == "BUY" and price >= trade.take_profit:
                should_exit = True
                exit_reason = "take_profit"
            elif trade.direction == "SELL" and price <= trade.take_profit:
                should_exit = True
                exit_reason = "take_profit"
            
            # Check max hold time (5 days)
            if trade.entry_date:
                hold_days = (current_date - trade.entry_date).days
                if hold_days >= 5:
                    should_exit = True
                    exit_reason = "max_hold_time"
            
            if should_exit:
                trade.exit_date = current_date
                trade.exit_price = price
                
                if trade.direction == "BUY":
                    trade.pnl = (price - trade.entry_price) * trade.quantity
                    trade.pnl_pct = (price - trade.entry_price) / trade.entry_price
                else:
                    trade.pnl = (trade.entry_price - price) * trade.quantity
                    trade.pnl_pct = (trade.entry_price - price) / trade.entry_price
                
                trade.is_winner = trade.pnl > 0
                trade.exit_reason = exit_reason
                closed.append(trade)
        
        return closed
    
    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        equity_curve: List[Dict],
        trades: List[BacktestTrade],
    ) -> BacktestResults:
        """Calculate all backtest metrics."""
        results = BacktestResults(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            symbols=symbols,
            equity_curve=equity_curve,
            trades=trades,
        )
        
        if not trades:
            return results
        
        # Final capital
        if equity_curve:
            results.final_capital = equity_curve[-1]["equity"]
        
        results.total_return = results.final_capital - results.initial_capital
        results.total_return_pct = results.total_return / results.initial_capital
        
        # Trade statistics
        results.total_trades = len(trades)
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        results.winning_trades = len(winners)
        results.losing_trades = len(losers)
        results.win_rate = len(winners) / len(trades) if trades else 0
        
        if winners:
            results.avg_win = np.mean([t.pnl_pct for t in winners])
        if losers:
            results.avg_loss = np.mean([t.pnl_pct for t in losers])
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk metrics
        if len(equity_curve) > 1:
            equities = [e["equity"] for e in equity_curve]
            returns = np.diff(equities) / equities[:-1]
            
            # Max drawdown
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)
            results.max_drawdown_pct = max_dd
            results.max_drawdown = max_dd * results.initial_capital
            
            # Sharpe ratio (annualized)
            if np.std(returns) > 0:
                results.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                results.sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        
        # PENTA analysis
        penta_wins = 0
        for trade in trades:
            if trade.penta_confluence == "PENTA":
                results.penta_trades += 1
                if trade.is_winner:
                    penta_wins += 1
            elif trade.penta_confluence == "QUAD":
                results.quad_trades += 1
            elif trade.penta_confluence == "TRIPLE":
                results.triple_trades += 1
            else:
                results.regular_trades += 1
        
        if results.penta_trades > 0:
            results.penta_win_rate = penta_wins / results.penta_trades
        
        return results


def run_gnosis_v2_backtest():
    """Run the GNOSIS V2 comprehensive backtest."""
    print("\n" + "="*70)
    print("  GNOSIS V2 ARCHITECTURE - COMPREHENSIVE BACKTEST")
    print("="*70)
    
    # Configuration
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 6, 30, tzinfo=timezone.utc)
    initial_capital = 100_000
    
    print(f"\nConfiguration:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    
    # Create backtester
    backtester = GnosisV2Backtester(
        initial_capital=initial_capital,
        max_positions=5,
        max_position_pct=0.10,
        min_confidence=0.50,
    )
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtester.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Print results
    results.print_summary()
    
    return results


if __name__ == "__main__":
    results = run_gnosis_v2_backtest()
