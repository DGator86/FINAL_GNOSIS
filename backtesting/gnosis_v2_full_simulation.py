#!/usr/bin/env python3
"""
GNOSIS V2 Architecture - Full Factor Simulation Suite

Comprehensive simulations testing ALL factors:
1. PENTA Methodology Components (Wyckoff, ICT, Order Flow, S&D, Liquidity Concepts)
2. Composer Agent Weights (Hedge, Sentiment, Liquidity)
3. Confluence Levels (PENTA, QUAD, TRIPLE, DUAL)
4. Risk Management Parameters (Stop Loss, Take Profit, Position Size)
5. Trading Modes (Standard, 0DTE Express, Cheap Calls)
6. Market Regimes (Trending, Volatile, Range-bound)
7. Symbol Performance
8. Time-based Analysis (Day of Week, Month)

Author: GNOSIS Trading System
Version: 2.0.0
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

# Suppress excessive logging for simulations
logger.remove()
logger.add(sys.stderr, level="WARNING")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SimulationTrade:
    """Single trade in simulation."""
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
    
    # Factors
    penta_confluence: Optional[str] = None
    hedge_signal: str = "neutral"
    sentiment_signal: str = "neutral"
    liquidity_signal: str = "neutral"
    market_regime: str = "neutral"
    
    # PENTA components
    wyckoff_aligned: bool = False
    ict_aligned: bool = False
    order_flow_aligned: bool = False
    supply_demand_aligned: bool = False
    liquidity_concepts_aligned: bool = False
    
    # Results
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_winner: bool = False
    exit_reason: str = ""
    hold_days: int = 0


@dataclass
class SimulationResults:
    """Results from a single simulation run."""
    name: str
    config: Dict[str, Any]
    
    # Performance
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Trades
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    # Details
    trades: List[SimulationTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    VOLATILE = "volatile"
    RANGE_BOUND = "range_bound"
    COMPRESSED = "compressed"


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class FullFactorSimulator:
    """
    Comprehensive simulator for all GNOSIS V2 factors.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize V2 components
        self._init_components()
        
        # Default parameters
        self.initial_capital = 100_000
        self.symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]
        
    def _init_components(self):
        """Initialize GNOSIS V2 components."""
        try:
            from agents.composer.composer_agent_v4 import ComposerAgentV4
            from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
            
            self.ComposerAgentV4 = ComposerAgentV4
            self.AlphaTradeAgentV2 = AlphaTradeAgentV2
            self.components_available = True
        except ImportError as e:
            print(f"Warning: Could not import V2 components: {e}")
            self.components_available = False
    
    def generate_price_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        regime: MarketRegime = MarketRegime.TRENDING_UP,
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic price data with specific market regime."""
        
        price_data = {}
        num_days = (end_date - start_date).days + 60  # Buffer
        dates = pd.date_range(start=start_date - timedelta(days=30), periods=num_days, freq='D')
        
        # Regime parameters
        regime_params = {
            MarketRegime.TRENDING_UP: {"drift": 0.0005, "vol": 0.012},
            MarketRegime.TRENDING_DOWN: {"drift": -0.0003, "vol": 0.014},
            MarketRegime.VOLATILE: {"drift": 0.0001, "vol": 0.025},
            MarketRegime.RANGE_BOUND: {"drift": 0.0, "vol": 0.008},
            MarketRegime.COMPRESSED: {"drift": 0.0, "vol": 0.005},
        }
        
        params = regime_params.get(regime, regime_params[MarketRegime.TRENDING_UP])
        
        base_prices = {
            "SPY": 450, "QQQ": 380, "AAPL": 180, "MSFT": 380,
            "NVDA": 500, "GOOGL": 140, "AMZN": 180, "TSLA": 250,
        }
        
        for symbol in symbols:
            base = base_prices.get(symbol, 100)
            returns = np.random.normal(params["drift"], params["vol"], num_days)
            
            prices = [base]
            for r in returns[1:]:
                prices.append(prices[-1] * (1 + r))
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
    
    def simulate_penta_signals(
        self,
        symbol: str,
        date: datetime,
        price_data: Dict[str, pd.DataFrame],
        penta_weights: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """Generate PENTA methodology signals."""
        
        if penta_weights is None:
            penta_weights = {
                "wyckoff": 0.18,
                "ict": 0.18,
                "order_flow": 0.18,
                "supply_demand": 0.18,
                "liquidity_concepts": 0.18,
            }
        
        df = price_data.get(symbol)
        if df is None or len(df) < 20:
            return self._neutral_penta_signals()
        
        mask = df.index <= date
        recent = df[mask].tail(20)
        if len(recent) < 10:
            return self._neutral_penta_signals()
        
        prices = recent['close'].values
        momentum = (prices[-1] - prices[0]) / prices[0]
        volatility = np.std(np.diff(prices) / prices[:-1])
        volumes = recent['volume'].values
        vol_trend = volumes[-5:].mean() / volumes.mean() if volumes.mean() > 0 else 1.0
        
        # Simulate each PENTA component
        signals = {}
        
        # Wyckoff (accumulation/distribution phases)
        wyckoff_score = momentum * 2 + (vol_trend - 1) * 0.5 + random.uniform(-0.2, 0.2)
        signals["wyckoff"] = {
            "direction": "bullish" if wyckoff_score > 0.1 else "bearish" if wyckoff_score < -0.1 else "neutral",
            "confidence": min(0.9, 0.5 + abs(wyckoff_score)),
            "phase": "markup" if wyckoff_score > 0.15 else "markdown" if wyckoff_score < -0.15 else "ranging",
            "aligned": wyckoff_score > 0.1 or wyckoff_score < -0.1,
        }
        
        # ICT (smart money concepts)
        ict_score = momentum * 1.5 + random.uniform(-0.15, 0.15)
        signals["ict"] = {
            "direction": "bullish" if ict_score > 0.08 else "bearish" if ict_score < -0.08 else "neutral",
            "confidence": min(0.85, 0.5 + abs(ict_score)),
            "bias": "bullish" if ict_score > 0 else "bearish",
            "aligned": ict_score > 0.08 or ict_score < -0.08,
        }
        
        # Order Flow (volume imbalance)
        order_flow_score = (vol_trend - 1) * 2 + momentum + random.uniform(-0.1, 0.1)
        signals["order_flow"] = {
            "direction": "bullish" if order_flow_score > 0.1 else "bearish" if order_flow_score < -0.1 else "neutral",
            "confidence": min(0.85, 0.5 + abs(order_flow_score) * 0.5),
            "imbalance": order_flow_score,
            "aligned": order_flow_score > 0.1 or order_flow_score < -0.1,
        }
        
        # Supply & Demand (zone analysis)
        sd_score = momentum * 1.2 + random.uniform(-0.12, 0.12)
        signals["supply_demand"] = {
            "direction": "bullish" if sd_score > 0.09 else "bearish" if sd_score < -0.09 else "neutral",
            "confidence": min(0.85, 0.5 + abs(sd_score)),
            "zone": "demand" if sd_score > 0 else "supply",
            "aligned": sd_score > 0.09 or sd_score < -0.09,
        }
        
        # Liquidity Concepts (pool analysis)
        lc_score = momentum + (vol_trend - 1) * 0.3 + random.uniform(-0.1, 0.1)
        signals["liquidity_concepts"] = {
            "direction": "bullish" if lc_score > 0.08 else "bearish" if lc_score < -0.08 else "neutral",
            "confidence": min(0.85, 0.5 + abs(lc_score)),
            "trend": "bullish" if lc_score > 0 else "bearish",
            "aligned": lc_score > 0.08 or lc_score < -0.08,
        }
        
        # Calculate confluence
        aligned_count = sum([
            signals["wyckoff"]["aligned"],
            signals["ict"]["aligned"],
            signals["order_flow"]["aligned"],
            signals["supply_demand"]["aligned"],
            signals["liquidity_concepts"]["aligned"],
        ])
        
        # Check direction agreement
        directions = [s["direction"] for s in signals.values() if s["direction"] != "neutral"]
        bullish_count = sum(1 for d in directions if d == "bullish")
        bearish_count = sum(1 for d in directions if d == "bearish")
        
        if aligned_count >= 5 and (bullish_count >= 4 or bearish_count >= 4):
            confluence = "PENTA"
        elif aligned_count >= 4 and (bullish_count >= 3 or bearish_count >= 3):
            confluence = "QUAD"
        elif aligned_count >= 3 and (bullish_count >= 2 or bearish_count >= 2):
            confluence = "TRIPLE"
        elif aligned_count >= 2:
            confluence = "DUAL"
        else:
            confluence = None
        
        signals["confluence"] = confluence
        signals["aligned_count"] = aligned_count
        
        return signals
    
    def _neutral_penta_signals(self) -> Dict[str, Any]:
        """Return neutral PENTA signals."""
        neutral = {"direction": "neutral", "confidence": 0.4, "aligned": False}
        return {
            "wyckoff": {**neutral, "phase": "ranging"},
            "ict": {**neutral, "bias": "neutral"},
            "order_flow": {**neutral, "imbalance": 0},
            "supply_demand": {**neutral, "zone": "neutral"},
            "liquidity_concepts": {**neutral, "trend": "neutral"},
            "confluence": None,
            "aligned_count": 0,
        }
    
    def run_simulation(
        self,
        name: str,
        config: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> SimulationResults:
        """Run a single simulation with given configuration."""
        
        # Extract config
        symbols = config.get("symbols", self.symbols)
        initial_capital = config.get("initial_capital", self.initial_capital)
        max_positions = config.get("max_positions", 5)
        max_position_pct = config.get("max_position_pct", 0.10)
        min_confidence = config.get("min_confidence", 0.50)
        stop_loss_pct = config.get("stop_loss_pct", 0.03)
        take_profit_pct = config.get("take_profit_pct", 0.05)
        max_hold_days = config.get("max_hold_days", 5)
        regime = config.get("regime", MarketRegime.TRENDING_UP)
        
        # Composer weights
        composer_weights = config.get("composer_weights", {
            "hedge": 0.40, "sentiment": 0.40, "liquidity": 0.20
        })
        
        # Generate price data if not provided
        if price_data is None:
            price_data = self.generate_price_data(symbols, start_date, end_date, regime)
        
        # Initialize state
        cash = initial_capital
        positions: Dict[str, SimulationTrade] = {}
        all_trades: List[SimulationTrade] = []
        equity_curve: List[float] = [initial_capital]
        
        # Create composer with custom weights
        if self.components_available:
            composer = self.ComposerAgentV4(weights=composer_weights)
            trade_agent = self.AlphaTradeAgentV2(config={"min_confidence": min_confidence})
        
        # Get trading days
        ref_symbol = symbols[0]
        df = price_data[ref_symbol]
        trading_days = [
            d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d
            for d in df.index
            if start_date <= (d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d) <= end_date
            and d.weekday() < 5
        ]
        
        # Main simulation loop
        for current_date in trading_days:
            # Calculate equity
            position_value = sum(
                self._get_price(t.symbol, current_date, price_data) * t.quantity
                for t in positions.values()
                if self._get_price(t.symbol, current_date, price_data)
            )
            equity = cash + position_value
            equity_curve.append(equity)
            
            # Check exits
            for symbol in list(positions.keys()):
                trade = positions[symbol]
                price = self._get_price(symbol, current_date, price_data)
                if price is None:
                    continue
                
                should_exit = False
                exit_reason = ""
                
                # Stop loss
                if trade.direction == "BUY" and price <= trade.stop_loss:
                    should_exit, exit_reason = True, "stop_loss"
                elif trade.direction == "SELL" and price >= trade.stop_loss:
                    should_exit, exit_reason = True, "stop_loss"
                
                # Take profit
                if trade.direction == "BUY" and price >= trade.take_profit:
                    should_exit, exit_reason = True, "take_profit"
                elif trade.direction == "SELL" and price <= trade.take_profit:
                    should_exit, exit_reason = True, "take_profit"
                
                # Max hold
                hold_days = (current_date - trade.entry_date).days
                if hold_days >= max_hold_days:
                    should_exit, exit_reason = True, "max_hold"
                
                if should_exit:
                    trade.exit_date = current_date
                    trade.exit_price = price
                    trade.hold_days = hold_days
                    
                    if trade.direction == "BUY":
                        trade.pnl = (price - trade.entry_price) * trade.quantity
                        trade.pnl_pct = (price - trade.entry_price) / trade.entry_price
                    else:
                        trade.pnl = (trade.entry_price - price) * trade.quantity
                        trade.pnl_pct = (trade.entry_price - price) / trade.entry_price
                    
                    trade.is_winner = trade.pnl > 0
                    trade.exit_reason = exit_reason
                    cash += trade.exit_price * trade.quantity
                    all_trades.append(trade)
                    del positions[symbol]
            
            # Generate new signals
            if len(positions) < max_positions:
                for symbol in symbols:
                    if symbol in positions or len(positions) >= max_positions:
                        continue
                    
                    price = self._get_price(symbol, current_date, price_data)
                    if price is None:
                        continue
                    
                    # Get PENTA signals
                    penta_signals = self.simulate_penta_signals(symbol, current_date, price_data)
                    
                    # Generate agent signals
                    hedge_signal = self._generate_hedge_signal(symbol, current_date, price_data)
                    sentiment_signal = self._generate_sentiment_signal(symbol, current_date, price_data)
                    liquidity_signal = self._aggregate_penta_to_liquidity(penta_signals)
                    
                    if not self.components_available:
                        continue
                    
                    # Use composer
                    composer_output = composer.compose(
                        hedge_signal=hedge_signal,
                        sentiment_signal=sentiment_signal,
                        liquidity_signal=liquidity_signal,
                        penta_confluence=penta_signals["confluence"],
                    )
                    
                    # Get trade signal
                    signal = trade_agent.process_composer_output(
                        composer_output, symbol=symbol, current_price=price
                    )
                    
                    if signal.direction == "HOLD" or signal.confidence < min_confidence:
                        continue
                    
                    # Position sizing
                    position_value = equity * max_position_pct
                    quantity = int(position_value / price)
                    if quantity <= 0 or price * quantity > cash:
                        continue
                    
                    # Create trade
                    trade = SimulationTrade(
                        symbol=symbol,
                        direction=signal.direction,
                        entry_date=current_date,
                        entry_price=price,
                        quantity=quantity,
                        stop_loss=price * (1 - stop_loss_pct) if signal.direction == "BUY" else price * (1 + stop_loss_pct),
                        take_profit=price * (1 + take_profit_pct) if signal.direction == "BUY" else price * (1 - take_profit_pct),
                        confidence=signal.confidence,
                        penta_confluence=penta_signals["confluence"],
                        hedge_signal=hedge_signal["direction"],
                        sentiment_signal=sentiment_signal["direction"],
                        liquidity_signal=liquidity_signal["direction"],
                        market_regime=regime.value if isinstance(regime, MarketRegime) else str(regime),
                        wyckoff_aligned=penta_signals["wyckoff"]["aligned"],
                        ict_aligned=penta_signals["ict"]["aligned"],
                        order_flow_aligned=penta_signals["order_flow"]["aligned"],
                        supply_demand_aligned=penta_signals["supply_demand"]["aligned"],
                        liquidity_concepts_aligned=penta_signals["liquidity_concepts"]["aligned"],
                    )
                    
                    positions[symbol] = trade
                    cash -= price * quantity
        
        # Close remaining positions
        for symbol, trade in positions.items():
            price = self._get_price(symbol, end_date, price_data)
            if price:
                trade.exit_date = end_date
                trade.exit_price = price
                trade.hold_days = (end_date - trade.entry_date).days
                if trade.direction == "BUY":
                    trade.pnl = (price - trade.entry_price) * trade.quantity
                    trade.pnl_pct = (price - trade.entry_price) / trade.entry_price
                else:
                    trade.pnl = (trade.entry_price - price) * trade.quantity
                    trade.pnl_pct = (trade.entry_price - price) / trade.entry_price
                trade.is_winner = trade.pnl > 0
                trade.exit_reason = "end_of_sim"
                all_trades.append(trade)
        
        # Calculate results
        return self._calculate_results(name, config, all_trades, equity_curve, initial_capital)
    
    def _get_price(self, symbol: str, date: datetime, price_data: Dict[str, pd.DataFrame]) -> Optional[float]:
        """Get price for symbol on date."""
        if symbol not in price_data:
            return None
        df = price_data[symbol]
        try:
            if date in df.index:
                return float(df.loc[date, 'close'])
            idx = df.index.get_indexer([date], method='ffill')[0]
            if 0 <= idx < len(df):
                return float(df['close'].iloc[idx])
        except:
            pass
        return None
    
    def _generate_hedge_signal(self, symbol: str, date: datetime, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate hedge signal based on momentum and volatility."""
        df = price_data.get(symbol)
        if df is None or len(df) < 20:
            return {"direction": "neutral", "confidence": 0.4}
        
        mask = df.index <= date
        recent = df[mask].tail(20)
        if len(recent) < 10:
            return {"direction": "neutral", "confidence": 0.4}
        
        prices = recent['close'].values
        momentum = (prices[-1] - prices[0]) / prices[0]
        vol_ratio = recent['volume'].tail(5).mean() / recent['volume'].mean()
        
        score = momentum * 3 + (vol_ratio - 1) * 0.5 + random.uniform(-0.1, 0.1)
        
        if score > 0.1:
            return {"direction": "bullish", "confidence": min(0.9, 0.5 + score)}
        elif score < -0.1:
            return {"direction": "bearish", "confidence": min(0.9, 0.5 + abs(score))}
        return {"direction": "neutral", "confidence": 0.4}
    
    def _generate_sentiment_signal(self, symbol: str, date: datetime, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate sentiment signal."""
        df = price_data.get(symbol)
        if df is None or len(df) < 10:
            return {"direction": "neutral", "confidence": 0.4}
        
        mask = df.index <= date
        recent = df[mask].tail(10)
        if len(recent) < 5:
            return {"direction": "neutral", "confidence": 0.4}
        
        prices = recent['close'].values
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        score = momentum * 2 + random.uniform(-0.15, 0.15)
        
        if score > 0.08:
            return {"direction": "bullish", "confidence": min(0.85, 0.5 + score)}
        elif score < -0.08:
            return {"direction": "bearish", "confidence": min(0.85, 0.5 + abs(score))}
        return {"direction": "neutral", "confidence": 0.45}
    
    def _aggregate_penta_to_liquidity(self, penta_signals: Dict) -> Dict:
        """Aggregate PENTA signals into liquidity signal."""
        directions = []
        confidences = []
        
        for key in ["wyckoff", "ict", "order_flow", "supply_demand", "liquidity_concepts"]:
            sig = penta_signals.get(key, {})
            if sig.get("direction") != "neutral":
                directions.append(sig.get("direction", "neutral"))
                confidences.append(sig.get("confidence", 0.5))
        
        if not directions:
            return {"direction": "neutral", "confidence": 0.4}
        
        bullish = sum(1 for d in directions if d == "bullish")
        bearish = sum(1 for d in directions if d == "bearish")
        avg_conf = np.mean(confidences) if confidences else 0.5
        
        if bullish > bearish:
            return {"direction": "bullish", "confidence": avg_conf}
        elif bearish > bullish:
            return {"direction": "bearish", "confidence": avg_conf}
        return {"direction": "neutral", "confidence": 0.5}
    
    def _calculate_results(
        self,
        name: str,
        config: Dict,
        trades: List[SimulationTrade],
        equity_curve: List[float],
        initial_capital: float,
    ) -> SimulationResults:
        """Calculate simulation results."""
        results = SimulationResults(name=name, config=config, trades=trades, equity_curve=equity_curve)
        
        if not trades:
            return results
        
        # Performance
        final_equity = equity_curve[-1] if equity_curve else initial_capital
        results.total_return_pct = (final_equity - initial_capital) / initial_capital
        
        # Risk metrics
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if np.std(returns) > 0:
                results.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            downside = returns[returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                results.sortino_ratio = np.mean(returns) / np.std(downside) * np.sqrt(252)
            
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)
            results.max_drawdown_pct = max_dd
        
        # Trade stats
        results.total_trades = len(trades)
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        results.win_rate = len(winners) / len(trades) if trades else 0
        
        if winners:
            results.avg_win_pct = np.mean([t.pnl_pct for t in winners])
        if losers:
            results.avg_loss_pct = np.mean([t.pnl_pct for t in losers])
        
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return results


# ============================================================================
# SIMULATION SUITE
# ============================================================================

def run_full_simulation_suite():
    """Run comprehensive simulation suite testing all factors."""
    
    print("\n" + "="*80)
    print("  GNOSIS V2 ARCHITECTURE - FULL FACTOR SIMULATION SUITE")
    print("="*80)
    
    simulator = FullFactorSimulator(seed=42)
    
    # Common parameters
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 6, 30, tzinfo=timezone.utc)
    
    all_results = {}
    
    # =========================================================================
    # 1. MARKET REGIME SIMULATIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 1: MARKET REGIME ANALYSIS")
    print("-"*80)
    
    regime_results = {}
    for regime in MarketRegime:
        config = {"regime": regime}
        result = simulator.run_simulation(
            name=f"Regime: {regime.value}",
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        regime_results[regime.value] = result
        print(f"  {regime.value:15} | Return: {result.total_return_pct*100:+6.2f}% | "
              f"Win Rate: {result.win_rate*100:5.1f}% | Sharpe: {result.sharpe_ratio:5.2f} | "
              f"Trades: {result.total_trades}")
    
    all_results["regime"] = regime_results
    
    # =========================================================================
    # 2. COMPOSER WEIGHT SIMULATIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 2: COMPOSER WEIGHT ANALYSIS")
    print("-"*80)
    
    weight_configs = [
        {"name": "Default (40/40/20)", "weights": {"hedge": 0.40, "sentiment": 0.40, "liquidity": 0.20}},
        {"name": "Hedge Heavy (60/20/20)", "weights": {"hedge": 0.60, "sentiment": 0.20, "liquidity": 0.20}},
        {"name": "Sentiment Heavy (20/60/20)", "weights": {"hedge": 0.20, "sentiment": 0.60, "liquidity": 0.20}},
        {"name": "Liquidity Heavy (20/20/60)", "weights": {"hedge": 0.20, "sentiment": 0.20, "liquidity": 0.60}},
        {"name": "Equal (33/33/33)", "weights": {"hedge": 0.33, "sentiment": 0.33, "liquidity": 0.34}},
        {"name": "Hedge Only (100/0/0)", "weights": {"hedge": 1.0, "sentiment": 0.0, "liquidity": 0.0}},
    ]
    
    weight_results = {}
    for wc in weight_configs:
        config = {"composer_weights": wc["weights"]}
        result = simulator.run_simulation(
            name=wc["name"],
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        weight_results[wc["name"]] = result
        print(f"  {wc['name']:25} | Return: {result.total_return_pct*100:+6.2f}% | "
              f"Win Rate: {result.win_rate*100:5.1f}% | Sharpe: {result.sharpe_ratio:5.2f}")
    
    all_results["weights"] = weight_results
    
    # =========================================================================
    # 3. CONFIDENCE THRESHOLD SIMULATIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 3: CONFIDENCE THRESHOLD ANALYSIS")
    print("-"*80)
    
    confidence_results = {}
    for conf in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        config = {"min_confidence": conf}
        result = simulator.run_simulation(
            name=f"Confidence >= {conf:.0%}",
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        confidence_results[conf] = result
        print(f"  Min Confidence: {conf:.0%}  | Return: {result.total_return_pct*100:+6.2f}% | "
              f"Win Rate: {result.win_rate*100:5.1f}% | Trades: {result.total_trades:3}")
    
    all_results["confidence"] = confidence_results
    
    # =========================================================================
    # 4. RISK MANAGEMENT SIMULATIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 4: RISK MANAGEMENT ANALYSIS")
    print("-"*80)
    
    risk_configs = [
        {"name": "Tight (2%/4%)", "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        {"name": "Default (3%/5%)", "stop_loss_pct": 0.03, "take_profit_pct": 0.05},
        {"name": "Wide (5%/10%)", "stop_loss_pct": 0.05, "take_profit_pct": 0.10},
        {"name": "Very Wide (8%/15%)", "stop_loss_pct": 0.08, "take_profit_pct": 0.15},
        {"name": "Asymmetric (2%/8%)", "stop_loss_pct": 0.02, "take_profit_pct": 0.08},
    ]
    
    risk_results = {}
    for rc in risk_configs:
        config = {"stop_loss_pct": rc["stop_loss_pct"], "take_profit_pct": rc["take_profit_pct"]}
        result = simulator.run_simulation(
            name=rc["name"],
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        risk_results[rc["name"]] = result
        print(f"  {rc['name']:20} | Return: {result.total_return_pct*100:+6.2f}% | "
              f"Win Rate: {result.win_rate*100:5.1f}% | Max DD: {result.max_drawdown_pct*100:5.2f}%")
    
    all_results["risk"] = risk_results
    
    # =========================================================================
    # 5. POSITION SIZE SIMULATIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 5: POSITION SIZE ANALYSIS")
    print("-"*80)
    
    position_results = {}
    for size in [0.05, 0.10, 0.15, 0.20, 0.25]:
        config = {"max_position_pct": size}
        result = simulator.run_simulation(
            name=f"Position Size: {size:.0%}",
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        position_results[size] = result
        print(f"  Position Size: {size:.0%}  | Return: {result.total_return_pct*100:+6.2f}% | "
              f"Max DD: {result.max_drawdown_pct*100:5.2f}% | Sharpe: {result.sharpe_ratio:5.2f}")
    
    all_results["position_size"] = position_results
    
    # =========================================================================
    # 6. MAX POSITIONS SIMULATIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 6: MAX POSITIONS ANALYSIS")
    print("-"*80)
    
    max_pos_results = {}
    for max_pos in [1, 3, 5, 8, 10]:
        config = {"max_positions": max_pos}
        result = simulator.run_simulation(
            name=f"Max Positions: {max_pos}",
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        max_pos_results[max_pos] = result
        print(f"  Max Positions: {max_pos:2}   | Return: {result.total_return_pct*100:+6.2f}% | "
              f"Trades: {result.total_trades:3} | Sharpe: {result.sharpe_ratio:5.2f}")
    
    all_results["max_positions"] = max_pos_results
    
    # =========================================================================
    # 7. HOLD TIME SIMULATIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 7: MAX HOLD TIME ANALYSIS")
    print("-"*80)
    
    hold_results = {}
    for hold_days in [1, 3, 5, 7, 10, 14]:
        config = {"max_hold_days": hold_days}
        result = simulator.run_simulation(
            name=f"Max Hold: {hold_days} days",
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        hold_results[hold_days] = result
        print(f"  Max Hold: {hold_days:2} days   | Return: {result.total_return_pct*100:+6.2f}% | "
              f"Win Rate: {result.win_rate*100:5.1f}% | Trades: {result.total_trades:3}")
    
    all_results["hold_time"] = hold_results
    
    # =========================================================================
    # 8. PENTA CONFLUENCE ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 8: PENTA CONFLUENCE ANALYSIS")
    print("-"*80)
    
    # Run default simulation and analyze by confluence level
    default_result = simulator.run_simulation(
        name="Default",
        config={},
        start_date=start_date,
        end_date=end_date,
    )
    
    confluence_stats = {
        "PENTA": {"trades": 0, "wins": 0, "total_pnl": 0},
        "QUAD": {"trades": 0, "wins": 0, "total_pnl": 0},
        "TRIPLE": {"trades": 0, "wins": 0, "total_pnl": 0},
        "DUAL": {"trades": 0, "wins": 0, "total_pnl": 0},
        "NONE": {"trades": 0, "wins": 0, "total_pnl": 0},
    }
    
    for trade in default_result.trades:
        level = trade.penta_confluence or "NONE"
        confluence_stats[level]["trades"] += 1
        if trade.is_winner:
            confluence_stats[level]["wins"] += 1
        confluence_stats[level]["total_pnl"] += trade.pnl
    
    for level, stats in confluence_stats.items():
        if stats["trades"] > 0:
            win_rate = stats["wins"] / stats["trades"]
            avg_pnl = stats["total_pnl"] / stats["trades"]
            print(f"  {level:6} | Trades: {stats['trades']:3} | Win Rate: {win_rate*100:5.1f}% | Avg P&L: ${avg_pnl:+8.2f}")
    
    all_results["confluence"] = confluence_stats
    
    # =========================================================================
    # 9. PENTA COMPONENT ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 9: PENTA COMPONENT CONTRIBUTION")
    print("-"*80)
    
    component_stats = {
        "wyckoff": {"aligned_wins": 0, "aligned_total": 0, "unaligned_wins": 0, "unaligned_total": 0},
        "ict": {"aligned_wins": 0, "aligned_total": 0, "unaligned_wins": 0, "unaligned_total": 0},
        "order_flow": {"aligned_wins": 0, "aligned_total": 0, "unaligned_wins": 0, "unaligned_total": 0},
        "supply_demand": {"aligned_wins": 0, "aligned_total": 0, "unaligned_wins": 0, "unaligned_total": 0},
        "liquidity_concepts": {"aligned_wins": 0, "aligned_total": 0, "unaligned_wins": 0, "unaligned_total": 0},
    }
    
    for trade in default_result.trades:
        for comp, attr in [
            ("wyckoff", "wyckoff_aligned"),
            ("ict", "ict_aligned"),
            ("order_flow", "order_flow_aligned"),
            ("supply_demand", "supply_demand_aligned"),
            ("liquidity_concepts", "liquidity_concepts_aligned"),
        ]:
            if getattr(trade, attr, False):
                component_stats[comp]["aligned_total"] += 1
                if trade.is_winner:
                    component_stats[comp]["aligned_wins"] += 1
            else:
                component_stats[comp]["unaligned_total"] += 1
                if trade.is_winner:
                    component_stats[comp]["unaligned_wins"] += 1
    
    for comp, stats in component_stats.items():
        aligned_wr = stats["aligned_wins"] / max(1, stats["aligned_total"])
        unaligned_wr = stats["unaligned_wins"] / max(1, stats["unaligned_total"])
        edge = aligned_wr - unaligned_wr
        print(f"  {comp:18} | Aligned WR: {aligned_wr*100:5.1f}% | Unaligned WR: {unaligned_wr*100:5.1f}% | Edge: {edge*100:+5.1f}%")
    
    all_results["components"] = component_stats
    
    # =========================================================================
    # 10. SYMBOL PERFORMANCE
    # =========================================================================
    print("\n" + "-"*80)
    print("  SIMULATION 10: SYMBOL PERFORMANCE ANALYSIS")
    print("-"*80)
    
    symbol_stats = {}
    for symbol in simulator.symbols:
        symbol_stats[symbol] = {"trades": 0, "wins": 0, "total_pnl": 0}
    
    for trade in default_result.trades:
        symbol_stats[trade.symbol]["trades"] += 1
        if trade.is_winner:
            symbol_stats[trade.symbol]["wins"] += 1
        symbol_stats[trade.symbol]["total_pnl"] += trade.pnl
    
    for symbol, stats in sorted(symbol_stats.items(), key=lambda x: -x[1]["total_pnl"]):
        if stats["trades"] > 0:
            win_rate = stats["wins"] / stats["trades"]
            print(f"  {symbol:5} | Trades: {stats['trades']:3} | Win Rate: {win_rate*100:5.1f}% | P&L: ${stats['total_pnl']:+10.2f}")
    
    all_results["symbols"] = symbol_stats
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("  SIMULATION SUMMARY")
    print("="*80)
    
    # Find best configurations
    print("\n  BEST CONFIGURATIONS:")
    
    # Best regime
    best_regime = max(regime_results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"    Best Regime: {best_regime[0]} (Sharpe: {best_regime[1].sharpe_ratio:.2f})")
    
    # Best weights
    best_weights = max(weight_results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"    Best Weights: {best_weights[0]} (Sharpe: {best_weights[1].sharpe_ratio:.2f})")
    
    # Best confidence
    best_conf = max(confidence_results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"    Best Confidence: {best_conf[0]:.0%} (Sharpe: {best_conf[1].sharpe_ratio:.2f})")
    
    # Best risk
    best_risk = max(risk_results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"    Best Risk Params: {best_risk[0]} (Sharpe: {best_risk[1].sharpe_ratio:.2f})")
    
    # Best position size
    best_pos = max(position_results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"    Best Position Size: {best_pos[0]:.0%} (Sharpe: {best_pos[1].sharpe_ratio:.2f})")
    
    print("\n" + "="*80)
    print("  FULL FACTOR SIMULATION COMPLETE")
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = run_full_simulation_suite()
