"""
Options-Enhanced Backtest Engine

Extends the Elite Backtest Engine with full historical options data integration:
- Real historical options chains from Polygon.io
- Synthetic fallback when real data unavailable
- Options strategy backtesting (calls, puts, spreads)
- Greeks-based position management
- Dealer positioning signals

Author: GNOSIS Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import json

import numpy as np
import pandas as pd
from loguru import logger

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.historical_options_manager import (
    HistoricalOptionsManager,
    HistoricalOptionsChain,
    HistoricalOptionContract,
    HistoricalOptionsConfig,
)
from backtesting.elite_backtest_engine import (
    EliteBacktestEngine,
    EliteBacktestConfig,
    EliteBacktestResults,
    SimulatedTrade,
    AssetType,
    HistoricalSnapshotGenerator,
)
from schemas.core_schemas import (
    DirectionEnum,
    PipelineResult,
    TradeIdea,
    StrategyType,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OptionsBacktestConfig(EliteBacktestConfig):
    """Extended configuration for options backtesting."""
    
    # Options data settings
    use_real_options_data: bool = True  # Try Polygon first
    options_dte_min: int = 7
    options_dte_max: int = 45
    options_strike_range_pct: float = 0.15
    
    # Options strategy settings
    enable_options_trades: bool = True
    preferred_delta_calls: float = 0.40  # 40 delta calls
    preferred_delta_puts: float = -0.40  # 40 delta puts
    max_iv_percentile: float = 0.70  # Avoid buying expensive options
    min_open_interest: int = 100  # Minimum OI for liquidity
    
    # Options cost modeling
    option_commission_per_contract: float = 0.65
    option_slippage_bps: float = 50.0  # Options have wider spreads
    
    # Greeks-based signals
    use_dealer_positioning: bool = True
    gamma_flip_weight: float = 0.2
    max_pain_weight: float = 0.15
    
    # Options caching
    cache_options_data: bool = True
    options_cache_dir: str = "cache/options_backtest"


@dataclass
class OptionsTradeResult:
    """Result of an options trade."""
    
    # Contract details
    contract_symbol: str = ""
    underlying: str = ""
    strike: float = 0.0
    expiration: datetime = None
    option_type: str = ""  # "call" or "put"
    
    # Entry
    entry_date: datetime = None
    entry_price: float = 0.0
    entry_delta: float = 0.0
    entry_iv: float = 0.0
    contracts: int = 1
    
    # Exit
    exit_date: datetime = None
    exit_price: float = 0.0
    exit_delta: float = 0.0
    exit_reason: str = ""
    
    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    commission: float = 0.0
    
    # Greeks at entry
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Context
    spot_at_entry: float = 0.0
    spot_at_exit: float = 0.0
    days_held: int = 0


# ============================================================================
# OPTIONS-ENHANCED BACKTEST ENGINE
# ============================================================================

class OptionsBacktestEngine(EliteBacktestEngine):
    """
    Options-Enhanced Backtest Engine.
    
    Extends EliteBacktestEngine with:
    - Historical options data integration (Polygon.io + synthetic)
    - Options strategy execution
    - Greeks-based signals
    - Dealer positioning analysis
    
    Usage:
        config = OptionsBacktestConfig(
            symbols=["SPY", "QQQ"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            use_real_options_data=True,
        )
        
        engine = OptionsBacktestEngine(config)
        results = engine.run_backtest()
    """
    
    def __init__(self, config: OptionsBacktestConfig):
        # Initialize parent
        super().__init__(config)
        
        self.options_config = config
        
        # Initialize options manager
        options_manager_config = HistoricalOptionsConfig(
            providers=["polygon", "synthetic"] if config.use_real_options_data else ["synthetic"],
            cache_enabled=config.cache_options_data,
            cache_dir=config.options_cache_dir,
            default_dte_min=config.options_dte_min,
            default_dte_max=config.options_dte_max,
            default_strike_range_pct=config.options_strike_range_pct,
        )
        self.options_manager = HistoricalOptionsManager(options_manager_config)
        
        # Options-specific state
        self.options_positions: Dict[str, OptionsTradeResult] = {}
        self.options_trades: List[OptionsTradeResult] = []
        self.options_chains_cache: Dict[str, HistoricalOptionsChain] = {}
        
        logger.info(
            f"OptionsBacktestEngine initialized | "
            f"providers={self.options_manager.get_available_providers()}"
        )
    
    def _get_options_chain(
        self,
        symbol: str,
        as_of_date: date,
        spot_price: float,
    ) -> Optional[HistoricalOptionsChain]:
        """Get options chain for a symbol and date."""
        
        cache_key = f"{symbol}_{as_of_date}"
        
        if cache_key in self.options_chains_cache:
            return self.options_chains_cache[cache_key]
        
        chain = self.options_manager.get_chain(
            underlying=symbol,
            as_of_date=as_of_date,
            spot_price=spot_price,
            dte_min=self.options_config.options_dte_min,
            dte_max=self.options_config.options_dte_max,
            strike_range_pct=self.options_config.options_strike_range_pct,
        )
        
        if chain:
            self.options_chains_cache[cache_key] = chain
        
        return chain
    
    def _find_target_option(
        self,
        chain: HistoricalOptionsChain,
        option_type: str,
        target_delta: float,
        min_dte: int = 7,
        max_dte: int = 45,
    ) -> Optional[HistoricalOptionContract]:
        """Find option contract closest to target delta."""
        
        contracts = chain.calls if option_type == "call" else chain.puts
        
        # Filter by DTE and OI
        valid_contracts = [
            c for c in contracts
            if min_dte <= c.days_to_expiry <= max_dte
            and c.open_interest >= self.options_config.min_open_interest
            and c.mid > 0
        ]
        
        if not valid_contracts:
            return None
        
        # Sort by delta proximity to target
        target_delta_abs = abs(target_delta)
        valid_contracts.sort(key=lambda c: abs(abs(c.delta) - target_delta_abs))
        
        # Filter by IV if available
        if self.options_config.max_iv_percentile < 1.0:
            ivs = [c.implied_volatility for c in valid_contracts if c.implied_volatility > 0]
            if ivs:
                iv_threshold = np.percentile(ivs, self.options_config.max_iv_percentile * 100)
                valid_contracts = [
                    c for c in valid_contracts
                    if c.implied_volatility <= iv_threshold or c.implied_volatility == 0
                ]
        
        return valid_contracts[0] if valid_contracts else None
    
    def _build_pipeline_result_with_options(
        self,
        symbol: str,
        bar: Dict[str, Any],
        history: pd.DataFrame,
        timestamp: datetime,
        options_chain: Optional[HistoricalOptionsChain] = None,
    ) -> PipelineResult:
        """Build pipeline result with options data enhancement."""
        
        # Get base pipeline result
        pipeline = self._build_pipeline_result(symbol, bar, history, timestamp)
        
        # Enhance with options data if available
        if options_chain and self.options_config.use_dealer_positioning:
            self._enhance_with_options_data(pipeline, options_chain, bar['close'])
        
        return pipeline
    
    def _enhance_with_options_data(
        self,
        pipeline: PipelineResult,
        chain: HistoricalOptionsChain,
        spot_price: float,
    ):
        """Enhance pipeline signals with options data."""
        
        consensus = pipeline.consensus or {}
        
        # Max pain signal
        if chain.max_pain > 0:
            max_pain_distance = (chain.max_pain - spot_price) / spot_price
            max_pain_signal = np.clip(max_pain_distance * 10, -1, 1)
            consensus["max_pain_signal"] = max_pain_signal
            consensus["max_pain"] = chain.max_pain
        
        # Gamma flip signal
        if chain.gamma_flip > 0:
            gamma_flip_distance = (chain.gamma_flip - spot_price) / spot_price
            gamma_flip_signal = np.clip(gamma_flip_distance * 5, -1, 1)
            consensus["gamma_flip_signal"] = gamma_flip_signal
            consensus["gamma_flip"] = chain.gamma_flip
        
        # Dealer positioning
        consensus["dealer_positioning"] = chain.dealer_positioning
        consensus["net_gamma_exposure"] = chain.net_gamma_exposure
        
        # Put/Call ratio signal
        if chain.put_call_ratio > 0:
            # High P/C = bearish sentiment (contrarian bullish)
            # Low P/C = bullish sentiment (contrarian bearish)
            pc_signal = np.clip((chain.put_call_ratio - 1.0) * 0.5, -1, 1)
            consensus["put_call_signal"] = pc_signal
            consensus["put_call_ratio"] = chain.put_call_ratio
        
        # Walls
        consensus["call_wall"] = chain.call_wall
        consensus["put_wall"] = chain.put_wall
        
        # Adjust overall consensus with options signals
        if self.options_config.use_dealer_positioning:
            base_score = consensus.get("score", 0)
            
            # Add options signal components
            options_adjustment = 0.0
            
            if "max_pain_signal" in consensus:
                options_adjustment += consensus["max_pain_signal"] * self.options_config.max_pain_weight
            
            if "gamma_flip_signal" in consensus:
                options_adjustment += consensus["gamma_flip_signal"] * self.options_config.gamma_flip_weight
            
            # Dealer positioning adjustment
            if chain.dealer_positioning == "short_gamma":
                # Short gamma = dealers will buy dips, sell rips (mean reverting)
                options_adjustment *= 0.8  # Reduce directional signals
            elif chain.dealer_positioning == "long_gamma":
                # Long gamma = dealers will amplify moves
                options_adjustment *= 1.2
            
            consensus["score"] = base_score + options_adjustment
            consensus["options_adjustment"] = options_adjustment
        
        pipeline.consensus = consensus
    
    def _open_options_position(
        self,
        symbol: str,
        chain: HistoricalOptionsChain,
        direction: str,
        timestamp: datetime,
        spot_price: float,
        confidence: float,
    ) -> Optional[OptionsTradeResult]:
        """Open an options position."""
        
        # Determine option type based on direction
        if direction in ("bullish", "long"):
            option_type = "call"
            target_delta = self.options_config.preferred_delta_calls
        else:
            option_type = "put"
            target_delta = self.options_config.preferred_delta_puts
        
        # Find target contract
        contract = self._find_target_option(
            chain,
            option_type,
            target_delta,
            self.options_config.options_dte_min,
            self.options_config.options_dte_max,
        )
        
        if not contract:
            return None
        
        # Calculate position size
        risk_per_trade = self.capital * self.config.max_position_pct
        contract_cost = contract.ask * 100  # Per contract
        
        if contract_cost <= 0:
            return None
        
        num_contracts = max(1, int(risk_per_trade / contract_cost))
        total_cost = num_contracts * contract_cost
        
        if total_cost > self.capital * 0.25:  # Max 25% in single trade
            num_contracts = max(1, int(self.capital * 0.25 / contract_cost))
            total_cost = num_contracts * contract_cost
        
        # Apply costs
        commission = num_contracts * self.options_config.option_commission_per_contract
        slippage = total_cost * self.options_config.option_slippage_bps / 10000
        entry_price = contract.ask * (1 + self.options_config.option_slippage_bps / 10000)
        
        # Deduct capital
        self.capital -= total_cost + commission + slippage
        
        # Create trade record
        trade = OptionsTradeResult(
            contract_symbol=contract.symbol,
            underlying=symbol,
            strike=contract.strike,
            expiration=contract.expiration,
            option_type=option_type,
            entry_date=timestamp,
            entry_price=entry_price,
            entry_delta=contract.delta,
            entry_iv=contract.implied_volatility,
            contracts=num_contracts,
            gamma=contract.gamma,
            theta=contract.theta,
            vega=contract.vega,
            spot_at_entry=spot_price,
            commission=commission,
        )
        
        # Track position
        self.options_positions[contract.symbol] = trade
        
        logger.debug(
            f"Opened {option_type} {symbol} ${contract.strike} exp {contract.expiration.date()} | "
            f"contracts={num_contracts} @ ${entry_price:.2f} | delta={contract.delta:.2f}"
        )
        
        return trade
    
    def _check_options_exit(
        self,
        position: OptionsTradeResult,
        current_chain: Optional[HistoricalOptionsChain],
        spot_price: float,
        timestamp: datetime,
    ) -> Tuple[bool, str, float]:
        """Check if options position should be closed."""
        
        # Find current contract in chain
        current_contract = None
        if current_chain:
            for c in current_chain.contracts:
                if (c.strike == position.strike and 
                    c.option_type == position.option_type and
                    c.expiration == position.expiration):
                    current_contract = c
                    break
        
        # Calculate current value
        if current_contract:
            current_price = current_contract.mid
            current_delta = current_contract.delta
        else:
            # Estimate value from spot movement
            if position.option_type == "call":
                intrinsic = max(0, spot_price - position.strike)
            else:
                intrinsic = max(0, position.strike - spot_price)
            
            # Add some time value estimate
            time_decay = position.theta * (timestamp - position.entry_date).days
            current_price = max(0.01, position.entry_price + time_decay + (intrinsic - max(0, position.spot_at_entry - position.strike if position.option_type == "call" else position.strike - position.spot_at_entry)))
            current_delta = position.entry_delta * 0.9  # Rough estimate
        
        # Calculate P&L
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # Exit conditions
        
        # 1. Near expiration (2 days before)
        days_to_exp = (position.expiration - timestamp).days if isinstance(timestamp, datetime) else 0
        if days_to_exp <= 2:
            return True, "near_expiration", current_price
        
        # 2. Profit target (50%+)
        if pnl_pct >= 0.50:
            return True, "profit_target", current_price
        
        # 3. Stop loss (-50%)
        if pnl_pct <= -0.50:
            return True, "stop_loss", current_price
        
        # 4. Delta too low (option lost most of its value)
        if current_delta and abs(current_delta) < 0.10:
            return True, "low_delta", current_price
        
        # 5. Max hold time
        days_held = (timestamp - position.entry_date).days if isinstance(timestamp, datetime) else 0
        if days_held >= 30:
            return True, "max_hold_time", current_price
        
        return False, "", current_price
    
    def _close_options_position(
        self,
        contract_symbol: str,
        exit_price: float,
        exit_reason: str,
        timestamp: datetime,
        spot_price: float,
    ) -> OptionsTradeResult:
        """Close an options position."""
        
        if contract_symbol not in self.options_positions:
            return None
        
        position = self.options_positions.pop(contract_symbol)
        
        # Apply exit costs
        exit_value = exit_price * position.contracts * 100
        commission = position.contracts * self.options_config.option_commission_per_contract
        slippage = exit_value * self.options_config.option_slippage_bps / 10000
        
        actual_exit_price = exit_price * (1 - self.options_config.option_slippage_bps / 10000)
        
        # Calculate P&L
        entry_value = position.entry_price * position.contracts * 100
        exit_value_net = actual_exit_price * position.contracts * 100
        
        gross_pnl = exit_value_net - entry_value
        total_commission = position.commission + commission
        net_pnl = gross_pnl - total_commission - slippage
        
        # Update position
        position.exit_date = timestamp
        position.exit_price = actual_exit_price
        position.exit_reason = exit_reason
        position.gross_pnl = gross_pnl
        position.net_pnl = net_pnl
        position.commission = total_commission
        position.spot_at_exit = spot_price
        position.days_held = (timestamp - position.entry_date).days if isinstance(timestamp, datetime) else 0
        
        # Return capital
        self.capital += exit_value_net - commission - slippage
        
        # Record trade
        self.options_trades.append(position)
        
        logger.debug(
            f"Closed {position.option_type} {position.underlying} ${position.strike} | "
            f"P&L=${net_pnl:.2f} ({net_pnl/entry_value*100:.1f}%) | {exit_reason}"
        )
        
        return position
    
    def run_backtest(self) -> EliteBacktestResults:
        """Run the full backtest with options integration."""
        
        # Fetch price data for all symbols
        symbol_data: Dict[str, pd.DataFrame] = {}
        for symbol in self.config.symbols:
            try:
                symbol_data[symbol] = self.fetch_historical_data(symbol)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        if not symbol_data:
            raise ValueError("No data fetched for any symbol")
        
        # Get common date range
        all_dates = set()
        for df in symbol_data.values():
            all_dates.update(df['timestamp'].tolist())
        all_dates = sorted(all_dates)
        
        if len(all_dates) < 50:
            raise ValueError(f"Insufficient data: {len(all_dates)} bars")
        
        logger.info(
            f"Running options backtest on {len(all_dates)} bars for {len(symbol_data)} symbols"
        )
        
        # Reset state
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.options_positions = {}
        self.options_trades = []
        self.options_chains_cache = {}
        
        warmup = 50
        
        # Process each bar
        for i in range(warmup, len(all_dates)):
            timestamp = all_dates[i]
            current_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
            
            # Get current bars and history for each symbol
            bar_data = {}
            for symbol, df in symbol_data.items():
                mask = df['timestamp'] <= timestamp
                if mask.sum() > 0:
                    current_df = df[mask]
                    bar_data[symbol] = {
                        'bar': current_df.iloc[-1].to_dict(),
                        'history': current_df,
                    }
            
            # Get options chains (weekly for efficiency)
            options_chains = {}
            if self.options_config.enable_options_trades and i % 5 == 0:
                for symbol in bar_data:
                    spot = bar_data[symbol]['bar']['close']
                    chain = self._get_options_chain(symbol, current_date, spot)
                    if chain:
                        options_chains[symbol] = chain
            
            # Check options position exits
            for contract_symbol in list(self.options_positions.keys()):
                position = self.options_positions[contract_symbol]
                symbol = position.underlying
                
                if symbol in bar_data:
                    spot = bar_data[symbol]['bar']['close']
                    chain = options_chains.get(symbol)
                    
                    should_exit, exit_reason, exit_price = self._check_options_exit(
                        position, chain, spot, timestamp
                    )
                    
                    if should_exit:
                        self._close_options_position(
                            contract_symbol, exit_price, exit_reason, timestamp, spot
                        )
            
            # Check equity position exits (from parent)
            for symbol in list(self.positions.keys()):
                if symbol in bar_data:
                    position = self.positions[symbol]
                    current_price = bar_data[symbol]['bar']['close']
                    
                    chain = options_chains.get(symbol)
                    pipeline = self._build_pipeline_result_with_options(
                        symbol,
                        bar_data[symbol]['bar'],
                        bar_data[symbol]['history'],
                        timestamp,
                        chain,
                    )
                    
                    should_exit, exit_reason = self._check_position_exit(
                        position, current_price, timestamp, pipeline.consensus
                    )
                    
                    if should_exit:
                        self._close_position(symbol, current_price, timestamp, exit_reason)
            
            # Check for new entries
            for symbol in symbol_data.keys():
                if symbol in bar_data:
                    bar = bar_data[symbol]['bar']
                    history = bar_data[symbol]['history']
                    spot = bar['close']
                    
                    chain = options_chains.get(symbol)
                    
                    # Build enhanced pipeline
                    pipeline = self._build_pipeline_result_with_options(
                        symbol, bar, history, timestamp, chain
                    )
                    
                    consensus = pipeline.consensus or {}
                    direction = consensus.get("direction", "neutral")
                    confidence = consensus.get("confidence", 0)
                    
                    # Skip if not confident enough
                    if direction == "neutral" or confidence < self.config.min_confidence:
                        continue
                    
                    # Entry logic
                    already_in_equity = symbol in self.positions
                    already_in_options = any(
                        p.underlying == symbol for p in self.options_positions.values()
                    )
                    
                    # Prefer options if available and enabled
                    if (self.options_config.enable_options_trades and 
                        chain and 
                        not already_in_options and
                        len(self.options_positions) < self.config.max_positions):
                        
                        self._open_options_position(
                            symbol, chain, direction, timestamp, spot, confidence
                        )
                    
                    # Fallback to equity if not using options
                    elif (not already_in_equity and 
                          len(self.positions) < self.config.max_positions):
                        
                        dir_enum = DirectionEnum.LONG if direction == "bullish" else DirectionEnum.SHORT
                        idea = TradeIdea(
                            symbol=symbol,
                            direction=dir_enum,
                            strategy_type=StrategyType.DIRECTIONAL,
                            confidence=confidence,
                            timestamp=timestamp,
                        )
                        self._open_position(symbol, idea, spot, timestamp, pipeline)
            
            # Record equity
            current_bars = {s: d['bar'] for s, d in bar_data.items()}
            self._record_equity_with_options(timestamp, current_bars)
        
        # Close remaining positions
        final_timestamp = all_dates[-1]
        
        for contract_symbol in list(self.options_positions.keys()):
            position = self.options_positions[contract_symbol]
            if position.underlying in bar_data:
                spot = bar_data[position.underlying]['bar']['close']
                # Estimate final price
                if position.option_type == "call":
                    final_price = max(0.01, spot - position.strike)
                else:
                    final_price = max(0.01, position.strike - spot)
                self._close_options_position(
                    contract_symbol, final_price, "end_of_test", final_timestamp, spot
                )
        
        for symbol in list(self.positions.keys()):
            if symbol in bar_data:
                current_price = bar_data[symbol]['bar']['close']
                self._close_position(symbol, current_price, final_timestamp, "end_of_test")
        
        # Calculate results
        results = self._calculate_results_with_options()
        
        # Monte Carlo
        if self.config.monte_carlo_runs > 0 and len(self.trades) + len(self.options_trades) > 10:
            self._run_monte_carlo(results)
        
        # Save results
        if self.config.save_trades or self.config.save_equity_curve:
            self._save_results_with_options(results)
        
        return results
    
    def _record_equity_with_options(
        self,
        timestamp: datetime,
        bar_data: Dict[str, Dict[str, Any]],
    ):
        """Record equity including options positions."""
        
        # Equity positions value
        equity_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in bar_data:
                current_price = bar_data[symbol]['close']
                if position.direction == "long":
                    unrealized = (current_price - position.entry_price) * position.position_size
                else:
                    unrealized = (position.entry_price - current_price) * position.position_size
                equity_value += position.entry_price * position.position_size + unrealized
        
        # Options positions value (simplified)
        options_value = 0.0
        for position in self.options_positions.values():
            # Rough estimate - would need current option prices for accuracy
            entry_value = position.entry_price * position.contracts * 100
            options_value += entry_value * 0.95  # Assume slight decay
        
        total_equity = self.capital + equity_value + options_value
        
        self.equity_curve.append({
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'equity': total_equity,
            'capital': self.capital,
            'equity_position_value': equity_value,
            'options_position_value': options_value,
            'num_equity_positions': len(self.positions),
            'num_options_positions': len(self.options_positions),
        })
    
    def _calculate_results_with_options(self) -> EliteBacktestResults:
        """Calculate results including options trades."""
        
        # Get base results
        results = self._calculate_results()
        
        # Add options-specific metrics
        if self.options_trades:
            options_winners = [t for t in self.options_trades if t.net_pnl > 0]
            options_losers = [t for t in self.options_trades if t.net_pnl <= 0]
            
            options_metrics = {
                'total_options_trades': len(self.options_trades),
                'options_winners': len(options_winners),
                'options_losers': len(options_losers),
                'options_win_rate': len(options_winners) / len(self.options_trades),
                'options_avg_win': np.mean([t.net_pnl for t in options_winners]) if options_winners else 0,
                'options_avg_loss': np.mean([abs(t.net_pnl) for t in options_losers]) if options_losers else 0,
                'options_total_pnl': sum(t.net_pnl for t in self.options_trades),
                'options_by_type': {
                    'calls': len([t for t in self.options_trades if t.option_type == "call"]),
                    'puts': len([t for t in self.options_trades if t.option_type == "put"]),
                },
                'avg_days_held': np.mean([t.days_held for t in self.options_trades]),
            }
            
            # Add to results
            results.strategy_returns['options'] = options_metrics['options_total_pnl']
            
            # Store in a way that can be accessed
            if not hasattr(results, 'options_metrics'):
                results.options_metrics = options_metrics
        
        return results
    
    def _save_results_with_options(self, results: EliteBacktestResults):
        """Save results including options trades."""
        
        # Save base results
        self._save_results(results)
        
        # Save options trades
        if self.options_trades:
            output_dir = Path(self.config.output_dir)
            tag = self.config.tag or f"options_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            options_data = []
            for t in self.options_trades:
                options_data.append({
                    'contract_symbol': t.contract_symbol,
                    'underlying': t.underlying,
                    'strike': t.strike,
                    'expiration': str(t.expiration.date()) if t.expiration else "",
                    'option_type': t.option_type,
                    'entry_date': str(t.entry_date),
                    'exit_date': str(t.exit_date),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'contracts': t.contracts,
                    'net_pnl': t.net_pnl,
                    'exit_reason': t.exit_reason,
                    'days_held': t.days_held,
                    'entry_delta': t.entry_delta,
                    'entry_iv': t.entry_iv,
                })
            
            options_path = output_dir / f"{tag}_options_trades.json"
            with open(options_path, 'w') as f:
                json.dump(options_data, f, indent=2)
            
            logger.info(f"Options trades saved to {options_path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_options_backtest(
    symbols: List[str] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 100_000.0,
    use_real_options: bool = True,
    **kwargs,
) -> EliteBacktestResults:
    """
    Run options-enhanced backtest.
    
    Args:
        symbols: List of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        use_real_options: Use Polygon.io for real data
        **kwargs: Additional config parameters
    
    Returns:
        Backtest results
    """
    if symbols is None:
        symbols = ["SPY"]
    
    config = OptionsBacktestConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        use_real_options_data=use_real_options,
        tag=f"options_{'-'.join(symbols)}_{start_date}_{end_date}",
        **kwargs,
    )
    
    engine = OptionsBacktestEngine(config)
    return engine.run_backtest()


def print_options_results(results: EliteBacktestResults):
    """Print formatted results including options metrics."""
    
    from backtesting.elite_backtest_engine import print_elite_results
    print_elite_results(results)
    
    # Print options-specific metrics
    if hasattr(results, 'options_metrics'):
        om = results.options_metrics
        print("\n" + "-"*70)
        print("OPTIONS TRADING SUMMARY")
        print(f"  Total Options Trades: {om['total_options_trades']}")
        print(f"  Options Win Rate:     {om['options_win_rate']*100:.1f}%")
        print(f"  Options Avg Win:      ${om['options_avg_win']:,.2f}")
        print(f"  Options Avg Loss:     ${om['options_avg_loss']:,.2f}")
        print(f"  Options Total P&L:    ${om['options_total_pnl']:,.2f}")
        print(f"  Calls Traded:         {om['options_by_type']['calls']}")
        print(f"  Puts Traded:          {om['options_by_type']['puts']}")
        print(f"  Avg Days Held:        {om['avg_days_held']:.1f}")
        print("-"*70)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  OPTIONS BACKTEST ENGINE - DEMO")
    print("="*70)
    
    # Check if Polygon API key is set
    from config.credentials import polygon_api_available
    
    if polygon_api_available():
        print("\n✅ Polygon.io API key detected - will use real historical options data")
    else:
        print("\n⚠️  No Polygon.io API key - will use synthetic options data")
        print("   Set POLYGON_API_KEY env var for real data")
    
    print("\nRunning short demo backtest...")
    
    try:
        results = run_options_backtest(
            symbols=["SPY"],
            start_date="2023-06-01",
            end_date="2023-09-01",
            initial_capital=100_000,
            use_real_options=polygon_api_available(),
        )
        
        print_options_results(results)
        
    except Exception as e:
        print(f"\n❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70 + "\n")
