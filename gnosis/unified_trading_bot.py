"""Unified Trading Bot - single bot managing multiple symbols from dynamic universe."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Set, Any
from loguru import logger

from alpaca.data.live import StockDataStream

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
from execution.broker_adapters.alpaca_options_adapter import AlpacaOptionsAdapter
from gnosis.timeframe_manager import TimeframeManager
from gnosis.dynamic_universe_manager import UniverseUpdate
from agents.hedge_agent_v4 import HedgeAgentV4
from agents.liquidity_agent_v2 import LiquidityAgentV2
from agents.sentiment_agent_v2 import SentimentAgentV2
from agents.composer.composer_agent_v2 import ComposerAgentV2
from trade.trade_agent_router import TradeAgentRouter
from schemas.core_schemas import OptionsOrderRequest, Position


@dataclass
class SymbolData:
    symbol: str
    timeframe_mgr: TimeframeManager


class UnifiedTradingBot:
    """
    Unified trading bot that manages multiple symbols, handles both equity and options,
    and uses a router for strategy generation.
    """

    def __init__(self, config: dict, enable_trading: bool = False, paper_mode: bool = True):
        self.config = config
        self.enable_trading = enable_trading
        self.paper_mode = paper_mode

        # State
        self.positions: Dict[str, Position] = {}
        self.active_strategies: Dict[str, Any] = {}
        self.active_symbols: Set[str] = set()
        self.symbol_data: Dict[str, SymbolData] = {}
        self.running = False

        # Risk Management
        self.risk_per_trade_pct = config.get("risk", {}).get("risk_per_trade_pct", 0.02)
        self.max_positions = config.get("risk", {}).get("max_positions", 5)

        # Circuit breaker state
        self.daily_start_value = 0.0
        self.daily_loss_limit_pct = config.get("risk", {}).get("daily_loss_limit", 0.05)
        self.circuit_breaker_triggered = False
        self.trailing_stop_pct = config.get("risk", {}).get("trailing_stop_pct", 0.01)
        self.trailing_stop_activation = config.get("risk", {}).get("trailing_stop_activation", 0.02)

        # Initialize adapters
        self.adapter = AlpacaBrokerAdapter(paper=paper_mode)
        self.options_adapter = AlpacaOptionsAdapter(paper=paper_mode)

        # Initialize Agents
        self.trade_agent = TradeAgentRouter(config=config, options_adapter=self.options_adapter)
        self.hedge_agent = HedgeAgentV4()
        self.liquidity_agent = LiquidityAgentV2()
        self.sentiment_agent = SentimentAgentV2()
        self.composer_agent = ComposerAgentV2()

        # Initialize Data Stream
        self.stream = StockDataStream(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))

    async def add_symbol(self, symbol: str):
        """Add a symbol to monitor and trade."""
        if symbol in self.active_symbols:
            return

        logger.info(f"Adding {symbol} to UnifiedTradingBot")
        self.active_symbols.add(symbol)

        # Initialize timeframe manager for this symbol
        tf_mgr = TimeframeManager(symbol)
        self.symbol_data[symbol] = SymbolData(symbol=symbol, timeframe_mgr=tf_mgr)

        # Subscribe to bars
        try:
            self.stream.subscribe_bars(self._handle_bar, symbol)
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")

    async def update_universe(self, update: UniverseUpdate):
        """Update the active universe."""
        # Remove symbols
        for symbol in update.removed:
            if symbol in self.active_symbols:
                logger.info(f"Removing {symbol} from active universe")
                self.active_symbols.remove(symbol)
                if symbol in self.symbol_data:
                    del self.symbol_data[symbol]

        # Add symbols
        for symbol in update.added:
            await self.add_symbol(symbol)

    async def _handle_bar(self, bar):
        """Handle incoming bar data."""
        symbol = bar.symbol
        if symbol not in self.symbol_data:
            return

        # Update timeframe manager
        self.symbol_data[symbol].timeframe_mgr.update(bar)

        # Run analysis
        await self.analyze_and_trade(symbol, bar.close)

        # Manage existing positions
        if symbol in self.positions:
            await self.manage_position(symbol, bar.close)

        # Check circuit breaker
        await self.check_circuit_breaker()

    async def analyze_and_trade(self, symbol: str, current_price: float):
        """Analyze market and generate trade signals."""
        # Get account info for capital
        try:
            account = self.adapter.get_account()
            # available_capital = float(account.cash) # Unused
        except Exception:
            pass
            # available_capital = 100000.0

        # Check max positions
        if len(self.positions) >= self.max_positions:
            return

        # Generate Strategy using Router
        # Note: In a real scenario, we would get signals from other agents first.
        # For now, we'll assume the router can generate a strategy if we ask it.
        # But the router expects a 'composer_decision'.

        # Placeholder for composer decision
        # We need to create a dummy decision or integrate with composer
        # For now, let's skip if we don't have a signal
        pass

    async def open_position(self, symbol: str, strategy, current_price: float) -> None:
        """Open a new position."""

        # Get account info for sizing
        try:
            account = self.adapter.get_account()
            # buying_power = float(account.buying_power) # Unused
            equity = float(account.equity)
        except Exception:
            # buying_power = 100000.0
            equity = 100000.0

        if isinstance(strategy, OptionsOrderRequest):
            logger.info(f"Opening OPTIONS strategy: {strategy.strategy_name} for {symbol}")

            # Calculate Quantity based on Risk
            # Risk amount = Equity * Risk %
            risk_amount = equity * self.risk_per_trade_pct

            # Cost basis per contract (BPR or Max Loss)
            # Use BPR as the primary cost constraint, or max_loss if BPR is 0 (e.g. long options)
            cost_per_unit = strategy.bpr if strategy.bpr > 0 else strategy.max_loss

            # If cost is still 0 or undefined, fallback to a safe default or 1
            if cost_per_unit <= 0:
                cost_per_unit = 500.0  # Fallback assumption $500 per trade

            # Calculate quantity
            raw_quantity = risk_amount / cost_per_unit
            quantity = max(1, int(raw_quantity))

            # Cap quantity at 10 for safety during testing
            quantity = min(quantity, 10)

            logger.info(
                f"Sizing: Risk ${risk_amount:.2f} / Cost ${cost_per_unit:.2f} = {raw_quantity:.2f} -> {quantity} contracts"
            )

            if self.enable_trading and self.options_adapter:
                legs_payload = [
                    {"symbol": leg.symbol, "side": leg.side, "qty": int(leg.ratio * quantity)}
                    for leg in strategy.legs
                ]

                await self.options_adapter.place_multileg_order(
                    legs=legs_payload, note=strategy.strategy_name
                )
            else:
                logger.info(f"DRY RUN: Would execute {strategy.strategy_name} x {quantity}")

            # Track position
            # For options, we track the primary leg or the strategy as a whole
            primary_leg = strategy.legs[0]
            pos = Position(
                symbol=symbol,
                side="long",  # Strategies are generally "long" the strategy itself (even if net short premium)
                size=strategy.bpr * quantity,
                entry_price=current_price,
                entry_time=datetime.now(),
                asset_class="option_strategy",
                option_symbol=primary_leg.symbol,
                quantity=quantity,
            )
            self.positions[symbol] = pos
            self.active_strategies[symbol] = strategy
            return

        # Equity Logic
        logger.info(f"Opening {strategy.direction} position in {symbol} ({strategy.asset_class})")

        trade_symbol = symbol
        if strategy.asset_class == "option" and strategy.option_symbol:
            trade_symbol = strategy.option_symbol

        if self.enable_trading:
            side = "buy" if strategy.direction == "LONG" else "sell"
            self.adapter.place_order(trade_symbol, strategy.quantity, side=side)
        else:
            logger.info(f"DRY RUN: Would open {strategy.direction} {trade_symbol}")

        pos = Position(
            symbol=symbol,
            side=strategy.direction.lower(),
            size=strategy.quantity * current_price,
            entry_price=strategy.entry_price,
            entry_time=datetime.now(),
            highest_price=current_price,
            stop_loss_price=strategy.stop_loss_price,
            take_profit_price=strategy.take_profit_price,
            trailing_stop_price=strategy.stop_loss_price,
            trailing_stop_active=False,
            asset_class=strategy.asset_class,
            option_symbol=strategy.option_symbol if strategy.asset_class == "option" else None,
            quantity=strategy.quantity,
        )

        self.positions[symbol] = pos
        self.active_strategies[symbol] = strategy

    async def manage_position(self, symbol: str, current_price: float) -> None:
        """Manage existing position with risk checks."""
        pos = self.positions[symbol]

        # Skip management for complex option strategies for now (handled by expiration/manual)
        if pos.asset_class == "option_strategy":
            return

        pos.update_highest_price(current_price)

        if await self.check_stop_loss(pos, current_price):
            return
        if await self.check_take_profit(pos, current_price):
            return
        if await self.check_trailing_stop(pos, current_price):
            return

        self.update_trailing_stop(pos, current_price)

    async def close_position(self, symbol: str, price: float, reason: str) -> None:
        """Close a position."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        logger.info(f"Closing {pos.side} {symbol} @ {price:.2f} | Reason: {reason}")

        if self.enable_trading:
            # Handle Option Strategy Closing
            if pos.asset_class == "option_strategy" and symbol in self.active_strategies:
                strategy = self.active_strategies[symbol]
                if isinstance(strategy, OptionsOrderRequest):
                    # Invert legs to close
                    close_legs = []
                    for leg in strategy.legs:
                        close_side = "sell" if leg.side == "buy" else "buy"
                        close_legs.append(
                            {
                                "symbol": leg.symbol,
                                "side": close_side,
                                "qty": int(leg.ratio * pos.quantity),
                            }
                        )

                    if self.options_adapter:
                        await self.options_adapter.place_multileg_order(
                            legs=close_legs, note=f"Closing {strategy.strategy_name}"
                        )
            else:
                # Equity closing
                qty = round(pos.size / max(price, 1e-6), 2)
                side = "sell" if pos.side == "long" else "buy"
                self.adapter.place_order(symbol, qty, side=side)

        del self.positions[symbol]
        if symbol in self.active_strategies:
            del self.active_strategies[symbol]

    async def check_stop_loss(self, pos: Position, current_price: float) -> bool:
        triggered = (pos.side == "long" and current_price <= pos.stop_loss_price) or (
            pos.side == "short" and current_price >= pos.stop_loss_price
        )
        if triggered:
            await self.close_position(pos.symbol, current_price, reason="Stop-loss")
            return True
        return False

    async def check_take_profit(self, pos: Position, current_price: float) -> bool:
        triggered = (pos.side == "long" and current_price >= pos.take_profit_price) or (
            pos.side == "short" and current_price <= pos.take_profit_price
        )
        if triggered:
            await self.close_position(pos.symbol, current_price, reason="Take-profit")
            return True
        return False

    def update_trailing_stop(self, pos: Position, current_price: float) -> None:
        if pos.side == "long":
            gain_pct = (current_price - pos.entry_price) / pos.entry_price
            if not pos.trailing_stop_active and gain_pct >= self.trailing_stop_activation:
                pos.trailing_stop_active = True
            if pos.trailing_stop_active:
                new_stop = current_price * (1 - self.trailing_stop_pct)
                if new_stop > pos.trailing_stop_price:
                    pos.trailing_stop_price = new_stop
        elif pos.side == "short":
            gain_pct = (pos.entry_price - current_price) / pos.entry_price
            if not pos.trailing_stop_active and gain_pct >= self.trailing_stop_activation:
                pos.trailing_stop_active = True
            if pos.trailing_stop_active:
                new_stop = current_price * (1 + self.trailing_stop_pct)
                if new_stop < pos.trailing_stop_price:
                    pos.trailing_stop_price = new_stop

    async def check_trailing_stop(self, pos: Position, current_price: float) -> bool:
        if not pos.trailing_stop_active:
            return False
        triggered = (pos.side == "long" and current_price <= pos.trailing_stop_price) or (
            pos.side == "short" and current_price >= pos.trailing_stop_price
        )
        if triggered:
            await self.close_position(pos.symbol, current_price, reason="Trailing stop")
            return True
        return False

    async def check_circuit_breaker(self) -> None:
        if self.daily_start_value == 0.0:
            try:
                account = self.adapter.get_account()
                self.daily_start_value = float(account.portfolio_value)
            except Exception:
                pass
            return

        try:
            account = self.adapter.get_account()
            current_value = float(account.portfolio_value)
            daily_pnl_pct = (current_value - self.daily_start_value) / self.daily_start_value

            if daily_pnl_pct <= -self.daily_loss_limit_pct and not self.circuit_breaker_triggered:
                self.circuit_breaker_triggered = True
                logger.error(f"⛔ CIRCUIT BREAKER | Daily loss {daily_pnl_pct * 100:.2f}%")
                for symbol in list(self.positions.keys()):
                    # We need current price to close.
                    # Simplified: use 0 or last known price if available
                    await self.close_position(symbol, 0.0, reason="Circuit breaker")
        except Exception:
            pass

    async def run(self):
        """Run the trading bot."""
        self.running = True
        logger.info("Starting UnifiedTradingBot stream...")
        await self.stream.run()

    async def stop(self):
        """Stop the trading bot."""
        self.running = False
        if self.stream:
            await self.stream.stop()
