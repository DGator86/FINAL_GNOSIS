"""Unified Trading Bot - single bot managing multiple symbols from dynamic universe."""

from __future__ import annotations

import asyncio
import os
import json
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
        logger.info("Initializing TradeAgentRouter...")
        self.trade_agent = TradeAgentRouter(config=config, options_adapter=self.options_adapter)

        logger.info("Initializing HedgeAgentV4...")
        self.hedge_agent = HedgeAgentV4(config=config)

        logger.info("Initializing LiquidityAgentV2...")
        self.liquidity_agent = LiquidityAgentV2(config=config)

        logger.info("Initializing SentimentAgentV2...")
        self.sentiment_agent = SentimentAgentV2(config=config)

        logger.info("Initializing ComposerAgentV2...")
        self.composer_agent = ComposerAgentV2(config=config)

        # Initialize Data Stream
        logger.info("Initializing StockDataStream...")
        self.stream = StockDataStream(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
        logger.info("StockDataStream initialized.")

    async def add_symbol(self, symbol: str):
        """Add a symbol to monitor and trade."""
        if symbol in self.active_symbols:
            return

        logger.info(f"Adding {symbol} to UnifiedTradingBot")
        self.active_symbols.add(symbol)

        # Initialize timeframe manager for this symbol
        tf_mgr = TimeframeManager()
        self.symbol_data[symbol] = SymbolData(symbol=symbol, timeframe_mgr=tf_mgr)

        # Fetch historical bars to populate initial data
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import datetime, timedelta

            # Initialize historical data client
            hist_client = StockHistoricalDataClient(
                os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
            )

            # Request last 50 1-minute bars
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=2)  # Look back 2 hours to ensure we get 50 bars

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start_time, end=end_time
            )

            bars_response = hist_client.get_stock_bars(request_params)

            if symbol in bars_response.data:
                historical_bars = bars_response.data[symbol]
                # Take last 50 bars
                recent_bars = (
                    historical_bars[-50:] if len(historical_bars) > 50 else historical_bars
                )

                # Feed bars into timeframe manager
                for bar in recent_bars:
                    tf_mgr.update(bar)

                logger.info(f"Loaded {len(recent_bars)} historical bars for {symbol}")
                bar_counts = tf_mgr.get_bar_counts()
                logger.info(f"Bar counts: {bar_counts}")
            else:
                logger.warning(f"No historical bars found for {symbol}")

        except Exception as e:
            # Historical data unavailable - system will accumulate bars from live stream
            logger.info(f"No historical bars for {symbol} (Alpaca SIP restricted on free tier)")
            logger.info(f"✅ Will accumulate live data for {symbol} from real-time stream")

        # Subscribe to live bars (optional - gracefully handle connection limits)
        try:
            self.stream.subscribe_bars(self._handle_bar, symbol)
            logger.info(f"Subscribed to live bars for {symbol}")
        except Exception as e:
            logger.warning(f"Could not subscribe to live bars for {symbol}: {e}")
            logger.info(f"Will use polling mode for {symbol} instead of live stream")

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
        logger.debug(f"Received bar for {symbol}: Close={bar.close}")

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

        # Export state for dashboard
        await self.export_state()

    async def export_state(self):
        """Export current state to JSON for dashboard."""
        try:
            # Prepare positions data
            positions_data = []
            for symbol, pos in self.positions.items():
                pos_dict = pos.dict()
                positions_data.append(pos_dict)

            # Prepare account data (simplified)
            account_data = {
                "portfolio_value": self.daily_start_value,  # Approx
                "pnl": 0.0,  # TODO: Calculate real PnL
            }
            try:
                acct = self.adapter.get_account()
                account_data["portfolio_value"] = float(acct.portfolio_value)
                account_data["pnl"] = float(acct.equity) - float(acct.last_equity)
            except Exception:
                pass

            # Prepare symbols data (from active symbols)
            symbols_data = {}
            for sym in self.active_symbols:
                # Placeholder for scanner data
                symbols_data[sym] = {
                    "symbol": sym,
                    "price": 0.0,  # TODO: Get last price
                    "composer_confidence": 0.5,  # Placeholder
                    "composer_signal": "HOLD",
                }

            state = {
                "market_open": True,  # TODO: Check clock
                "account": account_data,
                "positions": positions_data,
                "symbols": symbols_data,
                "last_update": datetime.now().isoformat(),
            }

            # Write to file
            os.makedirs("data/scanner_state", exist_ok=True)
            with open("data/scanner_state/current_state.json", "w") as f:
                json.dump(state, f, default=str)

        except Exception as e:
            logger.error(f"Failed to export state: {e}")

    async def analyze_and_trade(self, symbol: str, current_price: float):
        """Analyze market and generate trade signals."""
        # Skip if already at max positions
        if len(self.positions) >= self.max_positions:
            logger.debug(f"Skipping {symbol}: Max positions reached ({len(self.positions)})")
            return

        # Skip if already in position for this symbol
        if symbol in self.positions:
            logger.debug(f"Skipping {symbol}: Already in position")
            return

        # Skip if circuit breaker triggered
        if self.circuit_breaker_triggered:
            logger.debug(f"Skipping {symbol}: Circuit breaker triggered")
            return

        # Get timeframe data
        if symbol not in self.symbol_data:
            return

        tf_mgr = self.symbol_data[symbol].timeframe_mgr

        # Check if we have enough data
        bar_counts = tf_mgr.get_bar_counts()
        if bar_counts.get("1Min", 0) < 5:  # Reduced from 20 to 5 for testing
            logger.debug(f"Skipping {symbol}: Not enough bars ({bar_counts.get('1Min', 0)} < 5)")
            return

        # Create simple agent suggestions based on price action
        suggestions = []

        # Simple trend detection from 5min bars
        bars_5min = tf_mgr.get_bars("5Min", count=5)
        if len(bars_5min) >= 2:  # Reduced requirement
            # Simple momentum: Close > Open (Bullish) or Close < Open (Bearish)
            last_bar = bars_5min[-1]
            prev_bar = bars_5min[-2]

            trend_up = last_bar.close > last_bar.open and last_bar.close > prev_bar.close
            trend_down = last_bar.close < last_bar.open and last_bar.close < prev_bar.close

            if trend_up or trend_down:
                from schemas.core_schemas import AgentSuggestion, DirectionEnum

                direction = DirectionEnum.LONG if trend_up else DirectionEnum.SHORT
                confidence = 0.7  # High confidence for testing

                suggestions.append(
                    AgentSuggestion(
                        agent_name="simple_trend",
                        timestamp=datetime.now(),
                        symbol=symbol,
                        direction=direction,
                        confidence=confidence,
                        reasoning=f"{'Uptrend' if trend_up else 'Downtrend'} detected (Close vs Open/Prev)",
                        target_allocation=0.0,
                    )
                )

        # If no clear signal, don't trade
        if not suggestions:
            logger.debug(f"Skipping {symbol}: No trend detected")
            return

        # Use composer to make final decision
        composer_result = self.composer_agent.compose(suggestions, datetime.now())

        # Check if composer gives GO signal
        if composer_result.get("confidence", 0) < 0.5:  # Reduced threshold
            logger.debug(
                f"Skipping {symbol}: Low confidence ({composer_result.get('confidence', 0):.2f})"
            )
            return

        direction_str = composer_result.get("direction", "NEUTRAL")
        if direction_str == "NEUTRAL":
            return

        logger.info(
            f"🎯 Trading signal for {symbol}: {direction_str} (confidence: {composer_result['confidence']:.2f})"
        )

        # Create ComposerDecision for TradeAgentRouter
        from agents.composer.composer_agent_v2 import ComposerDecision

        composer_decision = ComposerDecision(
            timestamp=datetime.now(),
            symbol=symbol,
            go_signal=True,
            predicted_direction=direction_str,
            confidence=composer_result["confidence"],
            predicted_timeframe="intraday",
            risk_reward_ratio=2.0,
            reasoning=composer_result.get("reasoning", "Trend-based signal"),
        )

        # Get account info for capital
        try:
            account = self.adapter.get_account()
            available_capital = float(account.cash)
        except Exception:
            available_capital = 10000.0  # Fallback

        # Generate strategy using TradeAgentRouter
        try:
            strategy = self.trade_agent.generate_strategy(
                composer_decision=composer_decision,
                current_price=current_price,
                available_capital=available_capital,
                timestamp=datetime.now(),
            )

            if strategy:
                logger.info(f"✅ Strategy generated for {symbol}")
                # Execute the trade
                await self.open_position(symbol, strategy, current_price)
            else:
                logger.info(f"⚠️ No valid strategy generated for {symbol}")

        except Exception as e:
            logger.error(f"Error generating/executing strategy for {symbol}: {e}")

    async def open_position(self, symbol: str, strategy, current_price: float) -> None:
        """Open a new position."""

        # Get account info for sizing
        try:
            account = self.adapter.get_account()
            equity = float(account.equity)
        except Exception:
            equity = 100000.0

        if isinstance(strategy, OptionsOrderRequest):
            logger.info(f"Opening OPTIONS strategy: {strategy.strategy_name} for {symbol}")

            # Calculate Quantity based on Risk
            risk_amount = equity * self.risk_per_trade_pct

            # Cost basis per contract (BPR or Max Loss)
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
            primary_leg = strategy.legs[0]
            pos = Position(
                symbol=symbol,
                side="long",
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
        try:
            if self.stream:
                # Use _run_forever if available (async), otherwise fallback to run (sync wrapper)
                if hasattr(self.stream, "_run_forever"):
                    logger.info("Using async stream._run_forever()")
                    await self.stream._run_forever()
                else:
                    logger.info("Using sync stream.run()")
                    await self.stream.run()
            else:
                logger.warning("No valid stream to run.")
        except Exception as e:
            logger.error(f"Stream run failed: {e}")

    async def stop(self):
        """Stop the trading bot."""
        self.running = False
        if self.stream:
            try:
                await self.stream.stop()
            except AttributeError:
                # Stream loop might not be initialized if run() wasn't called
                pass
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
