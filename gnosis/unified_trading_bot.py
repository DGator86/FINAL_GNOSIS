"""Unified Trading Bot - single bot managing multiple symbols from dynamic universe."""

from __future__ import annotations

import asyncio
import contextlib
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Set, Any
from loguru import logger

from alpaca.data.live import StockDataStream

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
from execution.broker_adapters.alpaca_options_adapter import AlpacaOptionsAdapter
from execution.broker_adapters.settings import get_required_options_level
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

    Risk Management:
    - Equities: Broker-level bracket orders (stop loss & take profit) + manual trailing stops
    - Single-Leg Options: Manual stop loss & take profit monitoring based on premium price
    - Multi-Leg Options: Manual P&L-based stop loss & take profit (percentage of max risk)

    Scale-Out Strategy (Tiered Take Profits):
    - Level 1: 33% at 1.5R → Tighten trail to 1.5% → Move stop to breakeven
    - Level 2: 33% at 3.0R → Tighten trail to 1.0%
    - Level 3: 34% at 5.0R → Tighten trail to 0.5% → Let winners run

    After each scale-out, the trailing stop tightens, and stop loss moves up to protect profits.
    This allows taking partial profits while letting the best trades run for maximum gains.

    All positions are actively monitored and automatically closed when risk thresholds are hit.
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
        self.stopping = False
        self.stream_task: asyncio.Task | None = None

        # Risk Management
        self.risk_per_trade_pct = config.get("risk", {}).get("risk_per_trade_pct", 0.02)
        self.max_positions = config.get("risk", {}).get("max_positions", 5)

        # Circuit breaker state
        self.daily_start_value = 0.0
        self.daily_loss_limit_pct = config.get("risk", {}).get("daily_loss_limit", 0.05)
        self.circuit_breaker_triggered = False
        self.trailing_stop_pct = config.get("risk", {}).get("trailing_stop_pct", 0.01)
        self.trailing_stop_activation = config.get("risk", {}).get("trailing_stop_activation", 0.02)

        # Scale-Out Configuration (tiered take profits with trailing stops)
        # Each level: (pct_of_position, reward_multiple, trailing_stop_pct_after)
        # reward_multiple is in terms of risk (R): 1.5R = 1.5x the initial risk
        self.enable_scale_outs = config.get("risk", {}).get("enable_scale_outs", True)
        self.scale_out_levels = config.get("risk", {}).get(
            "scale_out_levels",
            [
                {"pct": 0.33, "reward_multiple": 1.5, "trailing_stop_pct": 0.015},  # 33% at 1.5R, 1.5% trail
                {"pct": 0.33, "reward_multiple": 3.0, "trailing_stop_pct": 0.01},   # 33% at 3R, 1% trail
                {"pct": 0.34, "reward_multiple": 5.0, "trailing_stop_pct": 0.005},  # 34% at 5R, 0.5% trail
            ],
        )

        # Initialize adapters
        self.adapter = AlpacaBrokerAdapter(paper=paper_mode)

        options_config_enabled = config.get("enable_options", True)
        self.options_adapter = None
        options_enabled = False
        account_level = None

        try:
            account = self.adapter.get_account()
            account_level = getattr(account, "options_trading_level", None)
            approved_level = getattr(account, "options_approved_level", None)
            required_level = get_required_options_level()

            if not options_config_enabled:
                options_reason = "Config enable_options=False"
            elif account_level is None:
                options_reason = "Alpaca account missing options_trading_level"
            elif account_level < required_level:
                options_reason = (
                    f"Alpaca options_trading_level={account_level} below required {required_level}"
                )
            else:
                options_enabled = True
                self.options_adapter = AlpacaOptionsAdapter(paper=paper_mode)
                options_reason = ""

            if options_enabled:
                logger.info(
                    "Options enabled: Alpaca level %s (approved=%s, required=%s)",
                    account_level,
                    approved_level,
                    required_level,
                )
            else:
                logger.warning(f"Options disabled: {options_reason}")
        except Exception as exc:
            logger.warning(f"Options disabled due to account check failure: {exc}")
            options_enabled = False
            self.options_adapter = None

        # Initialize Agents
        logger.info("Initializing TradeAgentRouter...")
        router_config = dict(config)
        router_config["enable_options"] = options_enabled and options_config_enabled
        self.trade_agent = TradeAgentRouter(config=router_config, options_adapter=self.options_adapter)

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
        if self.stopping:
            logger.warning(f"Ignoring add_symbol({symbol}) during shutdown")
            return
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
            logger.error(f"Failed to fetch historical bars for {symbol}: {e}")

        # Subscribe to live bars
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
            available_capital = float(account.equity)
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
        """Open a new position.

        For equity positions, uses bracket orders to set stop loss and take profit at broker level.
        This ensures positions are protected even if the bot crashes or loses connection.
        Trailing stops are still managed manually via manage_position().
        """

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

            # Calculate stop loss and take profit for multi-leg strategy
            # Use percentage of max risk (BPR or max_loss)
            stop_loss_pct = 0.50  # 50% loss on max risk
            take_profit_pct = 0.30  # 30% gain on max risk (conservative for spreads)

            entry_cost = strategy.bpr * quantity if strategy.bpr > 0 else strategy.max_loss * quantity
            stop_loss_value = entry_cost * (1 + stop_loss_pct)  # Loss threshold (cost + additional loss)
            take_profit_value = entry_cost * (1 - take_profit_pct)  # Profit threshold (cost - profit)

            # Track position
            primary_leg = strategy.legs[0]
            pos = Position(
                symbol=symbol,
                side="long",
                size=entry_cost,
                entry_price=entry_cost,  # Use cost as "entry price" for P&L tracking
                entry_time=datetime.now(),
                stop_loss_price=stop_loss_value,
                take_profit_price=take_profit_value,
                asset_class="option_strategy",
                option_symbol=primary_leg.symbol,
                quantity=quantity,
                original_quantity=quantity,
                remaining_quantity=quantity,
                scale_out_completed=0,
            )
            self.positions[symbol] = pos
            self.active_strategies[symbol] = strategy

            logger.info(
                f"Multi-leg strategy tracking: Entry=${entry_cost:.2f}, "
                f"SL=${stop_loss_value:.2f}, TP=${take_profit_value:.2f}"
            )
            return

        # Single-Leg Options Logic (Long Calls/Puts)
        if strategy.asset_class == "option":
            logger.info(f"Opening {strategy.direction} OPTION position: {strategy.option_symbol}")

            if self.enable_trading:
                side = "buy" if strategy.direction == "LONG" else "sell"
                # Options cannot use bracket orders - must use market orders and manual monitoring
                self.adapter.place_order(
                    symbol=strategy.option_symbol,
                    quantity=strategy.quantity,
                    side=side,
                    order_type="market",
                    time_in_force="gtc",
                )
            else:
                logger.info(
                    f"DRY RUN: Would open {strategy.direction} {strategy.option_symbol} "
                    f"| Entry: ${strategy.entry_price:.2f} | SL: ${strategy.stop_loss_price:.2f} | TP: ${strategy.take_profit_price:.2f}"
                )

            pos = Position(
                symbol=symbol,
                side=strategy.direction.lower(),
                size=strategy.quantity * strategy.entry_price * 100,  # Options are per 100 shares
                entry_price=strategy.entry_price,
                entry_time=datetime.now(),
                highest_price=strategy.entry_price,
                stop_loss_price=strategy.stop_loss_price,
                take_profit_price=strategy.take_profit_price,
                trailing_stop_price=None,  # Not using trailing for options
                trailing_stop_active=False,
                asset_class="option",
                option_symbol=strategy.option_symbol,
                quantity=strategy.quantity,
                original_quantity=strategy.quantity,
                remaining_quantity=strategy.quantity,
                scale_out_completed=0,
            )
            self.positions[symbol] = pos
            self.active_strategies[symbol] = strategy
            return

        # Equity Logic (Stocks/ETFs)
        logger.info(f"Opening {strategy.direction} EQUITY position in {symbol}")

        if self.enable_trading:
            side = "buy" if strategy.direction == "LONG" else "sell"
            # Use bracket orders to automatically set stop loss and take profit at broker level
            self.adapter.place_bracket_order(
                symbol=symbol,
                quantity=strategy.quantity,
                side=side,
                take_profit_price=strategy.take_profit_price,
                stop_loss_price=strategy.stop_loss_price,
                time_in_force="gtc",
            )
        else:
            logger.info(
                f"DRY RUN: Would open {strategy.direction} {symbol} "
                f"| SL: ${strategy.stop_loss_price:.2f} | TP: ${strategy.take_profit_price:.2f}"
            )

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
            original_quantity=strategy.quantity,
            remaining_quantity=strategy.quantity,
            scale_out_completed=0,
        )

        self.positions[symbol] = pos
        self.active_strategies[symbol] = strategy

    def get_option_price(self, option_symbol: str) -> Optional[float]:
        """Fetch current option price (mid-price) for monitoring."""
        try:
            price = self.adapter._fetch_option_price(option_symbol)
            if price and price > 0:
                return price
            logger.warning(f"Could not fetch valid price for option {option_symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching option price for {option_symbol}: {e}")
            return None

    async def manage_position(self, symbol: str, current_price: float) -> None:
        """Manage existing position with risk checks.

        For equities: Stop loss and take profit are handled by broker-level bracket orders.
                     Only trailing stops are managed manually.
        For options: All risk management (SL/TP) is handled manually via price monitoring.
        """
        pos = self.positions[symbol]

        # Options (single-leg) - monitor option premium price for stop loss and take profit
        if pos.asset_class == "option":
            if not pos.option_symbol:
                logger.warning(f"Option position {symbol} missing option_symbol")
                return

            # Fetch current option price (premium)
            option_price = self.get_option_price(pos.option_symbol)
            if option_price is None:
                logger.debug(f"Skipping option management for {symbol} - no price available")
                return

            logger.debug(f"Managing option {pos.option_symbol}: Entry=${pos.entry_price:.2f}, Current=${option_price:.2f}")

            # Check for scale-out opportunities on option premium
            if await self.check_scale_outs(pos, option_price):
                if symbol not in self.positions:
                    return  # Position fully closed
                pos = self.positions[symbol]

            # Check stop loss on option premium
            if await self.check_option_stop_loss(pos, option_price):
                return

            # Check take profit on option premium (full close if not scaling out)
            if not self.enable_scale_outs:
                if await self.check_option_take_profit(pos, option_price):
                    return

            return

        # Multi-leg option strategies - monitor P&L based on strategy value
        if pos.asset_class == "option_strategy":
            if symbol not in self.active_strategies:
                logger.warning(f"No active strategy found for {symbol}")
                return

            strategy = self.active_strategies[symbol]
            if not isinstance(strategy, OptionsOrderRequest):
                return

            # Calculate current value of all legs
            current_value = 0.0
            legs_priced = 0

            for leg in strategy.legs:
                leg_price = self.get_option_price(leg.symbol)
                if leg_price is not None:
                    # Each leg contributes to value: buy legs are negative (cost), sell legs are positive (credit)
                    leg_multiplier = -1 if leg.side == "buy" else 1
                    leg_value = leg_price * 100 * (leg.ratio * pos.quantity) * leg_multiplier
                    current_value += leg_value
                    legs_priced += 1

            if legs_priced < len(strategy.legs):
                logger.debug(f"Could not price all legs for {symbol} ({legs_priced}/{len(strategy.legs)} priced)")
                return

            # Current cost = entry cost + current value (negative value = loss, positive = gain)
            current_cost = pos.entry_price + current_value

            logger.debug(
                f"Managing multi-leg {symbol}: Entry=${pos.entry_price:.2f}, "
                f"Current=${current_cost:.2f}, SL=${pos.stop_loss_price:.2f}, TP=${pos.take_profit_price:.2f}"
            )

            # Check stop loss (current cost exceeds stop loss threshold)
            if current_cost >= pos.stop_loss_price:
                logger.warning(
                    f"Multi-leg STOP LOSS triggered for {symbol}: "
                    f"Cost=${current_cost:.2f} >= SL=${pos.stop_loss_price:.2f}"
                )
                await self.close_position(symbol, current_cost, reason="Multi-Leg Stop-Loss")
                return

            # Check take profit (current cost below take profit threshold = profit taken)
            if current_cost <= pos.take_profit_price:
                logger.info(
                    f"Multi-leg TAKE PROFIT triggered for {symbol}: "
                    f"Cost=${current_cost:.2f} <= TP=${pos.take_profit_price:.2f}"
                )
                await self.close_position(symbol, current_cost, reason="Multi-Leg Take-Profit")
                return

            return

        # Equities - manage scale-outs and trailing stops (broker handles initial SL/TP)
        pos.update_highest_price(current_price)

        # Check for scale-out opportunities (tiered take profits)
        if await self.check_scale_outs(pos, current_price):
            # Position may have been closed completely during scale-out
            if symbol not in self.positions:
                return
            # Continue to manage remaining position
            pos = self.positions[symbol]

        # Check trailing stop
        if await self.check_trailing_stop(pos, current_price):
            return

        # Update trailing stop level
        self.update_trailing_stop(pos, current_price)

    async def close_position(self, symbol: str, price: float, reason: str) -> None:
        """Close a position."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        logger.info(f"Closing {pos.side} {symbol} @ {price:.2f} | Reason: {reason}")

        if self.enable_trading:
            # Handle Multi-Leg Option Strategy Closing
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
            # Handle Single-Leg Option Closing
            elif pos.asset_class == "option" and pos.option_symbol:
                side = "sell" if pos.side == "long" else "buy"
                self.adapter.place_order(
                    symbol=pos.option_symbol,
                    quantity=pos.quantity,
                    side=side,
                    order_type="market",
                )
                logger.info(f"Closed option position {pos.option_symbol} x{pos.quantity}")
            # Equity closing
            else:
                qty = round(pos.size / max(price, 1e-6), 2)
                side = "sell" if pos.side == "long" else "buy"
                self.adapter.place_order(symbol, qty, side=side)

        del self.positions[symbol]
        if symbol in self.active_strategies:
            del self.active_strategies[symbol]

    async def scale_out_position(
        self, symbol: str, price: float, pct_to_close: float, reason: str
    ) -> bool:
        """Partially close a position (scale-out).

        Args:
            symbol: Position symbol
            price: Current price
            pct_to_close: Percentage of ORIGINAL position to close (0.33 = 33%)
            reason: Scale-out reason for logging

        Returns:
            True if scale-out executed successfully
        """
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]

        # Calculate quantity to close based on original position size
        close_qty = int(pos.original_quantity * pct_to_close)

        if close_qty < 1:
            logger.warning(f"Scale-out quantity < 1 for {symbol}, skipping")
            return False

        # Don't close more than remaining quantity
        close_qty = min(close_qty, int(pos.remaining_quantity))

        logger.info(
            f"SCALE-OUT {symbol} @ ${price:.2f} | {reason} | "
            f"Closing {close_qty} / {pos.remaining_quantity} (Level {pos.scale_out_completed + 1})"
        )

        if self.enable_trading:
            # Handle Multi-Leg Option Strategy partial close
            if pos.asset_class == "option_strategy" and symbol in self.active_strategies:
                strategy = self.active_strategies[symbol]
                if isinstance(strategy, OptionsOrderRequest):
                    # Close proportional amount of each leg
                    close_legs = []
                    for leg in strategy.legs:
                        close_side = "sell" if leg.side == "buy" else "buy"
                        leg_qty = int(leg.ratio * close_qty)
                        if leg_qty > 0:
                            close_legs.append(
                                {"symbol": leg.symbol, "side": close_side, "qty": leg_qty}
                            )

                    if self.options_adapter and close_legs:
                        await self.options_adapter.place_multileg_order(
                            legs=close_legs, note=f"Scale-out {strategy.strategy_name}"
                        )

            # Handle Single-Leg Option partial close
            elif pos.asset_class == "option" and pos.option_symbol:
                side = "sell" if pos.side == "long" else "buy"
                self.adapter.place_order(
                    symbol=pos.option_symbol,
                    quantity=close_qty,
                    side=side,
                    order_type="market",
                )

            # Handle Equity partial close
            else:
                side = "sell" if pos.side == "long" else "buy"
                self.adapter.place_order(symbol=symbol, quantity=close_qty, side=side)

        # Update position tracking
        pos.remaining_quantity -= close_qty
        pos.quantity = pos.remaining_quantity
        pos.scale_out_completed += 1

        # If completely closed, remove position
        if pos.remaining_quantity <= 0:
            logger.info(f"Position {symbol} fully closed after scale-outs")
            del self.positions[symbol]
            if symbol in self.active_strategies:
                del self.active_strategies[symbol]
            return True

        logger.info(
            f"Position {symbol} scaled out | Remaining: {pos.remaining_quantity} / {pos.original_quantity}"
        )
        return True

    async def check_option_stop_loss(self, pos: Position, option_price: float) -> bool:
        """Check stop loss for option positions based on premium price."""
        if not pos.stop_loss_price:
            return False

        # For long options, stop loss triggers when price drops below threshold
        # For short options (if supported), stop loss triggers when price rises above threshold
        triggered = option_price <= pos.stop_loss_price if pos.side == "long" else option_price >= pos.stop_loss_price

        if triggered:
            logger.warning(
                f"Option STOP LOSS triggered for {pos.symbol}: "
                f"Entry=${pos.entry_price:.2f}, Current=${option_price:.2f}, SL=${pos.stop_loss_price:.2f}"
            )
            await self.close_position(pos.symbol, option_price, reason="Option Stop-Loss")
            return True
        return False

    async def check_option_take_profit(self, pos: Position, option_price: float) -> bool:
        """Check take profit for option positions based on premium price."""
        if not pos.take_profit_price:
            return False

        # For long options, take profit triggers when price rises above threshold
        # For short options (if supported), take profit triggers when price drops below threshold
        triggered = option_price >= pos.take_profit_price if pos.side == "long" else option_price <= pos.take_profit_price

        if triggered:
            logger.info(
                f"Option TAKE PROFIT triggered for {pos.symbol}: "
                f"Entry=${pos.entry_price:.2f}, Current=${option_price:.2f}, TP=${pos.take_profit_price:.2f}"
            )
            await self.close_position(pos.symbol, option_price, reason="Option Take-Profit")
            return True
        return False

    async def check_scale_outs(self, pos: Position, current_price: float) -> bool:
        """Check if any scale-out levels have been hit.

        Args:
            pos: Position to check
            current_price: Current price (underlying for equities, premium for options)

        Returns:
            True if a scale-out was executed
        """
        if not self.enable_scale_outs:
            return False

        # Check if all scale-outs completed
        if pos.scale_out_completed >= len(self.scale_out_levels):
            return False

        # Calculate risk per share (distance from entry to stop loss)
        risk_per_share = abs(pos.entry_price - pos.stop_loss_price)
        if risk_per_share <= 0:
            return False

        # Check each configured scale-out level
        for i in range(pos.scale_out_completed, len(self.scale_out_levels)):
            level = self.scale_out_levels[i]
            reward_multiple = level["reward_multiple"]
            pct_to_close = level["pct"]
            new_trail_pct = level["trailing_stop_pct"]

            # Calculate target price for this reward multiple
            if pos.side == "long":
                target_price = pos.entry_price + (risk_per_share * reward_multiple)
                reached = current_price >= target_price
            else:  # short
                target_price = pos.entry_price - (risk_per_share * reward_multiple)
                reached = current_price <= target_price

            if reached:
                reason = f"{reward_multiple}R Scale-Out (Level {i + 1})"
                await self.scale_out_position(pos.symbol, current_price, pct_to_close, reason)

                # Tighten trailing stop after scale-out
                self.trailing_stop_pct = new_trail_pct
                pos.trailing_stop_active = True

                logger.info(
                    f"Trailing stop tightened to {new_trail_pct * 100:.1f}% after scale-out level {i + 1}"
                )

                # After first scale-out, move stop to breakeven
                if i == 0:
                    if pos.side == "long":
                        pos.stop_loss_price = max(pos.stop_loss_price, pos.entry_price)
                    else:
                        pos.stop_loss_price = min(pos.stop_loss_price, pos.entry_price)
                    logger.info(f"Stop loss moved to breakeven: ${pos.stop_loss_price:.2f}")

                return True

        return False

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
            if not self.stream:
                logger.warning("No valid stream to run.")
                return

            # Use _run_forever if available (async), otherwise fallback to run (sync wrapper)
            if hasattr(self.stream, "_run_forever"):
                logger.info("Using async stream._run_forever()")
                self.stream_task = asyncio.create_task(self.stream._run_forever())
            else:
                logger.info("Using sync stream.run()")
                self.stream_task = asyncio.create_task(self.stream.run())

            await self.stream_task
        except asyncio.CancelledError:
            logger.info("Stream task cancelled; stopping stream")
            await self._stop_stream()
            raise
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received; stopping stream")
            await self._stop_stream()
        except Exception as e:
            logger.error(f"Stream run failed: {e}")
            await self._stop_stream()

    async def stop(self):
        """Stop the trading bot."""
        self.running = False
        self.stopping = True
        await self._stop_stream()

    async def _stop_stream(self) -> None:
        """Safely cancel and stop the Alpaca stream without surfacing noisy tracebacks."""
        try:
            if self.stream_task and not self.stream_task.done():
                self.stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.stream_task
        except Exception as exc:
            logger.debug(f"Stream task cancellation emitted: {exc}")

        if self.stream:
            try:
                await self.stream.stop()
            except AttributeError:
                # Stream loop might not be initialized if run() wasn't called
                pass
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
