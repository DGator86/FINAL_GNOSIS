"""Autonomous Trader - Orchestrates full position lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from execution.position_manager import (
    ManagedPosition,
    PositionManager,
    PositionStatus,
)
from execution.trailing_stop_manager import (
    TrailingStopConfig,
    TrailingStopManager,
)
from execution.exit_rules_engine import (
    ExitReason,
    ExitRulesConfig,
    ExitRulesEngine,
    ExitSignal,
)
from execution.trading_attitude import (
    TradingAttitude,
    AttitudeProfile,
    AttitudeSelector,
    MarketConditions,
    create_conditions_from_pipeline,
    ATTITUDE_PROFILES,
)
from schemas.core_schemas import TradeIdea, PipelineResult


@dataclass
class TraderConfig:
    """Configuration for autonomous trader."""

    # Position limits
    max_positions: int = 5
    max_position_size_pct: float = 10.0  # % of portfolio per position
    max_daily_trades: int = 20
    max_daily_loss_pct: float = 5.0

    # Entry requirements
    min_confidence: float = 0.5
    min_mtf_alignment: float = 0.4
    require_mtf_agreement: bool = False

    # Default stops/targets
    default_stop_pct: float = 1.0
    default_target_1_pct: float = 0.5
    default_target_2_pct: float = 1.0
    default_trailing_pct: float = 0.5
    default_max_hold_minutes: int = 60

    # Execution settings
    use_bracket_orders: bool = True
    update_stops_at_broker: bool = True
    check_interval_seconds: int = 5

    # Safety
    paper_mode: bool = True
    confirm_before_trade: bool = False

    # Attitude settings
    default_attitude: TradingAttitude = TradingAttitude.DAY_TRADER
    enable_adaptive_attitude: bool = True  # Auto-select attitude from conditions


@dataclass
class TraderState:
    """Runtime state of the trader."""

    session_start: datetime = field(default_factory=datetime.utcnow)
    session_start_equity: float = 0.0
    current_equity: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0
    is_trading_halted: bool = False
    halt_reason: str = ""


class AutonomousTrader:
    """
    Orchestrates the full trading lifecycle:
    - Entry: Converts trade ideas to positions with stops/targets
    - Monitoring: Continuously checks prices and exit conditions
    - Exit: Executes exits based on rules (stop, target, time, reversal)
    - Tracking: Maintains P&L and position history

    Integrates:
    - PositionManager: Position lifecycle
    - TrailingStopManager: Dynamic stop adjustment
    - ExitRulesEngine: Exit condition evaluation
    """

    def __init__(
        self,
        broker_adapter: Any,
        config: TraderConfig = None,
        trailing_config: TrailingStopConfig = None,
        exit_config: ExitRulesConfig = None,
    ):
        self.broker = broker_adapter
        self.config = config or TraderConfig()

        # Initialize components
        self.position_manager = PositionManager(
            broker_adapter=broker_adapter,
            config={
                "max_positions": self.config.max_positions,
                "default_trailing_pct": self.config.default_trailing_pct,
                "default_max_hold_minutes": self.config.default_max_hold_minutes,
            },
        )

        self.trailing_manager = TrailingStopManager(
            broker_adapter=broker_adapter,
            config=trailing_config or TrailingStopConfig(
                trailing_pct=self.config.default_trailing_pct,
                initial_stop_pct=self.config.default_stop_pct,
            ),
        )

        self.exit_engine = ExitRulesEngine(
            config=exit_config or ExitRulesConfig(
                max_hold_minutes=self.config.default_max_hold_minutes,
                target_1_pct=self.config.default_target_1_pct,
                target_2_pct=self.config.default_target_2_pct,
            ),
        )

        # Attitude selector for adaptive parameter adjustment
        self.attitude_selector = AttitudeSelector(
            default_attitude=self.config.default_attitude
        )
        self.current_attitude_profile: Optional[AttitudeProfile] = None

        # State
        self.state = TraderState()
        self._init_session()

        # Event callbacks
        self.on_position_opened: Optional[callable] = None
        self.on_position_closed: Optional[callable] = None
        self.on_stop_updated: Optional[callable] = None

        mode = "PAPER" if self.config.paper_mode else "LIVE"
        logger.info(
            f"AutonomousTrader initialized ({mode} mode, "
            f"max_positions={self.config.max_positions})"
        )

    def update_attitude_from_conditions(
        self,
        pipeline_result: PipelineResult,
        time_to_close: Optional[int] = None,
    ) -> AttitudeProfile:
        """
        Update trading attitude based on current market conditions.

        Args:
            pipeline_result: Latest pipeline result with volatility data
            time_to_close: Minutes until market close

        Returns:
            The selected AttitudeProfile
        """
        if not self.config.enable_adaptive_attitude:
            # Use default profile
            profile = ATTITUDE_PROFILES[self.config.default_attitude]
            self.current_attitude_profile = profile
            return profile

        # Create conditions from pipeline
        conditions = create_conditions_from_pipeline(pipeline_result, time_to_close)

        # Select attitude
        profile = self.attitude_selector.select_attitude(conditions)
        self.current_attitude_profile = profile

        # Apply profile settings to components
        self._apply_attitude_profile(profile)

        return profile

    def _apply_attitude_profile(self, profile: AttitudeProfile) -> None:
        """Apply attitude profile settings to all components."""

        # Update config
        self.config.min_confidence = profile.min_confidence
        self.config.min_mtf_alignment = profile.min_mtf_alignment
        self.config.require_mtf_agreement = profile.require_mtf_agreement
        self.config.max_positions = profile.max_positions
        self.config.max_position_size_pct = profile.max_position_size_pct
        self.config.default_stop_pct = profile.initial_stop_pct
        self.config.default_trailing_pct = profile.trailing_stop_pct
        self.config.default_target_1_pct = profile.target_1_pct
        self.config.default_target_2_pct = profile.target_2_pct
        self.config.default_max_hold_minutes = profile.max_hold_minutes

        # Update trailing stop manager
        self.trailing_manager.config.initial_stop_pct = profile.initial_stop_pct
        self.trailing_manager.config.trailing_pct = profile.trailing_stop_pct
        self.trailing_manager.config.activation_pct = profile.trailing_activation_pct
        self.trailing_manager.config.breakeven_trigger_pct = profile.breakeven_trigger_pct
        self.trailing_manager.config.tighten_after_target_1 = profile.tighten_after_target_1
        self.trailing_manager.config.tightened_trail_pct = profile.tightened_trail_pct

        # Update exit rules engine
        self.exit_engine.config.max_hold_minutes = profile.max_hold_minutes
        self.exit_engine.config.target_1_pct = profile.target_1_pct
        self.exit_engine.config.target_2_pct = profile.target_2_pct
        self.exit_engine.config.partial_exit_at_target_1 = profile.partial_exit_at_target_1
        self.exit_engine.config.partial_exit_pct = profile.partial_exit_pct
        self.exit_engine.config.market_close_exit_minutes = profile.exit_before_close_minutes

        logger.debug(f"Applied attitude profile: {profile.describe()}")

    def set_attitude_override(self, attitude: Optional[TradingAttitude]) -> None:
        """Set manual attitude override (bypasses adaptive selection)."""
        self.attitude_selector.set_override(attitude)
        if attitude:
            profile = ATTITUDE_PROFILES[attitude]
            self._apply_attitude_profile(profile)
            self.current_attitude_profile = profile

    def get_current_attitude(self) -> str:
        """Get current trading attitude name and settings."""
        if self.current_attitude_profile:
            return self.current_attitude_profile.describe()
        return "Not set"

    def _init_session(self) -> None:
        """Initialize trading session state."""
        self.state.session_start = datetime.utcnow()
        self.state.daily_trades = 0
        self.state.daily_pnl = 0.0
        self.state.is_trading_halted = False

        if self.broker:
            try:
                account = self.broker.get_account()
                self.state.session_start_equity = account.equity
                self.state.current_equity = account.equity
            except Exception as e:
                logger.warning(f"Could not fetch account equity: {e}")

    def process_trade_ideas(
        self,
        trade_ideas: List[TradeIdea],
        pipeline_result: PipelineResult,
        timestamp: datetime,
        time_to_close: Optional[int] = None,
    ) -> List[ManagedPosition]:
        """
        Process trade ideas and open positions.

        Args:
            trade_ideas: List of trade ideas from TradeAgent
            pipeline_result: Full pipeline result for context
            timestamp: Current timestamp
            time_to_close: Minutes until market close (for attitude selection)

        Returns:
            List of newly opened positions
        """
        if self.state.is_trading_halted:
            logger.warning(f"Trading halted: {self.state.halt_reason}")
            return []

        # Update trading attitude based on market conditions
        if self.config.enable_adaptive_attitude:
            self.update_attitude_from_conditions(pipeline_result, time_to_close)

        opened_positions = []

        for idea in trade_ideas:
            # Check entry criteria
            if not self._validate_entry(idea, pipeline_result):
                continue

            # Check if we can open
            if not self.position_manager.can_open_position(idea.symbol):
                continue

            # Check daily limits
            if self.state.daily_trades >= self.config.max_daily_trades:
                logger.warning("Daily trade limit reached")
                break

            # Execute entry
            position = self._execute_entry(idea, pipeline_result, timestamp)
            if position:
                opened_positions.append(position)
                self.state.daily_trades += 1

                # Callback
                if self.on_position_opened:
                    self.on_position_opened(position)

        return opened_positions

    def _validate_entry(
        self,
        idea: TradeIdea,
        pipeline_result: PipelineResult,
    ) -> bool:
        """Validate if trade idea meets entry criteria."""

        # Check confidence
        if idea.confidence < self.config.min_confidence:
            logger.debug(f"{idea.symbol}: Confidence {idea.confidence:.2%} below minimum")
            return False

        # Check MTF alignment
        if idea.mtf_alignment and idea.mtf_alignment < self.config.min_mtf_alignment:
            logger.debug(f"{idea.symbol}: MTF alignment {idea.mtf_alignment:.2%} below minimum")
            return False

        # Check direction is not neutral
        if idea.direction.value == "neutral":
            return False

        return True

    def _execute_entry(
        self,
        idea: TradeIdea,
        pipeline_result: PipelineResult,
        timestamp: datetime,
    ) -> Optional[ManagedPosition]:
        """Execute entry order and create managed position."""

        symbol = idea.symbol
        side = "long" if idea.direction.value == "long" else "short"
        quantity = self._calculate_position_size(idea)
        entry_price = idea.entry_price or self._get_current_price(symbol)

        # Calculate stops and targets
        initial_stop = idea.stop_loss or self.trailing_manager.calculate_initial_stop(
            entry_price, side, symbol
        )

        target_1 = idea.take_profit
        if not target_1:
            if side == "long":
                target_1 = entry_price * (1 + self.config.default_target_1_pct / 100)
            else:
                target_1 = entry_price * (1 - self.config.default_target_1_pct / 100)

        target_2 = None
        if side == "long":
            target_2 = entry_price * (1 + self.config.default_target_2_pct / 100)
        else:
            target_2 = entry_price * (1 - self.config.default_target_2_pct / 100)

        # Get time-to-profit settings
        max_hold_minutes = self.config.default_max_hold_minutes
        trailing_pct = self.config.default_trailing_pct

        if idea.time_to_profit:
            max_hold_minutes = idea.time_to_profit.estimated_minutes or max_hold_minutes
            trailing_pct = idea.time_to_profit.trailing_stop_pct or trailing_pct

        # Execute order at broker
        order_id = None
        if self.broker:
            try:
                if self.config.use_bracket_orders and idea.stop_loss and idea.take_profit:
                    order_id = self.broker.place_bracket_order(
                        symbol=symbol,
                        quantity=quantity,
                        side="buy" if side == "long" else "sell",
                        take_profit_price=idea.take_profit,
                        stop_loss_price=idea.stop_loss,
                    )
                else:
                    order_id = self.broker.place_order(
                        symbol=symbol,
                        quantity=quantity,
                        side="buy" if side == "long" else "sell",
                        order_type="market",
                    )

                if not order_id:
                    logger.error(f"Failed to execute entry for {symbol}")
                    return None

            except Exception as e:
                logger.error(f"Entry execution error for {symbol}: {e}")
                return None

        # Create managed position
        position = self.position_manager.open_position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=timestamp,
            initial_stop=initial_stop,
            target_1=target_1,
            target_2=target_2,
            trailing_stop_pct=trailing_pct,
            max_hold_minutes=max_hold_minutes,
            source_timeframe=idea.source_timeframe or "",
            strategy_name=idea.options_strategy or idea.strategy_type.value,
            trade_idea_id=str(id(idea)),
            entry_order_id=order_id,
            metadata={
                "confidence": idea.confidence,
                "mtf_alignment": idea.mtf_alignment,
                "expected_move": str(idea.expected_move) if idea.expected_move else None,
            },
        )

        if position:
            logger.info(
                f"Opened position: {side.upper()} {quantity} {symbol} @ ${entry_price:.2f} | "
                f"Stop: ${initial_stop:.2f} | Target: ${target_1:.2f}"
            )

        return position

    def _calculate_position_size(self, idea: TradeIdea) -> float:
        """Calculate position size based on config and idea."""
        if idea.size:
            return idea.size

        # Default sizing based on portfolio percentage
        if self.broker:
            try:
                account = self.broker.get_account()
                max_value = account.portfolio_value * (self.config.max_position_size_pct / 100)
                price = idea.entry_price or 100
                return max(1, int(max_value / price))
            except Exception:
                pass

        return 100  # Default fallback

    def _get_current_price(self, symbol: str) -> float:
        """Get current price from broker."""
        if self.broker:
            try:
                quote = self.broker.get_latest_quote(symbol)
                if quote:
                    return (quote["bid"] + quote["ask"]) / 2
            except Exception:
                pass
        return 0.0

    def monitor_positions(
        self,
        timestamp: datetime,
        latest_signals: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Monitor all positions and process exits.

        Args:
            timestamp: Current timestamp
            latest_signals: Latest pipeline signals per symbol

        Returns:
            Summary of actions taken
        """
        summary = {
            "positions_checked": 0,
            "stops_updated": 0,
            "exits_triggered": 0,
            "exits_executed": 0,
        }

        if not self.position_manager.positions:
            return summary

        # Sync with broker
        self.position_manager.sync_with_broker()

        # Update signals in exit engine
        if latest_signals:
            for symbol, signal in latest_signals.items():
                self.exit_engine.update_signal(symbol, signal)

        # Check each position
        positions = self.position_manager.get_all_positions()
        summary["positions_checked"] = len(positions)

        for position in positions:
            if position.status != PositionStatus.OPEN:
                continue

            # Update price from broker
            self._update_position_price(position)

            # Process trailing stops
            trail_updates = self.trailing_manager.process_positions(
                [position],
                timestamp,
            )
            summary["stops_updated"] += len(trail_updates)

            for update in trail_updates:
                if self.on_stop_updated:
                    self.on_stop_updated(position, update)

            # Evaluate exit rules
            exit_signals = self.exit_engine.evaluate(position, timestamp)

            for signal in exit_signals:
                summary["exits_triggered"] += 1

                # Execute the exit
                success = self._execute_exit(position, signal, timestamp)
                if success:
                    summary["exits_executed"] += 1

                    if self.on_position_closed:
                        self.on_position_closed(position, signal)

                    break  # Only execute first exit signal per position

        # Update daily P&L
        self._update_daily_pnl()

        # Check circuit breaker
        self._check_circuit_breaker()

        return summary

    def _update_position_price(self, position: ManagedPosition) -> None:
        """Update position with current price."""
        if self.broker:
            try:
                quote = self.broker.get_latest_quote(position.symbol)
                if quote:
                    mid = (quote["bid"] + quote["ask"]) / 2
                    position.update_price(mid)
            except Exception:
                pass

    def _execute_exit(
        self,
        position: ManagedPosition,
        signal: ExitSignal,
        timestamp: datetime,
    ) -> bool:
        """Execute an exit order."""
        if not self.broker:
            logger.info(f"No broker - simulating exit for {position.symbol}")
            self.position_manager.close_position(
                position.symbol,
                signal.reason.value,
                signal.exit_price,
                timestamp,
            )
            return True

        try:
            # Determine exit quantity
            qty = signal.exit_quantity or position.quantity

            # Close position
            if qty >= position.quantity:
                # Full exit
                success = self.broker.close_position(position.symbol)
            else:
                # Partial exit
                side = "sell" if position.side == "long" else "buy"
                order_id = self.broker.place_order(
                    symbol=position.symbol,
                    quantity=qty,
                    side=side,
                    order_type="market",
                )
                success = order_id is not None

            if success:
                self.position_manager.close_position(
                    position.symbol,
                    signal.reason.value,
                    signal.exit_price,
                    timestamp,
                )
                logger.info(
                    f"Exited {position.symbol}: {signal.reason.value} @ ${signal.exit_price:.2f}"
                )
                return True
            else:
                logger.error(f"Failed to execute exit for {position.symbol}")
                return False

        except Exception as e:
            logger.error(f"Exit execution error for {position.symbol}: {e}")
            return False

    def _update_daily_pnl(self) -> None:
        """Update daily P&L from position manager."""
        self.state.daily_pnl = self.position_manager.total_pnl

        if self.broker:
            try:
                account = self.broker.get_account()
                self.state.current_equity = account.equity
            except Exception:
                pass

    def _check_circuit_breaker(self) -> None:
        """Check if daily loss limit exceeded."""
        if self.state.session_start_equity <= 0:
            return

        loss_pct = (
            (self.state.session_start_equity - self.state.current_equity)
            / self.state.session_start_equity * 100
        )

        if loss_pct >= self.config.max_daily_loss_pct:
            self.state.is_trading_halted = True
            self.state.halt_reason = f"Daily loss limit exceeded: {loss_pct:.2f}%"
            logger.warning(f"CIRCUIT BREAKER: {self.state.halt_reason}")

    def get_dashboard_summary(self) -> str:
        """Get formatted dashboard summary."""
        lines = [
            "=" * 60,
            "  AUTONOMOUS TRADER DASHBOARD",
            "=" * 60,
        ]

        # Session info
        session_duration = datetime.utcnow() - self.state.session_start
        lines.append(f"  Session Duration: {session_duration}")
        lines.append(f"  Mode: {'PAPER' if self.config.paper_mode else 'LIVE'}")

        # Current attitude
        if self.current_attitude_profile:
            lines.append(f"  Attitude: {self.current_attitude_profile.attitude.value.upper()}")
            lines.append(f"    Aggressiveness: {self.current_attitude_profile.aggressiveness:.0%}")
            lines.append(f"    Stop: {self.current_attitude_profile.initial_stop_pct:.1f}% | "
                        f"Trail: {self.current_attitude_profile.trailing_stop_pct:.1f}%")
            lines.append(f"    Max Hold: {self.current_attitude_profile.max_hold_minutes} min | "
                        f"DTE: {self.current_attitude_profile.preferred_dte_min}-"
                        f"{self.current_attitude_profile.preferred_dte_max}")

        if self.state.is_trading_halted:
            lines.append(f"  STATUS: HALTED - {self.state.halt_reason}")
        else:
            lines.append("  STATUS: ACTIVE")

        lines.append("")

        # Account
        lines.append("  ACCOUNT:")
        lines.append(f"    Start Equity: ${self.state.session_start_equity:,.2f}")
        lines.append(f"    Current Equity: ${self.state.current_equity:,.2f}")
        session_pnl = self.state.current_equity - self.state.session_start_equity
        lines.append(f"    Session P&L: ${session_pnl:+,.2f}")
        lines.append(f"    Daily Trades: {self.state.daily_trades}")
        lines.append("")

        # Position summary
        stats = self.position_manager.get_stats()
        lines.append("  POSITIONS:")
        lines.append(f"    Open: {stats['open_positions']}")
        lines.append(f"    Total Trades: {stats['total_trades']}")
        lines.append(f"    Win Rate: {stats['win_rate']:.1f}%")
        lines.append(f"    Realized P&L: ${stats['total_realized_pnl']:+,.2f}")
        lines.append(f"    Unrealized P&L: ${stats['total_unrealized_pnl']:+,.2f}")
        lines.append("")

        # Open positions detail
        if self.position_manager.positions:
            lines.append("  OPEN POSITIONS:")
            for pos in self.position_manager.get_all_positions():
                lines.append(
                    f"    {pos.symbol}: {pos.side.upper()} {pos.quantity} @ ${pos.entry_price:.2f} | "
                    f"P&L: ${pos.unrealized_pnl:+,.2f} ({pos.unrealized_pnl_pct:+.2f}%)"
                )
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def close_all_positions(self, reason: str = "manual") -> int:
        """
        Close all open positions.

        Returns:
            Number of positions closed
        """
        closed = 0
        timestamp = datetime.utcnow()

        for symbol in list(self.position_manager.positions.keys()):
            position = self.position_manager.positions[symbol]

            if self.broker:
                try:
                    self.broker.close_position(symbol)
                except Exception as e:
                    logger.error(f"Error closing {symbol}: {e}")
                    continue

            self.position_manager.close_position(symbol, reason, exit_time=timestamp)
            closed += 1

        logger.info(f"Closed {closed} positions (reason: {reason})")
        return closed
