"""Position Manager - Manages position lifecycle from entry to exit."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class PositionStatus(str, Enum):
    """Position lifecycle status."""
    PENDING = "pending"          # Order submitted, awaiting fill
    OPEN = "open"                # Position is active
    PARTIAL_EXIT = "partial"     # Partially closed
    CLOSING = "closing"          # Exit order submitted
    CLOSED = "closed"            # Fully closed
    EXPIRED = "expired"          # Options expired


@dataclass
class ManagedPosition:
    """
    A position with full lifecycle tracking.

    Tracks entry, current state, stops, targets, and exit conditions.
    """
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    entry_time: datetime

    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Stop management
    initial_stop: Optional[float] = None
    current_stop: Optional[float] = None
    trailing_stop_pct: float = 0.0
    highest_price: float = 0.0  # For trailing (long)
    lowest_price: float = float('inf')  # For trailing (short)

    # Targets
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_1_hit: bool = False
    target_2_hit: bool = False

    # Time management
    max_hold_minutes: int = 60
    time_stop_enabled: bool = True

    # Source tracking
    source_timeframe: str = ""
    strategy_name: str = ""
    trade_idea_id: str = ""

    # Status
    status: PositionStatus = PositionStatus.OPEN
    exit_reason: str = ""
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0

    # Order tracking
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_price(self, price: float) -> None:
        """Update current price and calculate P&L."""
        self.current_price = price

        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (price - self.entry_price) / self.entry_price * 100

            # Track high for trailing stop
            if price > self.highest_price:
                self.highest_price = price
        else:  # short
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            self.unrealized_pnl_pct = (self.entry_price - price) / self.entry_price * 100

            # Track low for trailing stop
            if price < self.lowest_price:
                self.lowest_price = price

    def calculate_trailing_stop(self) -> Optional[float]:
        """Calculate current trailing stop level."""
        if self.trailing_stop_pct <= 0:
            return self.current_stop

        if self.side == "long":
            # Trail from highest price
            trail_stop = self.highest_price * (1 - self.trailing_stop_pct / 100)
            # Only move stop up, never down
            if self.current_stop and trail_stop > self.current_stop:
                return trail_stop
            return self.current_stop or trail_stop
        else:
            # Trail from lowest price (short)
            trail_stop = self.lowest_price * (1 + self.trailing_stop_pct / 100)
            # Only move stop down, never up
            if self.current_stop and trail_stop < self.current_stop:
                return trail_stop
            return self.current_stop or trail_stop

    def is_stop_hit(self) -> bool:
        """Check if stop loss is hit."""
        if not self.current_stop or self.current_price <= 0:
            return False

        if self.side == "long":
            return self.current_price <= self.current_stop
        else:
            return self.current_price >= self.current_stop

    def is_target_hit(self, target_num: int = 1) -> bool:
        """Check if profit target is hit."""
        target = self.target_1 if target_num == 1 else self.target_2
        if not target or self.current_price <= 0:
            return False

        if self.side == "long":
            return self.current_price >= target
        else:
            return self.current_price <= target

    def is_time_expired(self, current_time: datetime) -> bool:
        """Check if max hold time exceeded."""
        if not self.time_stop_enabled or self.max_hold_minutes <= 0:
            return False

        elapsed = (current_time - self.entry_time).total_seconds() / 60
        return elapsed >= self.max_hold_minutes

    def get_hold_duration_minutes(self, current_time: datetime) -> float:
        """Get current hold duration in minutes."""
        return (current_time - self.entry_time).total_seconds() / 60


class PositionManager:
    """
    Manages all open positions and their lifecycle.

    Responsibilities:
    - Track all open positions
    - Update prices and P&L
    - Check exit conditions
    - Coordinate with broker for exits
    - Maintain position history
    """

    def __init__(
        self,
        broker_adapter: Any,
        config: Dict[str, Any] = None,
    ):
        self.broker = broker_adapter
        self.config = config or {}

        # Active positions by symbol
        self.positions: Dict[str, ManagedPosition] = {}

        # Closed position history
        self.closed_positions: List[ManagedPosition] = []

        # Configuration
        self.max_positions = self.config.get("max_positions", 5)
        self.default_trailing_pct = self.config.get("default_trailing_pct", 0.5)
        self.default_max_hold_minutes = self.config.get("default_max_hold_minutes", 60)

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        logger.info(
            f"PositionManager initialized (max_positions={self.max_positions}, "
            f"default_trailing={self.default_trailing_pct}%)"
        )

    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        # Already have position in this symbol?
        if symbol in self.positions:
            logger.warning(f"Already have position in {symbol}")
            return False

        # At max positions?
        if len(self.positions) >= self.max_positions:
            logger.warning(f"At max positions ({self.max_positions})")
            return False

        return True

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        initial_stop: Optional[float] = None,
        target_1: Optional[float] = None,
        target_2: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        max_hold_minutes: Optional[int] = None,
        source_timeframe: str = "",
        strategy_name: str = "",
        trade_idea_id: str = "",
        entry_order_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Optional[ManagedPosition]:
        """
        Register a new position for management.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            quantity: Position size
            entry_price: Entry price
            entry_time: Entry timestamp
            initial_stop: Initial stop loss price
            target_1: First profit target
            target_2: Second profit target
            trailing_stop_pct: Trailing stop percentage
            max_hold_minutes: Maximum hold time
            source_timeframe: Timeframe that generated the signal
            strategy_name: Strategy name
            trade_idea_id: Reference to trade idea
            entry_order_id: Broker order ID
            metadata: Additional data

        Returns:
            ManagedPosition if successful, None otherwise
        """
        if not self.can_open_position(symbol):
            return None

        position = ManagedPosition(
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            current_price=entry_price,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            trailing_stop_pct=trailing_stop_pct or self.default_trailing_pct,
            highest_price=entry_price,
            lowest_price=entry_price,
            target_1=target_1,
            target_2=target_2,
            max_hold_minutes=max_hold_minutes or self.default_max_hold_minutes,
            source_timeframe=source_timeframe,
            strategy_name=strategy_name,
            trade_idea_id=trade_idea_id,
            entry_order_id=entry_order_id,
            status=PositionStatus.OPEN,
            metadata=metadata or {},
        )

        self.positions[symbol] = position
        self.total_trades += 1

        logger.info(
            f"Opened position: {side.upper()} {quantity} {symbol} @ ${entry_price:.2f} | "
            f"Stop: ${initial_stop:.2f if initial_stop else 'None'} | "
            f"Target: ${target_1:.2f if target_1 else 'None'}"
        )

        return position

    def update_positions(self, current_time: datetime) -> List[Dict[str, Any]]:
        """
        Update all positions with current prices and check exit conditions.

        Returns:
            List of exit signals for positions that should be closed
        """
        exit_signals = []

        if not self.broker:
            logger.warning("No broker available for position updates")
            return exit_signals

        # Sync with broker positions
        try:
            broker_positions = self.broker.get_positions()
            broker_position_map = {p.symbol: p for p in broker_positions}
        except Exception as e:
            logger.error(f"Failed to fetch broker positions: {e}")
            return exit_signals

        # Update each managed position
        for symbol, position in list(self.positions.items()):
            if position.status != PositionStatus.OPEN:
                continue

            # Get current price from broker
            broker_pos = broker_position_map.get(symbol)
            if broker_pos:
                position.update_price(broker_pos.current_price)
            else:
                # Position may have been closed externally
                try:
                    quote = self.broker.get_latest_quote(symbol)
                    if quote:
                        mid_price = (quote["bid"] + quote["ask"]) / 2
                        position.update_price(mid_price)
                except Exception:
                    pass

            # Check exit conditions
            exit_signal = self._check_exit_conditions(position, current_time)
            if exit_signal:
                exit_signals.append(exit_signal)

        return exit_signals

    def _check_exit_conditions(
        self,
        position: ManagedPosition,
        current_time: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Check all exit conditions for a position."""

        # 1. Stop loss hit
        if position.is_stop_hit():
            return {
                "symbol": position.symbol,
                "position": position,
                "reason": "stop_loss",
                "exit_price": position.current_stop,
                "priority": 1,  # Highest priority
            }

        # 2. Target 2 hit (full exit)
        if position.target_2 and position.is_target_hit(2):
            return {
                "symbol": position.symbol,
                "position": position,
                "reason": "target_2",
                "exit_price": position.target_2,
                "priority": 2,
            }

        # 3. Target 1 hit (partial exit or trail tighter)
        if position.target_1 and position.is_target_hit(1) and not position.target_1_hit:
            position.target_1_hit = True
            # Tighten trailing stop after hitting first target
            position.trailing_stop_pct = max(0.25, position.trailing_stop_pct / 2)
            logger.info(
                f"{position.symbol}: Target 1 hit @ ${position.current_price:.2f}, "
                f"tightening trail to {position.trailing_stop_pct:.2f}%"
            )

        # 4. Time stop
        if position.is_time_expired(current_time):
            return {
                "symbol": position.symbol,
                "position": position,
                "reason": "time_stop",
                "exit_price": position.current_price,
                "priority": 3,
            }

        # 5. Update trailing stop
        new_stop = position.calculate_trailing_stop()
        if new_stop and new_stop != position.current_stop:
            old_stop = position.current_stop
            position.current_stop = new_stop
            logger.debug(
                f"{position.symbol}: Trailing stop updated ${old_stop:.2f} -> ${new_stop:.2f}"
            )

        return None

    def close_position(
        self,
        symbol: str,
        reason: str,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
    ) -> Optional[ManagedPosition]:
        """
        Close a position and move to history.

        Args:
            symbol: Symbol to close
            reason: Exit reason
            exit_price: Exit price (uses current if not provided)
            exit_time: Exit timestamp

        Returns:
            Closed ManagedPosition
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None

        position = self.positions[symbol]
        position.status = PositionStatus.CLOSED
        position.exit_reason = reason
        position.exit_price = exit_price or position.current_price
        position.exit_time = exit_time or datetime.utcnow()

        # Calculate realized P&L
        if position.side == "long":
            position.realized_pnl = (position.exit_price - position.entry_price) * position.quantity
        else:
            position.realized_pnl = (position.entry_price - position.exit_price) * position.quantity

        # Update stats
        self.total_pnl += position.realized_pnl
        if position.realized_pnl >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Move to history
        self.closed_positions.append(position)
        del self.positions[symbol]

        logger.info(
            f"Closed position: {symbol} | Reason: {reason} | "
            f"P&L: ${position.realized_pnl:+,.2f} ({position.unrealized_pnl_pct:+.2f}%)"
        )

        return position

    def execute_exit(
        self,
        exit_signal: Dict[str, Any],
        current_time: datetime,
    ) -> bool:
        """
        Execute an exit order through the broker.

        Args:
            exit_signal: Exit signal from _check_exit_conditions
            current_time: Current timestamp

        Returns:
            True if exit executed successfully
        """
        if not self.broker:
            logger.warning("No broker available for exit execution")
            return False

        position = exit_signal["position"]
        reason = exit_signal["reason"]

        try:
            # Close the position
            success = self.broker.close_position(position.symbol)

            if success:
                # Get actual exit price from quote
                quote = self.broker.get_latest_quote(position.symbol)
                exit_price = None
                if quote:
                    exit_price = (quote["bid"] + quote["ask"]) / 2

                self.close_position(
                    position.symbol,
                    reason,
                    exit_price=exit_price,
                    exit_time=current_time,
                )
                return True
            else:
                logger.error(f"Failed to close position {position.symbol}")
                return False

        except Exception as e:
            logger.error(f"Error executing exit for {position.symbol}: {e}")
            return False

    def sync_with_broker(self) -> None:
        """
        Sync managed positions with broker's actual positions.

        Handles:
        - Positions opened externally
        - Positions closed externally (stops hit, etc.)
        """
        if not self.broker:
            return

        try:
            broker_positions = self.broker.get_positions()
            broker_symbols = {p.symbol for p in broker_positions}
            managed_symbols = set(self.positions.keys())

            # Check for positions closed externally
            for symbol in managed_symbols - broker_symbols:
                position = self.positions[symbol]
                if position.status == PositionStatus.OPEN:
                    logger.info(f"Position {symbol} was closed externally (likely stop hit)")
                    self.close_position(symbol, "external_close")

            # Log positions not being managed
            for symbol in broker_symbols - managed_symbols:
                logger.debug(f"Broker has position in {symbol} not being managed")

        except Exception as e:
            logger.error(f"Error syncing with broker: {e}")

    def get_position(self, symbol: str) -> Optional[ManagedPosition]:
        """Get a managed position by symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[ManagedPosition]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_total_exposure(self) -> float:
        """Get total market exposure across all positions."""
        return sum(
            abs(p.current_price * p.quantity)
            for p in self.positions.values()
        )

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get position manager statistics."""
        win_rate = (
            self.winning_trades / self.total_trades * 100
            if self.total_trades > 0 else 0
        )

        return {
            "open_positions": len(self.positions),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "total_realized_pnl": self.total_pnl,
            "total_unrealized_pnl": self.get_total_unrealized_pnl(),
            "total_exposure": self.get_total_exposure(),
        }

    def get_position_summary(self) -> str:
        """Get formatted summary of all positions."""
        if not self.positions:
            return "No open positions"

        lines = ["Open Positions:"]
        for pos in self.positions.values():
            duration = pos.get_hold_duration_minutes(datetime.utcnow())
            lines.append(
                f"  {pos.symbol}: {pos.side.upper()} {pos.quantity} @ ${pos.entry_price:.2f} | "
                f"Current: ${pos.current_price:.2f} | P&L: ${pos.unrealized_pnl:+,.2f} "
                f"({pos.unrealized_pnl_pct:+.2f}%) | Stop: ${pos.current_stop:.2f if pos.current_stop else 'None'} | "
                f"Hold: {duration:.0f}min"
            )

        return "\n".join(lines)
