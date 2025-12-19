"""Trailing Stop Manager - Dynamic stop loss adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from execution.position_manager import ManagedPosition


class TrailingMode(str, Enum):
    """Trailing stop calculation modes."""
    PERCENTAGE = "percentage"      # Trail by fixed percentage
    ATR = "atr"                    # Trail by ATR multiple
    STRUCTURE = "structure"        # Trail to structure levels
    HYBRID = "hybrid"              # Combination approach
    CHANDELIER = "chandelier"      # Chandelier exit style


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop behavior."""

    # Basic settings
    mode: TrailingMode = TrailingMode.PERCENTAGE
    initial_stop_pct: float = 1.0      # Initial stop as % from entry
    trailing_pct: float = 0.5          # Trailing % from high/low

    # Activation
    activation_pct: float = 0.0        # Start trailing after X% profit (0 = immediate)
    breakeven_trigger_pct: float = 0.5  # Move to breakeven after X% profit

    # Progressive tightening
    tighten_after_target_1: bool = True
    tightened_trail_pct: float = 0.25

    # ATR-based settings
    atr_period: int = 14
    atr_multiplier: float = 2.0

    # Time-based adjustments
    tighten_near_expiry: bool = True
    expiry_hours_threshold: int = 4
    expiry_trail_pct: float = 0.2

    # Volatility adjustments
    vol_adjust_enabled: bool = True
    high_vol_multiplier: float = 1.5   # Widen stops in high vol
    low_vol_multiplier: float = 0.75   # Tighten stops in low vol


class TrailingStopManager:
    """
    Manages trailing stop calculations and broker order updates.

    Features:
    - Multiple trailing modes (percentage, ATR, structure)
    - Breakeven triggers
    - Progressive tightening
    - Volatility-adjusted stops
    - Time-based tightening near expiry
    """

    def __init__(
        self,
        broker_adapter: Any,
        config: TrailingStopConfig = None,
    ):
        self.broker = broker_adapter
        self.config = config or TrailingStopConfig()

        # ATR cache per symbol
        self.atr_cache: Dict[str, float] = {}

        # Structure levels cache
        self.structure_cache: Dict[str, Dict[str, List[float]]] = {}

        logger.info(
            f"TrailingStopManager initialized (mode={self.config.mode.value}, "
            f"trail={self.config.trailing_pct}%)"
        )

    def calculate_initial_stop(
        self,
        entry_price: float,
        side: str,
        symbol: str = "",
        volatility: float = 0.0,
    ) -> float:
        """
        Calculate initial stop loss price.

        Args:
            entry_price: Entry price
            side: "long" or "short"
            symbol: Trading symbol (for ATR lookup)
            volatility: Current volatility (for adjustments)

        Returns:
            Initial stop loss price
        """
        stop_pct = self.config.initial_stop_pct

        # Adjust for volatility if enabled
        if self.config.vol_adjust_enabled and volatility > 0:
            if volatility > 0.3:  # High vol
                stop_pct *= self.config.high_vol_multiplier
            elif volatility < 0.15:  # Low vol
                stop_pct *= self.config.low_vol_multiplier

        if side == "long":
            return entry_price * (1 - stop_pct / 100)
        else:
            return entry_price * (1 + stop_pct / 100)

    def calculate_trailing_stop(
        self,
        position: ManagedPosition,
        current_time: datetime,
        volatility: float = 0.0,
        expiry_time: Optional[datetime] = None,
    ) -> Tuple[Optional[float], str]:
        """
        Calculate new trailing stop level for a position.

        Args:
            position: The managed position
            current_time: Current timestamp
            volatility: Current volatility
            expiry_time: Option expiry time (for time-based tightening)

        Returns:
            Tuple of (new_stop_price, reason) or (None, "") if no change
        """
        if not position.current_stop:
            return None, ""

        side = position.side
        entry_price = position.entry_price
        current_price = position.current_price
        current_stop = position.current_stop

        # Calculate profit percentage
        if side == "long":
            profit_pct = (current_price - entry_price) / entry_price * 100
            reference_price = position.highest_price
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            reference_price = position.lowest_price

        # 1. Check if trailing should be activated
        if profit_pct < self.config.activation_pct:
            return None, ""

        # 2. Determine trailing percentage to use
        trail_pct = self.config.trailing_pct

        # Tighten after target 1 hit
        if position.target_1_hit and self.config.tighten_after_target_1:
            trail_pct = self.config.tightened_trail_pct

        # Tighten near expiry
        if expiry_time and self.config.tighten_near_expiry:
            hours_to_expiry = (expiry_time - current_time).total_seconds() / 3600
            if hours_to_expiry <= self.config.expiry_hours_threshold:
                trail_pct = self.config.expiry_trail_pct

        # Adjust for volatility
        if self.config.vol_adjust_enabled and volatility > 0:
            if volatility > 0.3:
                trail_pct *= self.config.high_vol_multiplier
            elif volatility < 0.15:
                trail_pct *= self.config.low_vol_multiplier

        # 3. Calculate new stop level based on mode
        if self.config.mode == TrailingMode.PERCENTAGE:
            new_stop = self._calculate_percentage_trail(
                side, reference_price, trail_pct
            )
        elif self.config.mode == TrailingMode.ATR:
            atr = self.atr_cache.get(position.symbol, 0)
            new_stop = self._calculate_atr_trail(
                side, reference_price, atr
            )
        elif self.config.mode == TrailingMode.CHANDELIER:
            atr = self.atr_cache.get(position.symbol, 0)
            new_stop = self._calculate_chandelier_trail(
                side, reference_price, atr
            )
        else:
            new_stop = self._calculate_percentage_trail(
                side, reference_price, trail_pct
            )

        # 4. Check breakeven trigger
        if profit_pct >= self.config.breakeven_trigger_pct:
            breakeven = entry_price
            if side == "long" and new_stop < breakeven:
                new_stop = breakeven
            elif side == "short" and new_stop > breakeven:
                new_stop = breakeven

        # 5. Ensure stop only moves in favorable direction
        if side == "long":
            if new_stop <= current_stop:
                return None, ""
            reason = f"Trail up: ${current_stop:.2f} -> ${new_stop:.2f}"
        else:
            if new_stop >= current_stop:
                return None, ""
            reason = f"Trail down: ${current_stop:.2f} -> ${new_stop:.2f}"

        return new_stop, reason

    def _calculate_percentage_trail(
        self,
        side: str,
        reference_price: float,
        trail_pct: float,
    ) -> float:
        """Calculate trail using fixed percentage."""
        if side == "long":
            return reference_price * (1 - trail_pct / 100)
        else:
            return reference_price * (1 + trail_pct / 100)

    def _calculate_atr_trail(
        self,
        side: str,
        reference_price: float,
        atr: float,
    ) -> float:
        """Calculate trail using ATR multiple."""
        if atr <= 0:
            # Fallback to percentage
            return self._calculate_percentage_trail(
                side, reference_price, self.config.trailing_pct
            )

        distance = atr * self.config.atr_multiplier
        if side == "long":
            return reference_price - distance
        else:
            return reference_price + distance

    def _calculate_chandelier_trail(
        self,
        side: str,
        reference_price: float,
        atr: float,
    ) -> float:
        """
        Calculate Chandelier Exit style stop.

        Long: Highest High - ATR * multiplier
        Short: Lowest Low + ATR * multiplier
        """
        if atr <= 0:
            return self._calculate_percentage_trail(
                side, reference_price, self.config.trailing_pct
            )

        distance = atr * self.config.atr_multiplier
        if side == "long":
            return reference_price - distance
        else:
            return reference_price + distance

    def update_atr(self, symbol: str, atr: float) -> None:
        """Update ATR cache for a symbol."""
        self.atr_cache[symbol] = atr

    def update_broker_stop(
        self,
        position: ManagedPosition,
        new_stop: float,
    ) -> bool:
        """
        Update the stop order at the broker.

        Args:
            position: The managed position
            new_stop: New stop price

        Returns:
            True if update successful
        """
        if not self.broker:
            logger.warning("No broker available for stop update")
            return False

        try:
            # Cancel existing stop order
            if position.stop_order_id:
                self.broker.cancel_order(position.stop_order_id)

            # Place new stop order
            side = "sell" if position.side == "long" else "buy"
            order_id = self.broker.place_order(
                symbol=position.symbol,
                quantity=position.quantity,
                side=side,
                order_type="stop",
                stop_price=new_stop,
                time_in_force="gtc",
            )

            if order_id:
                position.stop_order_id = order_id
                position.current_stop = new_stop
                logger.info(
                    f"{position.symbol}: Updated stop to ${new_stop:.2f} (order: {order_id})"
                )
                return True
            else:
                logger.error(f"Failed to place new stop order for {position.symbol}")
                return False

        except Exception as e:
            logger.error(f"Error updating broker stop for {position.symbol}: {e}")
            return False

    def process_positions(
        self,
        positions: List[ManagedPosition],
        current_time: datetime,
        volatility_map: Dict[str, float] = None,
        expiry_map: Dict[str, datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process all positions and update trailing stops.

        Args:
            positions: List of managed positions
            current_time: Current timestamp
            volatility_map: Symbol -> volatility mapping
            expiry_map: Symbol -> expiry time mapping

        Returns:
            List of stop update events
        """
        volatility_map = volatility_map or {}
        expiry_map = expiry_map or {}
        updates = []

        for position in positions:
            vol = volatility_map.get(position.symbol, 0.0)
            expiry = expiry_map.get(position.symbol)

            new_stop, reason = self.calculate_trailing_stop(
                position=position,
                current_time=current_time,
                volatility=vol,
                expiry_time=expiry,
            )

            if new_stop:
                # Update position object
                old_stop = position.current_stop
                position.current_stop = new_stop

                # Optionally update at broker
                if self.broker and position.stop_order_id:
                    self.update_broker_stop(position, new_stop)

                updates.append({
                    "symbol": position.symbol,
                    "old_stop": old_stop,
                    "new_stop": new_stop,
                    "reason": reason,
                    "time": current_time,
                })

        return updates
