"""Alpaca Options Adapter - Extension for Options Trading."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter, Position


class AlpacaOptionsAdapter(AlpacaBrokerAdapter):
    """Alpaca broker adapter with Options capabilities."""

    def __init__(self, paper: Optional[bool] = None):
        """Initialize Options Adapter."""
        super().__init__(paper)
        logger.info("AlpacaOptionsAdapter initialized")

    def get_option_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific option symbol."""
        return self.get_position(symbol)

    def place_option_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[str]:
        """
        Place a single-leg option order.
        Wrapper around place_order but with extra logging/validation for options.
        """
        # Basic validation for OCC symbol format (simple check)
        if len(symbol) < 15:
            logger.warning(f"Symbol {symbol} does not look like a valid OCC option symbol")

        return self.place_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
        )

    async def place_multileg_order(
        self,
        legs: List[Dict[str, Any]],
        order_type: str = "market",
        time_in_force: str = "day",
        note: str = "",
    ) -> List[str]:
        """
        Place a multi-leg options order (Sequentially for now).

        Args:
            legs: List of dicts with keys: symbol, side, qty
            order_type: Order type (market/limit)
            time_in_force: TIF
            note: Strategy description for logs

        Returns:
            List of Order IDs
        """
        logger.info(f"Executing Multi-Leg Order: {note}")
        order_ids = []

        # TODO: Implement atomic multi-leg when supported/verified
        # For now, execute sequentially
        for i, leg in enumerate(legs):
            symbol = leg["symbol"]
            side = leg["side"]
            qty = leg["qty"]

            logger.info(f"Submitting Leg {i + 1}/{len(legs)}: {side.upper()} {qty} {symbol}")

            order_id = self.place_option_order(
                symbol=symbol,
                quantity=qty,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
            )

            if order_id:
                order_ids.append(order_id)
            else:
                logger.error(f"Failed to submit leg {i + 1} ({symbol}). Stopping execution.")
                # In a real system, we might want to attempt to unwind previous legs here
                break

            # Small delay to ensure order acceptance/processing if needed
            # await asyncio.sleep(0.5)

        if len(order_ids) == len(legs):
            logger.info(f"All {len(legs)} legs submitted successfully")
        else:
            logger.warning(f"Partial execution: {len(order_ids)}/{len(legs)} legs submitted")

        return order_ids

    def get_options_buying_power(self) -> float:
        """Get available options buying power."""
        account = self.get_account()
        return (
            account.options_buying_power
            if account.options_buying_power is not None
            else account.buying_power
        )
