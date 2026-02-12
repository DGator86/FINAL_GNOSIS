"""Alpaca Options Adapter - Extension for Options Trading.

Supports:
- Single-leg options orders
- Multi-leg (MLeg) atomic options orders (spreads, iron condors, etc.)
- Options position management

Author: Super Gnosis Elite Trading System
Version: 2.0.0 - Added atomic multi-leg support
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter, Position


class PositionIntent(str, Enum):
    """Position intent for options orders."""
    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"


@dataclass
class OptionLeg:
    """Represents a single leg in a multi-leg options order."""
    symbol: str  # OCC option symbol (e.g., "AAPL250117C00200000")
    side: str  # "buy" or "sell"
    ratio_qty: int = 1  # Ratio quantity for this leg
    position_intent: PositionIntent = PositionIntent.BUY_TO_OPEN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "ratio_qty": str(self.ratio_qty),
            "position_intent": self.position_intent.value,
        }


@dataclass
class MultiLegOrderResult:
    """Result of a multi-leg order submission."""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    legs_count: int = 0
    order_class: str = "mleg"
    status: str = ""
    filled_qty: Optional[int] = None
    
    # For backward compatibility
    order_ids: List[str] = None
    
    def __post_init__(self):
        if self.order_ids is None:
            self.order_ids = [self.order_id] if self.order_id else []


class AlpacaOptionsAdapter(AlpacaBrokerAdapter):
    """
    Alpaca broker adapter with full Options capabilities.
    
    Supports:
    - Single-leg options orders
    - Atomic multi-leg (MLeg) orders for complex strategies
    - Position intent tracking (open/close)
    
    Multi-leg strategies supported:
    - Vertical spreads (call/put spreads)
    - Iron condors
    - Iron butterflies
    - Straddles/strangles
    - Calendar spreads (roll positions)
    """

    def __init__(self, paper: Optional[bool] = None):
        """Initialize Options Adapter."""
        super().__init__(paper)
        self._api_version = "v2"
        logger.info("AlpacaOptionsAdapter initialized with atomic multi-leg support")

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
        position_intent: Optional[PositionIntent] = None,
    ) -> Optional[str]:
        """
        Place a single-leg option order.
        
        Args:
            symbol: OCC option symbol
            quantity: Number of contracts
            side: "buy" or "sell"
            order_type: "market" or "limit"
            time_in_force: "day", "gtc", etc.
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            position_intent: Intent (buy_to_open, sell_to_close, etc.)
            
        Returns:
            Order ID if successful, None otherwise
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

    def place_multileg_order(
        self,
        legs: List[Union[Dict[str, Any], OptionLeg]],
        quantity: int = 1,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        note: str = "",
    ) -> MultiLegOrderResult:
        """
        Place an atomic multi-leg options order.
        
        This uses Alpaca's native mleg order_class for atomic execution,
        ensuring all legs fill together or none fill at all.
        
        Args:
            legs: List of OptionLeg objects or dicts with keys:
                  symbol, side, ratio_qty (optional), position_intent (optional)
            quantity: Number of spreads/combos to trade
            order_type: "market" or "limit" (limit recommended)
            limit_price: Net limit price for the entire combo
            time_in_force: "day", "gtc", etc.
            note: Strategy description for logs
            
        Returns:
            MultiLegOrderResult with order details
            
        Example - Bull Call Spread:
            legs = [
                OptionLeg("AAPL250117C00190000", "buy", 1, PositionIntent.BUY_TO_OPEN),
                OptionLeg("AAPL250117C00200000", "sell", 1, PositionIntent.SELL_TO_OPEN),
            ]
            result = adapter.place_multileg_order(legs, quantity=1, limit_price=2.50)
            
        Example - Iron Condor:
            legs = [
                OptionLeg("AAPL250117P00180000", "buy", 1, PositionIntent.BUY_TO_OPEN),
                OptionLeg("AAPL250117P00185000", "sell", 1, PositionIntent.SELL_TO_OPEN),
                OptionLeg("AAPL250117C00195000", "sell", 1, PositionIntent.SELL_TO_OPEN),
                OptionLeg("AAPL250117C00200000", "buy", 1, PositionIntent.BUY_TO_OPEN),
            ]
            result = adapter.place_multileg_order(legs, quantity=1, limit_price=1.80)
        """
        logger.info(f"Executing Atomic Multi-Leg Order: {note} | {len(legs)} legs")
        
        # Validate legs
        if not legs:
            return MultiLegOrderResult(
                success=False,
                message="No legs provided",
            )
        
        if len(legs) < 2:
            return MultiLegOrderResult(
                success=False,
                message="Multi-leg orders require at least 2 legs",
            )
        
        # Convert legs to API format
        legs_payload = []
        for leg in legs:
            if isinstance(leg, OptionLeg):
                legs_payload.append(leg.to_dict())
            elif isinstance(leg, dict):
                # Handle legacy dict format
                leg_dict = {
                    "symbol": leg["symbol"],
                    "side": leg["side"],
                    "ratio_qty": str(leg.get("ratio_qty", leg.get("qty", 1))),
                }
                # Add position_intent if provided
                if "position_intent" in leg:
                    leg_dict["position_intent"] = leg["position_intent"]
                else:
                    # Default intent based on side
                    leg_dict["position_intent"] = (
                        PositionIntent.BUY_TO_OPEN.value if leg["side"] == "buy"
                        else PositionIntent.SELL_TO_OPEN.value
                    )
                legs_payload.append(leg_dict)
            else:
                return MultiLegOrderResult(
                    success=False,
                    message=f"Invalid leg type: {type(leg)}",
                )
        
        # Validate ratio_qty are in simplest form (GCD = 1)
        ratios = [int(leg["ratio_qty"]) for leg in legs_payload]
        from math import gcd
        from functools import reduce
        overall_gcd = reduce(gcd, ratios)
        if overall_gcd > 1:
            logger.warning(f"Leg ratios {ratios} not in simplest form (GCD={overall_gcd}). Simplifying...")
            for leg in legs_payload:
                leg["ratio_qty"] = str(int(leg["ratio_qty"]) // overall_gcd)
        
        # Build order payload
        order_payload = {
            "order_class": "mleg",
            "qty": str(quantity),
            "type": order_type,
            "time_in_force": time_in_force,
            "legs": legs_payload,
        }
        
        if order_type == "limit" and limit_price is not None:
            order_payload["limit_price"] = str(limit_price)
        
        logger.debug(f"Multi-leg order payload: {json.dumps(order_payload, indent=2)}")
        
        try:
            # Use REST API directly for mleg orders
            result = self._submit_multileg_order(order_payload)
            
            if result.success:
                logger.info(
                    f"✅ Multi-leg order submitted | "
                    f"Order ID: {result.order_id} | "
                    f"Legs: {len(legs)} | "
                    f"Strategy: {note}"
                )
            else:
                logger.error(f"❌ Multi-leg order failed: {result.message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-leg order exception: {e}")
            return MultiLegOrderResult(
                success=False,
                message=str(e),
            )
    
    def _submit_multileg_order(self, payload: Dict[str, Any]) -> MultiLegOrderResult:
        """
        Submit multi-leg order via REST API.
        
        Args:
            payload: Order payload with order_class="mleg"
            
        Returns:
            MultiLegOrderResult
        """
        url = f"{self.base_url}/{self._api_version}/orders"
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code in [200, 201]:
                data = response.json()
                return MultiLegOrderResult(
                    success=True,
                    order_id=data.get("id"),
                    message="Order submitted successfully",
                    legs_count=len(payload.get("legs", [])),
                    status=data.get("status", ""),
                    filled_qty=data.get("filled_qty"),
                )
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                except Exception:
                    pass
                
                return MultiLegOrderResult(
                    success=False,
                    message=f"API error ({response.status_code}): {error_msg}",
                )
                
        except requests.exceptions.Timeout:
            return MultiLegOrderResult(
                success=False,
                message="Request timeout",
            )
        except requests.exceptions.RequestException as e:
            return MultiLegOrderResult(
                success=False,
                message=f"Request failed: {e}",
            )
    
    async def place_multileg_order_async(
        self,
        legs: List[Union[Dict[str, Any], OptionLeg]],
        quantity: int = 1,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        note: str = "",
    ) -> MultiLegOrderResult:
        """
        Async version of place_multileg_order.
        
        For backward compatibility - wraps the synchronous method.
        """
        return self.place_multileg_order(
            legs=legs,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            note=note,
        )
    
    def close_multileg_position(
        self,
        legs: List[Union[Dict[str, Any], OptionLeg]],
        quantity: int = 1,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        note: str = "",
    ) -> MultiLegOrderResult:
        """
        Close an existing multi-leg position.
        
        Automatically inverts the sides and uses close intents.
        
        Args:
            legs: Original legs of the position to close
            quantity: Number of spreads to close
            order_type: "market" or "limit"
            limit_price: Net limit price for closing
            time_in_force: TIF
            note: Description
            
        Returns:
            MultiLegOrderResult
        """
        close_legs = []
        
        for leg in legs:
            if isinstance(leg, OptionLeg):
                # Invert side and change intent to close
                close_side = "sell" if leg.side == "buy" else "buy"
                close_intent = (
                    PositionIntent.SELL_TO_CLOSE if leg.side == "buy"
                    else PositionIntent.BUY_TO_CLOSE
                )
                close_legs.append(OptionLeg(
                    symbol=leg.symbol,
                    side=close_side,
                    ratio_qty=leg.ratio_qty,
                    position_intent=close_intent,
                ))
            elif isinstance(leg, dict):
                close_side = "sell" if leg["side"] == "buy" else "buy"
                close_intent = (
                    PositionIntent.SELL_TO_CLOSE.value if leg["side"] == "buy"
                    else PositionIntent.BUY_TO_CLOSE.value
                )
                close_legs.append({
                    "symbol": leg["symbol"],
                    "side": close_side,
                    "ratio_qty": leg.get("ratio_qty", leg.get("qty", 1)),
                    "position_intent": close_intent,
                })
        
        return self.place_multileg_order(
            legs=close_legs,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            note=f"Close: {note}",
        )

    def get_options_buying_power(self) -> float:
        """Get available options buying power."""
        account = self.get_account()
        return (
            account.options_buying_power
            if account.options_buying_power is not None
            else account.buying_power
        )
    
    def get_multileg_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a multi-leg order.
        
        Args:
            order_id: The order ID to check
            
        Returns:
            Order details dict or None if not found
        """
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                "id": order.id,
                "status": order.status,
                "filled_qty": order.filled_qty,
                "order_class": getattr(order, "order_class", None),
                "legs": getattr(order, "legs", []),
            }
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None


# Helper functions for creating common strategies
def create_vertical_spread(
    underlying: str,
    expiration: str,
    option_type: str,  # "C" or "P"
    buy_strike: float,
    sell_strike: float,
    is_opening: bool = True,
) -> List[OptionLeg]:
    """
    Create legs for a vertical spread (bull/bear call/put spread).
    
    Args:
        underlying: Underlying symbol (e.g., "AAPL")
        expiration: Expiration date in YYMMDD format (e.g., "250117")
        option_type: "C" for call, "P" for put
        buy_strike: Strike price for the long leg
        sell_strike: Strike price for the short leg
        is_opening: True for opening, False for closing
        
    Returns:
        List of OptionLeg objects
    """
    def format_strike(strike: float) -> str:
        return f"{int(strike * 1000):08d}"
    
    buy_symbol = f"{underlying}{expiration}{option_type}{format_strike(buy_strike)}"
    sell_symbol = f"{underlying}{expiration}{option_type}{format_strike(sell_strike)}"
    
    if is_opening:
        return [
            OptionLeg(buy_symbol, "buy", 1, PositionIntent.BUY_TO_OPEN),
            OptionLeg(sell_symbol, "sell", 1, PositionIntent.SELL_TO_OPEN),
        ]
    else:
        return [
            OptionLeg(buy_symbol, "sell", 1, PositionIntent.SELL_TO_CLOSE),
            OptionLeg(sell_symbol, "buy", 1, PositionIntent.BUY_TO_CLOSE),
        ]


def create_iron_condor(
    underlying: str,
    expiration: str,
    put_buy_strike: float,
    put_sell_strike: float,
    call_sell_strike: float,
    call_buy_strike: float,
    is_opening: bool = True,
) -> List[OptionLeg]:
    """
    Create legs for an iron condor.
    
    Args:
        underlying: Underlying symbol
        expiration: Expiration date in YYMMDD format
        put_buy_strike: Lower put strike (buy)
        put_sell_strike: Higher put strike (sell)
        call_sell_strike: Lower call strike (sell)
        call_buy_strike: Higher call strike (buy)
        is_opening: True for opening, False for closing
        
    Returns:
        List of OptionLeg objects (4 legs)
    """
    def format_strike(strike: float) -> str:
        return f"{int(strike * 1000):08d}"
    
    put_buy = f"{underlying}{expiration}P{format_strike(put_buy_strike)}"
    put_sell = f"{underlying}{expiration}P{format_strike(put_sell_strike)}"
    call_sell = f"{underlying}{expiration}C{format_strike(call_sell_strike)}"
    call_buy = f"{underlying}{expiration}C{format_strike(call_buy_strike)}"
    
    if is_opening:
        return [
            OptionLeg(put_buy, "buy", 1, PositionIntent.BUY_TO_OPEN),
            OptionLeg(put_sell, "sell", 1, PositionIntent.SELL_TO_OPEN),
            OptionLeg(call_sell, "sell", 1, PositionIntent.SELL_TO_OPEN),
            OptionLeg(call_buy, "buy", 1, PositionIntent.BUY_TO_OPEN),
        ]
    else:
        return [
            OptionLeg(put_buy, "sell", 1, PositionIntent.SELL_TO_CLOSE),
            OptionLeg(put_sell, "buy", 1, PositionIntent.BUY_TO_CLOSE),
            OptionLeg(call_sell, "buy", 1, PositionIntent.BUY_TO_CLOSE),
            OptionLeg(call_buy, "sell", 1, PositionIntent.SELL_TO_CLOSE),
        ]


__all__ = [
    "AlpacaOptionsAdapter",
    "OptionLeg",
    "PositionIntent",
    "MultiLegOrderResult",
    "create_vertical_spread",
    "create_iron_condor",
]
