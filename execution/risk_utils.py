"""Lightweight risk sizing helpers shared across brokers/tests."""

from __future__ import annotations



def is_option_symbol(symbol: str) -> bool:
    sanitized = symbol.replace(" ", "").upper()
    return len(sanitized) >= 15 and sanitized[-9] in {"C", "P"}


def calculate_order_value(symbol: str, quantity: float, price: float) -> float:
    if is_option_symbol(symbol):
        return abs(quantity) * price * 100.0
    return abs(quantity) * price


def assert_within_max(symbol: str, order_value: float, portfolio_value: float, max_position_pct: float) -> None:
    max_position_value = portfolio_value * max_position_pct
    if order_value > max_position_value:
        raise ValueError(
            f"Order size ${order_value:,.2f} exceeds maximum position size of "
            f"${max_position_value:,.2f} ({max_position_pct*100:.1f}% of ${portfolio_value:,.2f})"
        )


__all__ = ["is_option_symbol", "calculate_order_value", "assert_within_max"]
