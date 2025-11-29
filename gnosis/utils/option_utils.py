"""Option utilities for symbol formatting and calculations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


class OptionUtils:
    """Utilities for handling option symbols and calculations."""

    @staticmethod
    def generate_occ_symbol(
        symbol: str, expiration: datetime, option_type: str, strike: float
    ) -> str:
        """Generate OCC standard option symbol.

        Format: SYMBOLYYMMDD[C/P]00000000
        Example: AAPL230616C00150000 (AAPL June 16 2023 150.00 Call)

        Args:
            symbol: Underlying symbol (e.g., AAPL)
            expiration: Expiration date
            option_type: 'call' or 'put'
            strike: Strike price

        Returns:
            OCC formatted symbol string
        """
        # Format date: YYMMDD
        date_str = expiration.strftime("%y%m%d")

        # Format type: C or P
        type_char = "C" if option_type.lower() == "call" else "P"

        # Format strike: 8 digits, multiplied by 1000, zero padded
        # 150.00 -> 150000 -> 00150000
        strike_int = int(strike * 1000)
        strike_str = f"{strike_int:08d}"

        # Pad symbol to 6 chars with spaces if needed (standard OCC)
        # But most APIs (Alpaca/Polygon) accept compact format: AAPL...
        # We'll use compact format as it's more commonly accepted by modern APIs
        return f"{symbol}{date_str}{type_char}{strike_str}"

    @staticmethod
    def parse_occ_symbol(occ_symbol: str) -> Dict[str, Any]:
        """Parse OCC symbol into components.

        Args:
            occ_symbol: OCC formatted symbol

        Returns:
            Dictionary with symbol, expiration, type, strike
        """
        try:
            # Find the position of the date (starts with 2 digits year)
            # This is tricky without fixed width, but we know the suffix is fixed length
            # Date(6) + Type(1) + Strike(8) = 15 chars
            root = occ_symbol[:-15]
            suffix = occ_symbol[-15:]

            date_str = suffix[:6]
            type_char = suffix[6]
            strike_str = suffix[7:]

            expiration = datetime.strptime(date_str, "%y%m%d")
            option_type = "call" if type_char == "C" else "put"
            strike = float(strike_str) / 1000.0

            return {
                "symbol": root,
                "expiration": expiration,
                "option_type": option_type,
                "strike": strike,
            }
        except Exception as e:
            raise ValueError(f"Invalid OCC symbol format: {occ_symbol}") from e

    @staticmethod
    def calculate_moneyness(current_price: float, strike: float, option_type: str) -> float:
        """Calculate moneyness percentage.

        Args:
            current_price: Underlying price
            strike: Option strike
            option_type: 'call' or 'put'

        Returns:
            Moneyness % (positive = ITM, negative = OTM)
        """
        if option_type.lower() == "call":
            return (current_price - strike) / strike
        else:
            return (strike - current_price) / strike
