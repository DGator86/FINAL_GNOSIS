"""Option utilities for symbol formatting and calculations."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict


class OptionUtils:
    """Utilities for handling option symbols and calculations."""

    @staticmethod
    def generate_occ_symbol(
        symbol: str, expiration: datetime, option_type: str, strike: float
    ) -> str:
        """Generate deterministic OCC option symbol compatible with Alpaca.

        The builder normalizes the underlying to uppercase, rounds strikes to
        three decimals (per OCC thousandth convention), and zero-pads to eight
        digits. Spaces in short underlyings are removed because Alpaca rejects
        space-padded roots.
        """

        root = symbol.upper()
        date_str = expiration.strftime("%y%m%d")
        type_char = "C" if option_type.lower() == "call" else "P"

        strike_dec = Decimal(str(strike)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        strike_int = int(strike_dec * 1000)
        strike_str = f"{strike_int:08d}"

        return f"{root}{date_str}{type_char}{strike_str}"

    @staticmethod
    def to_alpaca_symbol(symbol: str, expiration: datetime, option_type: str, strike: float) -> str:
        """Explicit helper for Alpaca/ OCC option symbol formatting.

        Alias for :meth:`generate_occ_symbol` to keep call sites readable.
        """

        return OptionUtils.generate_occ_symbol(symbol, expiration, option_type, strike)

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
