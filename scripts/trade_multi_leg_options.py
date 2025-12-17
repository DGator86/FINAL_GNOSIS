"""
Multi-Leg Options Trading Script
Execute complex options strategies (spreads, straddles, etc.) in paper trading

Usage:
    python scripts/trade_multi_leg_options.py
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brokers.alpaca_client import AlpacaClient  # noqa: E402


def get_current_spy_price() -> float:
    """Helper to get current SPY price for strike selection"""
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )

    request = StockLatestQuoteRequest(symbol_or_symbols="SPY")
    quote = client.get_stock_latest_quote(request)
    return quote["SPY"].ask_price


def place_bull_call_spread(
    client: AlpacaClient, underlying_price: float, quantity: int = 1
) -> Dict[str, Any]:
    """
    Place a bull call spread (bullish strategy with limited risk/reward)

    Example: If SPY is at $600
    - Buy $600 call
    - Sell $610 call

    Max profit: $10 - net premium paid
    Max loss: Net premium paid
    """
    atm_strike = round(underlying_price / 5) * 5
    otm_strike = atm_strike + 10

    # Note: Update expiration date as needed
    expiration = "241220"  # YYMMDD format

    legs = [
        {"symbol": f"SPY{expiration}C00{atm_strike:05d}00", "side": "buy", "ratio_qty": 1},
        {"symbol": f"SPY{expiration}C00{otm_strike:05d}00", "side": "sell", "ratio_qty": 1},
    ]

    logger.info("Placing Bull Call Spread:")
    logger.info(f"  Buy:  SPY ${atm_strike} Call")
    logger.info(f"  Sell: SPY ${otm_strike} Call")
    logger.info(f"  Quantity: {quantity}")

    result = client.place_multi_leg_option_order(legs, quantity=quantity)

    logger.success(f"Order placed! ID: {result['id']}")
    logger.info(f"Status: {result['status']}")

    return result


def place_bear_put_spread(
    client: AlpacaClient, underlying_price: float, quantity: int = 1
) -> Dict[str, Any]:
    """
    Place a bear put spread (bearish strategy with limited risk/reward)

    Example: If SPY is at $600
    - Buy $600 put
    - Sell $590 put

    Max profit: $10 - net premium paid
    Max loss: Net premium paid
    """
    atm_strike = round(underlying_price / 5) * 5
    otm_strike = atm_strike - 10

    expiration = "241220"

    legs = [
        {"symbol": f"SPY{expiration}P00{atm_strike:05d}00", "side": "buy", "ratio_qty": 1},
        {"symbol": f"SPY{expiration}P00{otm_strike:05d}00", "side": "sell", "ratio_qty": 1},
    ]

    logger.info("Placing Bear Put Spread:")
    logger.info(f"  Buy:  SPY ${atm_strike} Put")
    logger.info(f"  Sell: SPY ${otm_strike} Put")
    logger.info(f"  Quantity: {quantity}")

    result = client.place_multi_leg_option_order(legs, quantity=quantity)

    logger.success(f"Order placed! ID: {result['id']}")
    logger.info(f"Status: {result['status']}")

    return result


def place_long_straddle(
    client: AlpacaClient, underlying_price: float, quantity: int = 1
) -> Dict[str, Any]:
    """
    Place a long straddle (volatility play - profit from big moves either direction)

    Example: If SPY is at $600
    - Buy $600 call
    - Buy $600 put

    Max profit: Unlimited
    Max loss: Total premium paid
    """
    atm_strike = round(underlying_price / 5) * 5

    expiration = "241220"

    legs = [
        {"symbol": f"SPY{expiration}C00{atm_strike:05d}00", "side": "buy", "ratio_qty": 1},
        {"symbol": f"SPY{expiration}P00{atm_strike:05d}00", "side": "buy", "ratio_qty": 1},
    ]

    logger.info("Placing Long Straddle:")
    logger.info(f"  Buy: SPY ${atm_strike} Call")
    logger.info(f"  Buy: SPY ${atm_strike} Put")
    logger.info(f"  Quantity: {quantity}")

    result = client.place_multi_leg_option_order(legs, quantity=quantity)

    logger.success(f"Order placed! ID: {result['id']}")
    logger.info(f"Status: {result['status']}")

    return result


def place_iron_condor(
    client: AlpacaClient, underlying_price: float, quantity: int = 1
) -> Dict[str, Any]:
    """
    Place an iron condor (neutral strategy - profit from low volatility)

    Example: If SPY is at $600
    - Sell $610 call
    - Buy $620 call
    - Sell $590 put
    - Buy $580 put

    Max profit: Net premium received
    Max loss: Width of spread - net premium
    """
    atm_strike = round(underlying_price / 5) * 5

    expiration = "241220"

    legs = [
        # Call spread (above current price)
        {"symbol": f"SPY{expiration}C00{atm_strike + 10:05d}00", "side": "sell", "ratio_qty": 1},
        {"symbol": f"SPY{expiration}C00{atm_strike + 20:05d}00", "side": "buy", "ratio_qty": 1},
        # Put spread (below current price)
        {"symbol": f"SPY{expiration}P00{atm_strike - 10:05d}00", "side": "sell", "ratio_qty": 1},
        {"symbol": f"SPY{expiration}P00{atm_strike - 20:05d}00", "side": "buy", "ratio_qty": 1},
    ]

    logger.info("Placing Iron Condor:")
    logger.info(f"  Sell: SPY ${atm_strike + 10} Call")
    logger.info(f"  Buy:  SPY ${atm_strike + 20} Call")
    logger.info(f"  Sell: SPY ${atm_strike - 10} Put")
    logger.info(f"  Buy:  SPY ${atm_strike - 20} Put")
    logger.info(f"  Quantity: {quantity}")

    result = client.place_multi_leg_option_order(legs, quantity=quantity)

    logger.success(f"Order placed! ID: {result['id']}")
    logger.info(f"Status: {result['status']}")

    return result


def main() -> int:
    """Interactive menu for multi-leg options trading"""
    logger.info("=" * 60)
    logger.info("Multi-Leg Options Trading (Paper Trading)")
    logger.info("=" * 60)

    # Initialize client
    client = AlpacaClient.from_env(mode="paper")

    # Get current SPY price
    spy_price = get_current_spy_price()
    logger.info(f"Current SPY Price: ${spy_price:.2f}")
    logger.info("")

    # Display menu
    print("\nAvailable Strategies:")
    print("1. Bull Call Spread (Bullish, Limited Risk)")
    print("2. Bear Put Spread (Bearish, Limited Risk)")
    print("3. Long Straddle (High Volatility Expected)")
    print("4. Iron Condor (Low Volatility Expected)")
    print("5. Exit")

    choice = input("\nSelect strategy (1-5): ").strip()

    if choice == "1":
        place_bull_call_spread(client, spy_price, quantity=1)
    elif choice == "2":
        place_bear_put_spread(client, spy_price, quantity=1)
    elif choice == "3":
        place_long_straddle(client, spy_price, quantity=1)
    elif choice == "4":
        place_iron_condor(client, spy_price, quantity=1)
    elif choice == "5":
        logger.info("Exiting...")
        return 0
    else:
        logger.error("Invalid choice")
        return 1

    logger.info("")
    logger.success("Trade execution complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
