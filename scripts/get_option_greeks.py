"""
Get Real-Time Option Greeks
Fetch delta, gamma, theta, vega, rho for any option contract

Usage:
    python scripts/get_option_greeks.py SPY241220C00600000
    python scripts/get_option_greeks.py SPY241220C00600000 SPY241220P00600000
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brokers.alpaca_client import AlpacaClient
from loguru import logger


def display_option_snapshot(symbol: str, snapshot: dict):
    """Display option snapshot in a readable format"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Option: {symbol}")
    logger.info(f"{'=' * 60}")

    # Quote data
    quote = snapshot["latest_quote"]
    logger.info(f"\nLatest Quote:")
    logger.info(f"  Bid: ${quote['bid_price']:.2f} x {quote['bid_size']}")
    logger.info(f"  Ask: ${quote['ask_price']:.2f} x {quote['ask_size']}")
    logger.info(f"  Spread: ${quote['ask_price'] - quote['bid_price']:.2f}")
    logger.info(f"  Mid: ${(quote['bid_price'] + quote['ask_price']) / 2:.2f}")

    # Greeks
    greeks = snapshot["greeks"]
    logger.info(f"\nGreeks:")
    logger.info(f"  Delta: {greeks['delta']:.4f}")
    logger.info(f"  Gamma: {greeks['gamma']:.4f}")
    logger.info(f"  Theta: {greeks['theta']:.4f}")
    logger.info(f"  Vega:  {greeks['vega']:.4f}")
    logger.info(f"  Rho:   {greeks['rho']:.4f}")

    # Implied Volatility
    iv = snapshot["implied_volatility"]
    if iv:
        logger.info(f"\nImplied Volatility: {iv * 100:.2f}%")


def main():
    """Get option Greeks for specified symbols"""
    if len(sys.argv) < 2:
        logger.error("Usage: python get_option_greeks.py <OPTION_SYMBOL> [<OPTION_SYMBOL2> ...]")
        logger.info("\nExample:")
        logger.info("  python scripts/get_option_greeks.py SPY241220C00600000")
        logger.info("  python scripts/get_option_greeks.py SPY241220C00600000 SPY241220P00600000")
        return 1

    symbols = sys.argv[1:]

    logger.info("Fetching Option Greeks...")
    logger.info(f"Symbols: {', '.join(symbols)}")

    # Initialize client
    client = AlpacaClient.from_env(mode="paper")

    try:
        # Get snapshots
        snapshots = client.get_option_snapshot(symbols)

        # Display each snapshot
        for symbol in symbols:
            if symbol in snapshots:
                display_option_snapshot(symbol, snapshots[symbol])
            else:
                logger.warning(f"\nNo data found for {symbol}")

        logger.success("\n✓ Greeks retrieved successfully")
        return 0

    except Exception as e:
        logger.error(f"\n✗ Error fetching Greeks: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
