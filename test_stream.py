import asyncio
import os
import logging
from alpaca.data.live import StockDataStream
from dotenv import load_dotenv

# Load env
load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Initializing StockDataStream...")
    stream = StockDataStream(api_key, secret_key)

    async def handle_bar(bar):
        logger.info(f"Received bar: {bar}")

    symbol = "BTC/USD"  # Crypto trades 24/7, good for testing off-hours
    # Or use a crypto stream if StockDataStream doesn't support crypto?
    # StockDataStream is for SIP (Stocks). It won't stream BTC.
    # We need CryptoDataStream for BTC.
    # But for Stocks, we can't test after hours easily unless we use 'ETH/USD' on Crypto stream to verify network.
    # Let's try to subscribe to SPY anyway, maybe we get a heartbeat or late trade?
    # Actually, let's use CryptoDataStream to verify network/auth if possible, but we want to test StockDataStream.

    # Let's stick to StockDataStream and SPY. Even if market is closed, we might get a connection success message.

    logger.info(f"Subscribing to SPY...")
    stream.subscribe_bars(handle_bar, "SPY")

    logger.info("Starting stream (using _run_forever)...")
    if hasattr(stream, "_run_forever"):
        await stream._run_forever()
    else:
        await stream.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
