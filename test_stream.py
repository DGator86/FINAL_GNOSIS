import asyncio
import logging
import os

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

    symbol = "SPY"
    logger.info(f"Subscribing to {symbol}...")
    stream.subscribe_bars(handle_bar, symbol)

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
