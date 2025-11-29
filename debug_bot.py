import os
import sys
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load env
load_dotenv()

from gnosis.trading.live_bot import LiveTradingBot

async def main():
    try:
        print("Initializing LiveTradingBot for SPY...")
        bot = LiveTradingBot(
            symbol='SPY',
            bar_interval="1Min",
            enable_memory=True,
            enable_trading=True,
            paper_mode=True
        )
        print("Initialization successful!")
    except Exception as e:
        print(f"Error initializing bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
