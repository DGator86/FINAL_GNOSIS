"""
Debug script to isolate UnifiedTradingBot initialization.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from gnosis.unified_trading_bot import UnifiedTradingBot

# Mock config
config = {
    "risk": {
        "risk_per_trade_pct": 0.02,
        "max_positions": 5,
        "daily_loss_limit": 0.05,
        "trailing_stop_pct": 0.01,
        "trailing_stop_activation": 0.02,
    }
}

try:
    print("Attempting to instantiate UnifiedTradingBot...")
    bot = UnifiedTradingBot(config, enable_trading=False, paper_mode=True)
    print("SUCCESS: UnifiedTradingBot instantiated correctly.")
except TypeError as e:
    print(f"FAILURE: TypeError caught: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"FAILURE: Unexpected exception: {e}")
    import traceback

    traceback.print_exc()
