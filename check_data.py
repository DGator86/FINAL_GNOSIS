from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from datetime import datetime, timedelta
import pytz

try:
    alpaca = AlpacaMarketDataAdapter()
    end = datetime.now(pytz.utc)
    start = end - timedelta(days=5)
    bars = alpaca.get_bars("SPY", start, end, timeframe="1Day")
    if bars:
        print(f"Latest SPY Bar: {bars[-1].timestamp} Close: {bars[-1].close}")
    else:
        print("No bars returned for last 5 days.")
        
    # Check "Future" date
    future_start = datetime(2025, 4, 1, tzinfo=pytz.utc)
    future_end = datetime(2025, 4, 5, tzinfo=pytz.utc)
    bars_future = alpaca.get_bars("SPY", future_start, future_end, timeframe="1Day")
    print(f"Future Bars (2025-04): {len(bars_future)}")
    
except Exception as e:
    print(f"Error: {e}")
