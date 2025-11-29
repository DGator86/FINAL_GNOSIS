import os
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

print(f"API Key: {api_key[:5]}...")
print(f"Secret Key: {secret_key[:5]}...")

client = StockHistoricalDataClient(api_key, secret_key)

symbol = "SPY"
end = datetime.now(timezone.utc)
start = end - timedelta(minutes=200)

print(f"Requesting data for {symbol} from {start} to {end}")

# Test 1: Default (SIP?)
try:
    print("\n--- Test 1: Default Feed ---")
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end
    )
    bars = client.get_stock_bars(req)
    print(f"Response keys: {bars.keys()}")
    if symbol in bars:
        print(f"Got {len(bars[symbol])} bars")
        print(f"First bar: {bars[symbol][0]}")
    else:
        print("No bars found")
except Exception as e:
    print(f"Error: {e}")

# Test 2: IEX
try:
    print("\n--- Test 2: IEX Feed ---")
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX
    )
    bars = client.get_stock_bars(req)
    print(f"Response keys: {bars.keys()}")
    if symbol in bars:
        print(f"Got {len(bars[symbol])} bars")
        print(f"First bar: {bars[symbol][0]}")
    else:
        print("No bars found")
except Exception as e:
    print(f"Error: {e}")

# Test 3: SIP explicitly
try:
    print("\n--- Test 3: SIP Feed ---")
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.SIP
    )
    bars = client.get_stock_bars(req)
    print(f"Response keys: {bars.keys()}")
    if symbol in bars:
        print(f"Got {len(bars[symbol])} bars")
        print(f"First bar: {bars[symbol][0]}")
    else:
        print("No bars found")
except Exception as e:
    print(f"Error: {e}")
