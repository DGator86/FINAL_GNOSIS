import inspect

from alpaca.data.live import StockDataStream

print("StockDataStream._run_forever source:")
try:
    print(inspect.getsource(StockDataStream._run_forever))
except Exception as e:
    print(f"Could not get source: {e}")

print("\nStockDataStream._run_forever signature:")
try:
    print(inspect.signature(StockDataStream._run_forever))
except Exception as e:
    print(f"Could not get signature: {e}")
