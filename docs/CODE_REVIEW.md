# Code Review Notes

## Security: Hardcoded Alpaca API credentials (fixed)
- Removed baked-in Alpaca keys from `AlpacaBrokerAdapter`; credentials must now be supplied via environment variables or explicit constructor arguments, preventing accidental use of committed secrets.

## Reliability: Adapter initialization performs live network calls (fixed)
- Constructor no longer performs an eager `get_account()` call. Account retrieval (and options permission validation) now occurs lazily on the first `get_account()` invocation, so constructing the adapter no longer fails when the broker is temporarily unavailable.

## Shutdown handling: background event loop not fully cleaned up (fixed)
- `GnosisLiveTradingEngine.stop()` now cancels tasks, runs them to completion with `asyncio.gather(..., return_exceptions=True)`, and closes the loop after the thread joins, ensuring clean teardown between start/stop cycles.
