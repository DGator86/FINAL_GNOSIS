# Alpaca Paper Trading Assessment

## Why trades are not executing

1. **Launcher scripts import missing modules**
   - `start_paper_trading.py`, `start_full_trading_system.py`, and related launchers previously referenced `gnosis.trading.live_bot.LiveTradingBot`, but no such package or file existed in the repository. The scripts crashed on import before reaching any trading logic, so no orders could be sent to Alpaca. A minimal `gnosis` package with a working live bot has been added so the launchers now run and can submit paper orders when configured.

2. **Main CLI defaults to non-executing mode**
   - The production entrypoint (`main.py`) only turns on live execution when a broker adapter is provided. By default, `build_pipeline` receives no broker, so `auto_execute` is `False` and trade ideas are never submitted. Execution is enabled only when you run `main.py live-loop` without `--dry-run` and the Alpaca adapter successfully connects.

3. **Broker credentials are required at runtime**
   - The Alpaca adapter loads `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` from the environment and raises an error if they are missing. Without these values in a `.env` file (or shell), the adapter cannot connect to paper trading and the system stays in analysis-only mode.

## How to enable active paper trading

- Run the supported CLI instead of the broken launcher scripts:
  ```bash
  python main.py live-loop --symbol SPY
  ```
- Ensure a `.env` file (or environment variables) supplies `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` before starting.
- Keep `--dry-run` **off** so the broker adapter is created and `auto_execute` is enabled.

With those settings, the pipeline will generate trade ideas and submit paper orders through the Alpaca adapter when market hours and risk checks allow.
