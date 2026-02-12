"""Legacy SPY paper runner refactored to trade the full active universe."""

from __future__ import annotations

import argparse
import os
import time

from brokers.alpaca_client import AlpacaClient
from scripts.universe_paper_trading import run_universe_paper_cycle
from trade.trade_agent_v2 import TradeAgent


def main(args: argparse.Namespace) -> None:
    interval = args.interval
    dry_run = args.dry_run

    trade_agent = TradeAgent.from_config(args.config)
    alpaca = AlpacaClient.from_env()

    print("[INFO] Starting universe paper daemon (legacy SPY entrypoint)")
    print(f"[INFO] Alpaca endpoint: {alpaca.base_url} (paper-only)")
    if "paper" not in alpaca.base_url:
        print("[WARN] Alpaca endpoint does not look like paper trading - check ALPACA_BASE_URL")
    if dry_run:
        print("[INFO] DRY RUN enabled - Alpaca orders will NOT be sent.")
    else:
        print("[INFO] Live paper submission ENABLED - orders will be sent to Alpaca Paper.")

    while True:
        try:
            run_universe_paper_cycle(
                trade_agent=trade_agent,
                alpaca=alpaca,
                config_path=args.config,
                max_names=args.max_names,
                dry_run=dry_run,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[FATAL CYCLE ERROR] {exc}")
        finally:
            print(f"[INFO] Sleeping for {interval} seconds...\n")
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--interval", type=int, default=int(os.environ.get("GNOSIS_PAPER_INTERVAL", 300)))
    parser.add_argument("--max-names", type=int, default=int(os.environ.get("GNOSIS_PAPER_MAX_NAMES", 25)))
    parser.add_argument("--dry-run", action="store_true", help="Run without sending Alpaca paper orders")
    args = parser.parse_args()

    main(args)
