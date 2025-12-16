# scripts/trading_universe.py
"""
Universe-wide trading entrypoint.

Features:
- Uses the dynamic active watchlist from universe.watchlist_loader.load_active_watchlist.
- Supports manual symbol overrides via --symbols.
- Supports paper and live modes via --mode.
- Supports one-shot run (default) or daemon mode via --daemon.
- Submits broker-ready trades using TradeAgent v2 and AlpacaClient.

Examples
--------
# One-shot paper run over current universe (no orders sent)
python scripts/trading_universe.py --mode paper --dry-run

# One-shot paper run with real paper orders
python scripts/trading_universe.py --mode paper --execute

# One-shot live run, DRY RUN ONLY (safety)
python scripts/trading_universe.py --mode live --dry-run

# One-shot with explicit symbols (bypasses watchlist)
python scripts/trading_universe.py --mode paper --dry-run --symbols AAPL MSFT TSLA
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from typing import List

from brokers.alpaca_client import AlpacaClient
from pipeline.full_pipeline import run_full_pipeline_for_symbol
from trade.trade_agent_v2 import TradeAgent
from universe.watchlist_loader import load_active_watchlist


def run_universe_cycle(
    trade_agent: TradeAgent,
    alpaca: AlpacaClient,
    mode: str,
    config_path: str,
    max_names: int | None = None,
    dry_run: bool = False,
    symbols_override: List[str] | None = None,
) -> None:
    """
    Execute one full trading cycle across the active watchlist.

    - Loads the universe from load_active_watchlist() or an explicit override.
    - Runs the full pipeline for each symbol.
    - Generates trades via TradeAgent.
    - Submits orders via AlpacaClient (unless dry_run).
    """
    as_of = datetime.now(timezone.utc)

    if symbols_override is not None:
        symbols = symbols_override
        print(
            f"[INFO] Using symbol override ({len(symbols)}): {', '.join(symbols)}"
        )
    else:
        symbols = load_active_watchlist(max_names=max_names)
        print(
            f"[INFO] Active watchlist ({len(symbols)}): {', '.join(symbols)}"
        )

    if not symbols:
        print("[WARN] No symbols in active watchlist. Exiting cycle.")
        return

    print(f"[INFO] Universe {mode} cycle at {as_of.isoformat()}")

    for symbol in symbols:
        print(f"\n=== [{mode.upper()}] Running pipeline for {symbol} ===")
        try:
            directive, diagnostics = run_full_pipeline_for_symbol(
                symbol=symbol,
                as_of=as_of,
                mode=mode,
                config_path=config_path,
            )

            trades = trade_agent.generate_trades(
                symbol=symbol,
                directive=directive,
                diagnostics=diagnostics,
            )

            if not trades:
                print(f"[INFO] {symbol}: no trades generated.")
                continue

            if dry_run:
                print(f"[DRY RUN] {symbol}: {len(trades)} trades -> {trades}")
                continue

            for trade in trades:
                alpaca.submit_order_from_trade(trade)
                print(f"[ORDER] Submitted {mode} trade: {trade}")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ERROR] {symbol}: {exc}")


def normalize_symbols(raw_symbols: List[str]) -> List[str]:
    """Uppercase, dedupe, and sort symbol overrides from the CLI."""
    symbols = [s.strip().upper() for s in raw_symbols if s.strip()]
    return sorted(set(symbols))


def main(args: argparse.Namespace) -> None:
    mode = args.mode  # "paper" or "live"

    # Dry-run logic:
    # - If --execute is NOT passed, default to dry-run.
    # - --dry-run always wins (even if --execute is set).
    dry_run = args.dry_run or not args.execute

    # Interval only matters in daemon mode; default can be overridden via env
    interval = args.interval

    symbols_override = None
    if args.symbols:
        symbols_override = normalize_symbols(args.symbols)
        if args.max_names is not None:
            symbols_override = symbols_override[: args.max_names]

    # Initialize core components once
    trade_agent = TradeAgent.from_config(args.config)
    alpaca = AlpacaClient.from_env(mode=mode)

    print(f"[INFO] Starting universe trading")
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Alpaca endpoint: {alpaca.base_url}")
    if mode == "paper" and "paper" not in alpaca.base_url:
        print("[WARN] Mode is 'paper' but Alpaca endpoint does not look like paper trading - check ALPACA_BASE_URL")
    if mode == "live" and "paper" in alpaca.base_url:
        print("[WARN] Mode is 'live' but Alpaca endpoint looks like paper - check ALPACA_BASE_URL")
    if dry_run:
        print("[INFO] DRY RUN enabled - no orders will be sent.")
    else:
        print("[INFO] EXECUTE enabled - orders WILL be sent.")

    if not args.daemon:
        # One-shot universe cycle
        run_universe_cycle(
            trade_agent=trade_agent,
            alpaca=alpaca,
            mode=mode,
            config_path=args.config,
            max_names=args.max_names,
            dry_run=dry_run,
            symbols_override=symbols_override,
        )
        return

    # Daemon mode: loop forever
    print(f"[INFO] Daemon mode enabled, interval={interval} seconds")
    while True:
        try:
            run_universe_cycle(
                trade_agent=trade_agent,
                alpaca=alpaca,
                mode=mode,
                config_path=args.config,
                max_names=args.max_names,
                dry_run=dry_run,
                symbols_override=symbols_override,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[FATAL CYCLE ERROR] {exc}")
        finally:
            print(f"[INFO] Sleeping for {interval} seconds...\n")
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode: paper or live (affects Alpaca endpoint default and pipeline mode)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a daemon, looping forever with a fixed interval",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("GNOSIS_UNIVERSE_INTERVAL", 300)),
        help="Seconds between cycles in daemon mode",
    )
    parser.add_argument(
        "--max-names",
        type=int,
        default=int(os.environ.get("GNOSIS_UNIVERSE_MAX_NAMES", 25)),
        help="Max number of symbols from the active watchlist to trade",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send orders to Alpaca. If omitted, script defaults to DRY RUN.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run even if --execute is passed.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Explicit list of symbols to trade (bypasses watchlist and honors --max-names).",
    )
    args = parser.parse_args()

    main(args)
