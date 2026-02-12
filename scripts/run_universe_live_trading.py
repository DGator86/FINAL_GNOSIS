"""Run live trading loop across the active universe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import List

from pipeline.full_pipeline import run_full_pipeline_for_symbol
from trade.trade_agent_v2 import TradeAgent
from brokers.alpaca_client import AlpacaClient
from brokers.alpaca_client import AlpacaClient
from pipeline.full_pipeline import run_full_pipeline_for_symbol
from trade.trade_agent_v2 import TradeAgent
from universe.watchlist_loader import load_active_watchlist


def run_for_symbol(
    symbol: str,
    as_of: datetime,
    trade_agent: TradeAgent,
    alpaca: AlpacaClient,
    config_path: str,
    dry_run: bool = False,
) -> None:
    """Run pipeline + trade submission for a single symbol."""
    directive, diagnostics = run_full_pipeline_for_symbol(
        symbol=symbol,
        as_of=as_of,
        mode="live",
        config_path=config_path,
    )

    trades = trade_agent.generate_trades(
        symbol=symbol,
        directive=directive,
        diagnostics=diagnostics,
    )

    if not trades:
        print(f"[INFO] {symbol}: no trades generated.")
        return

    if dry_run:
        print(f"[DRY RUN] {symbol}: {len(trades)} trades -> {trades}")
        return

    for trade in trades:
        alpaca.submit_order_from_trade(trade)
        print(f"[ORDER] Submitted {trade}")


def main(args: argparse.Namespace) -> None:
    as_of = datetime.now(timezone.utc)

    symbols: List[str] = load_active_watchlist(max_names=args.max_names)

    if not symbols:
        print("[WARN] No symbols in active watchlist. Exiting.")
        return

    print(f"[INFO] Active watchlist ({len(symbols)}): {', '.join(symbols)}")

    trade_agent = TradeAgent.from_config(args.config)
    alpaca = AlpacaClient.from_env(mode=args.mode)

    print(
        f"[INFO] Submitting orders to {'paper' if args.mode == 'paper' else 'live'} endpoint: {alpaca.base_url}"
    )
    if args.dry_run:
        print("[INFO] DRY RUN enabled - Alpaca orders will NOT be sent.")
    alpaca = AlpacaClient.from_env()

    for symbol in symbols:
        print(f"\n=== Running pipeline for {symbol} ===")
        try:
            run_for_symbol(
                symbol=symbol,
                as_of=as_of,
                trade_agent=trade_agent,
                alpaca=alpaca,
                config_path=args.config,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--max-names", type=int, default=25)
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main(args)
