"""Shared helpers for paper trading the active universe."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from brokers.alpaca_client import AlpacaClient
from pipeline.full_pipeline import run_full_pipeline_for_symbol
from trade.trade_agent_v2 import TradeAgent
from universe.watchlist_loader import load_active_watchlist


def run_universe_paper_cycle(
    trade_agent: TradeAgent,
    alpaca: AlpacaClient,
    config_path: str | None = None,
    max_names: int | None = None,
    dry_run: bool = False,
) -> None:
    """Execute one full paper-trading cycle across the active watchlist.

    The provided ``config_path`` is forwarded to the pipeline runner so the
    daemon respects the same configuration used to instantiate the TradeAgent.
    """

    as_of = datetime.now(timezone.utc)

    symbols: List[str] = load_active_watchlist(max_names=max_names)
    if not symbols:
        print("[WARN] No symbols in active watchlist. Exiting cycle.")
        return

    print(f"[INFO] Universe paper cycle at {as_of.isoformat()}")
    print(f"[INFO] Active watchlist ({len(symbols)}): {', '.join(symbols)}")

    for symbol in symbols:
        print(f"\n=== [PAPER] Running pipeline for {symbol} ===")
        try:
            directive, diagnostics = run_full_pipeline_for_symbol(
                symbol=symbol,
                as_of=as_of,
                mode="paper",
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
                print(f"[ORDER] Submitted paper trade: {trade}")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ERROR] {symbol}: {exc}")
