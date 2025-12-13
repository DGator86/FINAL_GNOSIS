"""Lightweight orchestrator for running multiple live trading bots."""
from __future__ import annotations

import asyncio
import threading
from typing import Dict, List

from gnosis.trading.live_bot import LiveTradingBot


class GnosisLiveTradingEngine:
    """Manage one or more :class:`LiveTradingBot` instances in the background."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.bots: List[LiveTradingBot] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._tasks: List[asyncio.Task] = []
        self.running = False

        timeframe = self.config.get("timeframe", "1Min")
        enable_trading = self.config.get("enable_trading", True)
        paper_mode = self.config.get("paper_mode", True)

        for symbol in self.config.get("symbols", []):
            bot = LiveTradingBot(
                symbol=symbol,
                bar_interval=timeframe,
                enable_memory=False,
                enable_trading=enable_trading,
                paper_mode=paper_mode,
            )
            self.bots.append(bot)

    def start(self) -> None:
        """Start all bots on a background event loop."""
        if self.running or not self.bots:
            return

        self.running = True
        self._loop = asyncio.new_event_loop()

        def runner() -> None:
            assert self._loop is not None
            asyncio.set_event_loop(self._loop)
            self._tasks = [self._loop.create_task(bot.run()) for bot in self.bots]
            self._loop.run_forever()

            # Ensure all tasks are awaited before closing the loop
            if self._tasks:
                self._loop.run_until_complete(
                    asyncio.gather(*self._tasks, return_exceptions=True)
                )

            self._loop.close()

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop all running bots and close the event loop."""
        self.running = False
        if self._loop:
            for task in self._tasks:
                task.cancel()
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def get_performance_summary(self) -> Dict:
        """Return a simple account snapshot from the first bot adapter."""
        if not self.bots:
            return {}

        adapter = self.bots[0].adapter
        account = adapter.get_account()
        positions = adapter.get_positions()
        unrealized = sum(p.unrealized_pnl for p in positions)

        return {
            "account": account.model_dump(),
            "positions": [p.model_dump() for p in positions],
            "total_unrealized_pl": unrealized,
            "total_trades": len(self.bots[0].position_mgr.trades),
        }
