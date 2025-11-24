"""Minimal live trading bot for Alpaca paper trading.

The bot listens to live minute bars via Alpaca's websocket feed, runs a
lightweight moving-average strategy, and places paper trades through the
existing :class:`execution.broker_adapters.alpaca_adapter.AlpacaBrokerAdapter`.

It is intentionally self-contained so the launcher scripts in the repository
(`start_paper_trading.py`, `start_full_trading_system.py`,
`start_scanner_trading.py`, and `start_with_dashboard.py`) can run without the
previously missing ``gnosis`` package.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import inspect
from typing import Callable, Dict, List, Optional

from alpaca.data.live import StockDataStream
from loguru import logger

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter


@dataclass
class AgentView:
    """Simple representation of an agent vote."""

    agent_name: str
    signal: int
    confidence: float
    reasoning: str


@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    entry_time: datetime
    bars_held: int = 0

    @property
    def unrealized_pnl(self) -> float:
        return 0.0


class PositionManager:
    """Lightweight position tracker used by the live bot."""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict[str, object]] = []

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_position(self, symbol: str, side: str, size: float, price: float) -> Position:
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            entry_time=datetime.utcnow(),
        )
        self.positions[symbol] = position
        logger.info(f"Opened {side} position in {symbol} for ~${size:,.0f} @ {price:,.2f}")
        return position

    def close_position(self, symbol: str, price: float, reason: str) -> Optional[Dict[str, object]]:
        position = self.positions.pop(symbol, None)
        if not position:
            return None

        pnl = (price - position.entry_price) / position.entry_price
        trade = {
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": price,
            "entry_time": position.entry_time,
            "exit_time": datetime.utcnow(),
            "bars_held": position.bars_held,
            "pnl": pnl,
            "reason": reason,
        }
        self.trades.append(trade)
        logger.info(
            f"Closed {position.side} in {symbol} @ {price:,.2f} | PnL: {pnl:+.2%} | Reason: {reason}"
        )
        return trade

    def increment_bars(self) -> None:
        for position in self.positions.values():
            position.bars_held += 1

    def get_portfolio_summary(self) -> Dict[str, float]:
        realized = sum(t["pnl"] for t in self.trades) if self.trades else 0.0
        return {
            "total_pnl": realized,
            "daily_pnl": realized,
            "total_trades": len(self.trades),
            "win_rate": sum(1 for t in self.trades if t["pnl"] > 0) / len(self.trades) if self.trades else 0.0,
        }


class LiveTradingBot:
    """Live trading bot using Alpaca websocket bars and the broker adapter."""

    def __init__(
        self,
        symbol: str,
        bar_interval: str = "1Min",
        enable_memory: bool = False,
        enable_trading: bool = False,
        paper_mode: bool = True,
        on_regime: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.bar_interval = bar_interval
        self.enable_memory = enable_memory
        self.enable_trading = enable_trading
        self.paper_mode = paper_mode
        self.on_regime = on_regime

        self.adapter = AlpacaBrokerAdapter(paper=self.paper_mode)
        self.position_mgr = PositionManager()
        self.bars: List[Dict[str, object]] = []
        self.stream: Optional[StockDataStream] = None
        self.running: bool = False

        logger.info(
            f"LiveTradingBot initialized for {self.symbol} | interval={self.bar_interval} | "
            f"trading={'ON' if self.enable_trading else 'DRY-RUN'} | paper={self.paper_mode}"
        )

    async def run(self) -> None:
        """Start listening for bars and process them in real time."""

        async def on_bar(bar):
            await self.process_bar(bar)

        self.running = True
        retry_delay = 5

        while self.running:
            try:
                self.stream = StockDataStream(
                    api_key=self.adapter.api_key,
                    secret_key=self.adapter.secret_key,
                    feed="iex",  # free plan feed
                )

                logger.info("Starting Alpaca data stream... press Ctrl+C to stop.")
                self.stream.subscribe_bars(on_bar, self.symbol)
                await self.stream.run()

                if self.running:
                    logger.warning(
                        "Alpaca stream stopped unexpectedly; restarting in %ss", retry_delay
                    )
                    await asyncio.sleep(retry_delay)

            except asyncio.CancelledError:
                logger.info("LiveTradingBot task cancelled; stopping stream")
                self.running = False
                break
            except Exception as exc:  # pragma: no cover - defensive reconnect
                logger.error("Stream error for %s: %s", self.symbol, exc)
                await asyncio.sleep(retry_delay)
            finally:
                await self._stop_stream()

        logger.info("LiveTradingBot stopped for %s", self.symbol)

    async def process_bar(self, bar) -> None:
        """Handle an incoming bar event."""
        bar_dict = {
            "t_event": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        self.bars.append(bar_dict)
        self.position_mgr.increment_bars()

        logger.info(
            f"Bar {bar.timestamp} | O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}"
        )

        if len(self.bars) < 50:
            return

        price = bar_dict["close"]
        short_ma = self._sma(20)
        long_ma = self._sma(50)

        agent_views = self.evaluate_agents({"short_ma": short_ma, "long_ma": long_ma}, price)
        consensus = sum(av.signal for av in agent_views)

        if consensus > 0 and not self.position_mgr.has_position(self.symbol):
            await self.open_position(price, agent_views)
        elif consensus < 0 and self.position_mgr.has_position(self.symbol):
            await self.close_position(self.symbol, price, reason="Trend reversal")

    def evaluate_agents(self, features: Dict[str, float], price: float) -> List[AgentView]:
        """Very small ensemble: moving-average direction and momentum."""
        short_ma = features["short_ma"]
        long_ma = features["long_ma"]
        momentum = (price - long_ma) / long_ma if long_ma else 0.0

        ma_signal = 1 if short_ma > long_ma else -1
        momentum_signal = 1 if momentum > 0 else -1

        return [
            AgentView("ma_trend", ma_signal, 0.6, "20/50 crossover"),
            AgentView("momentum", momentum_signal, abs(momentum), "price vs 50SMA"),
        ]

    async def open_position(self, price: float, agent_views: List[AgentView]) -> None:
        """Size and optionally place a new long position."""
        account = self.adapter.get_account()
        capital = account.buying_power
        size_dollars = capital * 0.1  # risk 10% buying power per trade
        quantity = round(size_dollars / max(price, 1e-6), 2)
        logger.info(
            f"Opening position with {len(agent_views)} agreeing agents | qty={quantity} | price={price:,.2f}"
        )

        if self.enable_trading:
            order_id = self.adapter.place_order(self.symbol, quantity, side="buy")
            if order_id:
                self.position_mgr.open_position(self.symbol, "long", size_dollars, price)
        else:
            logger.info("DRY RUN: not sending buy order")
            self.position_mgr.open_position(self.symbol, "long", size_dollars, price)

    async def close_position(self, symbol: str, price: float, reason: str) -> None:
        """Close an open position if it exists."""
        if not self.position_mgr.has_position(symbol):
            return

        if self.enable_trading:
            pos = self.position_mgr.positions[symbol]
            qty = round(pos.size / max(price, 1e-6), 2)
            self.adapter.place_order(symbol, qty, side="sell")
        else:
            logger.info("DRY RUN: not sending sell order")

        self.position_mgr.close_position(symbol, price, reason)

    async def _stop_stream(self) -> None:
        """Gracefully stop the websocket stream if it is running."""
        if not self.stream:
            return

        stop_fn = getattr(self.stream, "stop", None)
        if callable(stop_fn):
            try:
                result = stop_fn()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.debug("Error stopping Alpaca stream for %s: %s", self.symbol, exc)

        self.stream = None

    def _sma(self, window: int) -> float:
        if len(self.bars) < window:
            return 0.0
        return sum(bar["close"] for bar in self.bars[-window:]) / window


__all__ = ["LiveTradingBot", "AgentView", "PositionManager", "Position"]
