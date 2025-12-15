"""Liquidity Engine v2 - production-grade liquidity and flow analytics.

# NEW: Implements Kyle's lambda, dynamic liquidity friction, ARIMA depth forecasts,
# and Unusual Whales flow alerts while keeping adapters/protocols swappable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from statsmodels.tsa.arima.model import ARIMA

from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.options_chain_adapter import OptionsChainAdapter
from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
from schemas.core_schemas import LiquiditySnapshot


@dataclass
class OrderBookDepth:
    """Minimal order book representation to avoid tight coupling to adapters."""

    bids: List[float]
    asks: List[float]
    bid_sizes: List[float]
    ask_sizes: List[float]


class LiquidityEngineV2:
    """Advanced liquidity engine with impact modeling and forecasting."""

    def __init__(
        self,
        market_data_adapter: MarketDataAdapter,
        options_adapter: OptionsChainAdapter,
        unusual_whales_adapter: Optional[UnusualWhalesAdapter] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.market_data_adapter = market_data_adapter
        self.options_adapter = options_adapter
        self.unusual_whales_adapter = unusual_whales_adapter
        self.config = config or {}
        self.flow_threshold = self.config.get("flow_threshold", 1_000_000)
        self.depth_horizon = self.config.get("depth_forecast_horizon", 5)
        logger.info("LiquidityEngineV2 initialized with ARIMA depth forecasting")

    def run(self, symbol: str, timestamp: datetime) -> LiquiditySnapshot:
        """Execute liquidity analysis for the provided symbol."""

        logger.debug("LiquidityEngineV2.run: fetching market data")
        try:
            quote = self.market_data_adapter.get_quote(symbol)
            bars = self.market_data_adapter.get_bars(
                symbol, timestamp - timedelta(days=5), timestamp, timeframe="1Min"
            )
            order_book = self._safe_order_book(symbol)
        except Exception as exc:  # pragma: no cover - adapter safety
            logger.error(f"LiquidityEngineV2 unable to fetch data for {symbol}: {exc}")
            return LiquiditySnapshot(timestamp=timestamp, symbol=symbol)

        mid_price = (quote.bid + quote.ask) / 2 if (quote.bid and quote.ask) else 0.0
        spread = max(1e-6, quote.ask - quote.bid)
        spread_pct = (spread / mid_price) * 100 if mid_price else 0.0

        signed_volumes = self._signed_volumes(bars)
        price_changes = self._price_changes(bars)
        impact_lambda = self._kyle_lambda(price_changes, signed_volumes)

        avg_volume = np.mean([bar.volume for bar in bars]) if bars else 0.0
        current_vol = bars[-1].volume if bars else 0.0
        vol_spike = (abs(current_vol - avg_volume) / avg_volume) if avg_volume else 0.0
        vol_spike_factor = 1 + vol_spike
        liquidity_friction = spread * vol_spike_factor

        unusual_signal = self._unusual_whales_signal(symbol)
        friction = liquidity_friction * (1.2 if unusual_signal else 1.0)

        depth_series = self._historical_depth_series(order_book)
        forecast_depth = self._forecast_depth(depth_series)
        percentile_score = self._depth_percentile(order_book, depth_series)

        liquidity_score = self._score_liquidity(
            spread_pct=spread_pct,
            depth=np.sum(order_book.bid_sizes) + np.sum(order_book.ask_sizes),
            vol_spike=vol_spike,
            unusual_signal=unusual_signal,
        )

        return LiquiditySnapshot(
            timestamp=timestamp,
            symbol=symbol,
            liquidity_score=liquidity_score,
            bid_ask_spread=spread_pct,
            volume=avg_volume,
            depth=float(np.sum(order_book.bid_sizes) + np.sum(order_book.ask_sizes)),
            impact_cost=spread_pct * 0.5,
            impact_lambda=impact_lambda,
            friction=friction,
            liquidity_friction=liquidity_friction,
            forecast_depth=forecast_depth,
            percentile_score=percentile_score,
        )

    def _safe_order_book(self, symbol: str) -> OrderBookDepth:
        """Fetch order book with graceful degradation when adapter lacks support."""

        default_depth = OrderBookDepth(bids=[0.0], asks=[0.0], bid_sizes=[0.0], ask_sizes=[0.0])
        adapter_fn = getattr(self.market_data_adapter, "get_order_book", None)
        if not adapter_fn:
            logger.debug("MarketDataAdapter missing get_order_book; returning default depth")
            return default_depth

        try:
            book = adapter_fn(symbol)
            return OrderBookDepth(
                bids=getattr(book, "bids", [0.0]),
                asks=getattr(book, "asks", [0.0]),
                bid_sizes=getattr(book, "bid_sizes", [0.0]),
                ask_sizes=getattr(book, "ask_sizes", [0.0]),
            )
        except Exception as exc:  # pragma: no cover - adapter safety
            logger.warning(f"Order book retrieval failed for {symbol}: {exc}")
            return default_depth

    def _signed_volumes(self, bars: List[Any]) -> np.ndarray:
        """Approximate signed volume using close-to-close returns as proxy."""

        if len(bars) < 2:
            return np.array([0.0])
        returns = np.diff([bar.close for bar in bars])
        volumes = np.array([bar.volume for bar in bars[1:]])
        return np.sign(returns) * volumes

    def _price_changes(self, bars: List[Any]) -> np.ndarray:
        if len(bars) < 2:
            return np.array([0.0])
        return np.diff([bar.close for bar in bars])

    def _kyle_lambda(self, price_changes: np.ndarray, signed_volumes: np.ndarray) -> float:
        """Estimate Kyle's lambda via regression of price changes on signed volume."""

        if price_changes.size != signed_volumes.size or price_changes.size < 2:
            return 0.0
        try:
            X = signed_volumes.reshape(-1, 1)
            y = price_changes
            # Using closed-form OLS; statsmodels would add overhead for tiny regressions
            beta = float(np.linalg.lstsq(X, y, rcond=None)[0])
            return beta
        except Exception as exc:
            logger.debug(f"Kyle lambda regression failed: {exc}")
            return 0.0

    def _unusual_whales_signal(self, symbol: str) -> bool:
        if not self.unusual_whales_adapter:
            return False
        try:
            flow = self.unusual_whales_adapter.get_unusual_activity(symbol) or []
            total_premium = sum(trade.get("premium", 0.0) for trade in flow)
            return total_premium > self.flow_threshold
        except Exception as exc:  # pragma: no cover
            logger.debug(f"UnusualWhales signal failed: {exc}")
            return False

    def _historical_depth_series(self, order_book: OrderBookDepth) -> List[float]:
        # Placeholder hook: adapters may expose richer history; we use present depth as proxy
        total_depth = float(np.sum(order_book.bid_sizes) + np.sum(order_book.ask_sizes))
        history = self.config.get("depth_history", [total_depth] * 15)
        history.append(total_depth)
        return history[-64:]

    def _forecast_depth(self, depth_series: List[float]) -> List[float]:
        if len(depth_series) < 5:
            return []
        try:
            model = ARIMA(depth_series, order=(1, 1, 0))
            result = model.fit()
            forecast = result.forecast(steps=self.depth_horizon)
            return [float(x) for x in forecast]
        except Exception as exc:  # pragma: no cover - statsmodels failures
            logger.debug(f"ARIMA forecast failed: {exc}")
            return []

    def _depth_percentile(self, order_book: OrderBookDepth, depth_series: List[float]) -> float:
        current_depth = float(np.sum(order_book.bid_sizes) + np.sum(order_book.ask_sizes))
        if not depth_series:
            return 0.0
        percentile = float(np.percentile(depth_series, 90))
        score = float(100 * (current_depth >= percentile))
        return score

    def _score_liquidity(
        self, *, spread_pct: float, depth: float, vol_spike: float, unusual_signal: bool
    ) -> float:
        spread_score = max(0.0, 1.0 - spread_pct / 2.0)
        depth_score = min(1.0, depth / 10_000)
        stability = max(0.0, 1.0 - vol_spike)
        alert_penalty = 0.2 if unusual_signal else 0.0
        score = (0.4 * spread_score) + (0.4 * depth_score) + (0.2 * stability)
        return max(0.0, min(1.0, score - alert_penalty))


__all__ = ["LiquidityEngineV2", "OrderBookDepth"]
