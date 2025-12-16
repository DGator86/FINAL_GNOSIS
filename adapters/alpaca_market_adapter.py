"""Alpaca market data adapter - Real implementation."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from loguru import logger

from config.credentials import get_alpaca_credentials
from adapters.market_data_adapter import OHLCV, Quote


class AlpacaMarketDataAdapter:
    """Alpaca market data adapter using official Alpaca SDK."""

    DEFAULT_DATA_FEED = "IEX"

    def __init__(self, *, client: StockHistoricalDataClient | None = None, data_feed: str | None = None) -> None:
        """Initialize Alpaca market data adapter."""
        creds = get_alpaca_credentials()

        if not creds.api_key or not creds.secret_key:
            raise ValueError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        self.client = client or StockHistoricalDataClient(api_key=creds.api_key, secret_key=creds.secret_key)
        self.data_feed = (data_feed or os.getenv("ALPACA_DATA_FEED", self.DEFAULT_DATA_FEED)).upper()

        logger.info("AlpacaMarketDataAdapter initialized")

    def get_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> List[OHLCV]:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Trading symbol
            start: Start timestamp
            end: End timestamp
            timeframe: Bar timeframe (e.g., "1Min", "1Hour", "1Day")

        Returns:
            List of OHLCV bars
        """
        try:
            # Map timeframe string to Alpaca TimeFrame
            timeframe_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
                "1Day": TimeFrame.Day,
                "1Week": TimeFrame.Week,
                "1Month": TimeFrame.Month,
            }

            tf = timeframe_map.get(timeframe, TimeFrame.Day)

            # Normalize datetimes to be timezone-aware (UTC) to avoid API rejection
            start_utc = self._ensure_utc(start)
            end_utc = self._ensure_utc(end)

            # Clamp end to now to avoid requesting future bars that return empty sets
            now_utc = datetime.now(timezone.utc)
            if end_utc > now_utc:
                logger.debug("Clamping end time for %s from %s to %s", symbol, end_utc, now_utc)
                end_utc = now_utc

            # Guard against inverted windows (can happen with naive timestamps)
            if end_utc <= start_utc:
                end_utc = start_utc + timedelta(minutes=1)
                logger.warning(
                    "Adjusted bar window for {symbol}: start >= end (tf={tf}) -> new_end={end}",
                    symbol=symbol,
                    tf=timeframe,
                    end=end_utc,
                )

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_utc,
                end=end_utc,
                feed=DataFeed[self.data_feed] if self.data_feed in DataFeed.__members__ else DataFeed.IEX,
            )

            # Get bars
            bars_response = self.client.get_stock_bars(request)
            bars = self._extract_bars(bars_response, symbol)

            if not bars:
                logger.warning(
                    "No bars found for {symbol} (tf={timeframe}, feed={feed}, start={start}, end={end}) | response={keys}",
                    symbol=symbol,
                    timeframe=timeframe,
                    feed=self.data_feed,
                    start=start_utc,
                    end=end_utc,
                    keys=self._describe_bars_response(bars_response),
                )

                # Retry with a slightly wider window for intraday gaps
                if timeframe.endswith("Min") or timeframe.endswith("Hour"):
                    fallback_start = min(start_utc, end_utc - timedelta(days=2))
                    fallback_request = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=tf,
                        start=fallback_start,
                        end=end_utc,
                        feed=DataFeed[self.data_feed] if self.data_feed in DataFeed.__members__ else DataFeed.IEX,
                    )
                    logger.info(
                        "Retrying {symbol} bars with expanded window (tf={timeframe}, start={start}, end={end}, feed={feed})",
                        symbol=symbol,
                        timeframe=timeframe,
                        start=fallback_start,
                        end=end_utc,
                        feed=self.data_feed,
                    )
                    bars_response = self.client.get_stock_bars(fallback_request)
                    bars = self._extract_bars(bars_response, symbol)

                if not bars:
                    logger.warning(
                        "No bars found for {symbol} after retry (tf={timeframe}, feed={feed}, start={start}, end={end}) | response={keys}",
                        symbol=symbol,
                        timeframe=timeframe,
                        feed=self.data_feed,
                        start=start_utc,
                        end=end_utc,
                        keys=self._describe_bars_response(bars_response),
                    )
                    return []

            # Convert to OHLCV objects
            result = []
            for bar in bars:
                result.append(
                    OHLCV(
                        timestamp=getattr(bar, "timestamp", getattr(bar, "t", None)),
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume),
                    )
                )

            logger.debug(f"Retrieved {len(result)} bars for {symbol}")
            return result

        except APIError as e:
            logger.error(f"Error getting bars for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error getting bars for {symbol} (tf={timeframe}): {e}"
            )
            return []

    @staticmethod
    def _extract_bars(response, symbol: str):
        """Return list-like collection of bars from any Alpaca response shape."""

        if response is None:
            return []

        if hasattr(response, "data"):
            data = response.data
            if isinstance(data, dict):
                return list(data.get(symbol, []))
            if isinstance(data, list):
                # MultiStockBars can sometimes expose a list of bars with symbol attributes
                return [bar for bar in data if getattr(bar, "symbol", symbol) == symbol]
            return list(data)

        if hasattr(response, "get"):
            try:
                bars = response.get(symbol)
                if bars is not None:
                    return list(bars)
            except Exception:
                pass

        if hasattr(response, "__getitem__"):
            try:
                return list(response[symbol])
            except Exception:
                pass

        if hasattr(response, "bars"):
            try:
                bars_attr = getattr(response, "bars")
                if isinstance(bars_attr, dict):
                    return list(bars_attr.get(symbol, []))
                if isinstance(bars_attr, list):
                    return [bar for bar in bars_attr if getattr(bar, "symbol", symbol) == symbol]
                return list(bars_attr)
            except Exception:
                pass

        if isinstance(response, dict):
            return list(response.get(symbol, []))

        if hasattr(response, "__iter__"):
            try:
                return list(response)
            except TypeError:
                return []

        return []

    @staticmethod
    def _describe_bars_response(response) -> str:
        """Provide lightweight diagnostics for logging empty responses."""

        if response is None:
            return "<none>"

        if hasattr(response, "keys"):
            try:
                return ",".join(sorted(map(str, response.keys())))
            except Exception:
                return str(type(response))

        if hasattr(response, "data"):
            data = getattr(response, "data")
            if isinstance(data, dict):
                return ",".join(sorted(map(str, data.keys())))
            if isinstance(data, list):
                return f"list[{len(data)}]"
            return str(type(data))

        if hasattr(response, "bars"):
            bars_attr = getattr(response, "bars")
            if isinstance(bars_attr, dict):
                return ",".join(sorted(map(str, bars_attr.keys())))
            if isinstance(bars_attr, list):
                return f"bars[{len(bars_attr)}]"
            return str(type(bars_attr))

        return str(type(response))

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """Ensure datetimes are timezone-aware in UTC for Alpaca requests."""

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote.

        Args:
            symbol: Trading symbol

        Returns:
            Current quote
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote_response = self.client.get_stock_latest_quote(request)

            if symbol not in quote_response:
                logger.warning(f"No quote found for {symbol}")
                # Return a dummy quote
                return Quote(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    bid=0.0,
                    ask=0.0,
                    bid_size=0.0,
                    ask_size=0.0,
                    last=0.0,
                    last_size=0.0,
                )

            q = quote_response[symbol]

            return Quote(
                timestamp=q.timestamp,
                symbol=symbol,
                bid=float(q.bid_price),
                ask=float(q.ask_price),
                bid_size=float(q.bid_size),
                ask_size=float(q.ask_size),
                last=float(q.ask_price),  # Use ask as last for now
                last_size=float(q.ask_size),
            )

        except APIError as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            # Return a dummy quote on error
            return Quote(
                timestamp=datetime.now(),
                symbol=symbol,
                bid=0.0,
                ask=0.0,
                bid_size=0.0,
                ask_size=0.0,
                last=0.0,
                last_size=0.0,
            )
