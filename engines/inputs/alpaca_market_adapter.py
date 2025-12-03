"""Alpaca market data adapter - Real implementation."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from loguru import logger

from engines.inputs.market_data_adapter import OHLCV, Quote


class AlpacaMarketDataAdapter:
    """Alpaca market data adapter using official Alpaca SDK."""

    def __init__(self) -> None:
        """Initialize Alpaca market data adapter."""
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        self.client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)

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

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )

            # Get bars
            bars_response = self.client.get_stock_bars(request)

            if symbol not in bars_response:
                logger.warning(f"No bars found for {symbol}")
                return []

            bars = bars_response[symbol]

            # Convert to OHLCV objects
            result = []
            for bar in bars:
                result.append(
                    OHLCV(
                        timestamp=bar.timestamp,
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
