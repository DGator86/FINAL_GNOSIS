"""MASSIVE.COM market data adapter - Comprehensive market data provider.

Provides market data, technical indicators, financials, news, and options data
using the official MASSIVE Python client library.
"""

from __future__ import annotations

import os
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from config.credentials import get_massive_api_keys, massive_api_enabled
from engines.inputs.market_data_adapter import OHLCV, Quote


class MassiveMarketDataAdapter:
    """MASSIVE market data adapter using official MASSIVE SDK.

    Provides comprehensive market data including:
    - Historical OHLCV bars (aggregates)
    - Real-time quotes
    - Technical indicators (SMA, EMA, RSI, MACD)
    - Financials (balance sheet, income, cash flow)
    - Benzinga news and analyst ratings
    - Options contracts and snapshots
    - Economic data (treasury yields, inflation)
    """

    def __init__(self, *, api_key: Optional[str] = None) -> None:
        """Initialize MASSIVE market data adapter.

        Args:
            api_key: MASSIVE API key (reads from MASSIVE_API_KEY if not provided)
        """
        primary_key, secondary_key = get_massive_api_keys(primary=api_key)
        self.api_key = primary_key or secondary_key
        self.enabled = massive_api_enabled(default=True)

        if not self.enabled:
            logger.info("MASSIVE API disabled (MASSIVE_API_ENABLED=false)")
            self.client = None
            return

        if not self.api_key:
            raise ValueError(
                "MASSIVE API key not found. Set MASSIVE_API_KEY or MASSIVE_API_KEY_SECONDARY."
            )

        try:
            from massive import RESTClient

            self.client = RESTClient(api_key=self.api_key)
            logger.info("MassiveMarketDataAdapter initialized with live data access")
        except ImportError:
            raise ImportError(
                "MASSIVE client not installed. Run: pip install massive"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MASSIVE client: {e}")
            raise

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day"
    ) -> List[OHLCV]:
        """Get historical OHLCV bars (aggregates).

        Args:
            symbol: Trading symbol (e.g., "AAPL")
            start: Start timestamp
            end: End timestamp
            timeframe: Bar timeframe (e.g., "1Min", "1Hour", "1Day")

        Returns:
            List of OHLCV bars
        """
        if not self.client:
            logger.warning("MASSIVE client not initialized")
            return []

        try:
            # Map timeframe to MASSIVE parameters
            timeframe_map = {
                "1Min": (1, "minute"),
                "5Min": (5, "minute"),
                "15Min": (15, "minute"),
                "30Min": (30, "minute"),
                "1Hour": (1, "hour"),
                "4Hour": (4, "hour"),
                "1Day": (1, "day"),
                "1Week": (1, "week"),
                "1Month": (1, "month"),
            }

            multiplier, timespan = timeframe_map.get(timeframe, (1, "day"))

            # Format dates for MASSIVE API
            from_date = self._format_date(start)
            to_date = self._format_date(end)

            # Get aggregates from MASSIVE
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                adjusted=True,
                sort="asc",
            )

            if not aggs:
                logger.warning(f"No bars found for {symbol} from MASSIVE")
                return []

            # Convert to OHLCV objects
            result = []
            for agg in aggs:
                result.append(
                    OHLCV(
                        timestamp=datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc),
                        open=float(agg.open),
                        high=float(agg.high),
                        low=float(agg.low),
                        close=float(agg.close),
                        volume=float(agg.volume),
                    )
                )

            logger.debug(f"Retrieved {len(result)} bars for {symbol} from MASSIVE")
            return result

        except Exception as e:
            logger.error(f"Error getting bars for {symbol} from MASSIVE: {e}")
            return []

    def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current quote
        """
        if not self.client:
            return self._empty_quote(symbol)

        try:
            # Get last quote from MASSIVE
            quote = self.client.get_last_quote(symbol)

            if not quote:
                logger.warning(f"No quote found for {symbol}")
                return self._empty_quote(symbol)

            return Quote(
                timestamp=datetime.fromtimestamp(quote.participant_timestamp / 1e9, tz=timezone.utc)
                if hasattr(quote, 'participant_timestamp') else datetime.now(timezone.utc),
                symbol=symbol,
                bid=float(quote.bid_price) if hasattr(quote, 'bid_price') else 0.0,
                ask=float(quote.ask_price) if hasattr(quote, 'ask_price') else 0.0,
                bid_size=float(quote.bid_size) if hasattr(quote, 'bid_size') else 0.0,
                ask_size=float(quote.ask_size) if hasattr(quote, 'ask_size') else 0.0,
                last=float(quote.ask_price) if hasattr(quote, 'ask_price') else 0.0,
                last_size=float(quote.ask_size) if hasattr(quote, 'ask_size') else 0.0,
            )

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return self._empty_quote(symbol)

    def get_previous_close(self, symbol: str) -> Optional[OHLCV]:
        """Get previous day's close for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Previous close OHLCV or None
        """
        if not self.client:
            return None

        try:
            prev = self.client.get_previous_close_agg(symbol, adjusted=True)

            if not prev:
                return None

            return OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=float(prev.open),
                high=float(prev.high),
                low=float(prev.low),
                close=float(prev.close),
                volume=float(prev.volume),
            )

        except Exception as e:
            logger.error(f"Error getting previous close for {symbol}: {e}")
            return None

    def get_sma(
        self,
        symbol: str,
        window: int = 50,
        timeframe: str = "day",
        series_type: str = "close",
    ) -> List[Dict[str, Any]]:
        """Get Simple Moving Average indicator.

        Args:
            symbol: Trading symbol
            window: SMA window size
            timeframe: Timeframe (minute, hour, day, week, month)
            series_type: Price series type (open, high, low, close)

        Returns:
            List of SMA values with timestamps
        """
        if not self.client:
            return []

        try:
            sma_data = self.client.get_sma(
                ticker=symbol,
                timespan=timeframe,
                window=window,
                series_type=series_type,
            )

            return [
                {"timestamp": v.timestamp, "value": v.value}
                for v in (sma_data.values if hasattr(sma_data, 'values') else [])
            ]

        except Exception as e:
            logger.error(f"Error getting SMA for {symbol}: {e}")
            return []

    def get_ema(
        self,
        symbol: str,
        window: int = 20,
        timeframe: str = "day",
        series_type: str = "close",
    ) -> List[Dict[str, Any]]:
        """Get Exponential Moving Average indicator.

        Args:
            symbol: Trading symbol
            window: EMA window size
            timeframe: Timeframe (minute, hour, day, week, month)
            series_type: Price series type (open, high, low, close)

        Returns:
            List of EMA values with timestamps
        """
        if not self.client:
            return []

        try:
            ema_data = self.client.get_ema(
                ticker=symbol,
                timespan=timeframe,
                window=window,
                series_type=series_type,
            )

            return [
                {"timestamp": v.timestamp, "value": v.value}
                for v in (ema_data.values if hasattr(ema_data, 'values') else [])
            ]

        except Exception as e:
            logger.error(f"Error getting EMA for {symbol}: {e}")
            return []

    def get_rsi(
        self,
        symbol: str,
        window: int = 14,
        timeframe: str = "day",
        series_type: str = "close",
    ) -> List[Dict[str, Any]]:
        """Get Relative Strength Index indicator.

        Args:
            symbol: Trading symbol
            window: RSI window size (default 14)
            timeframe: Timeframe (minute, hour, day, week, month)
            series_type: Price series type (open, high, low, close)

        Returns:
            List of RSI values with timestamps
        """
        if not self.client:
            return []

        try:
            rsi_data = self.client.get_rsi(
                ticker=symbol,
                timespan=timeframe,
                window=window,
                series_type=series_type,
            )

            return [
                {"timestamp": v.timestamp, "value": v.value}
                for v in (rsi_data.values if hasattr(rsi_data, 'values') else [])
            ]

        except Exception as e:
            logger.error(f"Error getting RSI for {symbol}: {e}")
            return []

    def get_macd(
        self,
        symbol: str,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        timeframe: str = "day",
        series_type: str = "close",
    ) -> List[Dict[str, Any]]:
        """Get MACD indicator.

        Args:
            symbol: Trading symbol
            short_window: Short EMA window (default 12)
            long_window: Long EMA window (default 26)
            signal_window: Signal line window (default 9)
            timeframe: Timeframe (minute, hour, day, week, month)
            series_type: Price series type (open, high, low, close)

        Returns:
            List of MACD values with histogram, signal, and value
        """
        if not self.client:
            return []

        try:
            macd_data = self.client.get_macd(
                ticker=symbol,
                timespan=timeframe,
                short_window=short_window,
                long_window=long_window,
                signal_window=signal_window,
                series_type=series_type,
            )

            return [
                {
                    "timestamp": v.timestamp,
                    "value": v.value,
                    "signal": v.signal,
                    "histogram": v.histogram,
                }
                for v in (macd_data.values if hasattr(macd_data, 'values') else [])
            ]

        except Exception as e:
            logger.error(f"Error getting MACD for {symbol}: {e}")
            return []

    def get_ticker_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a ticker.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker details dictionary or None
        """
        if not self.client:
            return None

        try:
            details = self.client.get_ticker_details(symbol)

            if not details:
                return None

            return {
                "symbol": getattr(details, 'ticker', symbol),
                "name": getattr(details, 'name', ''),
                "market": getattr(details, 'market', ''),
                "locale": getattr(details, 'locale', ''),
                "primary_exchange": getattr(details, 'primary_exchange', ''),
                "type": getattr(details, 'type', ''),
                "currency_name": getattr(details, 'currency_name', ''),
                "market_cap": getattr(details, 'market_cap', None),
                "share_class_shares_outstanding": getattr(details, 'share_class_shares_outstanding', None),
                "weighted_shares_outstanding": getattr(details, 'weighted_shares_outstanding', None),
                "description": getattr(details, 'description', ''),
                "sic_code": getattr(details, 'sic_code', ''),
                "sic_description": getattr(details, 'sic_description', ''),
                "homepage_url": getattr(details, 'homepage_url', ''),
                "total_employees": getattr(details, 'total_employees', None),
            }

        except Exception as e:
            logger.error(f"Error getting ticker details for {symbol}: {e}")
            return None

    def get_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get latest news articles.

        Args:
            symbol: Trading symbol (optional, gets all news if not provided)
            limit: Maximum number of articles to return

        Returns:
            List of news articles
        """
        if not self.client:
            return []

        try:
            news_list = list(self.client.list_ticker_news(
                ticker=symbol,
                limit=limit,
            ))

            return [
                {
                    "id": getattr(article, 'id', ''),
                    "title": getattr(article, 'title', ''),
                    "author": getattr(article, 'author', ''),
                    "published_utc": getattr(article, 'published_utc', ''),
                    "article_url": getattr(article, 'article_url', ''),
                    "tickers": getattr(article, 'tickers', []),
                    "description": getattr(article, 'description', ''),
                    "keywords": getattr(article, 'keywords', []),
                    "publisher": getattr(article, 'publisher', {}),
                }
                for article in news_list
            ]

        except Exception as e:
            logger.error(f"Error getting news: {e}")
            return []

    def get_benzinga_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get Benzinga news articles.

        Args:
            symbol: Trading symbol (optional)
            limit: Maximum number of articles

        Returns:
            List of Benzinga news articles
        """
        if not self.client:
            return []

        try:
            news_list = list(self.client.list_benzinga_news(
                tickers=symbol,
                limit=limit,
            ))

            return [
                {
                    "id": getattr(article, 'id', ''),
                    "title": getattr(article, 'title', ''),
                    "author": getattr(article, 'author', ''),
                    "created": getattr(article, 'created', ''),
                    "updated": getattr(article, 'updated', ''),
                    "url": getattr(article, 'url', ''),
                    "tickers": getattr(article, 'tickers', []),
                    "body": getattr(article, 'body', ''),
                    "channels": getattr(article, 'channels', []),
                }
                for article in news_list
            ]

        except Exception as e:
            logger.error(f"Error getting Benzinga news: {e}")
            return []

    def get_analyst_ratings(
        self,
        symbol: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get analyst ratings from Benzinga.

        Args:
            symbol: Trading symbol
            limit: Maximum number of ratings

        Returns:
            List of analyst ratings
        """
        if not self.client:
            return []

        try:
            ratings_list = list(self.client.list_benzinga_ratings(
                ticker=symbol,
                limit=limit,
            ))

            return [
                {
                    "id": getattr(rating, 'id', ''),
                    "ticker": getattr(rating, 'ticker', symbol),
                    "analyst": getattr(rating, 'analyst', ''),
                    "analyst_name": getattr(rating, 'analyst_name', ''),
                    "rating_current": getattr(rating, 'rating_current', ''),
                    "rating_prior": getattr(rating, 'rating_prior', ''),
                    "action_company": getattr(rating, 'action_company', ''),
                    "action_pt": getattr(rating, 'action_pt', ''),
                    "pt_current": getattr(rating, 'pt_current', None),
                    "pt_prior": getattr(rating, 'pt_prior', None),
                    "date": getattr(rating, 'date', ''),
                    "time": getattr(rating, 'time', ''),
                }
                for rating in ratings_list
            ]

        except Exception as e:
            logger.error(f"Error getting analyst ratings for {symbol}: {e}")
            return []

    def get_financials(
        self,
        symbol: str,
        statement_type: str = "income",
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """Get financial statements.

        Args:
            symbol: Trading symbol
            statement_type: Type of statement (income, balance_sheet, cash_flow)
            limit: Number of periods to return

        Returns:
            List of financial statements
        """
        if not self.client:
            return []

        try:
            if statement_type == "income":
                financials = list(self.client.list_financials_income_statements(
                    ticker=symbol,
                    limit=limit,
                ))
            elif statement_type == "balance_sheet":
                financials = list(self.client.list_financials_balance_sheets(
                    ticker=symbol,
                    limit=limit,
                ))
            elif statement_type == "cash_flow":
                financials = list(self.client.list_financials_cash_flow_statements(
                    ticker=symbol,
                    limit=limit,
                ))
            else:
                logger.warning(f"Unknown statement type: {statement_type}")
                return []

            return [
                {
                    "ticker": getattr(f, 'ticker', symbol),
                    "fiscal_period": getattr(f, 'fiscal_period', ''),
                    "fiscal_year": getattr(f, 'fiscal_year', ''),
                    "filing_date": getattr(f, 'filing_date', ''),
                    "data": f.__dict__ if hasattr(f, '__dict__') else {},
                }
                for f in financials
            ]

        except Exception as e:
            logger.error(f"Error getting financials for {symbol}: {e}")
            return []

    def get_options_contracts(
        self,
        underlying_symbol: str,
        expiration_date: Optional[str] = None,
        contract_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get options contracts for an underlying symbol.

        Args:
            underlying_symbol: Underlying stock symbol
            expiration_date: Optional expiration date (YYYY-MM-DD)
            contract_type: Optional contract type (call, put)
            limit: Maximum contracts to return

        Returns:
            List of options contracts
        """
        if not self.client:
            return []

        try:
            contracts = list(self.client.list_options_contracts(
                underlying_ticker=underlying_symbol,
                expiration_date=expiration_date,
                contract_type=contract_type,
                limit=limit,
            ))

            return [
                {
                    "ticker": getattr(c, 'ticker', ''),
                    "underlying_ticker": getattr(c, 'underlying_ticker', underlying_symbol),
                    "contract_type": getattr(c, 'contract_type', ''),
                    "expiration_date": getattr(c, 'expiration_date', ''),
                    "strike_price": getattr(c, 'strike_price', 0),
                    "shares_per_contract": getattr(c, 'shares_per_contract', 100),
                    "exercise_style": getattr(c, 'exercise_style', ''),
                    "primary_exchange": getattr(c, 'primary_exchange', ''),
                }
                for c in contracts
            ]

        except Exception as e:
            logger.error(f"Error getting options contracts for {underlying_symbol}: {e}")
            return []

    def get_options_snapshot(self, options_ticker: str) -> Optional[Dict[str, Any]]:
        """Get snapshot data for an options contract.

        Args:
            options_ticker: Options contract ticker (e.g., "O:AAPL230120C00150000")

        Returns:
            Options snapshot data or None
        """
        if not self.client:
            return None

        try:
            snapshot = self.client.get_snapshot_option(
                underlying_asset=options_ticker.split(':')[1][:4] if ':' in options_ticker else options_ticker[:4],
                option_contract=options_ticker,
            )

            if not snapshot:
                return None

            return {
                "ticker": options_ticker,
                "day": getattr(snapshot, 'day', {}),
                "greeks": getattr(snapshot, 'greeks', {}),
                "implied_volatility": getattr(snapshot, 'implied_volatility', None),
                "open_interest": getattr(snapshot, 'open_interest', None),
                "underlying_asset": getattr(snapshot, 'underlying_asset', {}),
                "break_even_price": getattr(snapshot, 'break_even_price', None),
            }

        except Exception as e:
            logger.error(f"Error getting options snapshot for {options_ticker}: {e}")
            return None

    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status.

        Returns:
            Market status dictionary
        """
        if not self.client:
            return {"status": "unknown"}

        try:
            status = self.client.get_market_status()

            return {
                "market": getattr(status, 'market', ''),
                "server_time": getattr(status, 'server_time', ''),
                "exchanges": getattr(status, 'exchanges', {}),
                "currencies": getattr(status, 'currencies', {}),
            }

        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {"status": "error", "error": str(e)}

    def get_treasury_yields(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get treasury yield data.

        Args:
            limit: Number of records to return

        Returns:
            List of treasury yield data
        """
        if not self.client:
            return []

        try:
            yields_data = list(self.client.list_treasury_yields(limit=limit))

            return [
                {
                    "date": getattr(y, 'date', ''),
                    "yield_1m": getattr(y, 'yield_1m', None),
                    "yield_3m": getattr(y, 'yield_3m', None),
                    "yield_6m": getattr(y, 'yield_6m', None),
                    "yield_1y": getattr(y, 'yield_1y', None),
                    "yield_2y": getattr(y, 'yield_2y', None),
                    "yield_5y": getattr(y, 'yield_5y', None),
                    "yield_10y": getattr(y, 'yield_10y', None),
                    "yield_30y": getattr(y, 'yield_30y', None),
                }
                for y in yields_data
            ]

        except Exception as e:
            logger.error(f"Error getting treasury yields: {e}")
            return []

    def get_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive snapshot for a ticker.

        Args:
            symbol: Trading symbol

        Returns:
            Snapshot data or None
        """
        if not self.client:
            return None

        try:
            snapshot = self.client.get_snapshot_ticker("stocks", symbol)

            if not snapshot:
                return None

            return {
                "ticker": symbol,
                "day": getattr(snapshot, 'day', {}),
                "prev_day": getattr(snapshot, 'prev_day', {}),
                "min": getattr(snapshot, 'min', {}),
                "last_quote": getattr(snapshot, 'last_quote', {}),
                "last_trade": getattr(snapshot, 'last_trade', {}),
                "todaysChange": getattr(snapshot, 'todaysChange', None),
                "todaysChangePerc": getattr(snapshot, 'todaysChangePerc', None),
                "updated": getattr(snapshot, 'updated', None),
            }

        except Exception as e:
            logger.error(f"Error getting snapshot for {symbol}: {e}")
            return None

    def _format_date(self, dt: datetime) -> str:
        """Format datetime for MASSIVE API.

        Args:
            dt: Datetime object

        Returns:
            Formatted date string (YYYY-MM-DD)
        """
        if isinstance(dt, datetime):
            return dt.strftime("%Y-%m-%d")
        if isinstance(dt, date):
            return dt.strftime("%Y-%m-%d")
        return str(dt)

    def _empty_quote(self, symbol: str) -> Quote:
        """Return an empty quote for error cases.

        Args:
            symbol: Trading symbol

        Returns:
            Empty Quote object
        """
        return Quote(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            bid=0.0,
            ask=0.0,
            bid_size=0.0,
            ask_size=0.0,
            last=0.0,
            last_size=0.0,
        )
