"""
Tests for Market Utilities Module.

Tests cover:
- MarketHoursChecker
- PriceFetcher
- PnLCalculator
- Integration scenarios

Author: Super Gnosis Elite Trading System
"""

from datetime import datetime, timedelta, timezone, time as dt_time
from unittest.mock import MagicMock, patch, Mock

import pytest

from gnosis.market_utils import (
    MarketStatus,
    MarketHours,
    PriceQuote,
    PositionPnL,
    PortfolioPnL,
    MarketHoursChecker,
    PriceFetcher,
    PnLCalculator,
    create_market_hours_checker,
    create_price_fetcher,
    create_pnl_calculator,
)


class TestMarketStatus:
    """Tests for MarketStatus enum."""
    
    def test_status_values(self):
        assert MarketStatus.PRE_MARKET == "pre_market"
        assert MarketStatus.OPEN == "open"
        assert MarketStatus.POST_MARKET == "post_market"
        assert MarketStatus.CLOSED == "closed"


class TestPriceQuote:
    """Tests for PriceQuote dataclass."""
    
    def test_creation(self):
        quote = PriceQuote(
            symbol="AAPL",
            bid=199.50,
            ask=200.50,
            last=200.0,
            mid=200.0,
            timestamp=datetime.now(timezone.utc),
            volume=1000000,
        )
        assert quote.symbol == "AAPL"
        assert quote.bid == 199.50
        assert quote.ask == 200.50
    
    def test_spread_calculation(self):
        quote = PriceQuote(
            symbol="AAPL",
            bid=199.50,
            ask=200.50,
            last=200.0,
            mid=200.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert quote.spread == 1.0
        assert quote.spread_pct == pytest.approx(0.005, rel=0.01)


class TestPositionPnL:
    """Tests for PositionPnL dataclass."""
    
    def test_long_position_gain(self):
        pnl = PositionPnL(
            symbol="AAPL",
            quantity=100,
            entry_price=200.0,
            current_price=210.0,
            side="long",
        )
        assert pnl.cost_basis == 20000.0
        assert pnl.market_value == 21000.0
        assert pnl.unrealized_pnl == 1000.0
        assert pnl.unrealized_pnl_pct == pytest.approx(0.05, rel=0.01)
    
    def test_long_position_loss(self):
        pnl = PositionPnL(
            symbol="AAPL",
            quantity=100,
            entry_price=200.0,
            current_price=190.0,
            side="long",
        )
        assert pnl.unrealized_pnl == -1000.0
        assert pnl.unrealized_pnl_pct == pytest.approx(-0.05, rel=0.01)
    
    def test_short_position_gain(self):
        pnl = PositionPnL(
            symbol="AAPL",
            quantity=100,
            entry_price=200.0,
            current_price=190.0,
            side="short",
        )
        assert pnl.unrealized_pnl == 1000.0  # Gain on short
    
    def test_short_position_loss(self):
        pnl = PositionPnL(
            symbol="AAPL",
            quantity=100,
            entry_price=200.0,
            current_price=210.0,
            side="short",
        )
        assert pnl.unrealized_pnl == -1000.0  # Loss on short


class TestPortfolioPnL:
    """Tests for PortfolioPnL dataclass."""
    
    def test_portfolio_pnl_gain(self):
        position_pnls = [
            PositionPnL(
                symbol="AAPL",
                quantity=100,
                entry_price=200.0,
                current_price=210.0,
                side="long",
            ),
        ]
        
        portfolio = PortfolioPnL(
            timestamp=datetime.now(timezone.utc),
            starting_equity=100000.0,
            current_equity=101000.0,
            position_pnls=position_pnls,
        )
        
        assert portfolio.total_pnl == 1000.0
        assert portfolio.total_pnl_pct == pytest.approx(0.01, rel=0.01)
        assert portfolio.unrealized_pnl == 1000.0
    
    def test_portfolio_empty(self):
        portfolio = PortfolioPnL(
            timestamp=datetime.now(timezone.utc),
            starting_equity=100000.0,
            current_equity=100000.0,
        )
        
        assert portfolio.total_pnl == 0.0
        assert portfolio.unrealized_pnl == 0.0


class TestMarketHoursChecker:
    """Tests for MarketHoursChecker."""
    
    @pytest.fixture
    def checker(self):
        return MarketHoursChecker()
    
    def test_initialization(self, checker):
        assert checker.broker_adapter is None
        assert checker._clock_cache is None
    
    def test_is_trading_day_weekday(self, checker):
        # Monday
        monday = datetime(2024, 12, 16)
        assert checker.is_trading_day(monday) is True
    
    def test_is_trading_day_weekend(self, checker):
        # Saturday
        saturday = datetime(2024, 12, 21)
        assert checker.is_trading_day(saturday) is False
        
        # Sunday
        sunday = datetime(2024, 12, 22)
        assert checker.is_trading_day(sunday) is False
    
    def test_is_trading_day_holiday(self, checker):
        # Christmas 2024
        christmas = datetime(2024, 12, 25)
        assert checker.is_trading_day(christmas) is False
    
    def test_get_market_hours(self, checker):
        hours = checker.get_market_hours(datetime(2024, 12, 16))
        
        assert hours.is_trading_day is True
        assert hours.market_open.hour == 9
        assert hours.market_open.minute == 30
        assert hours.market_close.hour == 16
    
    def test_get_market_status_fallback(self, checker):
        # Without broker adapter, uses standard hours logic
        status = checker.get_market_status()
        assert status in [
            MarketStatus.PRE_MARKET,
            MarketStatus.OPEN,
            MarketStatus.POST_MARKET,
            MarketStatus.CLOSED,
        ]
    
    def test_with_alpaca_clock(self):
        mock_adapter = MagicMock()
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_adapter.get_clock.return_value = mock_clock
        
        checker = MarketHoursChecker(broker_adapter=mock_adapter)
        assert checker.is_market_open() is True


class TestPriceFetcher:
    """Tests for PriceFetcher."""
    
    @pytest.fixture
    def fetcher(self):
        return PriceFetcher()
    
    def test_initialization(self, fetcher):
        assert fetcher.broker_adapter is None
        assert fetcher.market_adapter is None
        assert "SPY" in fetcher._fallback_prices
    
    def test_get_quote_fallback(self, fetcher):
        quote = fetcher.get_quote("SPY")
        
        assert quote is not None
        assert quote.symbol == "SPY"
        assert quote.mid == pytest.approx(600.0, rel=0.01)
    
    def test_get_price_fallback(self, fetcher):
        price = fetcher.get_price("AAPL")
        
        assert price is not None
        assert price == pytest.approx(250.0, rel=0.01)
    
    def test_get_price_unknown_symbol(self, fetcher):
        price = fetcher.get_price("UNKNOWN_SYMBOL_XYZ")
        assert price is None
    
    def test_get_prices_multiple(self, fetcher):
        prices = fetcher.get_prices(["SPY", "QQQ", "AAPL"])
        
        assert len(prices) == 3
        assert "SPY" in prices
        assert "QQQ" in prices
        assert "AAPL" in prices
    
    def test_update_fallback_price(self, fetcher):
        fetcher.update_fallback_price("NEWSTOCK", 123.45)
        
        price = fetcher.get_price("NEWSTOCK")
        assert price == pytest.approx(123.45, rel=0.01)
    
    def test_caching(self, fetcher):
        # First call
        quote1 = fetcher.get_quote("SPY")
        
        # Second call should use cache (same quote object from cache)
        quote2 = fetcher.get_quote("SPY", use_cache=True)
        
        # Both should have the same price (cached)
        assert quote1.mid == quote2.mid
        assert quote1.symbol == quote2.symbol
    
    def test_with_broker_adapter(self):
        mock_adapter = MagicMock()
        mock_adapter.get_latest_quote.return_value = {
            "bid": 199.50,
            "ask": 200.50,
            "volume": 1000000,
        }
        
        fetcher = PriceFetcher(broker_adapter=mock_adapter)
        quote = fetcher.get_quote("AAPL")
        
        assert quote is not None
        assert quote.bid == 199.50
        assert quote.ask == 200.50


class TestPnLCalculator:
    """Tests for PnLCalculator."""
    
    @pytest.fixture
    def calculator(self):
        fetcher = PriceFetcher()
        return PnLCalculator(price_fetcher=fetcher)
    
    def test_calculate_position_pnl_with_price(self, calculator):
        pnl = calculator.calculate_position_pnl(
            symbol="AAPL",
            quantity=100,
            entry_price=200.0,
            side="long",
            current_price=210.0,
        )
        
        assert pnl.unrealized_pnl == 1000.0
    
    def test_calculate_position_pnl_fetch_price(self, calculator):
        # Uses fallback price
        pnl = calculator.calculate_position_pnl(
            symbol="AAPL",
            quantity=100,
            entry_price=200.0,
            side="long",
        )
        
        # AAPL fallback is 250
        expected_pnl = (250.0 - 200.0) * 100
        assert pnl.unrealized_pnl == pytest.approx(expected_pnl, rel=0.01)
    
    def test_calculate_portfolio_pnl(self, calculator):
        positions = [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "entry_price": 200.0,
                "side": "long",
                "current_price": 210.0,
            },
            {
                "symbol": "MSFT",
                "quantity": 50,
                "entry_price": 400.0,
                "side": "long",
                "current_price": 420.0,
            },
        ]
        
        portfolio = calculator.calculate_portfolio_pnl(
            positions=positions,
            starting_equity=100000.0,
            current_equity=102000.0,
        )
        
        assert portfolio.total_pnl == 2000.0
        assert len(portfolio.position_pnls) == 2
        
        # Check individual positions
        aapl_pnl = portfolio.position_pnls[0]
        assert aapl_pnl.unrealized_pnl == 1000.0
        
        msft_pnl = portfolio.position_pnls[1]
        assert msft_pnl.unrealized_pnl == 1000.0


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_market_hours_checker(self):
        checker = create_market_hours_checker()
        assert isinstance(checker, MarketHoursChecker)
    
    def test_create_price_fetcher(self):
        fetcher = create_price_fetcher()
        assert isinstance(fetcher, PriceFetcher)
    
    def test_create_pnl_calculator(self):
        calculator = create_pnl_calculator()
        assert isinstance(calculator, PnLCalculator)
        assert isinstance(calculator.price_fetcher, PriceFetcher)


class TestIntegration:
    """Integration tests for market utilities."""
    
    def test_full_pnl_workflow(self):
        """Test complete P&L calculation workflow."""
        # Create calculator
        calculator = create_pnl_calculator()
        
        # Add positions
        positions = [
            {
                "symbol": "SPY",
                "quantity": 10,
                "entry_price": 590.0,
                "side": "long",
            },
            {
                "symbol": "QQQ",
                "quantity": 20,
                "entry_price": 510.0,
                "side": "long",
            },
        ]
        
        # Calculate portfolio P&L
        portfolio = calculator.calculate_portfolio_pnl(
            positions=positions,
            starting_equity=100000.0,
        )
        
        assert portfolio is not None
        assert len(portfolio.position_pnls) == 2
        assert portfolio.unrealized_pnl != 0  # Should have some P&L from fallback prices
    
    def test_market_hours_with_price_check(self):
        """Test combining market hours and price checking."""
        checker = create_market_hours_checker()
        fetcher = create_price_fetcher()
        
        # Get market status
        status = checker.get_market_status()
        
        # Get prices
        prices = fetcher.get_prices(["SPY", "QQQ"])
        
        # Both should work
        assert status in [
            MarketStatus.PRE_MARKET,
            MarketStatus.OPEN,
            MarketStatus.POST_MARKET,
            MarketStatus.CLOSED,
        ]
        assert len(prices) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
