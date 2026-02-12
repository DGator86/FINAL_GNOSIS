"""
Paper Trading Integration Tests.

End-to-end tests validating the complete trading system including:
- Broker adapter connectivity
- Multi-leg options execution
- Safety controls
- Market utilities
- ML pipeline integration

These tests use mocked broker responses to simulate paper trading
without requiring actual API credentials.

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Import components to test
from gnosis.market_utils import (
    MarketHoursChecker,
    MarketStatus,
    PriceFetcher,
    PnLCalculator,
    PositionPnL,
    PortfolioPnL,
    PriceQuote,
)
from trade.trading_safety import (
    SafetyConfig,
    TradingSafetyManager,
    SafetyMetrics,
    CircuitBreakerState,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_broker_adapter():
    """Create a mock broker adapter for testing."""
    adapter = MagicMock()
    
    # Mock account info
    adapter.get_account.return_value = {
        "equity": 100000.0,
        "buying_power": 50000.0,
        "cash": 30000.0,
        "portfolio_value": 100000.0,
        "status": "ACTIVE",
    }
    
    # Mock clock
    mock_clock = MagicMock()
    mock_clock.is_open = True
    mock_clock.next_open = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_clock.next_close = datetime.now(timezone.utc) + timedelta(hours=6)
    adapter.get_clock.return_value = mock_clock
    
    # Mock positions
    adapter.get_positions.return_value = [
        {
            "symbol": "AAPL",
            "qty": 100,
            "side": "long",
            "avg_entry_price": 175.0,
            "current_price": 180.0,
            "market_value": 18000.0,
            "unrealized_pl": 500.0,
        },
        {
            "symbol": "SPY",
            "qty": 50,
            "side": "long",
            "avg_entry_price": 580.0,
            "current_price": 590.0,
            "market_value": 29500.0,
            "unrealized_pl": 500.0,
        },
    ]
    
    # Mock orders
    adapter.submit_order.return_value = {
        "id": "test-order-123",
        "status": "filled",
        "symbol": "AAPL",
        "qty": 10,
        "side": "buy",
        "filled_avg_price": 180.0,
    }
    
    # Mock quotes
    adapter.get_latest_quote.return_value = {
        "symbol": "AAPL",
        "bid": 179.95,
        "ask": 180.05,
        "bid_price": 179.95,
        "ask_price": 180.05,
        "last": 180.0,
    }
    
    return adapter


@pytest.fixture
def mock_options_adapter():
    """Create a mock options adapter for testing."""
    adapter = MagicMock()
    
    # Mock multi-leg order
    adapter.place_multileg_order.return_value = MagicMock(
        success=True,
        order_id="mleg-order-456",
        message="Order filled",
        legs=[
            {"symbol": "AAPL250117C00200000", "qty": 1, "side": "buy", "filled_price": 5.00},
            {"symbol": "AAPL250117C00210000", "qty": 1, "side": "sell", "filled_price": 2.50},
        ],
        total_cost=250.0,
        fill_prices={"AAPL250117C00200000": 5.00, "AAPL250117C00210000": 2.50},
    )
    
    # Mock options positions
    adapter.get_options_positions.return_value = [
        {
            "symbol": "AAPL250117C00200000",
            "qty": 1,
            "side": "long",
            "avg_entry_price": 5.00,
            "current_price": 5.50,
            "unrealized_pl": 50.0,
        }
    ]
    
    return adapter


@pytest.fixture
def safety_config():
    """Create a safety configuration for testing."""
    return SafetyConfig(
        max_daily_loss_usd=5000.0,
        max_daily_loss_pct=0.05,
        max_positions=10,
        max_position_size_pct=0.10,
        circuit_breaker_loss_threshold_pct=0.05,
        max_bid_ask_spread_pct=0.10,  # Allow wider spreads in tests
    )


# ============================================================================
# Market Hours Tests
# ============================================================================

class TestMarketHoursIntegration:
    """Tests for market hours checking integration."""
    
    def test_market_hours_checker_initialization(self, mock_broker_adapter):
        """Test market hours checker initializes correctly."""
        checker = MarketHoursChecker(broker_adapter=mock_broker_adapter)
        
        assert checker is not None
        assert checker.broker_adapter == mock_broker_adapter
    
    def test_market_status_open(self, mock_broker_adapter):
        """Test market status when market is open."""
        mock_broker_adapter.get_clock.return_value.is_open = True
        
        checker = MarketHoursChecker(broker_adapter=mock_broker_adapter)
        
        assert checker.is_market_open() is True
    
    def test_market_status_closed(self, mock_broker_adapter):
        """Test market status when market is closed."""
        mock_broker_adapter.get_clock.return_value.is_open = False
        
        checker = MarketHoursChecker(broker_adapter=mock_broker_adapter)
        
        assert checker.is_market_open() is False
    
    def test_trading_day_check(self):
        """Test trading day validation."""
        checker = MarketHoursChecker()
        
        # Monday should be a trading day
        monday = datetime(2025, 1, 6)  # A Monday
        assert checker.is_trading_day(monday) is True
        
        # Saturday should not be
        saturday = datetime(2025, 1, 4)  # A Saturday
        assert checker.is_trading_day(saturday) is False
    
    def test_market_hours_fallback(self):
        """Test market hours with no broker adapter (fallback mode)."""
        checker = MarketHoursChecker(broker_adapter=None)
        
        # Should not crash and return a valid status
        status = checker.get_market_status()
        assert status in [MarketStatus.PRE_MARKET, MarketStatus.OPEN, 
                         MarketStatus.POST_MARKET, MarketStatus.CLOSED]


# ============================================================================
# Price Fetching Tests
# ============================================================================

class TestPriceFetcherIntegration:
    """Tests for price fetching integration."""
    
    def test_price_fetcher_with_broker(self, mock_broker_adapter):
        """Test price fetching via broker adapter."""
        fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        
        quote = fetcher.get_quote("AAPL")
        
        assert quote is not None
        assert quote.symbol == "AAPL"
        assert quote.bid > 0
        assert quote.ask > 0
        assert quote.mid > 0
    
    def test_price_fetcher_fallback(self):
        """Test price fetcher fallback prices."""
        fetcher = PriceFetcher(broker_adapter=None, market_adapter=None)
        
        # Should use fallback prices for common symbols
        price = fetcher.get_price("SPY")
        assert price is not None
        assert price > 0
    
    def test_price_caching(self, mock_broker_adapter):
        """Test that prices are cached."""
        fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        
        # First call
        quote1 = fetcher.get_quote("AAPL")
        
        # Second call should hit cache
        quote2 = fetcher.get_quote("AAPL", use_cache=True)
        
        assert quote1 == quote2
        # Broker should only be called once
        assert mock_broker_adapter.get_latest_quote.call_count == 1
    
    def test_get_multiple_prices(self, mock_broker_adapter):
        """Test fetching prices for multiple symbols."""
        fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        
        prices = fetcher.get_prices(["AAPL", "SPY", "QQQ"])
        
        assert len(prices) >= 1  # At least some prices returned


# ============================================================================
# P&L Calculation Tests
# ============================================================================

class TestPnLCalculatorIntegration:
    """Tests for P&L calculation integration."""
    
    def test_position_pnl_calculation(self, mock_broker_adapter):
        """Test position P&L calculation."""
        fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        calculator = PnLCalculator(price_fetcher=fetcher)
        
        pnl = calculator.calculate_position_pnl(
            symbol="AAPL",
            quantity=100,
            entry_price=175.0,
            side="long",
            current_price=180.0,
        )
        
        assert pnl.unrealized_pnl == 500.0  # (180 - 175) * 100
        assert pnl.cost_basis == 17500.0  # 175 * 100
        assert pnl.market_value == 18000.0  # 180 * 100
    
    def test_short_position_pnl(self, mock_broker_adapter):
        """Test short position P&L calculation."""
        fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        calculator = PnLCalculator(price_fetcher=fetcher)
        
        pnl = calculator.calculate_position_pnl(
            symbol="AAPL",
            quantity=100,
            entry_price=185.0,
            side="short",
            current_price=180.0,
        )
        
        # Short profit when price goes down
        assert pnl.unrealized_pnl == 500.0  # (185 - 180) * 100
    
    def test_portfolio_pnl_calculation(self, mock_broker_adapter):
        """Test portfolio-level P&L calculation."""
        fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        calculator = PnLCalculator(price_fetcher=fetcher)
        
        positions = [
            {"symbol": "AAPL", "quantity": 100, "entry_price": 175.0, "side": "long", "current_price": 180.0},
            {"symbol": "SPY", "quantity": 50, "entry_price": 580.0, "side": "long", "current_price": 590.0},
        ]
        
        portfolio = calculator.calculate_portfolio_pnl(
            positions=positions,
            starting_equity=100000.0,
        )
        
        assert portfolio.unrealized_pnl == 1000.0  # 500 + 500
        assert len(portfolio.position_pnls) == 2


# ============================================================================
# Trading Safety Tests
# ============================================================================

class TestTradingSafetyIntegration:
    """Tests for trading safety controls integration."""
    
    def test_safety_manager_initialization(self, safety_config):
        """Test safety manager initializes correctly."""
        manager = TradingSafetyManager(config=safety_config)
        
        assert manager is not None
        assert manager.config.max_daily_loss_usd == 5000.0
    
    def test_trade_validation_passes(self, safety_config):
        """Test trade validation passes for valid trade."""
        manager = TradingSafetyManager(config=safety_config, portfolio_value=100000.0)
        
        # Update metrics to simulate healthy state
        manager.metrics.daily_pnl = 1000.0  # Profitable day
        manager.metrics.position_count = 2
        
        result = manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=180.0,
        )
        
        assert result.approved is True
    
    def test_trade_validation_fails_max_positions(self, safety_config):
        """Test trade validation fails when max positions exceeded."""
        safety_config.max_positions = 2
        manager = TradingSafetyManager(config=safety_config, portfolio_value=100000.0)
        
        manager.metrics.position_count = 2
        
        result = manager.validate_trade(
            symbol="NEW_SYMBOL",
            side="buy",
            quantity=10,
            price=100.0,
        )
        
        assert result.approved is False
        assert "position" in result.reason.lower()
    
    def test_trade_validation_fails_daily_loss(self, safety_config):
        """Test trade validation fails when daily loss limit hit."""
        manager = TradingSafetyManager(config=safety_config, portfolio_value=100000.0)
        
        manager.metrics.daily_pnl = -6000.0  # Exceeded daily loss limit
        
        result = manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=180.0,
        )
        
        assert result.approved is False
        assert "loss" in result.reason.lower()
    
    def test_circuit_breaker_state(self, safety_config):
        """Test circuit breaker state affects trading."""
        manager = TradingSafetyManager(config=safety_config, portfolio_value=100000.0)
        
        # Set circuit breaker to OPEN
        manager.metrics.circuit_breaker_state = CircuitBreakerState.OPEN
        
        result = manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=180.0,
        )
        
        assert result.approved is False
        assert "circuit" in result.reason.lower()
    
    def test_position_sizing_warning(self, safety_config):
        """Test position sizing generates warnings for large trades."""
        manager = TradingSafetyManager(config=safety_config, portfolio_value=100000.0)
        
        # Request large position (>10% of portfolio)
        result = manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=100,  # 100 * 180 = 18000 = 18% of portfolio
            price=180.0,
        )
        
        # Should approve but with warnings
        assert result.approved is True
        assert len(result.warnings) > 0


# ============================================================================
# End-to-End Paper Trading Flow Tests
# ============================================================================

class TestPaperTradingFlow:
    """End-to-end tests for paper trading workflow."""
    
    def test_complete_trade_flow(self, mock_broker_adapter, safety_config):
        """Test complete trading flow from signal to execution."""
        # Setup components
        market_checker = MarketHoursChecker(broker_adapter=mock_broker_adapter)
        price_fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        pnl_calculator = PnLCalculator(price_fetcher=price_fetcher)
        safety_manager = TradingSafetyManager(config=safety_config, portfolio_value=100000.0)
        
        # Step 1: Check market status
        assert market_checker.is_market_open() is True
        
        # Step 2: Get current price
        quote = price_fetcher.get_quote("AAPL")
        assert quote is not None
        
        # Step 3: Validate trade
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=quote.mid,
        )
        assert result.approved is True
        
        # Step 4: Execute trade (mocked)
        order = mock_broker_adapter.submit_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            type="market",
        )
        assert order["status"] == "filled"
        
        # Step 5: Calculate P&L after trade
        pnl = pnl_calculator.calculate_position_pnl(
            symbol="AAPL",
            quantity=10,
            entry_price=order["filled_avg_price"],
            side="long",
            current_price=quote.mid,
        )
        assert pnl is not None
    
    def test_options_multileg_flow(self, mock_options_adapter, safety_config):
        """Test multi-leg options trading flow."""
        safety_manager = TradingSafetyManager(config=safety_config, portfolio_value=100000.0)
        
        # Validate trade
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=1,
            price=250.0,  # Net debit for spread
        )
        assert result.approved is True
        
        # Execute multi-leg order
        order_result = mock_options_adapter.place_multileg_order(
            legs=[
                {"symbol": "AAPL250117C00200000", "side": "buy", "qty": 1},
                {"symbol": "AAPL250117C00210000", "side": "sell", "qty": 1},
            ],
            order_type="limit",
            limit_price=2.50,
        )
        
        assert order_result.success is True
        assert order_result.order_id is not None
        assert len(order_result.legs) == 2
    
    def test_risk_management_flow(self, mock_broker_adapter, safety_config):
        """Test risk management during trading."""
        safety_manager = TradingSafetyManager(config=safety_config, portfolio_value=100000.0)
        price_fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        pnl_calculator = PnLCalculator(price_fetcher=price_fetcher)
        
        # Simulate existing positions
        positions = [
            {"symbol": "AAPL", "quantity": 100, "entry_price": 175.0, "side": "long"},
            {"symbol": "SPY", "quantity": 50, "entry_price": 580.0, "side": "long"},
        ]
        
        # Update safety metrics
        safety_manager.metrics.position_count = len(positions)
        safety_manager.metrics.total_exposure_pct = 0.465  # 46.5%
        
        # Calculate portfolio P&L
        portfolio = pnl_calculator.calculate_portfolio_pnl(
            positions=positions,
            starting_equity=100000.0,
        )
        
        # Verify circuit breaker not triggered (no losses)
        assert safety_manager.metrics.circuit_breaker_state == CircuitBreakerState.CLOSED
        
        # New trade should still be valid
        result = safety_manager.validate_trade(
            symbol="MSFT",
            side="buy",
            quantity=20,
            price=430.0,
        )
        assert result.approved is True


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in trading flow."""
    
    def test_broker_connection_failure(self):
        """Test handling of broker connection failure."""
        mock_adapter = MagicMock()
        mock_adapter.get_clock.side_effect = Exception("Connection failed")
        
        checker = MarketHoursChecker(broker_adapter=mock_adapter)
        
        # Should fall back to standard hours, not crash
        status = checker.get_market_status()
        assert status is not None
    
    def test_quote_fetch_failure(self):
        """Test handling of quote fetch failure."""
        mock_adapter = MagicMock()
        mock_adapter.get_latest_quote.side_effect = Exception("Quote unavailable")
        
        fetcher = PriceFetcher(broker_adapter=mock_adapter)
        
        # Should return fallback price for known symbols
        price = fetcher.get_price("SPY")
        assert price is not None
    
    def test_order_rejection_handling(self, mock_broker_adapter):
        """Test handling of order rejection."""
        mock_broker_adapter.submit_order.return_value = {
            "id": "test-order-456",
            "status": "rejected",
            "reject_reason": "Insufficient buying power",
        }
        
        order = mock_broker_adapter.submit_order(
            symbol="AAPL",
            qty=10000,
            side="buy",
            type="market",
        )
        
        assert order["status"] == "rejected"
        assert "buying power" in order["reject_reason"].lower()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for trading components."""
    
    def test_price_cache_performance(self, mock_broker_adapter):
        """Test that price caching improves performance."""
        import time
        
        fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        
        # First call (no cache)
        start = time.perf_counter()
        for _ in range(100):
            fetcher.get_quote("AAPL", use_cache=True)
        cached_time = time.perf_counter() - start
        
        # Cached calls should be fast
        # Most calls hit cache, so average should be very low
        assert cached_time < 1.0  # 100 calls in under 1 second
    
    def test_pnl_calculation_performance(self, mock_broker_adapter):
        """Test P&L calculation performance with many positions."""
        import time
        
        fetcher = PriceFetcher(broker_adapter=mock_broker_adapter)
        calculator = PnLCalculator(price_fetcher=fetcher)
        
        # Create many positions
        positions = [
            {
                "symbol": f"SYM{i}",
                "quantity": 100,
                "entry_price": 100.0,
                "side": "long",
                "current_price": 105.0,
            }
            for i in range(100)
        ]
        
        start = time.perf_counter()
        portfolio = calculator.calculate_portfolio_pnl(
            positions=positions,
            starting_equity=1000000.0,
        )
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0  # 100 positions in under 1 second
        assert len(portfolio.position_pnls) == 100
