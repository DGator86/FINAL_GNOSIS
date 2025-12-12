"""Integration tests for order execution and risk management."""

import os
from unittest.mock import Mock, patch

import pytest

from execution.broker_adapters.alpaca_adapter import Account, AlpacaBrokerAdapter


class TestOrderExecutionIntegration:
    """Integration tests for order execution with risk management."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock trading client."""
        client = Mock()

        # Mock account with realistic values
        account = Mock()
        account.id = "test-account-123"
        account.cash = 50000.0
        account.buying_power = 50000.0
        account.portfolio_value = 100000.0
        account.equity = 100000.0
        account.last_equity = 100000.0
        account.pattern_day_trader = False
        account.trading_blocked = False
        account.options_trading_level = 2
        account.options_approved_level = 2

        client.get_account.return_value = account
        client.get_all_positions.return_value = []

        return client

    @pytest.fixture
    def broker_adapter(self, mock_trading_client):
        """Create broker adapter with mocked client."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "test_key",
            "ALPACA_SECRET_KEY": "test_secret",
            "ALPACA_PAPER": "true",
            "MAX_POSITION_SIZE_PCT": "2.0",
            "MAX_DAILY_LOSS_USD": "5000.0",
        }):
            adapter = AlpacaBrokerAdapter(paper=True)
            adapter.trading_client = mock_trading_client
            adapter.data_client = Mock()
            return adapter

    def test_position_size_validation_passes(self, broker_adapter, mock_trading_client):
        """Test that valid position size passes validation."""
        # Portfolio value = $100k, max position = 2% = $2000
        # Order: 10 shares @ $150 = $1500 (< $2000) ✓

        # Mock price quote
        quote_mock = Mock()
        quote_mock["SPY"] = Mock(ask_price=150.0)
        broker_adapter.data_client.get_stock_latest_quote.return_value = quote_mock

        # Mock successful order submission
        order_mock = Mock()
        order_mock.id = "order-123"
        mock_trading_client.submit_order.return_value = order_mock

        # Should not raise exception
        order_id = broker_adapter.place_order(
            symbol="SPY",
            quantity=10,
            side="buy",
            order_type="market",
        )

        assert order_id == "order-123"
        assert mock_trading_client.submit_order.called

    def test_position_size_validation_fails(self, broker_adapter):
        """Test that oversized position fails validation."""
        # Portfolio value = $100k, max position = 2% = $2000
        # Order: 100 shares @ $450 = $45,000 (> $2000) ✗

        # Mock price quote
        quote_mock = Mock()
        quote_mock["SPY"] = Mock(ask_price=450.0)
        broker_adapter.data_client.get_stock_latest_quote.return_value = quote_mock

        # Should raise ValueError due to position size
        with pytest.raises(ValueError, match="exceeds maximum position size"):
            broker_adapter.place_order(
                symbol="SPY",
                quantity=100,
                side="buy",
                order_type="market",
            )

    def test_daily_loss_circuit_breaker_triggered(self, broker_adapter, mock_trading_client):
        """Test circuit breaker triggers when daily loss limit exceeded."""
        # Session started at $100k equity
        broker_adapter.session_start_equity = 100000.0

        # Current equity dropped to $94k (loss of $6k > $5k limit)
        account = mock_trading_client.get_account.return_value
        account.equity = 94000.0

        # Mock quote
        quote_mock = Mock()
        quote_mock["SPY"] = Mock(ask_price=450.0)
        broker_adapter.data_client.get_stock_latest_quote.return_value = quote_mock

        # Circuit breaker should trigger
        with pytest.raises(ValueError, match="CIRCUIT BREAKER TRIGGERED"):
            broker_adapter.place_order(
                symbol="SPY",
                quantity=1,
                side="buy",
                order_type="market",
            )

    def test_daily_loss_circuit_breaker_passes(self, broker_adapter, mock_trading_client):
        """Test circuit breaker doesn't trigger with acceptable loss."""
        # Session started at $100k equity
        broker_adapter.session_start_equity = 100000.0

        # Current equity at $96k (loss of $4k < $5k limit) ✓
        account = mock_trading_client.get_account.return_value
        account.equity = 96000.0

        # Mock quote and order
        quote_mock = Mock()
        quote_mock["SPY"] = Mock(ask_price=450.0)
        broker_adapter.data_client.get_stock_latest_quote.return_value = quote_mock

        order_mock = Mock()
        order_mock.id = "order-456"
        mock_trading_client.submit_order.return_value = order_mock

        # Should pass
        order_id = broker_adapter.place_order(
            symbol="SPY",
            quantity=1,
            side="buy",
            order_type="market",
        )

        assert order_id == "order-456"

    def test_sell_orders_bypass_validation(self, broker_adapter, mock_trading_client):
        """Test that sell orders don't trigger position size validation."""
        # Sell orders should bypass position size check (you can sell what you own)
        order_mock = Mock()
        order_mock.id = "order-789"
        mock_trading_client.submit_order.return_value = order_mock

        # Large sell should not raise exception
        order_id = broker_adapter.place_order(
            symbol="SPY",
            quantity=1000,
            side="sell",
            order_type="market",
        )

        assert order_id == "order-789"

    def test_limit_order_uses_limit_price_for_validation(self, broker_adapter, mock_trading_client):
        """Test that limit orders use limit price for position size validation."""
        # Limit order at $100 per share, 15 shares = $1500 (< $2000) ✓
        order_mock = Mock()
        order_mock.id = "order-limit-1"
        mock_trading_client.submit_order.return_value = order_mock

        order_id = broker_adapter.place_order(
            symbol="SPY",
            quantity=15,
            side="buy",
            order_type="limit",
            limit_price=100.0,
        )

        assert order_id == "order-limit-1"

    def test_risk_parameters_from_environment(self):
        """Test that risk parameters are loaded from environment."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "test_key",
            "ALPACA_SECRET_KEY": "test_secret",
            "ALPACA_PAPER": "true",
            "MAX_POSITION_SIZE_PCT": "5.0",
            "MAX_DAILY_LOSS_USD": "10000.0",
            "MAX_PORTFOLIO_LEVERAGE": "2.0",
        }):
            with patch("execution.broker_adapters.alpaca_adapter.TradingClient"):
                with patch("execution.broker_adapters.alpaca_adapter.StockHistoricalDataClient"):
                    adapter = AlpacaBrokerAdapter(paper=True)

                    assert adapter.max_position_size_pct == 0.05  # 5%
                    assert adapter.max_daily_loss_usd == 10000.0
                    assert adapter.max_portfolio_leverage == 2.0

    def test_account_info_retrieval(self, broker_adapter, mock_trading_client):
        """Test account information retrieval and mapping."""
        account = broker_adapter.get_account()

        assert isinstance(account, Account)
        assert account.account_id == "test-account-123"
        assert account.cash == 50000.0
        assert account.portfolio_value == 100000.0
        assert not account.trading_blocked

    def test_position_retrieval(self, broker_adapter, mock_trading_client):
        """Test position retrieval."""
        # Mock positions
        pos_mock = Mock()
        pos_mock.symbol = "SPY"
        pos_mock.qty = 10
        pos_mock.avg_entry_price = 450.0
        pos_mock.current_price = 455.0
        pos_mock.market_value = 4550.0
        pos_mock.cost_basis = 4500.0
        pos_mock.unrealized_pl = 50.0
        pos_mock.unrealized_plpc = 0.011
        pos_mock.side.value = "long"

        mock_trading_client.get_all_positions.return_value = [pos_mock]

        positions = broker_adapter.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "SPY"
        assert positions[0].quantity == 10.0
        assert positions[0].unrealized_pnl == 50.0
