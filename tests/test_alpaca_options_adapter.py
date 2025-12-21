"""Comprehensive tests for AlpacaOptionsAdapter multi-leg functionality.

Tests cover:
- OptionLeg dataclass
- MultiLegOrderResult dataclass
- Multi-leg order validation
- Multi-leg order submission (mocked)
- Position closing with inverted sides
- Helper functions for strategies
- Error handling and edge cases

Author: Super Gnosis Elite Trading System
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from execution.broker_adapters.alpaca_options_adapter import (
    AlpacaOptionsAdapter,
    OptionLeg,
    PositionIntent,
    MultiLegOrderResult,
    create_vertical_spread,
    create_iron_condor,
)


class TestOptionLeg:
    """Tests for OptionLeg dataclass."""

    def test_option_leg_creation(self):
        """Test basic OptionLeg creation."""
        leg = OptionLeg(
            symbol="AAPL250117C00200000",
            side="buy",
            ratio_qty=1,
            position_intent=PositionIntent.BUY_TO_OPEN,
        )
        assert leg.symbol == "AAPL250117C00200000"
        assert leg.side == "buy"
        assert leg.ratio_qty == 1
        assert leg.position_intent == PositionIntent.BUY_TO_OPEN

    def test_option_leg_default_values(self):
        """Test OptionLeg default values."""
        leg = OptionLeg(symbol="SPY250117P00500000", side="sell")
        assert leg.ratio_qty == 1
        assert leg.position_intent == PositionIntent.BUY_TO_OPEN

    def test_option_leg_to_dict(self):
        """Test OptionLeg to_dict conversion."""
        leg = OptionLeg(
            symbol="AAPL250117C00200000",
            side="buy",
            ratio_qty=2,
            position_intent=PositionIntent.SELL_TO_CLOSE,
        )
        result = leg.to_dict()
        assert result["symbol"] == "AAPL250117C00200000"
        assert result["side"] == "buy"
        assert result["ratio_qty"] == "2"  # String conversion
        assert result["position_intent"] == "sell_to_close"

    def test_all_position_intents(self):
        """Test all position intent values."""
        assert PositionIntent.BUY_TO_OPEN.value == "buy_to_open"
        assert PositionIntent.BUY_TO_CLOSE.value == "buy_to_close"
        assert PositionIntent.SELL_TO_OPEN.value == "sell_to_open"
        assert PositionIntent.SELL_TO_CLOSE.value == "sell_to_close"


class TestMultiLegOrderResult:
    """Tests for MultiLegOrderResult dataclass."""

    def test_successful_result(self):
        """Test successful multi-leg order result."""
        result = MultiLegOrderResult(
            success=True,
            order_id="ord_123456",
            message="Order submitted successfully",
            legs_count=4,
            status="pending_new",
        )
        assert result.success is True
        assert result.order_id == "ord_123456"
        assert result.legs_count == 4
        assert result.order_class == "mleg"
        assert result.order_ids == ["ord_123456"]

    def test_failed_result(self):
        """Test failed multi-leg order result."""
        result = MultiLegOrderResult(
            success=False,
            message="Insufficient buying power",
        )
        assert result.success is False
        assert result.order_id is None
        assert result.order_ids == []

    def test_order_ids_backward_compat(self):
        """Test backward compatibility for order_ids."""
        result = MultiLegOrderResult(
            success=True,
            order_id="ord_abc",
        )
        assert result.order_ids == ["ord_abc"]

        result_no_id = MultiLegOrderResult(success=False)
        assert result_no_id.order_ids == []


class TestCreateVerticalSpread:
    """Tests for create_vertical_spread helper function."""

    def test_bull_call_spread_open(self):
        """Test creating a bull call spread for opening."""
        legs = create_vertical_spread(
            underlying="AAPL",
            expiration="250117",
            option_type="C",
            buy_strike=190.0,
            sell_strike=200.0,
            is_opening=True,
        )
        assert len(legs) == 2
        
        # Buy lower strike
        assert legs[0].symbol == "AAPL250117C00190000"
        assert legs[0].side == "buy"
        assert legs[0].position_intent == PositionIntent.BUY_TO_OPEN
        
        # Sell higher strike
        assert legs[1].symbol == "AAPL250117C00200000"
        assert legs[1].side == "sell"
        assert legs[1].position_intent == PositionIntent.SELL_TO_OPEN

    def test_bear_put_spread_open(self):
        """Test creating a bear put spread for opening."""
        legs = create_vertical_spread(
            underlying="SPY",
            expiration="250221",
            option_type="P",
            buy_strike=500.0,
            sell_strike=490.0,
            is_opening=True,
        )
        assert len(legs) == 2
        assert legs[0].symbol == "SPY250221P00500000"
        assert legs[1].symbol == "SPY250221P00490000"

    def test_spread_close(self):
        """Test closing a vertical spread."""
        legs = create_vertical_spread(
            underlying="AAPL",
            expiration="250117",
            option_type="C",
            buy_strike=190.0,
            sell_strike=200.0,
            is_opening=False,
        )
        assert len(legs) == 2
        
        # Close by selling the long
        assert legs[0].side == "sell"
        assert legs[0].position_intent == PositionIntent.SELL_TO_CLOSE
        
        # Close by buying back the short
        assert legs[1].side == "buy"
        assert legs[1].position_intent == PositionIntent.BUY_TO_CLOSE

    def test_fractional_strike(self):
        """Test handling fractional strikes."""
        legs = create_vertical_spread(
            underlying="SPX",
            expiration="250117",
            option_type="C",
            buy_strike=5850.5,
            sell_strike=5860.5,
        )
        # 5850.5 * 1000 = 5850500
        assert "05850500" in legs[0].symbol
        assert "05860500" in legs[1].symbol


class TestCreateIronCondor:
    """Tests for create_iron_condor helper function."""

    def test_iron_condor_open(self):
        """Test creating an iron condor for opening."""
        legs = create_iron_condor(
            underlying="SPY",
            expiration="250117",
            put_buy_strike=480.0,
            put_sell_strike=485.0,
            call_sell_strike=510.0,
            call_buy_strike=515.0,
            is_opening=True,
        )
        assert len(legs) == 4
        
        # Put spread (buy lower, sell higher put)
        assert legs[0].symbol == "SPY250117P00480000"
        assert legs[0].side == "buy"
        assert legs[0].position_intent == PositionIntent.BUY_TO_OPEN
        
        assert legs[1].symbol == "SPY250117P00485000"
        assert legs[1].side == "sell"
        assert legs[1].position_intent == PositionIntent.SELL_TO_OPEN
        
        # Call spread (sell lower, buy higher call)
        assert legs[2].symbol == "SPY250117C00510000"
        assert legs[2].side == "sell"
        assert legs[2].position_intent == PositionIntent.SELL_TO_OPEN
        
        assert legs[3].symbol == "SPY250117C00515000"
        assert legs[3].side == "buy"
        assert legs[3].position_intent == PositionIntent.BUY_TO_OPEN

    def test_iron_condor_close(self):
        """Test closing an iron condor."""
        legs = create_iron_condor(
            underlying="SPY",
            expiration="250117",
            put_buy_strike=480.0,
            put_sell_strike=485.0,
            call_sell_strike=510.0,
            call_buy_strike=515.0,
            is_opening=False,
        )
        assert len(legs) == 4
        
        # All positions should be inverted
        assert legs[0].side == "sell"  # Sell to close long put
        assert legs[0].position_intent == PositionIntent.SELL_TO_CLOSE
        
        assert legs[1].side == "buy"  # Buy to close short put
        assert legs[1].position_intent == PositionIntent.BUY_TO_CLOSE
        
        assert legs[2].side == "buy"  # Buy to close short call
        assert legs[2].position_intent == PositionIntent.BUY_TO_CLOSE
        
        assert legs[3].side == "sell"  # Sell to close long call
        assert legs[3].position_intent == PositionIntent.SELL_TO_CLOSE


class TestAlpacaOptionsAdapterMultiLeg:
    """Tests for AlpacaOptionsAdapter multi-leg functionality."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mocked options adapter."""
        with patch.object(AlpacaOptionsAdapter, '__init__', lambda self, paper=None: None):
            adapter = AlpacaOptionsAdapter()
            adapter.api_key = "test_key"
            adapter.secret_key = "test_secret"
            adapter.base_url = "https://paper-api.alpaca.markets"
            adapter._api_version = "v2"
            return adapter

    def test_place_multileg_order_validation_no_legs(self, mock_adapter):
        """Test validation fails with no legs."""
        result = mock_adapter.place_multileg_order(legs=[])
        assert result.success is False
        assert "No legs provided" in result.message

    def test_place_multileg_order_validation_single_leg(self, mock_adapter):
        """Test validation fails with single leg."""
        legs = [OptionLeg("AAPL250117C00200000", "buy")]
        result = mock_adapter.place_multileg_order(legs=legs)
        assert result.success is False
        assert "at least 2 legs" in result.message

    def test_place_multileg_order_invalid_leg_type(self, mock_adapter):
        """Test validation fails with invalid leg type."""
        legs = ["invalid", "legs"]
        result = mock_adapter.place_multileg_order(legs=legs)
        assert result.success is False
        assert "Invalid leg type" in result.message

    @patch('requests.post')
    def test_place_multileg_order_success(self, mock_post, mock_adapter):
        """Test successful multi-leg order submission."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "ord_multileg_123",
            "status": "pending_new",
            "filled_qty": 0,
        }
        mock_post.return_value = mock_response

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = mock_adapter.place_multileg_order(
            legs=legs,
            quantity=5,
            order_type="limit",
            limit_price=2.50,
            note="Bull Call Spread",
        )

        assert result.success is True
        assert result.order_id == "ord_multileg_123"
        assert result.status == "pending_new"
        assert result.legs_count == 2

        # Verify API call
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["order_class"] == "mleg"
        assert payload["qty"] == "5"
        assert payload["type"] == "limit"
        assert payload["limit_price"] == "2.5"  # Float to string conversion
        assert len(payload["legs"]) == 2

    @patch('requests.post')
    def test_place_multileg_order_with_dict_legs(self, mock_post, mock_adapter):
        """Test multi-leg order with dict legs (legacy format)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "ord_dict_123", "status": "new"}
        mock_post.return_value = mock_response

        legs = [
            {"symbol": "AAPL250117C00190000", "side": "buy", "qty": 1},
            {"symbol": "AAPL250117C00200000", "side": "sell", "qty": 1},
        ]
        result = mock_adapter.place_multileg_order(legs=legs, quantity=1)

        assert result.success is True
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        # Should have default position_intent
        assert payload["legs"][0]["position_intent"] == "buy_to_open"
        assert payload["legs"][1]["position_intent"] == "sell_to_open"

    @patch('requests.post')
    def test_place_multileg_order_api_error(self, mock_post, mock_adapter):
        """Test handling API error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid order parameters"
        mock_response.json.return_value = {"message": "Invalid order parameters"}
        mock_post.return_value = mock_response

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = mock_adapter.place_multileg_order(legs=legs, quantity=1)

        assert result.success is False
        assert "400" in result.message
        assert "Invalid order parameters" in result.message

    @patch('requests.post')
    def test_place_multileg_order_timeout(self, mock_post, mock_adapter):
        """Test handling request timeout."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = mock_adapter.place_multileg_order(legs=legs, quantity=1)

        assert result.success is False
        assert "timeout" in result.message.lower()

    @patch('requests.post')
    def test_place_multileg_order_connection_error(self, mock_post, mock_adapter):
        """Test handling connection error."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = mock_adapter.place_multileg_order(legs=legs, quantity=1)

        assert result.success is False
        assert "Request failed" in result.message

    @patch('requests.post')
    def test_place_multileg_order_ratio_simplification(self, mock_post, mock_adapter):
        """Test that leg ratios are simplified to GCD=1."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_ratio_123", "status": "new"}
        mock_post.return_value = mock_response

        # Legs with non-simplified ratios (GCD=2)
        legs = [
            OptionLeg("AAPL250117C00190000", "buy", ratio_qty=2),
            OptionLeg("AAPL250117C00200000", "sell", ratio_qty=2),
        ]
        result = mock_adapter.place_multileg_order(legs=legs, quantity=1)

        assert result.success is True
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        # Ratios should be simplified
        assert payload["legs"][0]["ratio_qty"] == "1"
        assert payload["legs"][1]["ratio_qty"] == "1"

    @patch('requests.post')
    def test_place_multileg_iron_condor(self, mock_post, mock_adapter):
        """Test placing an iron condor order."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_ic_123", "status": "pending_new"}
        mock_post.return_value = mock_response

        legs = create_iron_condor(
            "SPY", "250117",
            put_buy_strike=480.0,
            put_sell_strike=485.0,
            call_sell_strike=510.0,
            call_buy_strike=515.0,
        )
        result = mock_adapter.place_multileg_order(
            legs=legs,
            quantity=10,
            order_type="limit",
            limit_price=1.80,
            note="Iron Condor",
        )

        assert result.success is True
        assert result.legs_count == 4
        
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["qty"] == "10"
        assert len(payload["legs"]) == 4


class TestCloseMultilegPosition:
    """Tests for close_multileg_position method."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mocked options adapter."""
        with patch.object(AlpacaOptionsAdapter, '__init__', lambda self, paper=None: None):
            adapter = AlpacaOptionsAdapter()
            adapter.api_key = "test_key"
            adapter.secret_key = "test_secret"
            adapter.base_url = "https://paper-api.alpaca.markets"
            adapter._api_version = "v2"
            return adapter

    @patch('requests.post')
    def test_close_vertical_spread_with_option_legs(self, mock_post, mock_adapter):
        """Test closing a vertical spread with OptionLeg objects."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_close_123", "status": "new"}
        mock_post.return_value = mock_response

        # Original opening legs
        original_legs = [
            OptionLeg("AAPL250117C00190000", "buy", 1, PositionIntent.BUY_TO_OPEN),
            OptionLeg("AAPL250117C00200000", "sell", 1, PositionIntent.SELL_TO_OPEN),
        ]
        
        result = mock_adapter.close_multileg_position(
            legs=original_legs,
            quantity=5,
            limit_price=3.00,
            note="Bull Call Spread",
        )

        assert result.success is True
        
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        
        # Verify sides are inverted
        assert payload["legs"][0]["side"] == "sell"  # Was buy
        assert payload["legs"][0]["position_intent"] == "sell_to_close"
        
        assert payload["legs"][1]["side"] == "buy"  # Was sell
        assert payload["legs"][1]["position_intent"] == "buy_to_close"

    @patch('requests.post')
    def test_close_vertical_spread_with_dict_legs(self, mock_post, mock_adapter):
        """Test closing a vertical spread with dict legs."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_close_dict_123", "status": "new"}
        mock_post.return_value = mock_response

        original_legs = [
            {"symbol": "AAPL250117C00190000", "side": "buy", "ratio_qty": 1},
            {"symbol": "AAPL250117C00200000", "side": "sell", "ratio_qty": 1},
        ]
        
        result = mock_adapter.close_multileg_position(legs=original_legs, quantity=1)

        assert result.success is True
        
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        
        # Verify sides are inverted
        assert payload["legs"][0]["side"] == "sell"
        assert payload["legs"][1]["side"] == "buy"


class TestAsyncMultiLeg:
    """Tests for async multi-leg order support."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mocked options adapter."""
        with patch.object(AlpacaOptionsAdapter, '__init__', lambda self, paper=None: None):
            adapter = AlpacaOptionsAdapter()
            adapter.api_key = "test_key"
            adapter.secret_key = "test_secret"
            adapter.base_url = "https://paper-api.alpaca.markets"
            adapter._api_version = "v2"
            return adapter

    @pytest.mark.asyncio
    @patch('requests.post')
    async def test_place_multileg_order_async(self, mock_post, mock_adapter):
        """Test async version of multi-leg order."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_async_123", "status": "new"}
        mock_post.return_value = mock_response

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = await mock_adapter.place_multileg_order_async(
            legs=legs,
            quantity=1,
            limit_price=2.50,
        )

        assert result.success is True
        assert result.order_id == "ord_async_123"


class TestGetMultilegOrderStatus:
    """Tests for getting multi-leg order status."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mocked options adapter."""
        with patch.object(AlpacaOptionsAdapter, '__init__', lambda self, paper=None: None):
            adapter = AlpacaOptionsAdapter()
            adapter.api_key = "test_key"
            adapter.secret_key = "test_secret"
            adapter.base_url = "https://paper-api.alpaca.markets"
            adapter._api_version = "v2"
            adapter.trading_client = Mock()
            return adapter

    def test_get_order_status_success(self, mock_adapter):
        """Test getting order status successfully."""
        mock_order = Mock()
        mock_order.id = "ord_status_123"
        mock_order.status = "filled"
        mock_order.filled_qty = 5
        mock_order.order_class = "mleg"
        mock_order.legs = [Mock(), Mock()]
        
        mock_adapter.trading_client.get_order_by_id.return_value = mock_order

        result = mock_adapter.get_multileg_order_status("ord_status_123")

        assert result is not None
        assert result["id"] == "ord_status_123"
        assert result["status"] == "filled"
        assert result["filled_qty"] == 5
        assert result["order_class"] == "mleg"

    def test_get_order_status_not_found(self, mock_adapter):
        """Test handling order not found."""
        mock_adapter.trading_client.get_order_by_id.side_effect = Exception("Order not found")

        result = mock_adapter.get_multileg_order_status("invalid_order_id")

        assert result is None


class TestMarketOrderType:
    """Tests for market order types."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mocked options adapter."""
        with patch.object(AlpacaOptionsAdapter, '__init__', lambda self, paper=None: None):
            adapter = AlpacaOptionsAdapter()
            adapter.api_key = "test_key"
            adapter.secret_key = "test_secret"
            adapter.base_url = "https://paper-api.alpaca.markets"
            adapter._api_version = "v2"
            return adapter

    @patch('requests.post')
    def test_market_order_no_limit_price(self, mock_post, mock_adapter):
        """Test that market orders don't include limit price."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_mkt_123", "status": "new"}
        mock_post.return_value = mock_response

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = mock_adapter.place_multileg_order(
            legs=legs,
            quantity=1,
            order_type="market",
        )

        assert result.success is True
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["type"] == "market"
        assert "limit_price" not in payload

    @patch('requests.post')
    def test_limit_order_with_price(self, mock_post, mock_adapter):
        """Test that limit orders include limit price."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_lmt_123", "status": "new"}
        mock_post.return_value = mock_response

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = mock_adapter.place_multileg_order(
            legs=legs,
            quantity=1,
            order_type="limit",
            limit_price=2.50,
        )

        assert result.success is True
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["type"] == "limit"
        assert payload["limit_price"] == "2.5"  # Float to string conversion


class TestTimeInForce:
    """Tests for time-in-force options."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mocked options adapter."""
        with patch.object(AlpacaOptionsAdapter, '__init__', lambda self, paper=None: None):
            adapter = AlpacaOptionsAdapter()
            adapter.api_key = "test_key"
            adapter.secret_key = "test_secret"
            adapter.base_url = "https://paper-api.alpaca.markets"
            adapter._api_version = "v2"
            return adapter

    @patch('requests.post')
    def test_gtc_time_in_force(self, mock_post, mock_adapter):
        """Test GTC time-in-force."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_gtc_123", "status": "new"}
        mock_post.return_value = mock_response

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = mock_adapter.place_multileg_order(
            legs=legs,
            quantity=1,
            time_in_force="gtc",
            limit_price=2.50,
        )

        assert result.success is True
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["time_in_force"] == "gtc"


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from execution.broker_adapters.alpaca_options_adapter import (
            AlpacaOptionsAdapter,
            OptionLeg,
            PositionIntent,
            MultiLegOrderResult,
            create_vertical_spread,
            create_iron_condor,
        )
        
        # All imports should succeed
        assert AlpacaOptionsAdapter is not None
        assert OptionLeg is not None
        assert PositionIntent is not None
        assert MultiLegOrderResult is not None
        assert create_vertical_spread is not None
        assert create_iron_condor is not None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mocked options adapter."""
        with patch.object(AlpacaOptionsAdapter, '__init__', lambda self, paper=None: None):
            adapter = AlpacaOptionsAdapter()
            adapter.api_key = "test_key"
            adapter.secret_key = "test_secret"
            adapter.base_url = "https://paper-api.alpaca.markets"
            adapter._api_version = "v2"
            return adapter

    def test_very_large_strike_price(self):
        """Test handling very large strike prices."""
        legs = create_vertical_spread(
            underlying="SPX",
            expiration="250117",
            option_type="C",
            buy_strike=9999.0,
            sell_strike=10000.0,
        )
        # 9999 * 1000 = 9999000
        assert "09999000" in legs[0].symbol
        assert "10000000" in legs[1].symbol

    def test_very_small_strike_price(self):
        """Test handling very small strike prices."""
        legs = create_vertical_spread(
            underlying="SOUN",
            expiration="250117",
            option_type="P",
            buy_strike=2.5,
            sell_strike=5.0,
        )
        # 2.5 * 1000 = 2500
        assert "00002500" in legs[0].symbol
        assert "00005000" in legs[1].symbol

    @patch('requests.post')
    def test_large_quantity(self, mock_post, mock_adapter):
        """Test handling large quantity orders."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_large_123", "status": "new"}
        mock_post.return_value = mock_response

        legs = create_vertical_spread("AAPL", "250117", "C", 190.0, 200.0)
        result = mock_adapter.place_multileg_order(
            legs=legs,
            quantity=1000,
            limit_price=2.50,
        )

        assert result.success is True
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["qty"] == "1000"

    @patch('requests.post')
    def test_mixed_leg_types_in_order(self, mock_post, mock_adapter):
        """Test order with mixed OptionLeg and dict legs."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "ord_mixed_123", "status": "new"}
        mock_post.return_value = mock_response

        legs = [
            OptionLeg("AAPL250117C00190000", "buy", 1, PositionIntent.BUY_TO_OPEN),
            {"symbol": "AAPL250117C00200000", "side": "sell", "ratio_qty": 1},
        ]
        result = mock_adapter.place_multileg_order(legs=legs, quantity=1)

        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
