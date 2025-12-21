"""
Comprehensive Tests for Options Greeks API Router

Tests cover:
- Black-Scholes Greeks calculation
- Quick Greeks endpoint (GET)
- Strategy analysis with multi-leg options
- Position aggregation
- Implied volatility calculation
- Option chain Greeks
- Edge cases and error handling

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

import pytest
import math
import sys
from unittest.mock import patch, MagicMock

# Mock db module before importing routers to avoid psycopg2 dependency
sys.modules['db'] = MagicMock()

from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import directly from the module file to avoid __init__.py cascade
import importlib.util
spec = importlib.util.spec_from_file_location("options_greeks", "routers/options_greeks.py")
options_greeks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(options_greeks)

router = options_greeks.router
GreeksRequest = options_greeks.GreeksRequest
GreeksResponse = options_greeks.GreeksResponse
OptionType = options_greeks.OptionType
OptionLeg = options_greeks.OptionLeg
StrategyAnalysisRequest = options_greeks.StrategyAnalysisRequest
StrategyAnalysisResponse = options_greeks.StrategyAnalysisResponse
PositionGreeksRequest = options_greeks.PositionGreeksRequest
ImpliedVolRequest = options_greeks.ImpliedVolRequest
_calculate_bs_price = options_greeks._calculate_bs_price
_calculate_implied_vol = options_greeks._calculate_implied_vol


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_app():
    """Create FastAPI test app with router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_greeks_request():
    """Sample Greeks calculation request."""
    return {
        "option_type": "call",
        "spot_price": 100.0,
        "strike": 100.0,
        "time_to_expiration": 0.25,  # 3 months
        "volatility": 0.25,  # 25%
        "risk_free_rate": 0.05,
        "dividend_yield": 0.0,
    }


@pytest.fixture
def sample_strategy_request():
    """Sample strategy analysis request."""
    return {
        "legs": [
            {"option_type": "call", "strike": 100.0, "premium": 5.0, "quantity": 1},
            {"option_type": "call", "strike": 110.0, "premium": -2.0, "quantity": -1},
        ],
        "spot_price": 100.0,
        "volatility": 0.25,
        "days_to_expiration": 30,
        "risk_free_rate": 0.05,
    }


# =============================================================================
# TEST CLASS: Pydantic Models
# =============================================================================

class TestPydanticModels:
    """Test Pydantic request/response models."""
    
    def test_greeks_request_validation(self):
        """Test GreeksRequest validation."""
        request = GreeksRequest(
            option_type=OptionType.CALL,
            spot_price=100.0,
            strike=100.0,
            time_to_expiration=0.25,
            volatility=0.25,
        )
        assert request.option_type == OptionType.CALL
        assert request.risk_free_rate == 0.05  # Default
        
    def test_greeks_request_invalid_spot(self):
        """Test GreeksRequest rejects invalid spot price."""
        with pytest.raises(ValueError):
            GreeksRequest(
                option_type=OptionType.CALL,
                spot_price=-100.0,  # Invalid
                strike=100.0,
                time_to_expiration=0.25,
                volatility=0.25,
            )
            
    def test_greeks_request_invalid_volatility(self):
        """Test GreeksRequest rejects invalid volatility."""
        with pytest.raises(ValueError):
            GreeksRequest(
                option_type=OptionType.CALL,
                spot_price=100.0,
                strike=100.0,
                time_to_expiration=0.25,
                volatility=10.0,  # Too high (max 5.0)
            )
            
    def test_option_leg_creation(self):
        """Test OptionLeg creation."""
        leg = OptionLeg(
            option_type=OptionType.PUT,
            strike=95.0,
            premium=3.0,
            quantity=-1,
        )
        assert leg.option_type == OptionType.PUT
        assert leg.quantity == -1
        
    def test_strategy_request_max_legs(self):
        """Test StrategyAnalysisRequest limits legs."""
        # 8 legs is the max
        legs = [
            {"option_type": "call", "strike": 100.0, "premium": 1.0, "quantity": 1}
            for _ in range(8)
        ]
        request = StrategyAnalysisRequest(
            legs=legs,
            spot_price=100.0,
            days_to_expiration=30,
        )
        assert len(request.legs) == 8


# =============================================================================
# TEST CLASS: Calculate Greeks Endpoint
# =============================================================================

class TestCalculateGreeksEndpoint:
    """Test POST /calculate endpoint."""
    
    def test_calculate_atm_call(self, client, sample_greeks_request):
        """Test calculating ATM call Greeks."""
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # ATM call delta should be around 0.5
        assert 0.45 < data["delta"] < 0.60
        # Gamma should be positive
        assert data["gamma"] > 0
        # Theta should be negative (time decay)
        assert data["theta"] < 0
        # Vega should be positive
        assert data["vega"] > 0
        
    def test_calculate_put_greeks(self, client, sample_greeks_request):
        """Test calculating put Greeks."""
        sample_greeks_request["option_type"] = "put"
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Put delta should be negative
        assert -0.60 < data["delta"] < -0.40
        
    def test_calculate_itm_call(self, client, sample_greeks_request):
        """Test calculating ITM call Greeks."""
        sample_greeks_request["strike"] = 90.0  # ITM
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # ITM call delta should be > 0.5
        assert data["delta"] > 0.5
        
    def test_calculate_otm_call(self, client, sample_greeks_request):
        """Test calculating OTM call Greeks."""
        sample_greeks_request["strike"] = 120.0  # OTM
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # OTM call delta should be < 0.5
        assert data["delta"] < 0.5
        
    def test_calculate_includes_option_price(self, client, sample_greeks_request):
        """Test response includes option price."""
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "option_price" in data
        assert data["option_price"] > 0
        
    def test_calculate_includes_intrinsic_and_time_value(self, client, sample_greeks_request):
        """Test response includes value breakdown."""
        sample_greeks_request["strike"] = 90.0  # ITM
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        data = response.json()
        
        assert "intrinsic_value" in data
        assert "time_value" in data
        # ITM call has intrinsic value
        assert data["intrinsic_value"] > 0
        
    def test_calculate_with_dividend(self, client, sample_greeks_request):
        """Test calculation with dividend yield."""
        sample_greeks_request["dividend_yield"] = 0.02  # 2%
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Delta should be slightly lower with dividends
        assert data["delta"] > 0


# =============================================================================
# TEST CLASS: Quick Greeks Endpoint
# =============================================================================

class TestQuickGreeksEndpoint:
    """Test GET /quick endpoint."""
    
    def test_quick_greeks_call(self, client):
        """Test quick Greeks for call option."""
        response = client.get(
            "/options/greeks/quick",
            params={
                "option_type": "call",
                "spot": 100.0,
                "strike": 100.0,
                "dte": 30,
                "iv": 25.0,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "delta" in data
        assert "gamma" in data
        
    def test_quick_greeks_put(self, client):
        """Test quick Greeks for put option."""
        response = client.get(
            "/options/greeks/quick",
            params={
                "option_type": "put",
                "spot": 100.0,
                "strike": 100.0,
                "dte": 30,
                "iv": 25.0,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Put delta is negative
        assert data["delta"] < 0
        
    def test_quick_greeks_custom_rate(self, client):
        """Test quick Greeks with custom risk-free rate."""
        response = client.get(
            "/options/greeks/quick",
            params={
                "option_type": "call",
                "spot": 100.0,
                "strike": 100.0,
                "dte": 30,
                "iv": 25.0,
                "rate": 3.0,  # 3%
            }
        )
        
        assert response.status_code == 200
        
    def test_quick_greeks_zero_dte(self, client):
        """Test quick Greeks at expiration."""
        response = client.get(
            "/options/greeks/quick",
            params={
                "option_type": "call",
                "spot": 105.0,
                "strike": 100.0,  # ITM
                "dte": 0,
                "iv": 25.0,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # At expiration, ITM call delta should be ~1
        assert data["delta"] >= 0.9


# =============================================================================
# TEST CLASS: Strategy Analysis Endpoint
# =============================================================================

class TestStrategyAnalysisEndpoint:
    """Test POST /strategy/analyze endpoint."""
    
    def test_analyze_bull_call_spread(self, client, sample_strategy_request):
        """Test analyzing bull call spread."""
        response = client.post("/options/greeks/strategy/analyze", json=sample_strategy_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Bull call spread has limited risk
        assert "max_profit" in data
        assert "max_loss" in data
        assert data["max_loss"] < 0  # Loss is negative P&L
        
    def test_analyze_returns_breakevens(self, client, sample_strategy_request):
        """Test analysis includes breakeven points."""
        response = client.post("/options/greeks/strategy/analyze", json=sample_strategy_request)
        
        data = response.json()
        
        assert "breakevens" in data
        assert isinstance(data["breakevens"], list)
        
    def test_analyze_returns_probability_of_profit(self, client, sample_strategy_request):
        """Test analysis includes PoP."""
        response = client.post("/options/greeks/strategy/analyze", json=sample_strategy_request)
        
        data = response.json()
        
        assert "probability_of_profit" in data
        assert 0 <= data["probability_of_profit"] <= 1
        
    def test_analyze_iron_condor(self, client):
        """Test analyzing iron condor."""
        request = {
            "legs": [
                {"option_type": "put", "strike": 90.0, "premium": -1.0, "quantity": 1},
                {"option_type": "put", "strike": 95.0, "premium": 2.0, "quantity": -1},
                {"option_type": "call", "strike": 105.0, "premium": 2.0, "quantity": -1},
                {"option_type": "call", "strike": 110.0, "premium": -1.0, "quantity": 1},
            ],
            "spot_price": 100.0,
            "volatility": 0.25,
            "days_to_expiration": 30,
        }
        
        response = client.post("/options/greeks/strategy/analyze", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Iron condor collects premium, max profit is positive
        assert data["max_profit"] > 0
        
    def test_analyze_returns_net_greeks(self, client, sample_strategy_request):
        """Test analysis includes aggregate Greeks."""
        response = client.post("/options/greeks/strategy/analyze", json=sample_strategy_request)
        
        data = response.json()
        
        assert "net_delta" in data
        assert "net_gamma" in data
        assert "net_theta" in data
        assert "net_vega" in data
        
    def test_analyze_straddle(self, client):
        """Test analyzing straddle."""
        request = {
            "legs": [
                {"option_type": "call", "strike": 100.0, "premium": 5.0, "quantity": 1},
                {"option_type": "put", "strike": 100.0, "premium": 5.0, "quantity": 1},
            ],
            "spot_price": 100.0,
            "volatility": 0.30,
            "days_to_expiration": 45,
        }
        
        response = client.post("/options/greeks/strategy/analyze", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Straddle should have near-zero net delta
        assert abs(data["net_delta"]) < 20  # Small delta


# =============================================================================
# TEST CLASS: Position Aggregation Endpoint
# =============================================================================

class TestPositionAggregationEndpoint:
    """Test POST /position/aggregate endpoint."""
    
    def test_aggregate_single_position(self, client):
        """Test aggregating single position."""
        request = {
            "positions": [
                {
                    "symbol": "AAPL240315C175",
                    "quantity": 10,
                    "greeks": {"delta": 0.55, "gamma": 0.02, "theta": -0.12, "vega": 0.15}
                }
            ]
        }
        
        response = client.post("/options/greeks/position/aggregate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Net delta = 0.55 * 10 * 100 = 550
        assert data["net_delta"] == 550.0
        
    def test_aggregate_multiple_positions(self, client):
        """Test aggregating multiple positions."""
        request = {
            "positions": [
                {
                    "symbol": "AAPL_CALL",
                    "quantity": 10,
                    "greeks": {"delta": 0.50, "gamma": 0.02, "theta": -0.10, "vega": 0.15}
                },
                {
                    "symbol": "AAPL_PUT",
                    "quantity": -5,
                    "greeks": {"delta": -0.40, "gamma": 0.02, "theta": -0.08, "vega": 0.12}
                },
            ]
        }
        
        response = client.post("/options/greeks/position/aggregate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "net_delta" in data
        assert "net_gamma" in data
        
    def test_aggregate_empty_positions(self, client):
        """Test aggregating empty position list."""
        request = {"positions": []}
        
        response = client.post("/options/greeks/position/aggregate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["net_delta"] == 0.0


# =============================================================================
# TEST CLASS: Implied Volatility Endpoint
# =============================================================================

class TestImpliedVolatilityEndpoint:
    """Test POST /implied-volatility endpoint."""
    
    def test_calculate_iv_call(self, client):
        """Test calculating IV for call option."""
        request = {
            "option_type": "call",
            "option_price": 5.50,
            "spot_price": 100.0,
            "strike": 100.0,
            "time_to_expiration": 0.25,
            "risk_free_rate": 0.05,
        }
        
        response = client.post("/options/greeks/implied-volatility", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "implied_volatility" in data
        assert data["implied_volatility"] > 0
        
    def test_calculate_iv_put(self, client):
        """Test calculating IV for put option."""
        request = {
            "option_type": "put",
            "option_price": 4.50,
            "spot_price": 100.0,
            "strike": 100.0,
            "time_to_expiration": 0.25,
            "risk_free_rate": 0.05,
        }
        
        response = client.post("/options/greeks/implied-volatility", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "implied_volatility" in data
        
    def test_iv_returns_greeks(self, client):
        """Test IV calculation returns Greeks at that IV."""
        request = {
            "option_type": "call",
            "option_price": 5.50,
            "spot_price": 100.0,
            "strike": 100.0,
            "time_to_expiration": 0.25,
        }
        
        response = client.post("/options/greeks/implied-volatility", json=request)
        
        data = response.json()
        
        assert "greeks" in data
        assert "delta" in data["greeks"]
        
    def test_iv_percentage_format(self, client):
        """Test IV is returned in both decimal and percentage."""
        request = {
            "option_type": "call",
            "option_price": 5.50,
            "spot_price": 100.0,
            "strike": 100.0,
            "time_to_expiration": 0.25,
        }
        
        response = client.post("/options/greeks/implied-volatility", json=request)
        
        data = response.json()
        
        assert "implied_volatility" in data
        assert "implied_volatility_pct" in data
        assert data["implied_volatility_pct"] == round(data["implied_volatility"] * 100, 2)


# =============================================================================
# TEST CLASS: Option Chain Greeks Endpoint
# =============================================================================

class TestChainGreeksEndpoint:
    """Test GET /chain/greeks endpoint."""
    
    def test_chain_greeks_single_strike(self, client):
        """Test chain Greeks for single strike."""
        response = client.get(
            "/options/greeks/chain/greeks",
            params={
                "spot": 100.0,
                "strikes": "100",
                "dte": 30,
                "iv": 25.0,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "calls" in data
        assert "puts" in data
        assert len(data["calls"]) == 1
        assert len(data["puts"]) == 1
        
    def test_chain_greeks_multiple_strikes(self, client):
        """Test chain Greeks for multiple strikes."""
        response = client.get(
            "/options/greeks/chain/greeks",
            params={
                "spot": 100.0,
                "strikes": "95,100,105,110",
                "dte": 30,
                "iv": 25.0,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["calls"]) == 4
        assert len(data["puts"]) == 4
        
    def test_chain_greeks_includes_price(self, client):
        """Test chain Greeks includes option prices."""
        response = client.get(
            "/options/greeks/chain/greeks",
            params={
                "spot": 100.0,
                "strikes": "100",
                "dte": 30,
                "iv": 25.0,
            }
        )
        
        data = response.json()
        
        assert "price" in data["calls"][0]
        assert "price" in data["puts"][0]
        
    def test_chain_greeks_call_put_parity(self, client):
        """Test put-call parity holds approximately."""
        response = client.get(
            "/options/greeks/chain/greeks",
            params={
                "spot": 100.0,
                "strikes": "100",
                "dte": 30,
                "iv": 25.0,
                "rate": 5.0,
            }
        )
        
        data = response.json()
        
        call_price = data["calls"][0]["price"]
        put_price = data["puts"][0]["price"]
        
        # ATM call and put prices should be similar (put-call parity)
        # With some allowance for time value differences
        assert abs(call_price - put_price) < 2.0


# =============================================================================
# TEST CLASS: Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Test internal helper functions."""
    
    def test_calculate_bs_price_call(self):
        """Test Black-Scholes price for call."""
        price = _calculate_bs_price(
            option_type="call",
            spot_price=100.0,
            strike=100.0,
            time_to_expiration=0.25,
            risk_free_rate=0.05,
            volatility=0.25,
        )
        
        assert price > 0
        # ATM 3-month call with 25% vol should be around $5-6
        assert 4.0 < price < 8.0
        
    def test_calculate_bs_price_put(self):
        """Test Black-Scholes price for put."""
        price = _calculate_bs_price(
            option_type="put",
            spot_price=100.0,
            strike=100.0,
            time_to_expiration=0.25,
            risk_free_rate=0.05,
            volatility=0.25,
        )
        
        assert price > 0
        
    def test_calculate_bs_price_at_expiration(self):
        """Test price at expiration is intrinsic."""
        call_price = _calculate_bs_price(
            option_type="call",
            spot_price=105.0,
            strike=100.0,
            time_to_expiration=0,
            risk_free_rate=0.05,
            volatility=0.25,
        )
        
        assert call_price == 5.0  # Intrinsic value
        
    def test_calculate_bs_price_otm_at_expiration(self):
        """Test OTM option at expiration is worthless."""
        call_price = _calculate_bs_price(
            option_type="call",
            spot_price=95.0,
            strike=100.0,
            time_to_expiration=0,
            risk_free_rate=0.05,
            volatility=0.25,
        )
        
        assert call_price == 0.0
        
    def test_calculate_implied_vol(self):
        """Test implied volatility calculation."""
        # First calculate a price at known vol
        known_vol = 0.25
        price = _calculate_bs_price(
            option_type="call",
            spot_price=100.0,
            strike=100.0,
            time_to_expiration=0.25,
            risk_free_rate=0.05,
            volatility=known_vol,
        )
        
        # Then back out the vol
        calculated_vol = _calculate_implied_vol(
            option_type="call",
            option_price=price,
            spot_price=100.0,
            strike=100.0,
            time_to_expiration=0.25,
            risk_free_rate=0.05,
        )
        
        assert calculated_vol is not None
        assert abs(calculated_vol - known_vol) < 0.01
        
    def test_calculate_implied_vol_at_expiration(self):
        """Test IV calculation at expiration returns None."""
        iv = _calculate_implied_vol(
            option_type="call",
            option_price=5.0,
            spot_price=100.0,
            strike=100.0,
            time_to_expiration=0,
            risk_free_rate=0.05,
        )
        
        assert iv is None


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_low_volatility(self, client, sample_greeks_request):
        """Test with very low volatility."""
        sample_greeks_request["volatility"] = 0.01  # 1%
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        
    def test_high_volatility(self, client, sample_greeks_request):
        """Test with high volatility."""
        sample_greeks_request["volatility"] = 1.5  # 150%
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        
    def test_deep_itm_call(self, client, sample_greeks_request):
        """Test deep ITM call."""
        sample_greeks_request["strike"] = 50.0  # Very ITM
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Deep ITM delta should be close to 1
        assert data["delta"] > 0.95
        
    def test_deep_otm_put(self, client, sample_greeks_request):
        """Test deep OTM put."""
        sample_greeks_request["option_type"] = "put"
        sample_greeks_request["strike"] = 50.0  # Very OTM put
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Deep OTM put delta should be close to 0
        assert data["delta"] > -0.05
        
    def test_long_dated_option(self, client, sample_greeks_request):
        """Test long-dated option (LEAPS)."""
        sample_greeks_request["time_to_expiration"] = 2.0  # 2 years
        response = client.post("/options/greeks/calculate", json=sample_greeks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Long-dated options have higher vega
        assert data["vega"] > 0
        
    def test_strategy_with_greeks_provided(self, client):
        """Test strategy analysis when Greeks are provided."""
        request = {
            "legs": [
                {
                    "option_type": "call",
                    "strike": 100.0,
                    "premium": 5.0,
                    "quantity": 1,
                    "delta": 0.55,
                    "gamma": 0.02,
                    "theta": -0.10,
                    "vega": 0.15,
                },
            ],
            "spot_price": 100.0,
            "volatility": 0.25,
            "days_to_expiration": 30,
        }
        
        response = client.post("/options/greeks/strategy/analyze", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Net delta should match provided delta * quantity * 100
        assert data["net_delta"] == 55.0


# =============================================================================
# TEST CLASS: Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the Greeks API."""
    
    def test_full_options_workflow(self, client):
        """Test complete options analysis workflow."""
        # 1. Calculate Greeks for individual options
        call_greeks = client.post("/options/greeks/calculate", json={
            "option_type": "call",
            "spot_price": 100.0,
            "strike": 105.0,
            "time_to_expiration": 0.0833,  # 1 month
            "volatility": 0.25,
        }).json()
        
        put_greeks = client.post("/options/greeks/calculate", json={
            "option_type": "put",
            "spot_price": 100.0,
            "strike": 95.0,
            "time_to_expiration": 0.0833,
            "volatility": 0.25,
        }).json()
        
        # 2. Analyze a strategy
        strategy = client.post("/options/greeks/strategy/analyze", json={
            "legs": [
                {"option_type": "call", "strike": 105.0, "premium": call_greeks["option_price"], "quantity": -1},
                {"option_type": "put", "strike": 95.0, "premium": put_greeks["option_price"], "quantity": -1},
            ],
            "spot_price": 100.0,
            "volatility": 0.25,
            "days_to_expiration": 30,
        }).json()
        
        # 3. Aggregate positions
        positions = client.post("/options/greeks/position/aggregate", json={
            "positions": [
                {"symbol": "CALL", "quantity": -1, "greeks": call_greeks},
                {"symbol": "PUT", "quantity": -1, "greeks": put_greeks},
            ]
        }).json()
        
        # Verify complete workflow
        assert call_greeks["delta"] > 0
        assert put_greeks["delta"] < 0
        assert "max_profit" in strategy
        assert "net_delta" in positions
        
    def test_chain_to_strategy_workflow(self, client):
        """Test workflow from chain to strategy analysis."""
        # 1. Get chain Greeks
        chain = client.get(
            "/options/greeks/chain/greeks",
            params={
                "spot": 100.0,
                "strikes": "95,100,105",
                "dte": 30,
                "iv": 25.0,
            }
        ).json()
        
        # 2. Build a spread from chain data
        short_call = next(c for c in chain["calls"] if c["strike"] == 100.0)
        long_call = next(c for c in chain["calls"] if c["strike"] == 105.0)
        
        # 3. Analyze the spread
        spread = client.post("/options/greeks/strategy/analyze", json={
            "legs": [
                {"option_type": "call", "strike": 100.0, "premium": -short_call["price"], "quantity": -1},
                {"option_type": "call", "strike": 105.0, "premium": long_call["price"], "quantity": 1},
            ],
            "spot_price": 100.0,
            "volatility": 0.25,
            "days_to_expiration": 30,
        }).json()
        
        assert "max_profit" in spread
        assert "breakevens" in spread


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
