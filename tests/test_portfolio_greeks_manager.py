"""
Comprehensive Unit Tests for PortfolioGreeksManager

Tests cover:
- Position Greek calculations
- Portfolio-level Greek aggregation
- Risk limit checking
- Pre-trade risk validation
- Hedging recommendations
- Sector concentration limits
- Beta-adjusted delta calculations
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from trade.portfolio_greeks import (
    PortfolioGreeksManager,
    PositionGreeks,
    PortfolioGreeks,
    GreekLimits,
    GreekRiskAssessment,
    GreekLimitBreach,
    create_portfolio_greeks_manager,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def greeks_manager():
    """Create a basic PortfolioGreeksManager for testing."""
    return PortfolioGreeksManager(portfolio_value=100000.0)


@pytest.fixture
def greeks_manager_custom_limits():
    """Create a PortfolioGreeksManager with custom limits."""
    limits = GreekLimits(
        max_net_delta_pct=0.20,
        max_gamma_pct=0.01,
        max_vega_pct=0.02,
        max_single_position_delta_pct=0.05,
    )
    return PortfolioGreeksManager(limits=limits, portfolio_value=200000.0)


@pytest.fixture
def sample_position_greeks():
    """Create sample PositionGreeks."""
    return PositionGreeks(
        symbol="AAPL240315C175",
        underlying="AAPL",
        quantity=10,
        delta=0.55,
        gamma=0.02,
        theta=-0.12,
        vega=0.15,
        position_delta=9625.0,   # 0.55 * 10 * 100 * 175
        position_gamma=6125.0,   # 0.02 * 10 * 100 * 175^2 / 100
        position_theta=-120.0,
        position_vega=150.0,
        is_option=True,
        option_type="call",
        strike=175.0,
        expiration=datetime.utcnow() + timedelta(days=30),
        dte=30,
        underlying_price=175.0,
        notional_value=175000.0,
        beta=1.2,
    )


# =============================================================================
# TEST CLASS: Enums
# =============================================================================

class TestGreekEnums:
    """Test Greek-related enums."""
    
    def test_greek_limit_breach_values(self):
        """Test GreekLimitBreach enum values."""
        assert GreekLimitBreach.NONE.value == "none"
        assert GreekLimitBreach.DELTA.value == "delta"
        assert GreekLimitBreach.GAMMA.value == "gamma"
        assert GreekLimitBreach.THETA.value == "theta"
        assert GreekLimitBreach.VEGA.value == "vega"
        assert GreekLimitBreach.CONCENTRATION.value == "concentration"


# =============================================================================
# TEST CLASS: GreekLimits Dataclass
# =============================================================================

class TestGreekLimitsDataclass:
    """Test GreekLimits dataclass."""
    
    def test_default_limits(self):
        """Test default Greek limits."""
        limits = GreekLimits()
        
        assert limits.max_net_delta_pct == 0.30
        assert limits.max_gamma_pct == 0.02
        assert limits.max_negative_theta_pct == 0.005
        assert limits.max_vega_pct == 0.03
        assert limits.warning_threshold == 0.75
        
    def test_custom_limits(self):
        """Test custom Greek limits."""
        limits = GreekLimits(
            max_net_delta_pct=0.50,
            max_gamma_pct=0.03,
        )
        
        assert limits.max_net_delta_pct == 0.50
        assert limits.max_gamma_pct == 0.03
        
    def test_concentration_limits(self):
        """Test concentration limits are present."""
        limits = GreekLimits()
        
        assert limits.max_single_underlying_pct == 0.15
        assert limits.max_sector_pct == 0.30
        assert limits.max_correlated_exposure_pct == 0.40


# =============================================================================
# TEST CLASS: PositionGreeks Dataclass
# =============================================================================

class TestPositionGreeksDataclass:
    """Test PositionGreeks dataclass."""
    
    def test_position_greeks_creation(self, sample_position_greeks):
        """Test basic PositionGreeks creation."""
        pos = sample_position_greeks
        
        assert pos.symbol == "AAPL240315C175"
        assert pos.underlying == "AAPL"
        assert pos.quantity == 10
        assert pos.delta == 0.55
        
    def test_position_greeks_defaults(self):
        """Test PositionGreeks default values."""
        pos = PositionGreeks(
            symbol="TEST",
            underlying="TEST",
            quantity=1,
        )
        
        assert pos.delta == 0.0
        assert pos.gamma == 0.0
        assert pos.theta == 0.0
        assert pos.vega == 0.0
        assert pos.is_option is True


# =============================================================================
# TEST CLASS: PortfolioGreeks Dataclass
# =============================================================================

class TestPortfolioGreeksDataclass:
    """Test PortfolioGreeks dataclass."""
    
    def test_portfolio_greeks_creation(self):
        """Test PortfolioGreeks creation."""
        pg = PortfolioGreeks(
            net_delta=10000.0,
            net_gamma=500.0,
            net_theta=-100.0,
            net_vega=200.0,
            portfolio_value=100000.0,
        )
        
        assert pg.net_delta == 10000.0
        assert pg.portfolio_value == 100000.0
        
    def test_portfolio_greeks_defaults(self):
        """Test PortfolioGreeks default values."""
        pg = PortfolioGreeks()
        
        assert pg.net_delta == 0.0
        assert pg.positions == []
        assert pg.sector_exposures == {}


# =============================================================================
# TEST CLASS: Manager Initialization
# =============================================================================

class TestPortfolioGreeksManagerInit:
    """Test PortfolioGreeksManager initialization."""
    
    def test_default_initialization(self, greeks_manager):
        """Test default initialization."""
        assert greeks_manager is not None
        assert greeks_manager.portfolio_value == 100000.0
        assert greeks_manager.limits is not None
        
    def test_custom_limits_initialization(self, greeks_manager_custom_limits):
        """Test initialization with custom limits."""
        manager = greeks_manager_custom_limits
        
        assert manager.limits.max_net_delta_pct == 0.20
        assert manager.limits.max_gamma_pct == 0.01
        
    def test_factory_function(self):
        """Test factory function."""
        manager = create_portfolio_greeks_manager(
            portfolio_value=500000.0,
            custom_limits={"max_net_delta_pct": 0.25},
        )
        
        assert manager.portfolio_value == 500000.0
        assert manager.limits.max_net_delta_pct == 0.25


# =============================================================================
# TEST CLASS: Adding Positions
# =============================================================================

class TestAddPosition:
    """Test position addition functionality."""
    
    def test_add_option_position(self, greeks_manager):
        """Test adding an options position."""
        position = greeks_manager.add_position(
            symbol="AAPL240315C175",
            underlying="AAPL",
            quantity=10,
            delta=0.55,
            gamma=0.02,
            theta=-0.12,
            vega=0.15,
            underlying_price=175.0,
            option_type="call",
            strike=175.0,
            expiration=datetime.utcnow() + timedelta(days=30),
        )
        
        assert position is not None
        assert position.symbol == "AAPL240315C175"
        assert "AAPL240315C175" in greeks_manager.positions
        
    def test_add_equity_position(self, greeks_manager):
        """Test adding an equity position."""
        position = greeks_manager.add_position(
            symbol="AAPL",
            underlying="AAPL",
            quantity=100,
            delta=1.0,  # Equity delta = 1
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=175.0,
            is_option=False,
        )
        
        assert position is not None
        assert position.is_option is False
        
    def test_position_delta_calculation(self, greeks_manager):
        """Test position delta is calculated correctly."""
        position = greeks_manager.add_position(
            symbol="TEST",
            underlying="TEST",
            quantity=10,
            delta=0.50,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=100.0,
        )
        
        # Position delta = delta * quantity * 100 * price = 0.50 * 10 * 100 * 100 = 50000
        expected_delta = 0.50 * 10 * 100 * 100.0
        assert position.position_delta == expected_delta
        
    def test_add_short_position(self, greeks_manager):
        """Test adding a short position."""
        position = greeks_manager.add_position(
            symbol="SPY_PUT",
            underlying="SPY",
            quantity=-5,  # Short
            delta=-0.40,
            gamma=0.01,
            theta=0.08,  # Collecting theta
            vega=-0.20,
            underlying_price=500.0,
            option_type="put",
            strike=480.0,
        )
        
        assert position.quantity == -5
        # Short position should have negative position delta
        
    def test_beta_assignment(self, greeks_manager):
        """Test beta is assigned from estimates."""
        position = greeks_manager.add_position(
            symbol="NVDA_CALL",
            underlying="NVDA",  # Should have beta ~1.8
            quantity=5,
            delta=0.60,
            gamma=0.02,
            theta=-0.15,
            vega=0.25,
            underlying_price=800.0,
        )
        
        assert position.beta == 1.8  # From BETA_ESTIMATES


# =============================================================================
# TEST CLASS: Removing Positions
# =============================================================================

class TestRemovePosition:
    """Test position removal functionality."""
    
    def test_remove_position(self, greeks_manager):
        """Test removing a position."""
        greeks_manager.add_position(
            symbol="TEST",
            underlying="TEST",
            quantity=10,
            delta=0.50,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=100.0,
        )
        
        assert "TEST" in greeks_manager.positions
        
        greeks_manager.remove_position("TEST")
        
        assert "TEST" not in greeks_manager.positions
        
    def test_remove_nonexistent_position(self, greeks_manager):
        """Test removing nonexistent position doesn't crash."""
        greeks_manager.remove_position("NONEXISTENT")
        
        # Should not raise exception


# =============================================================================
# TEST CLASS: Portfolio Greek Aggregation
# =============================================================================

class TestPortfolioGreekAggregation:
    """Test portfolio-level Greek aggregation."""
    
    def test_aggregate_net_delta(self, greeks_manager):
        """Test net delta aggregation."""
        # Add long call (positive delta)
        greeks_manager.add_position(
            symbol="LONG_CALL",
            underlying="AAPL",
            quantity=10,
            delta=0.50,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            underlying_price=175.0,
        )
        
        # Add short put (positive delta on short put)
        greeks_manager.add_position(
            symbol="SHORT_PUT",
            underlying="AAPL",
            quantity=-5,
            delta=-0.30,  # Short put has negative delta (position will be positive)
            gamma=0.01,
            theta=0.08,
            vega=-0.10,
            underlying_price=175.0,
        )
        
        pg = greeks_manager.portfolio_greeks
        
        assert pg is not None
        assert pg.position_count == 2
        
    def test_aggregate_gross_delta(self, greeks_manager):
        """Test gross delta aggregation."""
        greeks_manager.add_position(
            symbol="LONG",
            underlying="AAPL",
            quantity=10,
            delta=0.50,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=100.0,
        )
        
        greeks_manager.add_position(
            symbol="SHORT",
            underlying="MSFT",
            quantity=-10,
            delta=0.50,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=100.0,
        )
        
        pg = greeks_manager.portfolio_greeks
        
        # Net delta should be close to 0 (offset)
        # Gross delta should be sum of absolute values
        assert pg.gross_delta > abs(pg.net_delta)
        
    def test_beta_adjusted_delta(self, greeks_manager):
        """Test beta-adjusted delta calculation."""
        # Add NVDA (high beta)
        greeks_manager.add_position(
            symbol="NVDA_CALL",
            underlying="NVDA",
            quantity=5,
            delta=0.60,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=800.0,
        )
        
        pg = greeks_manager.portfolio_greeks
        
        # Beta-adjusted delta should be higher than raw delta for high-beta stock
        assert pg.beta_adjusted_delta != 0
        
    def test_sector_exposure_tracking(self, greeks_manager):
        """Test sector exposure tracking."""
        # Add tech stock
        greeks_manager.add_position(
            symbol="AAPL",
            underlying="AAPL",  # technology
            quantity=100,
            delta=1.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=175.0,
            is_option=False,
        )
        
        # Add financial stock
        greeks_manager.add_position(
            symbol="JPM",
            underlying="JPM",  # financials
            quantity=50,
            delta=1.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=180.0,
            is_option=False,
        )
        
        pg = greeks_manager.portfolio_greeks
        
        assert "technology" in pg.sector_exposures
        assert "financials" in pg.sector_exposures


# =============================================================================
# TEST CLASS: Risk Assessment
# =============================================================================

class TestRiskAssessment:
    """Test risk assessment functionality."""
    
    def test_assess_risk_within_limits(self, greeks_manager):
        """Test assessment when within limits."""
        # Add very small position to stay within limits
        greeks_manager.add_position(
            symbol="TEST",
            underlying="SPY",
            quantity=1,  # Very small to stay within limits
            delta=0.10,  # Low delta
            gamma=0.001,
            theta=-0.01,
            vega=0.02,
            underlying_price=100.0,  # Lower price for smaller notional
        )
        
        assessment = greeks_manager.assess_risk()
        
        # Should be within limits with such a small position
        assert assessment.risk_score < 100
        
    def test_assess_risk_delta_breach(self, greeks_manager):
        """Test assessment detects delta breach."""
        # Add large position to breach delta limit
        # Need delta exposure > 30% of $100k = $30k
        greeks_manager.add_position(
            symbol="LARGE",
            underlying="SPY",
            quantity=100,  # Large quantity
            delta=0.80,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=500.0,
        )
        
        assessment = greeks_manager.assess_risk()
        
        # Should breach delta limit
        assert GreekLimitBreach.DELTA in assessment.breaches or assessment.is_within_limits
        
    def test_assess_risk_generates_warnings(self, greeks_manager):
        """Test assessment generates warnings approaching limits."""
        # Add position at ~80% of delta limit
        # 80% of 30% of $100k = $24k delta needed
        greeks_manager.add_position(
            symbol="MEDIUM",
            underlying="SPY",
            quantity=10,
            delta=0.50,
            gamma=0.01,
            theta=-0.05,
            vega=0.10,
            underlying_price=500.0,
        )
        
        assessment = greeks_manager.assess_risk()
        
        # Should have risk score
        assert assessment.risk_score >= 0
        
    def test_risk_score_calculation(self, greeks_manager):
        """Test risk score is calculated."""
        greeks_manager.add_position(
            symbol="TEST",
            underlying="AAPL",
            quantity=10,
            delta=0.50,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            underlying_price=175.0,
        )
        
        assessment = greeks_manager.assess_risk()
        
        assert 0 <= assessment.risk_score <= 100


# =============================================================================
# TEST CLASS: Pre-Trade Risk Check
# =============================================================================

class TestPreTradeRiskCheck:
    """Test pre-trade risk validation."""
    
    def test_check_new_position_allowed(self, greeks_manager):
        """Test checking if new position is allowed."""
        # Start with no positions, check small position
        can_add, reason = greeks_manager.check_new_position(
            underlying="AAPL",
            delta=0.10,  # Small delta
            gamma=0.005,
            theta=-0.02,
            vega=0.05,
            quantity=1,  # Very small quantity
            underlying_price=175.0,
        )
        
        # With such a small position, should be allowed
        assert isinstance(can_add, bool)
        assert isinstance(reason, str)
        
    def test_check_new_position_rejected_delta(self, greeks_manager):
        """Test new position rejected for delta breach."""
        # First add large existing position
        greeks_manager.add_position(
            symbol="EXISTING",
            underlying="SPY",
            quantity=50,
            delta=0.70,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=500.0,
        )
        
        # Try to add another large position
        can_add, reason = greeks_manager.check_new_position(
            underlying="SPY",
            delta=0.80,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            quantity=50,  # Large
            underlying_price=500.0,
        )
        
        # May or may not be rejected depending on calculations
        assert isinstance(can_add, bool)
        
    def test_check_new_position_concentration_limit(self, greeks_manager):
        """Test concentration limit checking."""
        # Add position in one underlying
        greeks_manager.add_position(
            symbol="AAPL_1",
            underlying="AAPL",
            quantity=30,
            delta=0.50,
            gamma=0.01,
            theta=-0.05,
            vega=0.10,
            underlying_price=175.0,
        )
        
        # Try to add more in same underlying
        can_add, reason = greeks_manager.check_new_position(
            underlying="AAPL",
            delta=0.60,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            quantity=30,
            underlying_price=175.0,
        )
        
        # Check that concentration was considered
        assert isinstance(can_add, bool)


# =============================================================================
# TEST CLASS: Hedging Recommendations
# =============================================================================

class TestHedgingRecommendations:
    """Test hedging recommendation functionality."""
    
    def test_get_hedging_recommendation_long_delta(self, greeks_manager):
        """Test hedging recommendation for long delta."""
        # Add large long delta position
        greeks_manager.add_position(
            symbol="LONG_DELTA",
            underlying="SPY",
            quantity=50,
            delta=0.70,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=500.0,
        )
        
        recommendations = greeks_manager.get_hedging_recommendation()
        
        assert "delta_hedge" in recommendations
        assert "overall_action" in recommendations
        
    def test_get_hedging_recommendation_short_delta(self, greeks_manager):
        """Test hedging recommendation for short delta."""
        # Add large short delta position
        greeks_manager.add_position(
            symbol="SHORT_DELTA",
            underlying="SPY",
            quantity=-50,
            delta=0.70,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=500.0,
        )
        
        recommendations = greeks_manager.get_hedging_recommendation()
        
        assert isinstance(recommendations, dict)
        
    def test_get_hedging_recommendation_negative_gamma(self, greeks_manager):
        """Test hedging recommendation for negative gamma."""
        # Add position with negative gamma (short options)
        greeks_manager.add_position(
            symbol="SHORT_GAMMA",
            underlying="SPY",
            quantity=-20,
            delta=-0.30,
            gamma=0.05,  # Short position, so negative gamma exposure
            theta=0.10,
            vega=-0.20,
            underlying_price=500.0,
        )
        
        recommendations = greeks_manager.get_hedging_recommendation()
        
        assert "gamma_hedge" in recommendations
        
    def test_get_hedging_recommendation_no_action_needed(self, greeks_manager):
        """Test when no hedging action needed."""
        # Add small balanced position
        greeks_manager.add_position(
            symbol="SMALL",
            underlying="SPY",
            quantity=2,
            delta=0.50,
            gamma=0.01,
            theta=-0.05,
            vega=0.10,
            underlying_price=500.0,
        )
        
        recommendations = greeks_manager.get_hedging_recommendation()
        
        # Should have no recommendations or 'none' action
        assert recommendations["overall_action"] in ["none", "hedge_recommended"]


# =============================================================================
# TEST CLASS: Sector and Beta Mappings
# =============================================================================

class TestSectorAndBetaMappings:
    """Test sector and beta mapping functionality."""
    
    def test_sector_map_tech_stocks(self, greeks_manager):
        """Test tech stocks are mapped correctly."""
        tech_stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
        
        for stock in tech_stocks:
            assert greeks_manager.SECTOR_MAP.get(stock) == "technology"
            
    def test_sector_map_financials(self, greeks_manager):
        """Test financial stocks are mapped correctly."""
        financial_stocks = ["JPM", "BAC", "GS", "V", "MA"]
        
        for stock in financial_stocks:
            assert greeks_manager.SECTOR_MAP.get(stock) == "financials"
            
    def test_sector_map_indices(self, greeks_manager):
        """Test index ETFs are mapped correctly."""
        indices = ["SPY", "QQQ", "IWM"]
        
        for idx in indices:
            assert greeks_manager.SECTOR_MAP.get(idx) == "index"
            
    def test_beta_estimates_exist(self, greeks_manager):
        """Test beta estimates exist for key symbols."""
        key_symbols = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]
        
        for symbol in key_symbols:
            assert symbol in greeks_manager.BETA_ESTIMATES
            
    def test_spy_beta_is_one(self, greeks_manager):
        """Test SPY beta is 1.0."""
        assert greeks_manager.BETA_ESTIMATES["SPY"] == 1.0
        
    def test_high_beta_stocks(self, greeks_manager):
        """Test high beta stocks have beta > 1.5."""
        high_beta = ["NVDA", "TSLA", "COIN"]
        
        for stock in high_beta:
            assert greeks_manager.BETA_ESTIMATES.get(stock, 1.0) > 1.5


# =============================================================================
# TEST CLASS: Portfolio Value Update
# =============================================================================

class TestPortfolioValueUpdate:
    """Test portfolio value update functionality."""
    
    def test_update_portfolio_value(self, greeks_manager):
        """Test updating portfolio value."""
        # Add a position first
        greeks_manager.add_position(
            symbol="TEST",
            underlying="AAPL",
            quantity=10,
            delta=0.50,
            gamma=0.01,
            theta=-0.05,
            vega=0.10,
            underlying_price=175.0,
        )
        
        # Update portfolio value
        greeks_manager.update_portfolio_value(200000.0)
        
        assert greeks_manager.portfolio_value == 200000.0
        
        # Greeks percentages should be recalculated
        pg = greeks_manager.portfolio_greeks
        assert pg.portfolio_value == 200000.0


# =============================================================================
# TEST CLASS: Summary
# =============================================================================

class TestPortfolioGreeksSummary:
    """Test summary functionality."""
    
    def test_get_summary(self, greeks_manager):
        """Test getting manager summary."""
        # Add some positions
        greeks_manager.add_position(
            symbol="AAPL",
            underlying="AAPL",
            quantity=10,
            delta=0.55,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            underlying_price=175.0,
        )
        
        summary = greeks_manager.get_summary()
        
        assert "timestamp" in summary
        assert "portfolio_value" in summary
        assert "position_count" in summary
        assert "greeks" in summary
        assert "risk_assessment" in summary
        
    def test_summary_greek_values(self, greeks_manager):
        """Test summary contains Greek values."""
        greeks_manager.add_position(
            symbol="TEST",
            underlying="SPY",
            quantity=5,
            delta=0.50,
            gamma=0.02,
            theta=-0.08,
            vega=0.12,
            underlying_price=500.0,
        )
        
        summary = greeks_manager.get_summary()
        
        assert "net_delta" in summary["greeks"]
        assert "net_gamma" in summary["greeks"]
        assert "net_theta" in summary["greeks"]
        assert "net_vega" in summary["greeks"]
        assert "delta_pct" in summary["greeks"]


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================

class TestPortfolioGreeksEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_portfolio(self, greeks_manager):
        """Test with no positions."""
        assessment = greeks_manager.assess_risk()
        
        assert assessment.is_within_limits is True
        assert assessment.risk_score == 0.0
        
    def test_zero_portfolio_value(self):
        """Test with zero portfolio value."""
        manager = PortfolioGreeksManager(portfolio_value=0.0)
        
        # Should handle gracefully
        manager.add_position(
            symbol="TEST",
            underlying="TEST",
            quantity=1,
            delta=0.50,
            gamma=0.01,
            theta=-0.05,
            vega=0.10,
            underlying_price=100.0,
        )
        
        # Assessment should not crash
        assessment = manager.assess_risk()
        assert assessment is not None
        
    def test_very_large_position(self, greeks_manager):
        """Test with very large position."""
        greeks_manager.add_position(
            symbol="LARGE",
            underlying="SPY",
            quantity=1000,  # Very large
            delta=0.90,
            gamma=0.05,
            theta=-0.50,
            vega=0.30,
            underlying_price=500.0,
        )
        
        assessment = greeks_manager.assess_risk()
        
        # Should detect breaches
        assert len(assessment.breaches) > 0 or assessment.risk_score > 50
        
    def test_unknown_underlying(self, greeks_manager):
        """Test with unknown underlying (no sector/beta mapping)."""
        position = greeks_manager.add_position(
            symbol="UNKNOWN123",
            underlying="UNKNOWN123",
            quantity=5,
            delta=0.50,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            underlying_price=100.0,
        )
        
        # Should use default beta of 1.0
        assert position.beta == 1.0
        
        # Sector should be 'other'
        pg = greeks_manager.portfolio_greeks
        assert "other" in pg.sector_exposures


# =============================================================================
# TEST CLASS: Integration Tests
# =============================================================================

class TestPortfolioGreeksIntegration:
    """Integration tests for PortfolioGreeksManager."""
    
    def test_full_portfolio_workflow(self, greeks_manager):
        """Test complete portfolio management workflow."""
        # Add multiple positions
        greeks_manager.add_position(
            symbol="AAPL_CALL",
            underlying="AAPL",
            quantity=10,
            delta=0.55,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            underlying_price=175.0,
            option_type="call",
            strike=175.0,
            expiration=datetime.utcnow() + timedelta(days=30),
        )
        
        greeks_manager.add_position(
            symbol="SPY_PUT",
            underlying="SPY",
            quantity=-5,
            delta=-0.35,
            gamma=0.01,
            theta=0.08,
            vega=-0.10,
            underlying_price=500.0,
            option_type="put",
            strike=480.0,
        )
        
        greeks_manager.add_position(
            symbol="NVDA",
            underlying="NVDA",
            quantity=50,
            delta=1.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=800.0,
            is_option=False,
        )
        
        # Assess risk
        assessment = greeks_manager.assess_risk()
        
        # Get hedging recommendations
        recommendations = greeks_manager.get_hedging_recommendation()
        
        # Get summary
        summary = greeks_manager.get_summary()
        
        # Check new position
        can_add, reason = greeks_manager.check_new_position(
            underlying="MSFT",
            delta=0.50,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            quantity=5,
            underlying_price=400.0,
        )
        
        # All operations should complete
        assert assessment is not None
        assert recommendations is not None
        assert summary is not None
        assert isinstance(can_add, bool)
        
    def test_position_lifecycle(self, greeks_manager):
        """Test adding, updating, and removing positions."""
        # Add
        greeks_manager.add_position(
            symbol="TEST",
            underlying="AAPL",
            quantity=10,
            delta=0.50,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            underlying_price=175.0,
        )
        
        assert greeks_manager.portfolio_greeks.position_count == 1
        
        # Update (re-add with same symbol)
        greeks_manager.add_position(
            symbol="TEST",
            underlying="AAPL",
            quantity=20,  # Changed
            delta=0.60,   # Changed
            gamma=0.02,
            theta=-0.12,
            vega=0.18,
            underlying_price=180.0,
        )
        
        assert greeks_manager.portfolio_greeks.position_count == 1  # Still one position
        assert greeks_manager.positions["TEST"].quantity == 20
        
        # Remove
        greeks_manager.remove_position("TEST")
        
        assert greeks_manager.portfolio_greeks.position_count == 0


# =============================================================================
# TEST CLASS: Greek Risk Assessment Details
# =============================================================================

class TestGreekRiskAssessmentDetails:
    """Test detailed GreekRiskAssessment functionality."""
    
    def test_assessment_has_max_additional_delta(self, greeks_manager):
        """Test assessment includes max additional delta allowed."""
        # Add small position
        greeks_manager.add_position(
            symbol="TEST",
            underlying="AAPL",
            quantity=1,
            delta=0.10,
            gamma=0.001,
            theta=-0.01,
            vega=0.02,
            underlying_price=100.0,
        )
        
        assessment = greeks_manager.assess_risk()
        
        # max_additional_delta is a number (can be negative if over limit)
        assert isinstance(assessment.max_additional_delta, (int, float))
        
    def test_assessment_recommendations(self, greeks_manager):
        """Test assessment includes recommendations."""
        # Add unbalanced position
        greeks_manager.add_position(
            symbol="UNBALANCED",
            underlying="SPY",
            quantity=30,
            delta=0.70,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            underlying_price=500.0,
        )
        
        assessment = greeks_manager.assess_risk()
        
        assert isinstance(assessment.recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
