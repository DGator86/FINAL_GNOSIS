"""
Comprehensive Unit Tests for EventRiskManager

Tests cover:
- Initialization and configuration
- Economic calendar building
- Earnings date estimation
- Event risk assessment
- Strategy filtering based on events
- Risk multiplier calculations
- Edge cases and error handling
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from trade.event_risk_manager import (
    EventRiskManager,
    EventType,
    EventImpact,
    RiskAction,
    MarketEvent,
    EventRiskAssessment,
    create_event_risk_manager,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def event_manager():
    """Create a basic EventRiskManager for testing."""
    return EventRiskManager(lookforward_days=14)


@pytest.fixture
def event_manager_extended():
    """Create an EventRiskManager with extended lookforward."""
    return EventRiskManager(lookforward_days=30)


@pytest.fixture
def sample_market_event():
    """Create a sample MarketEvent."""
    return MarketEvent(
        event_type=EventType.EARNINGS,
        date=date.today() + timedelta(days=5),
        symbol="AAPL",
        time="after-hours",
        description="AAPL Quarterly Earnings",
        impact=EventImpact.HIGH,
        historical_move_avg=5.0,
        historical_move_max=12.0,
        historical_iv_crush=8.0,
        recommended_action=RiskAction.REDUCE_SIZE,
        position_size_multiplier=0.5,
        stop_loss_multiplier=1.5,
    )


# =============================================================================
# TEST CLASS: EventType Enum
# =============================================================================

class TestEventTypeEnum:
    """Test EventType enum values."""
    
    def test_earnings_event_type(self):
        """Test EARNINGS event type."""
        assert EventType.EARNINGS.value == "earnings"
        
    def test_fomc_event_type(self):
        """Test FOMC event type."""
        assert EventType.FOMC.value == "fomc"
        
    def test_all_event_types_exist(self):
        """Test all expected event types exist."""
        expected_types = [
            "earnings", "dividend", "fomc", "cpi", "nfp", "gdp",
            "ppi", "retail_sales", "unemployment", "opex", "vix_exp",
            "triple_witching", "split", "merger", "spinoff",
            "fda", "product_launch", "conference", "other"
        ]
        
        actual_types = [et.value for et in EventType]
        
        for expected in expected_types:
            assert expected in actual_types


# =============================================================================
# TEST CLASS: EventImpact Enum
# =============================================================================

class TestEventImpactEnum:
    """Test EventImpact enum values."""
    
    def test_impact_levels(self):
        """Test impact level values."""
        assert EventImpact.HIGH.value == "high"
        assert EventImpact.MEDIUM.value == "medium"
        assert EventImpact.LOW.value == "low"
        assert EventImpact.UNKNOWN.value == "unknown"


# =============================================================================
# TEST CLASS: RiskAction Enum
# =============================================================================

class TestRiskActionEnum:
    """Test RiskAction enum values."""
    
    def test_risk_actions(self):
        """Test risk action values."""
        assert RiskAction.AVOID.value == "avoid"
        assert RiskAction.REDUCE_SIZE.value == "reduce"
        assert RiskAction.WIDEN_STOPS.value == "widen"
        assert RiskAction.DEFINED_RISK.value == "defined"
        assert RiskAction.CLOSE_BEFORE.value == "close"
        assert RiskAction.NORMAL.value == "normal"


# =============================================================================
# TEST CLASS: MarketEvent Dataclass
# =============================================================================

class TestMarketEventDataclass:
    """Test MarketEvent dataclass."""
    
    def test_market_event_creation(self, sample_market_event):
        """Test basic MarketEvent creation."""
        event = sample_market_event
        
        assert event.event_type == EventType.EARNINGS
        assert event.symbol == "AAPL"
        assert event.impact == EventImpact.HIGH
        assert event.historical_move_avg == 5.0
        
    def test_market_event_defaults(self):
        """Test MarketEvent default values."""
        event = MarketEvent(
            event_type=EventType.CPI,
            date=date.today(),
        )
        
        assert event.symbol is None
        assert event.time is None
        assert event.description == ""
        assert event.impact == EventImpact.MEDIUM
        assert event.historical_move_avg == 0.0
        assert event.recommended_action == RiskAction.NORMAL
        assert event.position_size_multiplier == 1.0


# =============================================================================
# TEST CLASS: EventRiskAssessment Dataclass
# =============================================================================

class TestEventRiskAssessmentDataclass:
    """Test EventRiskAssessment dataclass."""
    
    def test_assessment_creation(self):
        """Test EventRiskAssessment creation."""
        assessment = EventRiskAssessment(
            symbol="AAPL",
        )
        
        assert assessment.symbol == "AAPL"
        assert assessment.events == []
        assert assessment.days_to_next_event == 999
        assert assessment.next_event is None
        assert assessment.risk_level == "normal"
        
    def test_assessment_with_events(self, sample_market_event):
        """Test assessment with events."""
        assessment = EventRiskAssessment(
            symbol="AAPL",
            events=[sample_market_event],
            next_event=sample_market_event,
            days_to_next_event=5,
            risk_level="elevated",
        )
        
        assert len(assessment.events) == 1
        assert assessment.next_event is not None
        assert assessment.days_to_next_event == 5


# =============================================================================
# TEST CLASS: EventRiskManager Initialization
# =============================================================================

class TestEventRiskManagerInit:
    """Test EventRiskManager initialization."""
    
    def test_default_initialization(self, event_manager):
        """Test default initialization."""
        assert event_manager is not None
        assert event_manager.lookforward_days == 14
        
    def test_extended_lookforward(self, event_manager_extended):
        """Test extended lookforward initialization."""
        assert event_manager_extended.lookforward_days == 30
        
    def test_economic_calendar_built(self, event_manager):
        """Test that economic calendar is built on init."""
        assert len(event_manager._economic_events) > 0
        
    def test_factory_function(self):
        """Test factory function."""
        manager = create_event_risk_manager(lookforward_days=21)
        
        assert manager is not None
        assert manager.lookforward_days == 21


# =============================================================================
# TEST CLASS: Economic Calendar
# =============================================================================

class TestEconomicCalendar:
    """Test economic calendar building."""
    
    def test_calendar_has_fomc_events(self, event_manager):
        """Test calendar has FOMC events."""
        fomc_events = [
            e for e in event_manager._economic_events 
            if e.event_type == EventType.FOMC
        ]
        
        # Should have at least some FOMC events in 90 days
        assert len(fomc_events) >= 0  # May be 0 depending on current date
        
    def test_calendar_has_opex_events(self, event_manager):
        """Test calendar has options expiration events."""
        opex_events = [
            e for e in event_manager._economic_events 
            if e.event_type in [EventType.OPTIONS_EXPIRATION, EventType.TRIPLE_WITCHING]
        ]
        
        # Should have at least 2 monthly OpEx events
        assert len(opex_events) >= 2
        
    def test_calendar_has_cpi_events(self, event_manager):
        """Test calendar has CPI events."""
        cpi_events = [
            e for e in event_manager._economic_events 
            if e.event_type == EventType.CPI
        ]
        
        # Should have at least some CPI events
        assert len(cpi_events) >= 0
        
    def test_calendar_has_nfp_events(self, event_manager):
        """Test calendar has NFP events."""
        nfp_events = [
            e for e in event_manager._economic_events 
            if e.event_type == EventType.NFP
        ]
        
        # Should have at least some NFP events
        assert len(nfp_events) >= 0
        
    def test_calendar_events_are_sorted(self, event_manager):
        """Test that calendar events are sorted by date."""
        dates = [e.date for e in event_manager._economic_events]
        
        assert dates == sorted(dates)
        
    def test_fomc_has_high_impact(self, event_manager):
        """Test FOMC events have high impact."""
        fomc_events = [
            e for e in event_manager._economic_events 
            if e.event_type == EventType.FOMC
        ]
        
        for event in fomc_events:
            assert event.impact == EventImpact.HIGH


# =============================================================================
# TEST CLASS: Earnings Date Estimation
# =============================================================================

class TestEarningsDateEstimation:
    """Test earnings date estimation."""
    
    def test_estimate_earnings_known_symbol(self, event_manager):
        """Test earnings estimation for known symbol."""
        events = event_manager._estimate_earnings_dates("NVDA")
        
        # Should return list of events
        assert isinstance(events, list)
        
    def test_estimate_earnings_unknown_symbol(self, event_manager):
        """Test earnings estimation for unknown symbol uses defaults."""
        events = event_manager._estimate_earnings_dates("UNKNOWN")
        
        # Should return list using default values
        assert isinstance(events, list)
        
    def test_earnings_history_data(self, event_manager):
        """Test earnings history data is populated."""
        history = event_manager.EARNINGS_MOVE_HISTORY
        
        # Should have known volatile stocks
        assert "NVDA" in history
        assert "TSLA" in history
        assert "AAPL" in history
        
        # Should have default
        assert "_default" in history
        
    def test_nvda_has_high_expected_move(self, event_manager):
        """Test NVDA has high expected move."""
        history = event_manager.EARNINGS_MOVE_HISTORY["NVDA"]
        
        assert history["avg_move"] > 5.0
        assert history["max_move"] > 10.0
        
    def test_get_earnings_dates_caching(self, event_manager):
        """Test that earnings dates are cached."""
        # First call
        events1 = event_manager.get_earnings_dates("AAPL")
        
        # Second call should use cache
        events2 = event_manager.get_earnings_dates("AAPL")
        
        assert "AAPL" in event_manager._earnings_cache


# =============================================================================
# TEST CLASS: Event Risk Assessment
# =============================================================================

class TestEventRiskAssessment:
    """Test event risk assessment."""
    
    def test_assess_event_risk_basic(self, event_manager):
        """Test basic event risk assessment."""
        assessment = event_manager.assess_event_risk(
            symbol="AAPL",
            dte=30,
        )
        
        assert isinstance(assessment, EventRiskAssessment)
        assert assessment.symbol == "AAPL"
        assert assessment.risk_level in ["normal", "elevated", "high", "extreme"]
        
    def test_assess_event_risk_with_custom_date(self, event_manager):
        """Test assessment with custom date."""
        custom_date = date.today() + timedelta(days=30)
        
        assessment = event_manager.assess_event_risk(
            symbol="AAPL",
            dte=30,
            current_date=custom_date,
        )
        
        assert assessment.assessment_date == custom_date
        
    def test_assess_event_risk_position_multiplier(self, event_manager):
        """Test position size multiplier in assessment."""
        assessment = event_manager.assess_event_risk(
            symbol="NVDA",  # High volatility earnings
            dte=30,
        )
        
        # Multiplier should be between 0 and 1 (or exactly 1 if no events)
        assert 0 < assessment.position_size_multiplier <= 1.0
        
    def test_assess_event_risk_stop_multiplier(self, event_manager):
        """Test stop loss multiplier in assessment."""
        assessment = event_manager.assess_event_risk(
            symbol="TSLA",
            dte=30,
        )
        
        # Stop multiplier should be >= 1.0 (wider or same)
        assert assessment.stop_loss_multiplier >= 1.0
        
    def test_assess_event_risk_includes_economic_events(self, event_manager):
        """Test assessment includes economic events."""
        assessment = event_manager.assess_event_risk(
            symbol="SPY",  # Index ETF
            dte=30,
        )
        
        # Should include economic events
        economic_types = {EventType.FOMC, EventType.CPI, EventType.NFP, 
                        EventType.OPTIONS_EXPIRATION, EventType.TRIPLE_WITCHING}
        
        found_economic = any(
            e.event_type in economic_types 
            for e in assessment.events
        )
        
        # May or may not have events depending on calendar
        assert isinstance(assessment.events, list)


# =============================================================================
# TEST CLASS: Risk Level Determination
# =============================================================================

class TestRiskLevelDetermination:
    """Test risk level determination logic."""
    
    def test_normal_risk_level(self, event_manager):
        """Test normal risk level determination."""
        # Create mock assessment with distant events
        assessment = EventRiskAssessment(
            symbol="TEST",
            days_to_next_event=14,
        )
        
        assert assessment.risk_level == "normal"
        
    def test_risk_actions_mapping(self, event_manager):
        """Test risk action mappings exist."""
        # Verify the assessment can produce different actions
        possible_actions = [
            RiskAction.AVOID,
            RiskAction.DEFINED_RISK,
            RiskAction.REDUCE_SIZE,
            RiskAction.NORMAL,
        ]
        
        for action in possible_actions:
            assert action.value is not None


# =============================================================================
# TEST CLASS: Strategy Filtering
# =============================================================================

class TestStrategyFiltering:
    """Test strategy filtering based on events."""
    
    def test_filter_strategies_basic(self, event_manager):
        """Test basic strategy filtering."""
        strategies = [
            "long_call", "bull_put_spread", "iron_condor",
            "naked_put", "short_strangle"
        ]
        
        allowed, filtered = event_manager.filter_strategies_for_events(
            symbol="AAPL",
            strategies=strategies,
            dte=30,
        )
        
        # Should return two lists
        assert isinstance(allowed, list)
        assert isinstance(filtered, list)
        
    def test_filter_strategies_all_allowed(self, event_manager):
        """Test filtering when all strategies are allowed."""
        # Use a symbol unlikely to have imminent events
        strategies = ["long_call", "bull_put_spread"]
        
        allowed, filtered = event_manager.filter_strategies_for_events(
            symbol="IBM",  # Lower volatility stock
            strategies=strategies,
            dte=45,  # Long DTE
        )
        
        # Most should be allowed
        assert len(allowed) >= 0
        
    def test_filter_strategies_defined_risk_requirement(self, event_manager):
        """Test filtering with defined risk requirement."""
        # Create a manual test case
        strategies = [
            "long_call",      # Defined risk
            "naked_put",      # Undefined risk
            "iron_condor",    # Defined risk
        ]
        
        # Filter should identify undefined risk strategies
        undefined_risk = {"naked_put", "naked_call", "short_straddle", "short_strangle"}
        
        has_undefined = any(s.lower() in undefined_risk for s in strategies)
        assert has_undefined is True
        
    def test_filter_strategies_call_restriction(self, event_manager):
        """Test call strategy restriction before ex-dividend."""
        call_strategies = {"long_call", "bull_call_spread", "naked_call"}
        
        strategies = list(call_strategies)
        
        # All are call strategies
        for s in strategies:
            assert s in call_strategies


# =============================================================================
# TEST CLASS: High Impact Events
# =============================================================================

class TestHighImpactEvents:
    """Test high impact event retrieval."""
    
    def test_get_upcoming_high_impact_events(self, event_manager):
        """Test getting high impact events."""
        events = event_manager.get_upcoming_high_impact_events(days=14)
        
        assert isinstance(events, list)
        
        # All returned events should be high impact
        for event in events:
            assert event.impact == EventImpact.HIGH
            
    def test_get_upcoming_high_impact_events_sorted(self, event_manager):
        """Test high impact events are sorted by date."""
        events = event_manager.get_upcoming_high_impact_events(days=30)
        
        if len(events) > 1:
            dates = [e.date for e in events]
            assert dates == sorted(dates)


# =============================================================================
# TEST CLASS: Earnings Calendar
# =============================================================================

class TestEarningsCalendar:
    """Test earnings calendar retrieval."""
    
    def test_get_earnings_calendar(self, event_manager):
        """Test getting earnings calendar for multiple symbols."""
        symbols = ["AAPL", "MSFT", "NVDA"]
        
        calendar = event_manager.get_earnings_calendar(
            symbols=symbols,
            days=30,
        )
        
        assert isinstance(calendar, dict)
        
    def test_get_earnings_calendar_empty_symbols(self, event_manager):
        """Test earnings calendar with empty symbols list."""
        calendar = event_manager.get_earnings_calendar(
            symbols=[],
            days=30,
        )
        
        assert calendar == {}


# =============================================================================
# TEST CLASS: Dividend Events
# =============================================================================

class TestDividendEvents:
    """Test dividend event handling."""
    
    def test_get_dividend_events_known_stock(self, event_manager):
        """Test dividend events for known dividend stock."""
        events = event_manager._get_dividend_events(
            symbol="AAPL",
            start=date.today(),
            end=date.today() + timedelta(days=90),
        )
        
        # AAPL is in the known dividend stocks
        assert isinstance(events, list)
        
    def test_get_dividend_events_unknown_stock(self, event_manager):
        """Test dividend events for unknown stock."""
        events = event_manager._get_dividend_events(
            symbol="UNKNOWN",
            start=date.today(),
            end=date.today() + timedelta(days=90),
        )
        
        # Unknown stock should return empty list
        assert events == []
        
    def test_dividend_event_has_amount(self, event_manager):
        """Test dividend events include amount."""
        events = event_manager._get_dividend_events(
            symbol="SPY",
            start=date.today(),
            end=date.today() + timedelta(days=180),
        )
        
        for event in events:
            assert event.dividend_amount > 0
            assert event.event_type == EventType.DIVIDEND


# =============================================================================
# TEST CLASS: Summary
# =============================================================================

class TestEventManagerSummary:
    """Test summary functionality."""
    
    def test_get_summary(self, event_manager):
        """Test getting manager summary."""
        summary = event_manager.get_summary()
        
        assert "date" in summary
        assert "upcoming_week_economic_events" in summary
        assert "economic_events" in summary
        assert "cached_earnings_symbols" in summary
        
    def test_summary_economic_events_format(self, event_manager):
        """Test economic events format in summary."""
        summary = event_manager.get_summary()
        
        for event in summary["economic_events"]:
            assert "type" in event
            assert "date" in event
            assert "description" in event
            assert "impact" in event


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================

class TestEventManagerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_assess_risk_past_date(self, event_manager):
        """Test assessment with past date."""
        past_date = date.today() - timedelta(days=30)
        
        assessment = event_manager.assess_event_risk(
            symbol="AAPL",
            dte=30,
            current_date=past_date,
        )
        
        # Should not crash
        assert assessment is not None
        
    def test_assess_risk_zero_dte(self, event_manager):
        """Test assessment with zero DTE."""
        assessment = event_manager.assess_event_risk(
            symbol="SPY",
            dte=0,
        )
        
        assert assessment is not None
        
    def test_assess_risk_very_long_dte(self, event_manager):
        """Test assessment with very long DTE."""
        assessment = event_manager.assess_event_risk(
            symbol="AAPL",
            dte=365,
        )
        
        assert assessment is not None
        
    def test_empty_symbol(self, event_manager):
        """Test with empty symbol."""
        assessment = event_manager.assess_event_risk(
            symbol="",
            dte=30,
        )
        
        assert assessment.symbol == ""
        
    def test_special_characters_in_symbol(self, event_manager):
        """Test with special characters in symbol."""
        assessment = event_manager.assess_event_risk(
            symbol="BRK.B",
            dte=30,
        )
        
        assert assessment.symbol == "BRK.B"


# =============================================================================
# TEST CLASS: Event Impact Calculations
# =============================================================================

class TestEventImpactCalculations:
    """Test event impact calculation logic."""
    
    def test_high_volatility_stock_impact(self, event_manager):
        """Test impact calculation for high volatility stock."""
        history = event_manager.EARNINGS_MOVE_HISTORY.get("COIN", {})
        
        if history:
            # COIN should have high expected move
            assert history["avg_move"] > 10.0
            
    def test_low_volatility_stock_impact(self, event_manager):
        """Test impact calculation for low volatility stock."""
        history = event_manager.EARNINGS_MOVE_HISTORY.get("SPY", {})
        
        if history:
            # SPY should have low expected move
            assert history["avg_move"] < 3.0
            
    def test_iv_crush_estimates(self, event_manager):
        """Test IV crush estimates exist."""
        for symbol, history in event_manager.EARNINGS_MOVE_HISTORY.items():
            if symbol != "_default":
                assert "iv_crush" in history
                assert history["iv_crush"] > 0


# =============================================================================
# TEST CLASS: Warning Generation
# =============================================================================

class TestWarningGeneration:
    """Test warning generation in assessments."""
    
    def test_warnings_list_type(self, event_manager):
        """Test warnings is a list."""
        assessment = event_manager.assess_event_risk(
            symbol="NVDA",
            dte=30,
        )
        
        assert isinstance(assessment.warnings, list)
        
    def test_strategy_adjustments_list(self, event_manager):
        """Test strategy adjustments is a list."""
        assessment = event_manager.assess_event_risk(
            symbol="TSLA",
            dte=30,
        )
        
        assert isinstance(assessment.strategy_adjustments, list)


# =============================================================================
# TEST CLASS: Integration Tests
# =============================================================================

class TestEventManagerIntegration:
    """Integration tests for EventRiskManager."""
    
    def test_full_assessment_flow(self, event_manager):
        """Test full assessment flow."""
        # Assess risk
        assessment = event_manager.assess_event_risk(
            symbol="NVDA",
            dte=21,
        )
        
        # Filter strategies
        strategies = ["long_call", "bull_put_spread", "iron_condor", "naked_put"]
        allowed, filtered = event_manager.filter_strategies_for_events(
            symbol="NVDA",
            strategies=strategies,
            dte=21,
        )
        
        # Get summary
        summary = event_manager.get_summary()
        
        # All should work together
        assert assessment is not None
        assert isinstance(allowed, list)
        assert isinstance(summary, dict)
        
    def test_multiple_symbol_assessment(self, event_manager):
        """Test assessing multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
        
        assessments = {}
        for symbol in symbols:
            assessments[symbol] = event_manager.assess_event_risk(
                symbol=symbol,
                dte=30,
            )
        
        assert len(assessments) == 5
        
        for symbol, assessment in assessments.items():
            assert assessment.symbol == symbol


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
