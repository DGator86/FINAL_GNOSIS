"""
Tests for Liquidity Engine V3

Comprehensive tests for liquidity analysis including:
- Basic liquidity scoring
- Bid-ask spread analysis
- Volume analysis
- 0DTE depth calculation
- Gamma squeeze detection
- Error handling

Author: Super Gnosis Elite Trading System
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch

from engines.liquidity.liquidity_engine_v3 import LiquidityEngineV3
from schemas.core_schemas import LiquiditySnapshot


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_market_adapter():
    """Create a mock market data adapter."""
    adapter = Mock()
    
    # Default quote
    quote = Mock()
    quote.bid = 450.0
    quote.ask = 450.10
    quote.bid_size = 1000
    quote.ask_size = 1200
    adapter.get_quote.return_value = quote
    
    # Default bars
    bars = []
    for i in range(5):
        bar = Mock()
        bar.volume = 1_000_000 + i * 100_000
        bar.close = 450.0 + i
        bar.high = 451.0 + i
        bar.low = 449.0 + i
        bars.append(bar)
    adapter.get_bars.return_value = bars
    
    return adapter


@pytest.fixture
def mock_options_adapter():
    """Create a mock options chain adapter."""
    adapter = Mock()
    
    # Default chain with 0DTE contracts
    contracts = []
    for i in range(10):
        contract = Mock()
        contract.expiration = datetime.now(timezone.utc)  # 0DTE
        contract.open_interest = 5000 + i * 1000
        contract.strike = 445.0 + i * 5
        contract.option_type = "call" if i % 2 == 0 else "put"
        contracts.append(contract)
    adapter.get_chain.return_value = contracts
    
    return adapter


@pytest.fixture
def default_config():
    """Create default engine configuration."""
    return {
        "volume_threshold": 1_000_000,
        "spread_threshold": 0.05,
    }


@pytest.fixture
def liquidity_engine(mock_market_adapter, mock_options_adapter, default_config):
    """Create a LiquidityEngineV3 instance with mocked dependencies."""
    return LiquidityEngineV3(
        market_adapter=mock_market_adapter,
        options_adapter=mock_options_adapter,
        config=default_config,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestLiquidityEngineInitialization:
    """Tests for LiquidityEngineV3 initialization."""
    
    def test_initialization(self, mock_market_adapter, mock_options_adapter, default_config):
        """Test basic engine initialization."""
        engine = LiquidityEngineV3(
            market_adapter=mock_market_adapter,
            options_adapter=mock_options_adapter,
            config=default_config,
        )
        
        assert engine.market_adapter is mock_market_adapter
        assert engine.options_adapter is mock_options_adapter
        assert engine.config == default_config
    
    def test_initialization_empty_config(self, mock_market_adapter, mock_options_adapter):
        """Test initialization with empty config."""
        engine = LiquidityEngineV3(
            market_adapter=mock_market_adapter,
            options_adapter=mock_options_adapter,
            config={},
        )
        
        assert engine.config == {}


# =============================================================================
# BASIC LIQUIDITY ANALYSIS TESTS
# =============================================================================

class TestBasicLiquidityAnalysis:
    """Tests for basic liquidity scoring."""
    
    def test_run_returns_liquidity_snapshot(self, liquidity_engine):
        """Test that run returns a LiquiditySnapshot."""
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        assert isinstance(result, LiquiditySnapshot)
        assert result.symbol == "SPY"
        assert result.timestamp == timestamp
    
    def test_liquidity_score_in_valid_range(self, liquidity_engine):
        """Test that liquidity score is between 0 and 1."""
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        assert 0.0 <= result.liquidity_score <= 1.0
    
    def test_bid_ask_spread_calculated(self, liquidity_engine, mock_market_adapter):
        """Test bid-ask spread calculation."""
        quote = Mock()
        quote.bid = 100.0
        quote.ask = 100.20
        quote.bid_size = 1000
        quote.ask_size = 1000
        mock_market_adapter.get_quote.return_value = quote
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        # Spread should be (100.20 - 100.0) / 100.10 * 100 ≈ 0.2%
        assert result.bid_ask_spread > 0
        assert result.bid_ask_spread < 1.0  # Should be small for tight spread
    
    def test_volume_from_bars(self, liquidity_engine, mock_market_adapter):
        """Test volume calculation from historical bars."""
        bars = []
        for i in range(5):
            bar = Mock()
            bar.volume = 2_000_000
            bars.append(bar)
        mock_market_adapter.get_bars.return_value = bars
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        # Average volume should be 2M
        assert result.volume == 2_000_000
    
    def test_depth_from_quote_sizes(self, liquidity_engine, mock_market_adapter):
        """Test depth calculation from bid/ask sizes."""
        quote = Mock()
        quote.bid = 100.0
        quote.ask = 100.10
        quote.bid_size = 5000
        quote.ask_size = 3000
        mock_market_adapter.get_quote.return_value = quote
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        # Depth = bid_size + ask_size
        assert result.depth == 8000
    
    def test_impact_cost_calculated(self, liquidity_engine):
        """Test impact cost calculation."""
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        # Impact cost should be positive
        assert result.impact_cost >= 0


# =============================================================================
# LIQUIDITY SCORE CALCULATION TESTS
# =============================================================================

class TestLiquidityScoreCalculation:
    """Tests for liquidity score calculation logic."""
    
    def test_high_volume_increases_score(self, liquidity_engine, mock_market_adapter):
        """Test that high volume increases liquidity score."""
        # Low volume
        low_vol_bars = [Mock(volume=100_000) for _ in range(5)]
        mock_market_adapter.get_bars.return_value = low_vol_bars
        
        timestamp = datetime.now(timezone.utc)
        low_vol_result = liquidity_engine.run("SPY", timestamp)
        
        # High volume
        high_vol_bars = [Mock(volume=10_000_000) for _ in range(5)]
        mock_market_adapter.get_bars.return_value = high_vol_bars
        
        high_vol_result = liquidity_engine.run("SPY", timestamp)
        
        assert high_vol_result.liquidity_score > low_vol_result.liquidity_score
    
    def test_tight_spread_increases_score(self, liquidity_engine, mock_market_adapter):
        """Test that tight spread increases liquidity score."""
        # Wide spread
        wide_quote = Mock()
        wide_quote.bid = 100.0
        wide_quote.ask = 101.0  # 1% spread
        wide_quote.bid_size = 1000
        wide_quote.ask_size = 1000
        mock_market_adapter.get_quote.return_value = wide_quote
        
        timestamp = datetime.now(timezone.utc)
        wide_result = liquidity_engine.run("SPY", timestamp)
        
        # Tight spread
        tight_quote = Mock()
        tight_quote.bid = 100.0
        tight_quote.ask = 100.01  # 0.01% spread
        tight_quote.bid_size = 1000
        tight_quote.ask_size = 1000
        mock_market_adapter.get_quote.return_value = tight_quote
        
        tight_result = liquidity_engine.run("SPY", timestamp)
        
        assert tight_result.liquidity_score > wide_result.liquidity_score


# =============================================================================
# 0DTE DEPTH TESTS
# =============================================================================

class TestZeroDTEDepth:
    """Tests for 0DTE options depth calculation."""
    
    def test_0dte_depth_calculated(self, liquidity_engine, mock_options_adapter):
        """Test 0DTE depth is calculated from options chain."""
        timestamp = datetime.now(timezone.utc)
        
        # Create 0DTE contracts
        contracts = []
        for i in range(5):
            contract = Mock()
            contract.expiration = Mock()
            contract.expiration.date.return_value = timestamp.date()
            contract.open_interest = 10000
            contracts.append(contract)
        mock_options_adapter.get_chain.return_value = contracts
        
        result = liquidity_engine._calculate_0dte_depth("SPY", timestamp)
        
        # Should sum open interest: 5 * 10000 = 50000
        assert result == 50000
    
    def test_0dte_depth_excludes_non_0dte(self, liquidity_engine, mock_options_adapter):
        """Test that non-0DTE contracts are excluded."""
        timestamp = datetime.now(timezone.utc)
        
        contracts = []
        # 0DTE contract
        zero_dte = Mock()
        zero_dte.expiration = Mock()
        zero_dte.expiration.date.return_value = timestamp.date()
        zero_dte.open_interest = 10000
        contracts.append(zero_dte)
        
        # Non-0DTE contract (tomorrow)
        non_zero_dte = Mock()
        non_zero_dte.expiration = Mock()
        non_zero_dte.expiration.date.return_value = (timestamp + timedelta(days=1)).date()
        non_zero_dte.open_interest = 50000
        contracts.append(non_zero_dte)
        
        mock_options_adapter.get_chain.return_value = contracts
        
        result = liquidity_engine._calculate_0dte_depth("SPY", timestamp)
        
        # Should only include 0DTE
        assert result == 10000
    
    def test_0dte_depth_no_chain(self, liquidity_engine, mock_options_adapter):
        """Test 0DTE depth when no chain available."""
        mock_options_adapter.get_chain.return_value = None
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine._calculate_0dte_depth("SPY", timestamp)
        
        assert result == 0.0
    
    def test_0dte_depth_empty_chain(self, liquidity_engine, mock_options_adapter):
        """Test 0DTE depth with empty chain."""
        mock_options_adapter.get_chain.return_value = []
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine._calculate_0dte_depth("SPY", timestamp)
        
        assert result == 0.0
    
    def test_0dte_depth_exception_handling(self, liquidity_engine, mock_options_adapter):
        """Test 0DTE depth handles exceptions gracefully."""
        mock_options_adapter.get_chain.side_effect = Exception("API Error")
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine._calculate_0dte_depth("SPY", timestamp)
        
        assert result == 0.0


# =============================================================================
# GAMMA SQUEEZE DETECTION TESTS
# =============================================================================

class TestGammaSqueezeDetection:
    """Tests for gamma squeeze risk detection."""
    
    def test_gamma_squeeze_detected_high_volume(self, liquidity_engine, mock_market_adapter):
        """Test gamma squeeze detection with high volume spike."""
        timestamp = datetime.now(timezone.utc)
        
        # Set up high volume spike (4x average)
        bars = [Mock(volume=10_000_000)]  # Current day
        mock_market_adapter.get_bars.return_value = bars
        
        avg_volume = 2_000_000  # Average
        
        result = liquidity_engine._detect_gamma_squeeze_risk("SPY", timestamp, avg_volume)
        
        # 10M / 2M = 5x > 3x threshold
        assert result is True
    
    def test_no_gamma_squeeze_normal_volume(self, liquidity_engine, mock_market_adapter):
        """Test no gamma squeeze with normal volume."""
        timestamp = datetime.now(timezone.utc)
        
        bars = [Mock(volume=2_500_000)]  # Slightly above average
        mock_market_adapter.get_bars.return_value = bars
        
        avg_volume = 2_000_000
        
        result = liquidity_engine._detect_gamma_squeeze_risk("SPY", timestamp, avg_volume)
        
        # 2.5M / 2M = 1.25x < 3x threshold
        assert result is False
    
    def test_gamma_squeeze_no_bars(self, liquidity_engine, mock_market_adapter):
        """Test gamma squeeze detection with no bars."""
        mock_market_adapter.get_bars.return_value = []
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine._detect_gamma_squeeze_risk("SPY", timestamp, 1_000_000)
        
        assert result is False
    
    def test_gamma_squeeze_zero_avg_volume(self, liquidity_engine, mock_market_adapter):
        """Test gamma squeeze detection with zero average volume."""
        bars = [Mock(volume=1_000_000)]
        mock_market_adapter.get_bars.return_value = bars
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine._detect_gamma_squeeze_risk("SPY", timestamp, 0)
        
        assert result is False
    
    def test_gamma_squeeze_exception_handling(self, liquidity_engine, mock_market_adapter):
        """Test gamma squeeze handles exceptions."""
        mock_market_adapter.get_bars.side_effect = Exception("API Error")
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine._detect_gamma_squeeze_risk("SPY", timestamp, 1_000_000)
        
        assert result is False


# =============================================================================
# GAMMA SQUEEZE IMPACT ON LIQUIDITY TESTS
# =============================================================================

class TestGammaSqueezeImpact:
    """Tests for gamma squeeze impact on liquidity score."""
    
    def test_gamma_squeeze_reduces_liquidity_score(self, liquidity_engine, mock_market_adapter):
        """Test that gamma squeeze risk reduces liquidity score."""
        timestamp = datetime.now(timezone.utc)
        
        # Normal volume for initial run
        normal_bars = [Mock(volume=2_000_000) for _ in range(5)]
        mock_market_adapter.get_bars.return_value = normal_bars
        
        normal_result = liquidity_engine.run("SPY", timestamp)
        
        # High volume spike (gamma squeeze)
        def get_bars_with_spike(symbol, start, end, timeframe):
            if timeframe == "1Day" and (end - start).days <= 1:
                # For gamma squeeze detection
                return [Mock(volume=15_000_000)]
            return normal_bars
        
        mock_market_adapter.get_bars.side_effect = get_bars_with_spike
        
        squeeze_result = liquidity_engine.run("SPY", timestamp)
        
        # Gamma squeeze should reduce score by 30%
        assert squeeze_result.liquidity_score < normal_result.liquidity_score


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in liquidity engine."""
    
    def test_run_handles_quote_exception(self, liquidity_engine, mock_market_adapter):
        """Test run handles quote exception gracefully."""
        mock_market_adapter.get_quote.side_effect = Exception("Quote API Error")
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        # Should return default snapshot
        assert isinstance(result, LiquiditySnapshot)
        assert result.symbol == "SPY"
    
    def test_run_handles_bars_exception(self, liquidity_engine, mock_market_adapter):
        """Test run handles bars exception gracefully."""
        mock_market_adapter.get_bars.side_effect = Exception("Bars API Error")
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        assert isinstance(result, LiquiditySnapshot)
    
    def test_run_handles_zero_mid_price(self, liquidity_engine, mock_market_adapter):
        """Test handling of zero mid price."""
        quote = Mock()
        quote.bid = 0.0
        quote.ask = 0.0
        quote.bid_size = 0
        quote.ask_size = 0
        mock_market_adapter.get_quote.return_value = quote
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        # Should handle gracefully
        assert result.bid_ask_spread == 0.0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_bars(self, liquidity_engine, mock_market_adapter):
        """Test with empty bars list."""
        mock_market_adapter.get_bars.return_value = []
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        assert result.volume == 0.0
    
    def test_single_bar(self, liquidity_engine, mock_market_adapter):
        """Test with single bar."""
        mock_market_adapter.get_bars.return_value = [Mock(volume=1_000_000)]
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        assert result.volume == 1_000_000
    
    def test_very_wide_spread(self, liquidity_engine, mock_market_adapter):
        """Test with very wide spread (illiquid stock)."""
        quote = Mock()
        quote.bid = 10.0
        quote.ask = 12.0  # 20% spread
        quote.bid_size = 100
        quote.ask_size = 100
        mock_market_adapter.get_quote.return_value = quote
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        # Score should be low due to wide spread
        assert result.liquidity_score < 0.5
    
    def test_very_high_volume(self, liquidity_engine, mock_market_adapter):
        """Test with very high volume."""
        bars = [Mock(volume=100_000_000) for _ in range(5)]  # 100M
        mock_market_adapter.get_bars.return_value = bars
        
        timestamp = datetime.now(timezone.utc)
        result = liquidity_engine.run("SPY", timestamp)
        
        # Volume score should be capped at 1.0
        assert result.liquidity_score <= 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for liquidity engine."""
    
    def test_full_analysis_flow(self, liquidity_engine):
        """Test complete analysis flow."""
        timestamp = datetime.now(timezone.utc)
        
        result = liquidity_engine.run("SPY", timestamp)
        
        # Verify all fields are populated
        assert result.symbol == "SPY"
        assert result.timestamp == timestamp
        assert result.liquidity_score > 0
        assert result.bid_ask_spread >= 0
        assert result.volume >= 0
        assert result.depth >= 0
        assert result.impact_cost >= 0
    
    def test_multiple_symbols(self, liquidity_engine):
        """Test analysis for multiple symbols."""
        timestamp = datetime.now(timezone.utc)
        symbols = ["SPY", "QQQ", "AAPL"]
        
        results = [liquidity_engine.run(symbol, timestamp) for symbol in symbols]
        
        for i, result in enumerate(results):
            assert result.symbol == symbols[i]
            assert isinstance(result, LiquiditySnapshot)
    
    def test_different_timestamps(self, liquidity_engine):
        """Test analysis at different timestamps."""
        base_time = datetime.now(timezone.utc)
        timestamps = [
            base_time,
            base_time - timedelta(hours=1),
            base_time - timedelta(days=1),
        ]
        
        for ts in timestamps:
            result = liquidity_engine.run("SPY", ts)
            assert result.timestamp == ts
