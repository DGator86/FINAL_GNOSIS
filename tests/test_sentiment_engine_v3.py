"""
Tests for Sentiment Engine V3

Comprehensive tests for sentiment analysis including:
- Multi-source sentiment aggregation
- News, flow, and technical sentiment processing
- Unusual Whales integration
- Confidence scoring
- Error handling

Author: Super Gnosis Elite Trading System
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from engines.sentiment.sentiment_engine_v3 import SentimentEngineV3
from schemas.core_schemas import SentimentSnapshot


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_news_processor():
    """Create a mock news sentiment processor."""
    processor = Mock()
    processor.__class__.__name__ = "NewsSentimentProcessor"
    processor.process.return_value = 0.3  # Slightly bullish
    return processor


@pytest.fixture
def mock_flow_processor():
    """Create a mock flow sentiment processor."""
    processor = Mock()
    processor.__class__.__name__ = "FlowSentimentProcessor"
    processor.process.return_value = 0.5  # Bullish
    return processor


@pytest.fixture
def mock_technical_processor():
    """Create a mock technical sentiment processor."""
    processor = Mock()
    processor.__class__.__name__ = "TechnicalSentimentProcessor"
    processor.process.return_value = 0.2  # Slightly bullish
    return processor


@pytest.fixture
def mock_unusual_whales_adapter():
    """Create a mock Unusual Whales adapter."""
    adapter = Mock()
    adapter.get_unusual_activity.return_value = [
        {"type": "call_sweep", "premium": 500000},
        {"type": "put_sweep", "premium": 200000},
    ]
    return adapter


@pytest.fixture
def default_config():
    """Create default engine configuration."""
    return {
        "news_weight": 0.3,
        "flow_weight": 0.4,
        "technical_weight": 0.3,
        "enable_social_media": False,  # Disable for deterministic tests
    }


@pytest.fixture
def sentiment_engine(
    mock_news_processor,
    mock_flow_processor, 
    mock_technical_processor,
    mock_unusual_whales_adapter,
    default_config
):
    """Create a SentimentEngineV3 instance with mocked dependencies."""
    # Ensure social media is disabled for deterministic tests
    config = {**default_config, "enable_social_media": False}
    return SentimentEngineV3(
        processors=[mock_news_processor, mock_flow_processor, mock_technical_processor],
        unusual_whales_adapter=mock_unusual_whales_adapter,
        config=config,
    )


@pytest.fixture
def sentiment_engine_no_uw(
    mock_news_processor,
    mock_flow_processor,
    mock_technical_processor,
    default_config
):
    """Create a SentimentEngineV3 without Unusual Whales."""
    # Ensure social media is disabled for deterministic tests
    config = {**default_config, "enable_social_media": False}
    return SentimentEngineV3(
        processors=[mock_news_processor, mock_flow_processor, mock_technical_processor],
        unusual_whales_adapter=None,
        config=config,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestSentimentEngineInitialization:
    """Tests for SentimentEngineV3 initialization."""
    
    def test_initialization_with_all_components(
        self, mock_news_processor, mock_unusual_whales_adapter, default_config
    ):
        """Test initialization with all components."""
        engine = SentimentEngineV3(
            processors=[mock_news_processor],
            unusual_whales_adapter=mock_unusual_whales_adapter,
            config=default_config,
        )
        
        assert len(engine.processors) == 1
        assert engine.unusual_whales_adapter is mock_unusual_whales_adapter
        assert engine.config == default_config
    
    def test_initialization_without_unusual_whales(
        self, mock_news_processor, default_config
    ):
        """Test initialization without Unusual Whales adapter."""
        engine = SentimentEngineV3(
            processors=[mock_news_processor],
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        assert engine.unusual_whales_adapter is None
    
    def test_initialization_empty_processors(self, default_config):
        """Test initialization with no processors."""
        engine = SentimentEngineV3(
            processors=[],
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        assert len(engine.processors) == 0
    
    def test_initialization_empty_config(self, mock_news_processor):
        """Test initialization with empty config uses defaults."""
        engine = SentimentEngineV3(
            processors=[mock_news_processor],
            unusual_whales_adapter=None,
            config={},
        )
        
        assert engine.config == {}


# =============================================================================
# BASIC SENTIMENT ANALYSIS TESTS
# =============================================================================

class TestBasicSentimentAnalysis:
    """Tests for basic sentiment analysis."""
    
    def test_run_returns_sentiment_snapshot(self, sentiment_engine):
        """Test that run returns a SentimentSnapshot."""
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        assert isinstance(result, SentimentSnapshot)
        assert result.symbol == "SPY"
        assert result.timestamp == timestamp
    
    def test_sentiment_score_in_valid_range(self, sentiment_engine):
        """Test that sentiment score is in valid range."""
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        # Score should be between -1 and 1
        assert -1.0 <= result.sentiment_score <= 1.0
    
    def test_confidence_in_valid_range(self, sentiment_engine):
        """Test that confidence is between 0 and 1."""
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        assert 0.0 <= result.confidence <= 1.0
    
    def test_news_sentiment_populated(self, sentiment_engine):
        """Test news sentiment is populated."""
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        assert result.news_sentiment == 0.3  # From mock
    
    def test_flow_sentiment_populated(self, sentiment_engine):
        """Test flow sentiment is populated."""
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        # Flow is blended with unusual whales (0.5 * 0.5 + UW * 0.5)
        assert result.flow_sentiment != 0
    
    def test_technical_sentiment_populated(self, sentiment_engine):
        """Test technical sentiment is populated."""
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        assert result.technical_sentiment == 0.2  # From mock


# =============================================================================
# WEIGHTED SENTIMENT CALCULATION TESTS
# =============================================================================

class TestWeightedSentimentCalculation:
    """Tests for weighted sentiment calculation."""
    
    def test_default_weights_applied(self, sentiment_engine_no_uw):
        """Test that default weights are applied correctly."""
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine_no_uw.run("SPY", timestamp)
        
        # With social media disabled, calculation is deterministic:
        # (news*0.3 + flow*0.4 + tech*0.3) / 1.0
        # = (0.3*0.3 + 0.5*0.4 + 0.2*0.3) = 0.09 + 0.20 + 0.06 = 0.35
        expected = 0.3 * 0.3 + 0.5 * 0.4 + 0.2 * 0.3
        assert abs(result.sentiment_score - expected) < 0.01
        # Check that processors contributed correctly
        assert result.news_sentiment == 0.3  # From mock
        assert result.flow_sentiment == 0.5  # From mock (no UW blending)
    
    def test_custom_weights_applied(
        self, mock_news_processor, mock_flow_processor, mock_technical_processor
    ):
        """Test custom weights are applied."""
        custom_config = {
            "news_weight": 0.5,  # Heavy news weight
            "flow_weight": 0.3,
            "technical_weight": 0.2,
            "enable_social_media": False,  # Disable social for exact calculation
        }
        
        engine = SentimentEngineV3(
            processors=[mock_news_processor, mock_flow_processor, mock_technical_processor],
            unusual_whales_adapter=None,
            config=custom_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Without social media, calculation is: (news*0.5 + flow*0.3 + tech*0.2) / 1.0
        expected = 0.3 * 0.5 + 0.5 * 0.3 + 0.2 * 0.2
        assert abs(result.sentiment_score - expected) < 0.01
    
    def test_all_positive_sentiment(self, default_config):
        """Test with all positive sentiment."""
        processors = []
        for name in ["News", "Flow", "Technical"]:
            p = Mock()
            p.__class__.__name__ = f"{name}Processor"
            p.process.return_value = 0.8
            processors.append(p)
        
        engine = SentimentEngineV3(
            processors=processors,
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Should be positive
        assert result.sentiment_score > 0.5
    
    def test_all_negative_sentiment(self, default_config):
        """Test with all negative sentiment."""
        processors = []
        for name in ["News", "Flow", "Technical"]:
            p = Mock()
            p.__class__.__name__ = f"{name}Processor"
            p.process.return_value = -0.7
            processors.append(p)
        
        engine = SentimentEngineV3(
            processors=processors,
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Should be negative
        assert result.sentiment_score < -0.5
    
    def test_mixed_sentiment(self, default_config):
        """Test with mixed sentiment."""
        news_p = Mock()
        news_p.__class__.__name__ = "NewsProcessor"
        news_p.process.return_value = 0.8  # Very bullish
        
        flow_p = Mock()
        flow_p.__class__.__name__ = "FlowProcessor"
        flow_p.process.return_value = -0.6  # Bearish
        
        tech_p = Mock()
        tech_p.__class__.__name__ = "TechnicalProcessor"
        tech_p.process.return_value = 0.0  # Neutral
        
        engine = SentimentEngineV3(
            processors=[news_p, flow_p, tech_p],
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Mixed should result in moderate score
        assert -0.3 <= result.sentiment_score <= 0.3


# =============================================================================
# UNUSUAL WHALES INTEGRATION TESTS
# =============================================================================

class TestUnusualWhalesIntegration:
    """Tests for Unusual Whales flow integration."""
    
    def test_unusual_flow_bullish(self, sentiment_engine, mock_unusual_whales_adapter):
        """Test bullish unusual flow sentiment."""
        # More calls than puts
        mock_unusual_whales_adapter.get_unusual_activity.return_value = [
            {"type": "call", "premium": 800000},
            {"type": "put", "premium": 200000},
        ]
        
        result = sentiment_engine._analyze_unusual_flow("SPY")
        
        # (800k - 200k) / 1M = 0.6
        assert result > 0.5
    
    def test_unusual_flow_bearish(self, sentiment_engine, mock_unusual_whales_adapter):
        """Test bearish unusual flow sentiment."""
        # More puts than calls
        mock_unusual_whales_adapter.get_unusual_activity.return_value = [
            {"type": "call", "premium": 200000},
            {"type": "put", "premium": 800000},
        ]
        
        result = sentiment_engine._analyze_unusual_flow("SPY")
        
        # (200k - 800k) / 1M = -0.6
        assert result < -0.5
    
    def test_unusual_flow_neutral(self, sentiment_engine, mock_unusual_whales_adapter):
        """Test neutral unusual flow sentiment."""
        # Equal calls and puts
        mock_unusual_whales_adapter.get_unusual_activity.return_value = [
            {"type": "call", "premium": 500000},
            {"type": "put", "premium": 500000},
        ]
        
        result = sentiment_engine._analyze_unusual_flow("SPY")
        
        # (500k - 500k) / 1M = 0
        assert abs(result) < 0.1
    
    def test_unusual_flow_no_data(self, sentiment_engine, mock_unusual_whales_adapter):
        """Test unusual flow with no data."""
        mock_unusual_whales_adapter.get_unusual_activity.return_value = []
        
        result = sentiment_engine._analyze_unusual_flow("SPY")
        
        assert result is None
    
    def test_unusual_flow_no_adapter(self, sentiment_engine_no_uw):
        """Test unusual flow without adapter."""
        result = sentiment_engine_no_uw._analyze_unusual_flow("SPY")
        
        assert result is None
    
    def test_unusual_flow_exception(self, sentiment_engine, mock_unusual_whales_adapter):
        """Test unusual flow handles exceptions."""
        mock_unusual_whales_adapter.get_unusual_activity.side_effect = Exception("API Error")
        
        result = sentiment_engine._analyze_unusual_flow("SPY")
        
        assert result is None
    
    def test_unusual_flow_zero_total(self, sentiment_engine, mock_unusual_whales_adapter):
        """Test unusual flow with zero total flow."""
        mock_unusual_whales_adapter.get_unusual_activity.return_value = [
            {"type": "neutral", "premium": 100000},  # No call/put type
        ]
        
        result = sentiment_engine._analyze_unusual_flow("SPY")
        
        # Total flow is 0, should return 0
        assert result == 0.0
    
    def test_unusual_flow_blending(self, sentiment_engine, mock_unusual_whales_adapter):
        """Test unusual flow blending with standard flow."""
        # Strong bullish unusual flow
        mock_unusual_whales_adapter.get_unusual_activity.return_value = [
            {"type": "bullish", "premium": 1000000},
        ]
        
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        # Flow should be blended (standard 0.5 + UW 1.0) / 2 = 0.75
        # This affects the overall score
        assert result.flow_sentiment > 0


# =============================================================================
# CONFIDENCE CALCULATION TESTS
# =============================================================================

class TestConfidenceCalculation:
    """Tests for confidence score calculation."""
    
    def test_high_confidence_when_aligned(self, default_config):
        """Test high confidence when all sources agree."""
        processors = []
        for name in ["News", "Flow", "Technical"]:
            p = Mock()
            p.__class__.__name__ = f"{name}Processor"
            p.process.return_value = 0.5  # All agree
            processors.append(p)
        
        engine = SentimentEngineV3(
            processors=processors,
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # High agreement = high confidence
        assert result.confidence > 0.8
    
    def test_low_confidence_when_divergent(self, default_config):
        """Test low confidence when sources diverge."""
        news_p = Mock()
        news_p.__class__.__name__ = "NewsProcessor"
        news_p.process.return_value = 0.8  # Very bullish
        
        flow_p = Mock()
        flow_p.__class__.__name__ = "FlowProcessor"
        flow_p.process.return_value = -0.8  # Very bearish
        
        tech_p = Mock()
        tech_p.__class__.__name__ = "TechnicalProcessor"
        tech_p.process.return_value = 0.0  # Neutral
        
        engine = SentimentEngineV3(
            processors=[news_p, flow_p, tech_p],
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # High divergence = lower confidence
        assert result.confidence < 0.7
    
    def test_confidence_boost_from_unusual_flow(
        self, sentiment_engine, mock_unusual_whales_adapter
    ):
        """Test confidence boost when unusual flow confirms."""
        # Set up confirming unusual flow
        mock_unusual_whales_adapter.get_unusual_activity.return_value = [
            {"type": "call", "premium": 900000},
            {"type": "put", "premium": 100000},
        ]
        
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        # Should have boosted confidence
        assert result.confidence > 0.1


# =============================================================================
# PROCESSOR ERROR HANDLING TESTS
# =============================================================================

class TestProcessorErrorHandling:
    """Tests for handling processor errors."""
    
    def test_processor_exception_handled(self, default_config):
        """Test that processor exceptions are handled gracefully."""
        failing_processor = Mock()
        failing_processor.__class__.__name__ = "NewsProcessor"
        failing_processor.process.side_effect = Exception("Processor Error")
        
        working_processor = Mock()
        working_processor.__class__.__name__ = "FlowProcessor"
        working_processor.process.return_value = 0.5
        
        engine = SentimentEngineV3(
            processors=[failing_processor, working_processor],
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Should still return result
        assert isinstance(result, SentimentSnapshot)
    
    def test_all_processors_fail(self, default_config):
        """Test when all processors fail."""
        failing1 = Mock()
        failing1.__class__.__name__ = "NewsProcessor"
        failing1.process.side_effect = Exception("Error 1")
        
        failing2 = Mock()
        failing2.__class__.__name__ = "FlowProcessor"
        failing2.process.side_effect = Exception("Error 2")
        
        # Disable social media for this test
        config_no_social = {**default_config, "enable_social_media": False}
        
        engine = SentimentEngineV3(
            processors=[failing1, failing2],
            unusual_whales_adapter=None,
            config=config_no_social,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Should return with zeroed sentiment (no social media)
        assert result.sentiment_score == 0.0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_no_processors(self, mock_unusual_whales_adapter, default_config):
        """Test with no processors."""
        engine = SentimentEngineV3(
            processors=[],
            unusual_whales_adapter=mock_unusual_whales_adapter,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Should return with minimal sentiment from UW only
        assert isinstance(result, SentimentSnapshot)
    
    def test_extreme_sentiment_values(self, default_config):
        """Test with extreme sentiment values."""
        extreme_p = Mock()
        extreme_p.__class__.__name__ = "FlowProcessor"
        extreme_p.process.return_value = 1.0  # Maximum
        
        engine = SentimentEngineV3(
            processors=[extreme_p],
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Should handle extreme values
        assert -1.0 <= result.sentiment_score <= 1.0
    
    def test_unknown_processor_type(self, default_config):
        """Test with unknown processor type."""
        unknown_p = Mock()
        unknown_p.__class__.__name__ = "UnknownProcessor"
        unknown_p.process.return_value = 0.5
        
        engine = SentimentEngineV3(
            processors=[unknown_p],
            unusual_whales_adapter=None,
            config=default_config,
        )
        
        timestamp = datetime.now(timezone.utc)
        result = engine.run("SPY", timestamp)
        
        # Should handle unknown processors
        assert isinstance(result, SentimentSnapshot)
    
    def test_unusual_activity_with_bullish_type(
        self, sentiment_engine, mock_unusual_whales_adapter
    ):
        """Test unusual activity parsing with 'bullish' type."""
        mock_unusual_whales_adapter.get_unusual_activity.return_value = [
            {"type": "bullish_sweep", "premium": 500000},
        ]
        
        result = sentiment_engine._analyze_unusual_flow("SPY")
        
        assert result == 1.0  # All bullish
    
    def test_unusual_activity_with_bearish_type(
        self, sentiment_engine, mock_unusual_whales_adapter
    ):
        """Test unusual activity parsing with 'bearish' type."""
        mock_unusual_whales_adapter.get_unusual_activity.return_value = [
            {"type": "bearish_sweep", "premium": 500000},
        ]
        
        result = sentiment_engine._analyze_unusual_flow("SPY")
        
        assert result == -1.0  # All bearish


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for sentiment engine."""
    
    def test_full_analysis_flow(self, sentiment_engine):
        """Test complete analysis flow."""
        timestamp = datetime.now(timezone.utc)
        result = sentiment_engine.run("SPY", timestamp)
        
        # Verify all fields populated
        assert result.symbol == "SPY"
        assert result.timestamp == timestamp
        assert -1.0 <= result.sentiment_score <= 1.0
        assert -1.0 <= result.news_sentiment <= 1.0
        assert -1.0 <= result.flow_sentiment <= 1.0
        assert -1.0 <= result.technical_sentiment <= 1.0
        assert 0.0 <= result.confidence <= 1.0
    
    def test_multiple_symbols(self, sentiment_engine):
        """Test analysis for multiple symbols."""
        timestamp = datetime.now(timezone.utc)
        symbols = ["SPY", "QQQ", "AAPL"]
        
        results = [sentiment_engine.run(symbol, timestamp) for symbol in symbols]
        
        for i, result in enumerate(results):
            assert result.symbol == symbols[i]
            assert isinstance(result, SentimentSnapshot)
    
    def test_different_timestamps(self, sentiment_engine):
        """Test analysis at different timestamps."""
        from datetime import timedelta
        
        base_time = datetime.now(timezone.utc)
        timestamps = [
            base_time,
            base_time - timedelta(hours=1),
            base_time - timedelta(days=1),
        ]
        
        for ts in timestamps:
            result = sentiment_engine.run("SPY", ts)
            assert result.timestamp == ts
    
    def test_with_and_without_unusual_whales(
        self, sentiment_engine, sentiment_engine_no_uw
    ):
        """Test comparison with and without Unusual Whales."""
        timestamp = datetime.now(timezone.utc)
        
        with_uw = sentiment_engine.run("SPY", timestamp)
        without_uw = sentiment_engine_no_uw.run("SPY", timestamp)
        
        # Both should return valid snapshots
        assert isinstance(with_uw, SentimentSnapshot)
        assert isinstance(without_uw, SentimentSnapshot)
        
        # Scores may differ due to UW contribution
        # (This verifies integration path works)


# =============================================================================
# RETAIL SENTIMENT TESTS (PLACEHOLDER)
# =============================================================================

class TestRetailSentiment:
    """Tests for retail sentiment calculation (now uses social media)."""
    
    def test_retail_sentiment_returns_social_media(self, sentiment_engine):
        """Test retail sentiment now returns social media sentiment."""
        result = sentiment_engine._calculate_retail_sentiment("SPY")
        
        # V3.1: Now returns social media sentiment (simulation mode)
        # Should be in valid range (-1 to 1)
        assert -1.0 <= result <= 1.0
    
    def test_retail_sentiment_disabled(self):
        """Test retail sentiment returns zero when social media disabled."""
        engine = SentimentEngineV3(
            processors=[],
            config={"enable_social_media": False},
        )
        
        result = engine._calculate_retail_sentiment("SPY")
        assert result == 0.0
