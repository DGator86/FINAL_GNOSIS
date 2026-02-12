"""Tests for social media sentiment integration.

Tests cover:
- TwitterAdapter (simulated mode)
- RedditAdapter (simulated mode)
- SocialMediaSentimentAggregator
- SentimentEngineV3 social media integration
- Express weights application in ComposerAgentV3

Author: Super Gnosis Elite Trading System
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from engines.sentiment.social_media_adapter import (
    TwitterAdapter,
    RedditAdapter,
    SocialMediaSentimentAggregator,
    SocialPost,
    SocialSentimentResult,
    create_social_media_aggregator,
)
from engines.sentiment.sentiment_engine_v3 import SentimentEngineV3
from agents.composer.composer_agent_v3 import ComposerAgentV3, ComposerDecision
from agents.confidence_builder import TimeframeSignal


class TestSocialPost:
    """Tests for SocialPost dataclass."""

    def test_social_post_creation(self):
        """Test basic SocialPost creation."""
        post = SocialPost(
            platform="twitter",
            content="$AAPL looking bullish! 🚀",
            timestamp=datetime.utcnow(),
            sentiment=0.7,
            engagement=0.5,
        )
        assert post.platform == "twitter"
        assert post.sentiment == 0.7
        assert post.engagement == 0.5
        assert post.author_credibility == 0.5  # Default

    def test_social_post_with_cashtags(self):
        """Test SocialPost with cashtags."""
        post = SocialPost(
            platform="twitter",
            content="Long $AAPL $MSFT",
            timestamp=datetime.utcnow(),
            sentiment=0.6,
            engagement=0.4,
            cashtags=["$AAPL", "$MSFT"],
        )
        assert len(post.cashtags) == 2
        assert "$AAPL" in post.cashtags

    def test_reddit_post_with_subreddit(self):
        """Test Reddit post with subreddit."""
        post = SocialPost(
            platform="reddit",
            content="YOLO on $GME",
            timestamp=datetime.utcnow(),
            sentiment=0.9,
            engagement=0.8,
            subreddit="wallstreetbets",
        )
        assert post.platform == "reddit"
        assert post.subreddit == "wallstreetbets"


class TestTwitterAdapter:
    """Tests for TwitterAdapter."""

    def test_init_simulation_mode(self):
        """Test initialization in simulation mode (no API key)."""
        adapter = TwitterAdapter()
        assert adapter.is_available() is False

    def test_get_posts_simulated(self):
        """Test getting simulated posts."""
        adapter = TwitterAdapter()
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        posts = adapter.get_posts("AAPL", start_time, end_time, limit=5)
        
        assert len(posts) > 0
        assert all(p.platform == "twitter" for p in posts)
        assert all("$AAPL" in p.cashtags for p in posts)

    def test_sentiment_analysis_bullish(self):
        """Test bullish sentiment detection."""
        adapter = TwitterAdapter()
        
        sentiment = adapter._analyze_text_sentiment("$AAPL is going to moon! 🚀 Bullish!")
        assert sentiment > 0.3  # Should be bullish

    def test_sentiment_analysis_bearish(self):
        """Test bearish sentiment detection."""
        adapter = TwitterAdapter()
        
        sentiment = adapter._analyze_text_sentiment("$AAPL crash incoming, bearish puts")
        assert sentiment < -0.3  # Should be bearish

    def test_sentiment_analysis_neutral(self):
        """Test neutral sentiment detection."""
        adapter = TwitterAdapter()
        
        sentiment = adapter._analyze_text_sentiment("What time does market open?")
        assert -0.3 <= sentiment <= 0.3  # Should be neutral

    def test_cashtag_extraction(self):
        """Test cashtag extraction."""
        adapter = TwitterAdapter()
        
        cashtags = adapter._extract_cashtags("Long $AAPL $MSFT, watching $GOOGL")
        assert "$AAPL" not in cashtags  # Method returns without $
        assert "AAPL" in cashtags
        assert "MSFT" in cashtags
        assert "GOOGL" in cashtags


class TestRedditAdapter:
    """Tests for RedditAdapter."""

    def test_init_simulation_mode(self):
        """Test initialization in simulation mode."""
        adapter = RedditAdapter()
        assert adapter.is_available() is False

    def test_default_subreddits(self):
        """Test default subreddits are set."""
        adapter = RedditAdapter()
        assert "wallstreetbets" in adapter.subreddits
        assert "stocks" in adapter.subreddits

    def test_get_posts_simulated(self):
        """Test getting simulated Reddit posts."""
        adapter = RedditAdapter()
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=2)
        
        posts = adapter.get_posts("SPY", start_time, end_time, limit=5)
        
        assert len(posts) > 0
        assert all(p.platform == "reddit" for p in posts)
        assert all(p.subreddit is not None for p in posts)

    def test_sentiment_analysis(self):
        """Test Reddit sentiment analysis."""
        adapter = RedditAdapter()
        
        bullish = adapter._analyze_text_sentiment("YOLO $GME to the moon 🚀🚀🚀")
        assert bullish > 0

        bearish = adapter._analyze_text_sentiment("Puts are printing, this stock is dead")
        assert bearish < 0


class TestSocialMediaSentimentAggregator:
    """Tests for SocialMediaSentimentAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator with simulated adapters."""
        return create_social_media_aggregator()

    def test_init_with_default_weights(self, aggregator):
        """Test initialization with default weights."""
        assert aggregator.twitter_weight == 0.4
        assert aggregator.reddit_weight == 0.6

    def test_get_sentiment(self, aggregator):
        """Test getting aggregated sentiment."""
        result = aggregator.get_sentiment("AAPL", lookback_hours=24)
        
        assert isinstance(result, SocialSentimentResult)
        assert result.symbol == "AAPL"
        assert -1.0 <= result.overall_sentiment <= 1.0
        assert result.post_count > 0
        assert 0.0 <= result.confidence <= 1.0

    def test_sentiment_distribution(self, aggregator):
        """Test sentiment distribution counts."""
        result = aggregator.get_sentiment("TSLA", lookback_hours=24)
        
        total = result.bullish_count + result.bearish_count + result.neutral_count
        assert total == result.post_count

    def test_twitter_only(self):
        """Test with only Twitter adapter."""
        twitter = TwitterAdapter()
        aggregator = SocialMediaSentimentAggregator(
            twitter_adapter=twitter,
            reddit_adapter=None,
        )
        
        result = aggregator.get_sentiment("NVDA")
        assert result.twitter_sentiment is not None
        assert result.reddit_sentiment is None

    def test_reddit_only(self):
        """Test with only Reddit adapter."""
        reddit = RedditAdapter()
        aggregator = SocialMediaSentimentAggregator(
            twitter_adapter=None,
            reddit_adapter=reddit,
        )
        
        result = aggregator.get_sentiment("GME")
        assert result.twitter_sentiment is None
        assert result.reddit_sentiment is not None


class TestSentimentEngineV3Integration:
    """Tests for SentimentEngineV3 social media integration."""

    @pytest.fixture
    def engine(self):
        """Create SentimentEngineV3 with social media."""
        return SentimentEngineV3(
            processors=[],
            unusual_whales_adapter=None,
            config={"enable_social_media": True},
        )

    def test_social_media_enabled(self, engine):
        """Test social media aggregator is created."""
        assert engine.social_media_aggregator is not None

    def test_run_includes_social_sentiment(self, engine):
        """Test run method includes social sentiment."""
        result = engine.run("AAPL", datetime.utcnow())
        
        assert result.metadata is not None
        assert "social_sentiment" in result.metadata
        assert "social_confidence" in result.metadata

    def test_get_social_media_details(self, engine):
        """Test getting detailed social media breakdown."""
        details = engine.get_social_media_details("TSLA")
        
        assert details is not None
        assert "symbol" in details
        assert "twitter_sentiment" in details
        assert "reddit_sentiment" in details
        assert "sample_posts" in details

    def test_social_media_disabled(self):
        """Test engine works with social media disabled."""
        engine = SentimentEngineV3(
            processors=[],
            config={"enable_social_media": False},
        )
        
        result = engine.run("AAPL", datetime.utcnow())
        
        # Should still work, just without social sentiment
        assert result.sentiment_score is not None


class TestComposerAgentV3ExpressWeights:
    """Tests for ComposerAgentV3 express weights functionality."""

    @pytest.fixture
    def composer(self):
        """Create ComposerAgentV3."""
        return ComposerAgentV3()

    def test_express_weights_defined(self, composer):
        """Test express weights are defined."""
        assert "0dte" in composer.express_weights
        assert "cheap_call" in composer.express_weights

    def test_0dte_weights(self, composer):
        """Test 0DTE emphasizes liquidity."""
        weights = composer.express_weights["0dte"]
        assert weights["liquidity"] > weights["hedge"]
        assert weights["liquidity"] > weights["sentiment"]

    def test_cheap_call_weights(self, composer):
        """Test cheap_call emphasizes sentiment."""
        weights = composer.express_weights["cheap_call"]
        assert weights["sentiment"] > weights["hedge"]
        assert weights["sentiment"] > weights["liquidity"]

    def test_apply_express_weights_standard(self, composer):
        """Test standard strategy doesn't modify weights."""
        signals = [
            TimeframeSignal(
                timeframe="5Min",
                direction=0.5,
                strength=0.7,
                confidence=0.8,
                reasoning="Liquidity signal - good volume",
            ),
        ]
        
        result = composer._apply_express_weights(signals, "standard")
        
        # Should return original signals
        assert result[0].strength == signals[0].strength

    def test_apply_express_weights_0dte(self, composer):
        """Test 0DTE strategy modifies liquidity weight."""
        signals = [
            TimeframeSignal(
                timeframe="1Min",
                direction=0.6,
                strength=0.5,
                confidence=0.8,
                reasoning="Liquidity signal - high volume",
            ),
        ]
        
        result = composer._apply_express_weights(signals, "0dte")
        
        # Liquidity should be boosted for 0DTE
        # 0DTE liquidity weight (0.5) / standard (0.2) = 2.5x
        # But capped at 1.0
        assert result[0].strength >= signals[0].strength

    def test_detect_agent_source_hedge(self, composer):
        """Test hedge agent detection."""
        signal = TimeframeSignal(
            timeframe="5Min",
            direction=0.5,
            strength=0.7,
            confidence=0.8,
            reasoning="Gamma exposure is negative, delta skew bullish",
        )
        
        source = composer._detect_agent_source(signal)
        assert source == "hedge"

    def test_detect_agent_source_liquidity(self, composer):
        """Test liquidity agent detection."""
        signal = TimeframeSignal(
            timeframe="5Min",
            direction=0.5,
            strength=0.7,
            confidence=0.8,
            reasoning="Volume above average, tight spread, good depth",
        )
        
        source = composer._detect_agent_source(signal)
        assert source == "liquidity"

    def test_detect_agent_source_sentiment(self, composer):
        """Test sentiment agent detection."""
        signal = TimeframeSignal(
            timeframe="15Min",
            direction=0.7,
            strength=0.8,
            confidence=0.9,
            reasoning="News sentiment bullish, positive flow",
        )
        
        source = composer._detect_agent_source(signal)
        assert source == "sentiment"

    def test_compose_multiframe_with_express(self, composer):
        """Test compose_multiframe with express strategy."""
        signals = [
            TimeframeSignal("1Min", 0.6, 0.7, 0.8, "Liquidity signal"),
            TimeframeSignal("5Min", 0.5, 0.6, 0.7, "Sentiment bullish"),
        ]
        
        decision = composer.compose_multiframe(
            all_timeframe_signals=signals,
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            current_price=150.0,
            strategy_source="0dte",
        )
        
        assert isinstance(decision, ComposerDecision)
        assert decision.strategy_source == "0dte"
        assert decision.predicted_timeframe == "1Min"  # Forced for 0DTE


class TestEngineFactorySentiment:
    """Tests for EngineFactory sentiment initialization."""

    def test_create_sentiment_engine_v1(self):
        """Test creating V1 sentiment engine."""
        from engines.engine_factory import EngineFactory
        
        factory = EngineFactory({})
        engine = factory.create_sentiment_engine(version="v1")
        
        from engines.sentiment import SentimentEngineV1
        assert isinstance(engine, SentimentEngineV1)

    def test_create_sentiment_engine_v3(self):
        """Test creating V3 sentiment engine."""
        from engines.engine_factory import EngineFactory
        
        factory = EngineFactory({})
        engine = factory.create_sentiment_engine(version="v3")
        
        assert isinstance(engine, SentimentEngineV3)
        assert engine.social_media_aggregator is not None

    def test_processors_initialized(self):
        """Test sentiment processors are initialized."""
        from engines.engine_factory import EngineFactory
        
        factory = EngineFactory({})
        engine = factory.create_sentiment_engine(version="v3")
        
        # Should have at least technical processor
        assert len(engine.processors) >= 1


class TestFactoryFunction:
    """Tests for create_social_media_aggregator factory function."""

    def test_create_with_no_credentials(self):
        """Test creating aggregator without credentials."""
        aggregator = create_social_media_aggregator()
        
        assert aggregator is not None
        assert aggregator.twitter_adapter is not None
        assert aggregator.reddit_adapter is not None

    def test_create_with_custom_config(self):
        """Test creating aggregator with custom config."""
        config = {
            "twitter_weight": 0.3,
            "reddit_weight": 0.7,
        }
        
        aggregator = create_social_media_aggregator(config=config)
        
        assert aggregator.twitter_weight == 0.3
        assert aggregator.reddit_weight == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
