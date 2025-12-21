"""Social Media Sentiment Adapter - Twitter/X and Reddit integration.

Provides sentiment analysis from social media sources:
- Twitter/X API for real-time market sentiment
- Reddit API for retail investor sentiment (WallStreetBets, stocks, etc.)

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class SocialPost:
    """Represents a social media post with sentiment."""
    platform: str  # "twitter", "reddit"
    content: str
    timestamp: datetime
    sentiment: float  # -1.0 to 1.0
    engagement: float  # Normalized engagement score (likes, retweets, upvotes)
    author_credibility: float = 0.5  # 0.0 to 1.0
    cashtags: List[str] = field(default_factory=list)  # $AAPL, $SPY, etc.
    subreddit: Optional[str] = None  # For Reddit posts


@dataclass
class SocialSentimentResult:
    """Aggregated social media sentiment result."""
    symbol: str
    timestamp: datetime
    overall_sentiment: float  # -1.0 to 1.0
    twitter_sentiment: Optional[float] = None
    reddit_sentiment: Optional[float] = None
    post_count: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    engagement_weighted_sentiment: float = 0.0
    confidence: float = 0.0
    trending: bool = False
    posts: List[SocialPost] = field(default_factory=list)


class BaseSocialMediaAdapter(ABC):
    """Base class for social media adapters."""
    
    @abstractmethod
    def get_posts(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[SocialPost]:
        """Fetch posts mentioning the symbol."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the adapter is configured and available."""
        pass


class TwitterAdapter(BaseSocialMediaAdapter):
    """
    Twitter/X API adapter for sentiment analysis.
    
    Requires Twitter API v2 credentials (Bearer Token).
    Falls back to simulated data if not configured.
    """
    
    # Bullish keywords
    BULLISH_KEYWORDS = [
        'bullish', 'moon', 'buy', 'calls', 'long', 'breakout', 'squeeze',
        'rocket', '🚀', 'green', 'pump', 'rip', 'ath', 'all time high',
        'upgrade', 'beat', 'strong', 'growth', 'bull', 'yolo',
    ]
    
    # Bearish keywords
    BEARISH_KEYWORDS = [
        'bearish', 'crash', 'sell', 'puts', 'short', 'breakdown', 'dump',
        'red', 'tank', 'dead', 'drill', 'downgrade', 'miss', 'weak',
        'bear', 'fade', 'drop', 'falling', 'overvalued',
    ]
    
    def __init__(
        self,
        bearer_token: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Twitter adapter.
        
        Args:
            bearer_token: Twitter API v2 Bearer Token
            config: Additional configuration
        """
        self.bearer_token = bearer_token
        self.config = config or {}
        self._client = None
        
        if bearer_token:
            try:
                import tweepy
                self._client = tweepy.Client(bearer_token=bearer_token)
                logger.info("TwitterAdapter initialized with API credentials")
            except ImportError:
                logger.warning("tweepy not installed - Twitter API unavailable")
            except Exception as e:
                logger.warning(f"Failed to initialize Twitter client: {e}")
        else:
            logger.info("TwitterAdapter initialized in simulation mode (no API key)")
    
    def is_available(self) -> bool:
        """Check if Twitter API is available."""
        return self._client is not None
    
    def get_posts(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[SocialPost]:
        """
        Fetch tweets mentioning the symbol.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_time: Start of time window
            end_time: End of time window
            limit: Maximum posts to fetch
            
        Returns:
            List of SocialPost objects
        """
        if self._client:
            return self._fetch_real_tweets(symbol, start_time, end_time, limit)
        else:
            return self._generate_simulated_tweets(symbol, start_time, end_time, limit)
    
    def _fetch_real_tweets(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int,
    ) -> List[SocialPost]:
        """Fetch real tweets from Twitter API."""
        posts = []
        
        try:
            # Search for cashtag
            query = f"${symbol} -is:retweet lang:en"
            
            response = self._client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                start_time=start_time.isoformat() + "Z",
                end_time=end_time.isoformat() + "Z",
                tweet_fields=["created_at", "public_metrics", "author_id"],
            )
            
            if response.data:
                for tweet in response.data:
                    sentiment = self._analyze_text_sentiment(tweet.text)
                    engagement = self._calculate_engagement(tweet.public_metrics)
                    
                    posts.append(SocialPost(
                        platform="twitter",
                        content=tweet.text,
                        timestamp=tweet.created_at,
                        sentiment=sentiment,
                        engagement=engagement,
                        cashtags=self._extract_cashtags(tweet.text),
                    ))
            
            logger.debug(f"Fetched {len(posts)} tweets for ${symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching tweets for {symbol}: {e}")
        
        return posts
    
    def _generate_simulated_tweets(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int,
    ) -> List[SocialPost]:
        """Generate simulated tweet data for testing."""
        import random
        
        posts = []
        templates = [
            (f"${symbol} looking bullish today! 🚀", 0.7),
            (f"Just bought more ${symbol} calls. Moon soon!", 0.8),
            (f"${symbol} breaking out! Let's go!", 0.6),
            (f"Bearish on ${symbol}, selling my position", -0.6),
            (f"${symbol} is overvalued, puts printing 🐻", -0.7),
            (f"${symbol} chart looking weak, might dump", -0.5),
            (f"Holding ${symbol} for the long term", 0.3),
            (f"What do you think about ${symbol}?", 0.0),
            (f"${symbol} earnings coming up, could go either way", 0.0),
            (f"${symbol} dip buying opportunity? 🤔", 0.2),
        ]
        
        for i in range(min(limit, 10)):
            template, sentiment = random.choice(templates)
            # Add some noise to sentiment
            sentiment += random.uniform(-0.15, 0.15)
            sentiment = max(-1.0, min(1.0, sentiment))
            
            posts.append(SocialPost(
                platform="twitter",
                content=template,
                timestamp=start_time + timedelta(minutes=random.randint(0, 60)),
                sentiment=sentiment,
                engagement=random.uniform(0.1, 0.8),
                cashtags=[f"${symbol}"],
            ))
        
        return posts
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of tweet text using keyword matching.
        
        Returns:
            Sentiment score from -1.0 to 1.0
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        sentiment = (bullish_count - bearish_count) / total
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_engagement(self, metrics: Optional[Dict]) -> float:
        """Calculate normalized engagement score."""
        if not metrics:
            return 0.0
        
        likes = metrics.get("like_count", 0)
        retweets = metrics.get("retweet_count", 0)
        replies = metrics.get("reply_count", 0)
        
        # Weighted engagement
        score = likes + (retweets * 2) + (replies * 1.5)
        
        # Normalize (assuming max engagement ~1000)
        return min(1.0, score / 1000)
    
    def _extract_cashtags(self, text: str) -> List[str]:
        """Extract cashtags from text."""
        return re.findall(r'\$([A-Z]{1,5})', text)


class RedditAdapter(BaseSocialMediaAdapter):
    """
    Reddit API adapter for sentiment analysis.
    
    Monitors subreddits like r/wallstreetbets, r/stocks, r/options.
    Requires Reddit API credentials (client_id, client_secret).
    """
    
    DEFAULT_SUBREDDITS = [
        "wallstreetbets",
        "stocks",
        "options",
        "investing",
        "stockmarket",
    ]
    
    # Same keywords as Twitter
    BULLISH_KEYWORDS = TwitterAdapter.BULLISH_KEYWORDS
    BEARISH_KEYWORDS = TwitterAdapter.BEARISH_KEYWORDS
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "GnosisTrading/1.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Reddit adapter.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent for API requests
            config: Additional configuration
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.config = config or {}
        self._reddit = None
        
        self.subreddits = self.config.get("subreddits", self.DEFAULT_SUBREDDITS)
        
        if client_id and client_secret:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                logger.info(f"RedditAdapter initialized with API credentials | subreddits={self.subreddits}")
            except ImportError:
                logger.warning("praw not installed - Reddit API unavailable")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit client: {e}")
        else:
            logger.info("RedditAdapter initialized in simulation mode (no API key)")
    
    def is_available(self) -> bool:
        """Check if Reddit API is available."""
        return self._reddit is not None
    
    def get_posts(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[SocialPost]:
        """
        Fetch Reddit posts mentioning the symbol.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_time: Start of time window
            end_time: End of time window
            limit: Maximum posts to fetch
            
        Returns:
            List of SocialPost objects
        """
        if self._reddit:
            return self._fetch_real_posts(symbol, start_time, end_time, limit)
        else:
            return self._generate_simulated_posts(symbol, start_time, end_time, limit)
    
    def _fetch_real_posts(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int,
    ) -> List[SocialPost]:
        """Fetch real posts from Reddit API."""
        posts = []
        
        try:
            for subreddit_name in self.subreddits:
                subreddit = self._reddit.subreddit(subreddit_name)
                
                # Search for symbol mentions
                for submission in subreddit.search(
                    f"${symbol} OR {symbol}",
                    sort="new",
                    time_filter="day",
                    limit=limit // len(self.subreddits),
                ):
                    post_time = datetime.fromtimestamp(submission.created_utc)
                    
                    if start_time <= post_time <= end_time:
                        content = f"{submission.title} {submission.selftext}"
                        sentiment = self._analyze_text_sentiment(content)
                        engagement = self._calculate_engagement(submission)
                        
                        posts.append(SocialPost(
                            platform="reddit",
                            content=content[:500],  # Truncate long posts
                            timestamp=post_time,
                            sentiment=sentiment,
                            engagement=engagement,
                            subreddit=subreddit_name,
                            cashtags=self._extract_tickers(content),
                        ))
            
            logger.debug(f"Fetched {len(posts)} Reddit posts for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts for {symbol}: {e}")
        
        return posts
    
    def _generate_simulated_posts(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int,
    ) -> List[SocialPost]:
        """Generate simulated Reddit data for testing."""
        import random
        
        posts = []
        templates = [
            (f"DD on ${symbol} - This is going to moon! 🚀🚀🚀", "wallstreetbets", 0.9),
            (f"YOLO: Just put my entire portfolio into ${symbol} calls", "wallstreetbets", 0.85),
            (f"Why I'm bearish on ${symbol}", "stocks", -0.6),
            (f"${symbol} technical analysis - support holding", "stocks", 0.4),
            (f"Is ${symbol} a good long-term investment?", "investing", 0.1),
            (f"${symbol} puts for earnings?", "options", -0.3),
            (f"Diamond hands on ${symbol} 💎🙌", "wallstreetbets", 0.7),
            (f"${symbol} - undervalued gem or value trap?", "stocks", 0.0),
        ]
        
        for i in range(min(limit, 8)):
            template, subreddit, sentiment = random.choice(templates)
            sentiment += random.uniform(-0.15, 0.15)
            sentiment = max(-1.0, min(1.0, sentiment))
            
            posts.append(SocialPost(
                platform="reddit",
                content=template,
                timestamp=start_time + timedelta(minutes=random.randint(0, 120)),
                sentiment=sentiment,
                engagement=random.uniform(0.1, 0.9),
                subreddit=subreddit,
                cashtags=[f"${symbol}"],
            ))
        
        return posts
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of post text."""
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return max(-1.0, min(1.0, (bullish_count - bearish_count) / total))
    
    def _calculate_engagement(self, submission) -> float:
        """Calculate normalized engagement score."""
        upvotes = getattr(submission, "score", 0)
        comments = getattr(submission, "num_comments", 0)
        
        # Weighted engagement
        score = upvotes + (comments * 2)
        
        # Normalize (assuming max engagement ~5000)
        return min(1.0, score / 5000)
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        return re.findall(r'\$([A-Z]{1,5})', text)


class SocialMediaSentimentAggregator:
    """
    Aggregates sentiment from multiple social media sources.
    
    Combines Twitter and Reddit sentiment with configurable weights.
    """
    
    def __init__(
        self,
        twitter_adapter: Optional[TwitterAdapter] = None,
        reddit_adapter: Optional[RedditAdapter] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize aggregator.
        
        Args:
            twitter_adapter: Twitter sentiment adapter
            reddit_adapter: Reddit sentiment adapter
            config: Configuration including weights
        """
        self.twitter_adapter = twitter_adapter
        self.reddit_adapter = reddit_adapter
        self.config = config or {}
        
        # Default weights
        self.twitter_weight = self.config.get("twitter_weight", 0.4)
        self.reddit_weight = self.config.get("reddit_weight", 0.6)  # WSB can move markets
        
        logger.info(
            f"SocialMediaSentimentAggregator initialized | "
            f"twitter_weight={self.twitter_weight}, reddit_weight={self.reddit_weight}"
        )
    
    def get_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24,
    ) -> SocialSentimentResult:
        """
        Get aggregated social media sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_hours: Hours of history to analyze
            
        Returns:
            SocialSentimentResult with combined sentiment
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        all_posts = []
        twitter_sentiment = None
        reddit_sentiment = None
        
        # Fetch Twitter posts
        if self.twitter_adapter:
            twitter_posts = self.twitter_adapter.get_posts(symbol, start_time, end_time)
            all_posts.extend(twitter_posts)
            
            if twitter_posts:
                twitter_sentiment = self._calculate_weighted_sentiment(twitter_posts)
                logger.debug(f"Twitter sentiment for {symbol}: {twitter_sentiment:.2f} ({len(twitter_posts)} posts)")
        
        # Fetch Reddit posts
        if self.reddit_adapter:
            reddit_posts = self.reddit_adapter.get_posts(symbol, start_time, end_time)
            all_posts.extend(reddit_posts)
            
            if reddit_posts:
                reddit_sentiment = self._calculate_weighted_sentiment(reddit_posts)
                logger.debug(f"Reddit sentiment for {symbol}: {reddit_sentiment:.2f} ({len(reddit_posts)} posts)")
        
        # Calculate overall sentiment
        overall_sentiment = self._combine_sentiments(twitter_sentiment, reddit_sentiment)
        
        # Calculate engagement-weighted sentiment
        engagement_weighted = self._calculate_weighted_sentiment(all_posts) if all_posts else 0.0
        
        # Count sentiment distribution
        bullish_count = sum(1 for p in all_posts if p.sentiment > 0.2)
        bearish_count = sum(1 for p in all_posts if p.sentiment < -0.2)
        neutral_count = len(all_posts) - bullish_count - bearish_count
        
        # Calculate confidence based on post volume and agreement
        confidence = self._calculate_confidence(all_posts, overall_sentiment)
        
        # Check if symbol is trending (high volume of posts)
        trending = len(all_posts) > 20  # Arbitrary threshold
        
        result = SocialSentimentResult(
            symbol=symbol,
            timestamp=end_time,
            overall_sentiment=overall_sentiment,
            twitter_sentiment=twitter_sentiment,
            reddit_sentiment=reddit_sentiment,
            post_count=len(all_posts),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            engagement_weighted_sentiment=engagement_weighted,
            confidence=confidence,
            trending=trending,
            posts=all_posts[:50],  # Keep top 50 posts
        )
        
        logger.info(
            f"Social sentiment for {symbol}: {overall_sentiment:.2f} | "
            f"posts={len(all_posts)} | confidence={confidence:.2f} | "
            f"trending={trending}"
        )
        
        return result
    
    def _calculate_weighted_sentiment(self, posts: List[SocialPost]) -> float:
        """Calculate engagement-weighted sentiment."""
        if not posts:
            return 0.0
        
        total_weight = 0.0
        weighted_sentiment = 0.0
        
        for post in posts:
            # Weight by engagement and author credibility
            weight = post.engagement * post.author_credibility
            weight = max(0.1, weight)  # Minimum weight
            
            weighted_sentiment += post.sentiment * weight
            total_weight += weight
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _combine_sentiments(
        self,
        twitter_sentiment: Optional[float],
        reddit_sentiment: Optional[float],
    ) -> float:
        """Combine Twitter and Reddit sentiments with weights."""
        if twitter_sentiment is None and reddit_sentiment is None:
            return 0.0
        
        if twitter_sentiment is None:
            return reddit_sentiment
        
        if reddit_sentiment is None:
            return twitter_sentiment
        
        # Weighted combination
        total_weight = self.twitter_weight + self.reddit_weight
        combined = (
            twitter_sentiment * self.twitter_weight +
            reddit_sentiment * self.reddit_weight
        ) / total_weight
        
        return combined
    
    def _calculate_confidence(
        self,
        posts: List[SocialPost],
        overall_sentiment: float,
    ) -> float:
        """Calculate confidence based on volume and agreement."""
        if not posts:
            return 0.0
        
        # Volume factor (more posts = higher confidence, up to a point)
        volume_factor = min(1.0, len(posts) / 50)
        
        # Agreement factor (how much posts agree with overall sentiment)
        if overall_sentiment != 0:
            agreement_scores = []
            for post in posts:
                # Posts agreeing with overall direction increase confidence
                if (post.sentiment > 0 and overall_sentiment > 0) or \
                   (post.sentiment < 0 and overall_sentiment < 0):
                    agreement_scores.append(1.0)
                elif abs(post.sentiment) < 0.2:  # Neutral
                    agreement_scores.append(0.5)
                else:
                    agreement_scores.append(0.0)
            
            agreement_factor = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5
        else:
            agreement_factor = 0.5
        
        # Combine factors
        confidence = (volume_factor * 0.4) + (agreement_factor * 0.6)
        
        return min(1.0, confidence)


# Factory function for easy creation
def create_social_media_aggregator(
    twitter_bearer_token: Optional[str] = None,
    reddit_client_id: Optional[str] = None,
    reddit_client_secret: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SocialMediaSentimentAggregator:
    """
    Create a configured SocialMediaSentimentAggregator.
    
    Args:
        twitter_bearer_token: Twitter API bearer token
        reddit_client_id: Reddit API client ID
        reddit_client_secret: Reddit API client secret
        config: Additional configuration
        
    Returns:
        Configured aggregator (with simulation mode if no credentials)
    """
    twitter_adapter = TwitterAdapter(bearer_token=twitter_bearer_token, config=config)
    reddit_adapter = RedditAdapter(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        config=config,
    )
    
    return SocialMediaSentimentAggregator(
        twitter_adapter=twitter_adapter,
        reddit_adapter=reddit_adapter,
        config=config,
    )


__all__ = [
    "SocialPost",
    "SocialSentimentResult",
    "TwitterAdapter",
    "RedditAdapter",
    "SocialMediaSentimentAggregator",
    "create_social_media_aggregator",
]
