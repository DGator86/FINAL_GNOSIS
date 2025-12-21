"""Sentiment engine package."""

from engines.sentiment.sentiment_engine_v1 import SentimentEngineV1
from engines.sentiment.sentiment_engine_v2 import SentimentEngineV2
from engines.sentiment.sentiment_engine_v3 import SentimentEngineV3
from engines.sentiment.social_media_adapter import (
    SocialMediaSentimentAggregator,
    TwitterAdapter,
    RedditAdapter,
    SocialPost,
    SocialSentimentResult,
    create_social_media_aggregator,
)
from engines.sentiment.processors import (
    NewsSentimentProcessor,
    FlowSentimentProcessor,
    TechnicalSentimentProcessor,
)

__all__ = [
    "SentimentEngineV1",
    "SentimentEngineV2",
    "SentimentEngineV3",
    "SocialMediaSentimentAggregator",
    "TwitterAdapter",
    "RedditAdapter",
    "SocialPost",
    "SocialSentimentResult",
    "create_social_media_aggregator",
    "NewsSentimentProcessor",
    "FlowSentimentProcessor",
    "TechnicalSentimentProcessor",
]
