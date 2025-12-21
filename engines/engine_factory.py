"""Engine Factory - centralized initialization of all engines for OpportunityScanner.

Provides factory methods for creating properly initialized trading engines
with full sentiment processor integration including social media.

Author: Super Gnosis Elite Trading System
Version: 2.0.0 - Added full sentiment processor initialization
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from loguru import logger

from engines.elasticity import ElasticityEngineV1
from engines.hedge import HedgeEngineV3
from engines.inputs.adapter_factory import (
    create_market_data_adapter,
    create_options_adapter,
)
from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.options_chain_adapter import OptionsChainAdapter
from engines.liquidity import LiquidityEngineV1
from engines.scanner import OpportunityScanner
from engines.sentiment import (
    SentimentEngineV1,
    SentimentEngineV3,
    NewsSentimentProcessor,
    FlowSentimentProcessor,
    TechnicalSentimentProcessor,
    create_social_media_aggregator,
)


class EngineFactory:
    """Factory for creating properly initialized trading engines.
    
    V2.0 Features:
    - Full sentiment processor initialization
    - Social media integration (Twitter/Reddit)
    - Configurable engine versions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize engine factory with configuration.
        
        Args:
            config: Full system configuration dictionary
        """
        self.config = config
        self._market_adapter: Optional[MarketDataAdapter] = None
        self._options_adapter: Optional[OptionsChainAdapter] = None
        logger.info("EngineFactory initialized")
    
    def create_market_adapter(self) -> MarketDataAdapter:
        """Create market data adapter."""
        return create_market_data_adapter(prefer_real=True)
    
    def create_options_adapter(self) -> OptionsChainAdapter:
        """Create options chain adapter."""
        return create_options_adapter(prefer_real=True)
    
    def create_hedge_engine(self, options_adapter: OptionsChainAdapter) -> HedgeEngineV3:
        """Create hedge engine with proper configuration.
        
        Args:
            options_adapter: Options chain data provider
            
        Returns:
            Initialized HedgeEngineV3
        """
        hedge_config = self.config.get('engines', {}).get('hedge', {})
        return HedgeEngineV3(options_adapter=options_adapter, config=hedge_config)
    
    def create_liquidity_engine(self, market_adapter: MarketDataAdapter) -> LiquidityEngineV1:
        """Create liquidity engine with proper configuration.
        
        Args:
            market_adapter: Market data provider
            
        Returns:
            Initialized LiquidityEngineV1
        """
        liquidity_config = self.config.get('engines', {}).get('liquidity', {})
        return LiquidityEngineV1(market_adapter=market_adapter, config=liquidity_config)
    
    def create_sentiment_engine(
        self,
        market_adapter: Optional[MarketDataAdapter] = None,
        version: str = "v3",
    ) -> Union[SentimentEngineV1, SentimentEngineV3]:
        """Create sentiment engine with full processor initialization.
        
        V2.0: Now initializes all sentiment processors:
        - NewsSentimentProcessor (news sentiment from news adapter)
        - FlowSentimentProcessor (options flow sentiment)
        - TechnicalSentimentProcessor (technical analysis sentiment)
        - Social Media (Twitter/Reddit via SentimentEngineV3)
        
        Args:
            market_adapter: Market data provider (for technical processor)
            version: Engine version ("v1" or "v3", default "v3")
            
        Returns:
            Initialized SentimentEngine (V1 or V3)
        """
        sentiment_config = self.config.get('engines', {}).get('sentiment', {})
        
        # Get or create market adapter for technical processor
        if market_adapter is None:
            market_adapter = self.create_market_adapter()
        
        # Initialize sentiment processors
        processors = self._create_sentiment_processors(market_adapter, sentiment_config)
        
        if version == "v3":
            # V3: Full-featured engine with social media
            return self._create_sentiment_engine_v3(processors, sentiment_config)
        else:
            # V1: Basic engine
            return SentimentEngineV1(processors=processors, config=sentiment_config)
    
    def _create_sentiment_processors(
        self,
        market_adapter: MarketDataAdapter,
        config: Dict[str, Any],
    ) -> List[Any]:
        """Create list of sentiment processors.
        
        Args:
            market_adapter: Market data provider
            config: Sentiment engine configuration
            
        Returns:
            List of initialized sentiment processors
        """
        processors = []
        
        # 1. News Sentiment Processor
        try:
            from engines.inputs.news_adapter import NewsAdapter
            news_adapter = self._create_news_adapter(config)
            if news_adapter:
                news_processor = NewsSentimentProcessor(
                    news_adapter=news_adapter,
                    config=config.get("news", {}),
                )
                processors.append(news_processor)
                logger.debug("NewsSentimentProcessor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NewsSentimentProcessor: {e}")
        
        # 2. Flow Sentiment Processor
        try:
            flow_adapter = self._create_flow_adapter(config)
            flow_processor = FlowSentimentProcessor(
                config=config.get("flow", {}),
                flow_adapter=flow_adapter,
            )
            processors.append(flow_processor)
            logger.debug("FlowSentimentProcessor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize FlowSentimentProcessor: {e}")
        
        # 3. Technical Sentiment Processor
        try:
            technical_processor = TechnicalSentimentProcessor(
                market_adapter=market_adapter,
                config=config.get("technical", {}),
            )
            processors.append(technical_processor)
            logger.debug("TechnicalSentimentProcessor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize TechnicalSentimentProcessor: {e}")
        
        logger.info(f"Initialized {len(processors)} sentiment processors")
        return processors
    
    def _create_sentiment_engine_v3(
        self,
        processors: List[Any],
        config: Dict[str, Any],
    ) -> SentimentEngineV3:
        """Create SentimentEngineV3 with all features.
        
        Args:
            processors: List of sentiment processors
            config: Sentiment configuration
            
        Returns:
            Initialized SentimentEngineV3
        """
        # Try to get Unusual Whales adapter
        unusual_whales_adapter = None
        try:
            from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
            uw_config = config.get("unusual_whales", {})
            if uw_config.get("api_key"):
                unusual_whales_adapter = UnusualWhalesAdapter(
                    api_key=uw_config["api_key"],
                    config=uw_config,
                )
                logger.debug("UnusualWhalesAdapter initialized")
        except Exception as e:
            logger.debug(f"UnusualWhalesAdapter not available: {e}")
        
        # Create social media aggregator
        social_media_aggregator = None
        if config.get("enable_social_media", True):
            social_media_aggregator = create_social_media_aggregator(
                twitter_bearer_token=config.get("twitter_bearer_token"),
                reddit_client_id=config.get("reddit_client_id"),
                reddit_client_secret=config.get("reddit_client_secret"),
                config=config.get("social_media_config"),
            )
            logger.debug("SocialMediaSentimentAggregator initialized")
        
        return SentimentEngineV3(
            processors=processors,
            unusual_whales_adapter=unusual_whales_adapter,
            config=config,
            social_media_aggregator=social_media_aggregator,
        )
    
    def _create_news_adapter(self, config: Dict[str, Any]) -> Optional[Any]:
        """Create news adapter if configured.
        
        Args:
            config: Sentiment configuration
            
        Returns:
            NewsAdapter or None
        """
        try:
            from engines.inputs.news_adapter import NewsAdapter
            news_config = config.get("news", {})
            
            # Check if news adapter is configured
            if news_config.get("enabled", False):
                return NewsAdapter(config=news_config)
        except ImportError:
            logger.debug("NewsAdapter not available")
        except Exception as e:
            logger.debug(f"Failed to create NewsAdapter: {e}")
        
        return None
    
    def _create_flow_adapter(self, config: Dict[str, Any]) -> Optional[Any]:
        """Create flow adapter if configured.
        
        Args:
            config: Sentiment configuration
            
        Returns:
            Flow adapter or None
        """
        try:
            # Try Unusual Whales first
            uw_config = config.get("unusual_whales", {})
            if uw_config.get("api_key"):
                from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
                return UnusualWhalesAdapter(
                    api_key=uw_config["api_key"],
                    config=uw_config,
                )
        except Exception as e:
            logger.debug(f"UnusualWhalesAdapter not available for flow: {e}")
        
        return None
    
    def create_elasticity_engine(self, market_adapter: MarketDataAdapter) -> ElasticityEngineV1:
        """Create elasticity engine with proper configuration.
        
        Args:
            market_adapter: Market data provider
            
        Returns:
            Initialized ElasticityEngineV1
        """
        elasticity_config = self.config.get('engines', {}).get('elasticity', {})
        return ElasticityEngineV1(market_adapter=market_adapter, config=elasticity_config)
    
    def create_scanner(self, sentiment_version: str = "v3") -> OpportunityScanner:
        """Create fully initialized OpportunityScanner with all engines.
        
        V2.0: Uses SentimentEngineV3 by default with full processor initialization.
        
        Args:
            sentiment_version: Sentiment engine version ("v1" or "v3")
        
        Returns:
            OpportunityScanner with all engines initialized
        """
        logger.info("Creating OpportunityScanner with all engines...")
        
        # Create adapters (shared across engines)
        market_adapter = self.create_market_adapter()
        options_adapter = self.create_options_adapter()
        
        # Create engines
        hedge_engine = self.create_hedge_engine(options_adapter)
        liquidity_engine = self.create_liquidity_engine(market_adapter)
        sentiment_engine = self.create_sentiment_engine(
            market_adapter=market_adapter,
            version=sentiment_version,
        )
        elasticity_engine = self.create_elasticity_engine(market_adapter)
        
        # Create scanner
        scanner = OpportunityScanner(
            hedge_engine=hedge_engine,
            liquidity_engine=liquidity_engine,
            sentiment_engine=sentiment_engine,
            elasticity_engine=elasticity_engine,
            options_adapter=options_adapter,
            market_adapter=market_adapter
        )
        
        logger.info(
            f"OpportunityScanner created successfully | "
            f"sentiment_version={sentiment_version}"
        )
        return scanner


# Convenience function for quick factory creation
def create_engine_factory(config: Optional[Dict[str, Any]] = None) -> EngineFactory:
    """Create an EngineFactory with optional configuration.
    
    Args:
        config: Configuration dictionary (uses defaults if not provided)
        
    Returns:
        Configured EngineFactory
    """
    return EngineFactory(config or {})


__all__ = ["EngineFactory", "create_engine_factory"]
