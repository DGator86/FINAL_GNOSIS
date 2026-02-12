"""Engine Factory - centralized initialization of all engines for OpportunityScanner.

Provides factory methods for creating properly initialized trading engines
with full sentiment processor integration including social media.

Author: Super Gnosis Elite Trading System
Version: 6.0.0 - Unified LiquidityEngineV5 with PENTA Methodology

Architecture:
    Data Adapters → Engines → Primary Agents → Composer → Trade Agents → Monitors

Engine Layer:
- Hedge Engine: Dealer flow, gamma, options Greeks
- Sentiment Engine: News, Social Media, Technical Indicators
- Liquidity Engine V5: Market Quality + PENTA Methodology:
  - Wyckoff (VSA, Phase Tracking, Event Detection)
  - ICT (FVGs, Order Blocks, OTE, Liquidity Sweeps)
  - Order Flow (Footprint, CVD, Volume Profile)
  - Supply/Demand (Economic principle-based zone detection)
  - Liquidity Concepts (Pools, Voids, Strong/Weak Swings, Inducements)
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
from engines.liquidity import (
    LiquidityEngineV1, 
    LiquidityEngineV4,
    LiquidityEngineV5,  # Unified PENTA engine
    LiquidityEngineV5Snapshot,
    PENTAState,
    ICTEngine, 
    OrderFlowEngine, 
    create_order_flow_engine,
    SupplyDemandEngine,
    create_supply_demand_engine,
    LiquidityConceptsEngine,
    create_liquidity_concepts_engine,
)
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
    
    V5.0 Features:
    - Full sentiment processor initialization
    - Social media integration (Twitter/Reddit)
    - Configurable engine versions
    - ICT (Inner Circle Trader) methodology engine
    - Order Flow analysis (Footprint, CVD, Volume Profile)
    - Supply and Demand zone detection
    - Combined Wyckoff + ICT + Order Flow + S&D analysis support
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
    
    def create_liquidity_engine(
        self,
        market_adapter: MarketDataAdapter,
        options_adapter: Optional[OptionsChainAdapter] = None,
        version: str = "v4",
    ) -> Union[LiquidityEngineV1, LiquidityEngineV4]:
        """Create liquidity engine with proper configuration.
        
        V4 includes full Wyckoff methodology integration:
        - Volume Spread Analysis (VSA)
        - Seven Logical Events Detection
        - Five Phase Tracking
        - Accumulation/Distribution Structure Recognition
        - Spring/Upthrust Detection
        
        Args:
            market_adapter: Market data provider
            options_adapter: Options chain adapter (required for v4)
            version: Engine version ("v1" or "v4", default "v4")
            
        Returns:
            Initialized LiquidityEngine (V1 or V4)
        """
        liquidity_config = self.config.get('engines', {}).get('liquidity', {})
        
        if version == "v4" and options_adapter is not None:
            logger.info("Creating LiquidityEngineV4 with Wyckoff methodology")
            return LiquidityEngineV4(
                market_adapter=market_adapter,
                options_adapter=options_adapter,
                config=liquidity_config,
            )
        else:
            if version == "v4" and options_adapter is None:
                logger.warning("LiquidityEngineV4 requires options_adapter, falling back to V1")
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
    
    def create_ict_engine(self) -> ICTEngine:
        """Create ICT (Inner Circle Trader) methodology engine.
        
        ICT Engine provides:
        - Swing Points & Liquidity Levels
        - Fair Value Gaps (FVG) - BISI/SIBI
        - Order Blocks (High/Low Probability)
        - Premium/Discount Zones & OTE
        - Daily Bias Calculation
        - Liquidity Sweep Detection
        
        Returns:
            Initialized ICTEngine
        """
        ict_config = self.config.get('engines', {}).get('ict', {})
        logger.info("Creating ICTEngine with ICT methodology")
        return ICTEngine(config=ict_config)
    
    def create_order_flow_engine(
        self,
        imbalance_threshold: float = 2.0,
        stacked_min_count: int = 3,
        cvd_smoothing: int = 14,
        value_area_percent: float = 0.70,
    ) -> OrderFlowEngine:
        """Create Order Flow analysis engine.
        
        Order Flow Engine provides:
        - Footprint Analysis (bid/ask aggression, imbalance, absorption)
        - CVD Analysis (cumulative delta, exhaustion, divergence)
        - Volume Profile (POC, Value Area, HVN/LVN)
        - Auction Market Theory integration
        
        Args:
            imbalance_threshold: Ratio for bid/ask imbalance (default 2.0 = 200%)
            stacked_min_count: Minimum consecutive imbalances for stacked pattern
            cvd_smoothing: Period for CVD smoothing
            value_area_percent: Percentage of volume for value area (default 70%)
        
        Returns:
            Initialized OrderFlowEngine
        """
        of_config = self.config.get('engines', {}).get('order_flow', {})
        
        # Override with config values if present
        imbalance_threshold = of_config.get('imbalance_threshold', imbalance_threshold)
        stacked_min_count = of_config.get('stacked_min_count', stacked_min_count)
        cvd_smoothing = of_config.get('cvd_smoothing', cvd_smoothing)
        value_area_percent = of_config.get('value_area_percent', value_area_percent)
        
        logger.info(
            f"Creating OrderFlowEngine with Footprint, CVD, and Volume Profile | "
            f"imbalance_threshold={imbalance_threshold}, value_area={value_area_percent:.0%}"
        )
        return create_order_flow_engine(
            imbalance_threshold=imbalance_threshold,
            stacked_min_count=stacked_min_count,
            cvd_smoothing=cvd_smoothing,
            value_area_percent=value_area_percent,
        )
    
    def create_supply_demand_engine(
        self,
        swing_lookback: int = 3,
        min_swing_distance: int = 3,
        max_zones: int = 10,
        volatility_multiplier: float = 1.5,
        default_risk_reward: float = 3.0,
    ) -> SupplyDemandEngine:
        """Create Supply and Demand zone detection engine.
        
        Supply/Demand Engine provides:
        - Demand Zones (low between two highs, second higher than first)
        - Supply Zones (high between two lows, second lower than first)
        - Zone Strength validation (momentum confirmation)
        - Zone Boundary calculation (volatility shift detection)
        - Zone Status tracking (fresh, tested, retested, broken)
        - Risk Management integration (built-in R:R levels)
        
        Based on economic principles:
        - Law of Demand: Higher price = lower quantity demanded
        - Law of Supply: Higher price = higher quantity supplied
        - Market Equilibrium: Price seeks supply/demand balance
        
        Args:
            swing_lookback: Bars to confirm swing points (default 3)
            min_swing_distance: Minimum bars between swings
            max_zones: Maximum zones to track per type
            volatility_multiplier: For boundary calculation
            default_risk_reward: Default R:R ratio for targets
        
        Returns:
            Initialized SupplyDemandEngine
        """
        sd_config = self.config.get('engines', {}).get('supply_demand', {})
        
        # Override with config values if present
        swing_lookback = sd_config.get('swing_lookback', swing_lookback)
        min_swing_distance = sd_config.get('min_swing_distance', min_swing_distance)
        max_zones = sd_config.get('max_zones', max_zones)
        volatility_multiplier = sd_config.get('volatility_multiplier', volatility_multiplier)
        default_risk_reward = sd_config.get('default_risk_reward', default_risk_reward)
        
        logger.info(
            f"Creating SupplyDemandEngine with zone detection | "
            f"swing_lookback={swing_lookback}, max_zones={max_zones}, R:R={default_risk_reward}"
        )
        return create_supply_demand_engine(
            swing_lookback=swing_lookback,
            min_swing_distance=min_swing_distance,
            max_zones=max_zones,
            volatility_multiplier=volatility_multiplier,
            default_risk_reward=default_risk_reward,
        )
    
    def create_liquidity_concepts_engine(
        self,
        swing_lookback: int = 3,
        cluster_threshold_pct: float = 0.003,
        min_void_size_pct: float = 0.005,
    ) -> LiquidityConceptsEngine:
        """Create Liquidity Concepts analysis engine.
        
        Liquidity Concepts Engine provides:
        - Latent Liquidity Pools (buy-side/sell-side above/below highs/lows)
        - Strong/Weak Swing Classification (based on Break of Structure)
        - Liquidity Voids (areas of shallow depth - price travels easily)
        - Fractal Market Structure (smooth vs rough analysis)
        - Liquidity Inducement Detection (stop hunts, false breakouts, sweeps)
        
        Key Smart Money Insights:
        - Price follows VALUE, not liquidity - liquidity is fuel
        - Major pools create deeper liquidity than minor pools
        - Rough structure = more internal pools = zone more likely to hold
        - Inducements are traps - trade opposite direction after reversal
        
        Args:
            swing_lookback: Bars to confirm swing points (default 3)
            cluster_threshold_pct: % distance to consider swings clustered
            min_void_size_pct: Minimum void size as % of price
        
        Returns:
            Initialized LiquidityConceptsEngine
        """
        lc_config = self.config.get('engines', {}).get('liquidity_concepts', {})
        
        # Override with config values if present
        swing_lookback = lc_config.get('swing_lookback', swing_lookback)
        cluster_threshold_pct = lc_config.get('cluster_threshold_pct', cluster_threshold_pct)
        min_void_size_pct = lc_config.get('min_void_size_pct', min_void_size_pct)
        
        logger.info(
            f"Creating LiquidityConceptsEngine with smart money analysis | "
            f"swing_lookback={swing_lookback}, cluster_pct={cluster_threshold_pct:.1%}"
        )
        return create_liquidity_concepts_engine(
            swing_lookback=swing_lookback,
            cluster_threshold_pct=cluster_threshold_pct,
            min_void_size_pct=min_void_size_pct,
        )
    
    def create_liquidity_engine_v5(
        self,
        market_adapter: Optional[MarketDataAdapter] = None,
    ) -> LiquidityEngineV5:
        """Create unified LiquidityEngineV5 with PENTA methodology.
        
        LiquidityEngineV5 is the main liquidity engine that combines:
        1. Market Quality Analysis (bid-ask, depth, tradability)
        2. PENTA Methodology (5 sub-engines):
           - Wyckoff (VSA, Phases, Events)
           - ICT (FVGs, Order Blocks, OTE)
           - Order Flow (Footprint, CVD, Volume Profile)
           - Supply & Demand (Zones, Strength)
           - Liquidity Concepts (Pools, Voids, Inducements)
        
        Architecture:
            Data Adapters → LiquidityEngineV5 → LiquidityAgentV5 → Composer
        
        Args:
            market_adapter: Optional market data adapter
            
        Returns:
            Initialized LiquidityEngineV5
        """
        liq_config = self.config.get('engines', {}).get('liquidity', {})
        
        if market_adapter is None:
            market_adapter = self.create_market_adapter()
        
        logger.info("Creating LiquidityEngineV5 with unified PENTA methodology")
        return LiquidityEngineV5(
            market_adapter=market_adapter,
            config=liq_config,
        )
    
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
        liquidity_engine = self.create_liquidity_engine(
            market_adapter=market_adapter,
            options_adapter=options_adapter,
            version="v4",  # Use Wyckoff-enhanced engine by default
        )
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


def create_unified_analysis_engines(
    config: Optional[Dict[str, Any]] = None,
    use_unified_v5: bool = True,
) -> Dict[str, Any]:
    """Create all methodology engines for unified analysis.
    
    This is a convenience function for setting up combined methodology analysis.
    
    Architecture:
        Data Adapters → Engines → Primary Agents → Composer → Trade Agents
    
    Args:
        config: Configuration dictionary
        use_unified_v5: If True, use LiquidityEngineV5 (recommended)
        
    Returns:
        Dictionary with all engines and adapters:
        - 'liquidity_engine_v5': LiquidityEngineV5 (unified PENTA engine) - MAIN
        - 'wyckoff_engine': LiquidityEngineV4 with Wyckoff methodology (legacy)
        - 'ict_engine': ICTEngine with ICT methodology (legacy)
        - 'order_flow_engine': OrderFlowEngine with Footprint/CVD/Profile (legacy)
        - 'supply_demand_engine': SupplyDemandEngine with S&D zones (legacy)
        - 'liquidity_concepts_engine': LiquidityConceptsEngine with smart money (legacy)
        - 'market_adapter': MarketDataAdapter
        - 'options_adapter': OptionsChainAdapter
    """
    factory = EngineFactory(config or {})
    market_adapter = factory.create_market_adapter()
    options_adapter = factory.create_options_adapter()
    
    result = {
        'market_adapter': market_adapter,
        'options_adapter': options_adapter,
    }
    
    if use_unified_v5:
        # Preferred: Use unified LiquidityEngineV5
        liquidity_v5 = factory.create_liquidity_engine_v5(market_adapter)
        result['liquidity_engine_v5'] = liquidity_v5
        
        # Also expose individual engines for backward compatibility
        penta_engines = liquidity_v5.get_penta_engines()
        result['wyckoff_engine'] = penta_engines.get('wyckoff')
        result['ict_engine'] = penta_engines.get('ict')
        result['order_flow_engine'] = penta_engines.get('order_flow')
        result['supply_demand_engine'] = penta_engines.get('supply_demand')
        result['liquidity_concepts_engine'] = penta_engines.get('liquidity_concepts')
    else:
        # Legacy: Create individual engines
        result['wyckoff_engine'] = factory.create_liquidity_engine(
            market_adapter=market_adapter,
            options_adapter=options_adapter,
            version="v4"
        )
        result['ict_engine'] = factory.create_ict_engine()
        result['order_flow_engine'] = factory.create_order_flow_engine()
        result['supply_demand_engine'] = factory.create_supply_demand_engine()
        result['liquidity_concepts_engine'] = factory.create_liquidity_concepts_engine()
    
    return result


__all__ = [
    "EngineFactory", 
    "create_engine_factory", 
    "create_unified_analysis_engines",
    "LiquidityEngineV5",
    "LiquidityEngineV5Snapshot",
    "PENTAState",
]
