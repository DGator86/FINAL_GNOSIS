"""Engine Factory - centralized initialization of all engines for OpportunityScanner."""
from __future__ import annotations

from typing import Dict, Any
from loguru import logger

from engines.hedge import HedgeEngineV3
from engines.liquidity import LiquidityEngineV1
from engines.sentiment import SentimentEngineV1
from engines.elasticity import ElasticityEngineV1
from engines.inputs.options_chain_adapter import OptionsChainAdapter
from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.adapter_factory import create_market_data_adapter, create_options_adapter
from engines.scanner import OpportunityScanner


class EngineFactory:
    """Factory for creating properly initialized trading engines."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize engine factory with configuration.
        
        Args:
            config: Full system configuration dictionary
        """
        self.config = config
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
    
    def create_sentiment_engine(self) -> SentimentEngineV1:
        """Create sentiment engine with proper configuration.
        
        Returns:
            Initialized SentimentEngineV1
        """
        sentiment_config = self.config.get('engines', {}).get('sentiment', {})
        
        # For now, use empty processors list
        # TODO: Initialize actual sentiment processors (news, flow, technical)
        processors = []
        
        return SentimentEngineV1(processors=processors, config=sentiment_config)
    
    def create_elasticity_engine(self, market_adapter: MarketDataAdapter) -> ElasticityEngineV1:
        """Create elasticity engine with proper configuration.
        
        Args:
            market_adapter: Market data provider
            
        Returns:
            Initialized ElasticityEngineV1
        """
        elasticity_config = self.config.get('engines', {}).get('elasticity', {})
        return ElasticityEngineV1(market_adapter=market_adapter, config=elasticity_config)
    
    def create_scanner(self) -> OpportunityScanner:
        """Create fully initialized OpportunityScanner with all engines.
        
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
        sentiment_engine = self.create_sentiment_engine()
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
        
        logger.info("OpportunityScanner created successfully")
        return scanner


__all__ = ["EngineFactory"]
