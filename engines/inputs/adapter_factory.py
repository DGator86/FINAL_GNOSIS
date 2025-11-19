"""Adapter factory with automatic fallback to stubs."""

from __future__ import annotations

from typing import Any

from loguru import logger


def create_market_data_adapter(prefer_real: bool = True) -> Any:
    """
    Create market data adapter with fallback.
    
    Args:
        prefer_real: Try real adapter first
        
    Returns:
        Market data adapter instance
    """
    if prefer_real:
        try:
            from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
            adapter = AlpacaMarketDataAdapter()
            logger.info("Using AlpacaMarketDataAdapter (real)")
            return adapter
        except Exception as e:
            logger.warning(f"Failed to initialize Alpaca market data adapter: {e}")
            logger.info("Falling back to StaticMarketDataAdapter")
    
    from engines.inputs.stub_adapters import StaticMarketDataAdapter
    logger.info("Using StaticMarketDataAdapter (stub)")
    return StaticMarketDataAdapter()


def create_options_adapter(prefer_real: bool = True) -> Any:
    """
    Create options chain adapter with fallback.
    
    Args:
        prefer_real: Try real adapter first
        
    Returns:
        Options chain adapter instance
    """
    if prefer_real:
        try:
            from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
            adapter = UnusualWhalesAdapter()
            logger.info("Using UnusualWhalesAdapter (real)")
            return adapter
        except Exception as e:
            logger.warning(f"Failed to initialize Unusual Whales adapter: {e}")
            logger.info("Falling back to StaticOptionsAdapter")
    
    from engines.inputs.stub_adapters import StaticOptionsAdapter
    logger.info("Using StaticOptionsAdapter (stub)")
    return StaticOptionsAdapter()


def create_news_adapter(prefer_real: bool = True) -> Any:
    """
    Create news adapter with fallback.
    
    Args:
        prefer_real: Try real adapter first
        
    Returns:
        News adapter instance
    """
    # For now, always use stub (can add real news adapter later)
    from engines.inputs.stub_adapters import StaticNewsAdapter
    logger.info("Using StaticNewsAdapter (stub)")
    return StaticNewsAdapter()


def create_broker_adapter(paper: bool = True, prefer_real: bool = True) -> Any:
    """
    Create broker adapter with fallback.
    
    Args:
        paper: Use paper trading
        prefer_real: Try real adapter first
        
    Returns:
        Broker adapter instance
    """
    if prefer_real:
        try:
            from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
            adapter = AlpacaBrokerAdapter(paper=paper)
            logger.info(f"Using AlpacaBrokerAdapter (real, paper={paper})")
            return adapter
        except Exception as e:
            logger.warning(f"Failed to initialize Alpaca broker adapter: {e}")
            logger.info("No fallback broker available")
            return None
    
    logger.info("Real broker not requested")
    return None
