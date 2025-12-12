"""Adapter factory with automatic fallback to stubs."""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger


def create_market_data_adapter(prefer_real: bool = True, provider: Optional[str] = None) -> Any:
    """
    Create market data adapter with fallback.

    Args:
        prefer_real: Try real adapter first
        provider: Preferred provider ("massive", "alpaca", or None for auto)

    Returns:
        Market data adapter instance
    """
    # Check for preferred provider from environment
    if provider is None:
        provider = os.getenv("MARKET_DATA_PROVIDER", "auto").lower()

    if prefer_real:
        # Try MASSIVE first if enabled or preferred
        if provider in ("massive", "auto") and os.getenv("MASSIVE_API_ENABLED", "false").lower() == "true":
            try:
                from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
                adapter = MassiveMarketDataAdapter()
                logger.info("Using MassiveMarketDataAdapter (real)")
                return adapter
            except Exception as e:
                logger.warning(f"Failed to initialize MASSIVE market data adapter: {e}")
                if provider == "massive":
                    logger.info("Falling back to Alpaca adapter")

        # Try Alpaca
        if provider in ("alpaca", "auto"):
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
    if prefer_real and os.getenv("MASSIVE_API_ENABLED", "false").lower() == "true":
        try:
            from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
            adapter = MassiveMarketDataAdapter()
            logger.info("Using MassiveMarketDataAdapter for news (real)")
            return adapter
        except Exception as e:
            logger.warning(f"Failed to initialize MASSIVE adapter for news: {e}")

    from engines.inputs.stub_adapters import StaticNewsAdapter
    logger.info("Using StaticNewsAdapter (stub)")
    return StaticNewsAdapter()


def create_massive_adapter() -> Any:
    """
    Create MASSIVE market data adapter for comprehensive data access.

    Returns:
        MassiveMarketDataAdapter instance or None if not configured
    """
    if os.getenv("MASSIVE_API_ENABLED", "false").lower() != "true":
        logger.warning("MASSIVE API not enabled. Set MASSIVE_API_ENABLED=true in .env")
        return None

    try:
        from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
        adapter = MassiveMarketDataAdapter()
        logger.info("MassiveMarketDataAdapter created successfully")
        return adapter
    except Exception as e:
        logger.error(f"Failed to create MASSIVE adapter: {e}")
        return None


def create_broker_adapter(paper: Optional[bool] = None, prefer_real: bool = True) -> Any:
    """
    Create broker adapter with fallback.
    
    Args:
        paper: Use paper trading
        prefer_real: Try real adapter first
        
    Returns:
        Broker adapter instance
    """
    if paper is None:
        from execution.broker_adapters.settings import get_alpaca_paper_setting

        paper = get_alpaca_paper_setting()

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
