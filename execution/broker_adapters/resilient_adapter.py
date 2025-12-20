"""Resilient broker adapter wrapper with circuit breaker and rate limiting.

Provides production-grade resilience for broker API calls:
- Circuit breaker to prevent cascading failures
- Rate limiting to respect API limits
- Automatic retry with exponential backoff
- Fallback behavior for non-critical operations
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar

from loguru import logger

from utils.cache import RateLimiter, get_alpaca_rate_limiter
from utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    get_alpaca_circuit_breaker,
)

T = TypeVar("T")


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retryable_exceptions: Exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class ResilientBrokerAdapter:
    """Wrapper that adds resilience to any broker adapter.
    
    Features:
    - Circuit breaker to prevent cascading failures when broker is down
    - Rate limiting to respect API limits
    - Automatic retry with exponential backoff for transient failures
    - Graceful degradation for non-critical operations
    
    Example:
        adapter = AlpacaBrokerAdapter(paper=True)
        resilient = ResilientBrokerAdapter(adapter)
        
        # All calls now have circuit breaker and rate limiting
        account = resilient.get_account()
        resilient.place_order("SPY", 10, "buy")
    """
    
    def __init__(
        self,
        adapter: Any,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = 3,
    ):
        """Initialize resilient adapter wrapper.
        
        Args:
            adapter: Underlying broker adapter
            circuit_breaker: Circuit breaker instance (default: Alpaca breaker)
            rate_limiter: Rate limiter instance (default: Alpaca limiter)
            max_retries: Maximum retry attempts for transient failures
        """
        self._adapter = adapter
        self._circuit_breaker = circuit_breaker or get_alpaca_circuit_breaker()
        self._rate_limiter = rate_limiter or get_alpaca_rate_limiter()
        self._max_retries = max_retries
        
        logger.info(
            f"ResilientBrokerAdapter wrapping {type(adapter).__name__} "
            f"(circuit_breaker={self._circuit_breaker.name}, max_retries={max_retries})"
        )
    
    def _call_with_resilience(
        self, 
        method_name: str, 
        *args, 
        critical: bool = True,
        **kwargs
    ) -> Any:
        """Execute a method with circuit breaker, rate limiting, and retry.
        
        Args:
            method_name: Name of method to call on underlying adapter
            *args: Positional arguments
            critical: Whether to raise exceptions (True) or return None (False)
            **kwargs: Keyword arguments
            
        Returns:
            Method result
            
        Raises:
            CircuitBreakerOpen: If circuit is open and critical=True
            Exception: Any exception from the method if critical=True
        """
        method = getattr(self._adapter, method_name)
        
        # Check circuit breaker first
        if self._circuit_breaker.is_open:
            if critical:
                raise CircuitBreakerOpen(
                    self._circuit_breaker.name,
                    time.time() + self._circuit_breaker.timeout
                )
            logger.warning(
                f"Circuit breaker open, skipping {method_name}"
            )
            return None
        
        # Apply rate limiting
        if not self._rate_limiter.acquire(timeout=10.0):
            if critical:
                raise TimeoutError(f"Rate limit timeout for {method_name}")
            logger.warning(f"Rate limit timeout, skipping {method_name}")
            return None
        
        # Execute with circuit breaker and retry
        last_exception = None
        
        for attempt in range(self._max_retries + 1):
            try:
                with self._circuit_breaker:
                    result = method(*args, **kwargs)
                    return result
                    
            except CircuitBreakerOpen:
                if critical:
                    raise
                return None
                
            except Exception as e:
                last_exception = e
                
                if attempt == self._max_retries:
                    if critical:
                        raise
                    logger.error(f"{method_name} failed after retries: {e}")
                    return None
                
                # Exponential backoff
                delay = min(1.0 * (2 ** attempt), 30.0)
                logger.warning(
                    f"{method_name} failed (attempt {attempt + 1}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
        
        return None
    
    # ========== Account Methods ==========
    
    def get_account(self):
        """Get account information with resilience."""
        return self._call_with_resilience("get_account", critical=True)
    
    def get_positions(self) -> List:
        """Get positions with resilience."""
        result = self._call_with_resilience("get_positions", critical=False)
        return result if result is not None else []
    
    def get_position(self, symbol: str):
        """Get position for symbol with resilience."""
        return self._call_with_resilience("get_position", symbol, critical=False)
    
    # ========== Order Methods ==========
    
    def place_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[str]:
        """Place order with resilience.
        
        Order placement is critical - will raise on circuit breaker open.
        """
        return self._call_with_resilience(
            "place_order",
            symbol,
            quantity,
            side,
            order_type,
            time_in_force,
            limit_price,
            stop_price,
            critical=True,
        )
    
    def place_bracket_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "gtc",
    ) -> Optional[str]:
        """Place bracket order with resilience."""
        return self._call_with_resilience(
            "place_bracket_order",
            symbol,
            quantity,
            side,
            take_profit_price,
            stop_loss_price,
            time_in_force,
            critical=True,
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order with resilience."""
        result = self._call_with_resilience(
            "cancel_order", 
            order_id, 
            critical=False
        )
        return result if result is not None else False
    
    def close_position(self, symbol: str, qty: Optional[float] = None) -> bool:
        """Close position with resilience."""
        result = self._call_with_resilience(
            "close_position",
            symbol,
            qty,
            critical=True,
        )
        return result if result is not None else False
    
    def close_all_positions(self) -> bool:
        """Close all positions with resilience."""
        result = self._call_with_resilience(
            "close_all_positions",
            critical=True,
        )
        return result if result is not None else False
    
    # ========== Data Methods ==========
    
    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """Get latest quote with resilience (non-critical)."""
        return self._call_with_resilience(
            "get_latest_quote",
            symbol,
            critical=False,
        )
    
    def get_market_clock(self) -> Optional[dict]:
        """Get market clock with resilience (non-critical)."""
        return self._call_with_resilience(
            "get_market_clock",
            critical=False,
        )
    
    def is_market_open(self) -> bool:
        """Check if market is open with resilience."""
        result = self._call_with_resilience(
            "is_market_open",
            critical=False,
        )
        return result if result is not None else False
    
    # ========== Utility Methods ==========
    
    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get circuit breaker for inspection."""
        return self._circuit_breaker
    
    @property
    def is_healthy(self) -> bool:
        """Check if broker connection is healthy."""
        return self._circuit_breaker.is_closed
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        self._circuit_breaker.reset()
    
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying adapter."""
        return getattr(self._adapter, name)


def create_resilient_broker_adapter(
    paper: Optional[bool] = None,
    prefer_real: bool = True,
) -> Optional[ResilientBrokerAdapter]:
    """Create a resilient broker adapter with circuit breaker and rate limiting.
    
    Args:
        paper: Use paper trading
        prefer_real: Try real adapter first
        
    Returns:
        ResilientBrokerAdapter instance or None if adapter creation fails
    """
    from engines.inputs.adapter_factory import create_broker_adapter
    
    adapter = create_broker_adapter(paper=paper, prefer_real=prefer_real)
    if adapter is None:
        return None
    
    return ResilientBrokerAdapter(adapter)
