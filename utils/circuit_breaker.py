"""Circuit breaker pattern implementation for resilient API calls.

Provides:
- Automatic failure detection and recovery
- Configurable thresholds and timeouts
- Half-open state for gradual recovery testing
- Metrics and state reporting
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: int = 0


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, name: str, reset_time: float):
        self.name = name
        self.reset_time = reset_time
        remaining = max(0, reset_time - time.time())
        super().__init__(
            f"Circuit breaker '{name}' is open. "
            f"Reset in {remaining:.1f} seconds."
        )


class CircuitBreaker:
    """Circuit breaker for resilient API operations.
    
    Implements the circuit breaker pattern to prevent cascading failures
    when external services are unavailable.
    
    States:
    - CLOSED: Normal operation. Failures are counted.
    - OPEN: Service is down. All calls are rejected immediately.
    - HALF_OPEN: Testing if service has recovered. Limited calls allowed.
    
    Example:
        breaker = CircuitBreaker("alpaca_api", failure_threshold=5)
        
        @breaker
        def call_api():
            return requests.get("https://api.example.com")
        
        # Or use as context manager:
        with breaker:
            response = requests.get("https://api.example.com")
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        half_open_max_calls: int = 3,
        excluded_exceptions: tuple = (),
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Identifier for this breaker (used in logs and errors)
            failure_threshold: Number of failures before opening circuit
            success_threshold: Successes in half-open state to close circuit
            timeout: Seconds before attempting recovery (open -> half-open)
            half_open_max_calls: Max concurrent calls in half-open state
            excluded_exceptions: Exception types that don't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions
        
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        self._last_failure_time: Optional[float] = None
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._stats = CircuitStats()
        
        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={timeout}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, updating if timeout elapsed."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        return self._stats
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.timeout
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1
        
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
        
        logger.info(
            f"CircuitBreaker '{self.name}': {old_state.value} -> {new_state.value}"
        )
    
    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.time()
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = time.time()
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Immediate transition back to open on any failure
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            
            logger.warning(
                f"CircuitBreaker '{self.name}' recorded failure: {exception}"
            )
    
    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        state = self.state  # This may trigger state transition
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            return False
        
        # HALF_OPEN - allow limited calls
        with self._lock:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            self._stats.rejected_calls += 1
            return False
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function through the circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception from the function
        """
        if not self._can_execute():
            raise CircuitBreakerOpen(
                self.name, 
                (self._last_failure_time or 0) + self.timeout
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.excluded_exceptions:
            # Don't count excluded exceptions as failures
            raise
        except Exception as e:
            self._record_failure(e)
            raise
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap a function with circuit breaker.
        
        Example:
            @circuit_breaker
            def api_call():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        
        # Attach breaker reference for inspection
        wrapper.circuit_breaker = self
        return wrapper
    
    def __enter__(self):
        """Context manager entry - check if call is allowed."""
        if not self._can_execute():
            raise CircuitBreakerOpen(
                self.name,
                (self._last_failure_time or 0) + self.timeout
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record success or failure."""
        if exc_type is None:
            self._record_success()
        elif exc_type not in self.excluded_exceptions:
            self._record_failure(exc_val)
        return False  # Don't suppress exceptions
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info(f"CircuitBreaker '{self.name}' manually reset")
    
    def force_open(self) -> None:
        """Manually force the circuit breaker to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            self._last_failure_time = time.time()
            logger.info(f"CircuitBreaker '{self.name}' manually opened")


# Pre-configured circuit breakers for common services
_alpaca_breaker = CircuitBreaker(
    name="alpaca_api",
    failure_threshold=5,
    success_threshold=2,
    timeout=30.0,
)

_unusual_whales_breaker = CircuitBreaker(
    name="unusual_whales_api",
    failure_threshold=3,
    success_threshold=2,
    timeout=60.0,
)


def get_alpaca_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Alpaca API."""
    return _alpaca_breaker


def get_unusual_whales_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Unusual Whales API."""
    return _unusual_whales_breaker
