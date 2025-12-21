"""
Rate Limiting Middleware

Provides:
- Token bucket rate limiting
- Per-user and per-endpoint limits
- Redis-backed distributed rate limiting
- Configurable limits per role

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

from fastapi import Depends, HTTPException, Request, status
from loguru import logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    # Default limits
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    
    # Burst allowance
    burst_size: int = 10
    
    # Per-role limits (multipliers)
    role_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "viewer": 1.0,
        "trader": 2.0,
        "analyst": 1.5,
        "admin": 10.0,
        "service": 100.0,
    })
    
    # Per-endpoint limits (override default)
    endpoint_limits: Dict[str, int] = field(default_factory=lambda: {
        "/health": 1000,         # High limit for health checks
        "/metrics": 100,          # Moderate for metrics
        "/hub/market-data": 600,  # High for market data
        "/hub/options-flow": 600, # High for flow data
        "/ws/stream": 10,         # Low for WebSocket connections
    })
    
    # Exempt endpoints (no rate limiting)
    exempt_endpoints: list = field(default_factory=lambda: [
        "/health",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
    ])
    
    # Block duration for exceeded limits (seconds)
    block_duration: int = 60
    
    # Enable Redis for distributed rate limiting
    use_redis: bool = False


# =============================================================================
# Token Bucket Implementation
# =============================================================================

@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: float           # Maximum tokens
    tokens: float             # Current tokens
    refill_rate: float        # Tokens per second
    last_update: float        # Last update timestamp
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens.
        
        Returns:
            True if tokens were consumed, False if insufficient
        """
        now = time.time()
        
        # Refill tokens based on time elapsed
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to become available.
        
        Returns:
            Seconds to wait
        """
        if self.tokens >= tokens:
            return 0.0
        
        needed = tokens - self.tokens
        return needed / self.refill_rate


@dataclass
class RateLimitState:
    """State for a rate-limited client."""
    client_id: str
    buckets: Dict[str, TokenBucket] = field(default_factory=dict)
    request_count: int = 0
    blocked_until: Optional[float] = None
    violations: int = 0
    first_request: float = field(default_factory=time.time)
    
    def is_blocked(self) -> bool:
        """Check if client is currently blocked."""
        if self.blocked_until is None:
            return False
        return time.time() < self.blocked_until
    
    def block(self, duration: int):
        """Block client for specified duration."""
        self.blocked_until = time.time() + duration
        self.violations += 1


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Rate limiter with token bucket algorithm.
    
    Features:
    - Per-minute, per-hour, per-day limits
    - Per-user and per-endpoint limits
    - Role-based limit multipliers
    - Burst allowance
    - Automatic blocking for violations
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._states: Dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()
        
        # Metrics
        self._metrics = {
            "total_requests": 0,
            "rate_limited": 0,
            "blocked": 0,
        }
        
        logger.info(f"RateLimiter initialized: {self.config.requests_per_minute}/min")
    
    def _get_client_id(self, request: Request, user_id: Optional[str] = None) -> str:
        """Get client identifier for rate limiting.
        
        Uses user_id if authenticated, otherwise IP address.
        """
        if user_id:
            return f"user:{user_id}"
        
        # Get IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"
    
    def _get_or_create_state(
        self,
        client_id: str,
        role: Optional[str] = None,
    ) -> RateLimitState:
        """Get or create rate limit state for client."""
        if client_id not in self._states:
            state = RateLimitState(client_id=client_id)
            
            # Get role multiplier
            multiplier = self.config.role_multipliers.get(role or "viewer", 1.0)
            
            # Create buckets for different time windows
            rpm = int(self.config.requests_per_minute * multiplier)
            rph = int(self.config.requests_per_hour * multiplier)
            rpd = int(self.config.requests_per_day * multiplier)
            burst = int(self.config.burst_size * multiplier)
            
            state.buckets = {
                "minute": TokenBucket(
                    capacity=rpm + burst,
                    tokens=rpm + burst,
                    refill_rate=rpm / 60.0,
                    last_update=time.time(),
                ),
                "hour": TokenBucket(
                    capacity=rph,
                    tokens=rph,
                    refill_rate=rph / 3600.0,
                    last_update=time.time(),
                ),
                "day": TokenBucket(
                    capacity=rpd,
                    tokens=rpd,
                    refill_rate=rpd / 86400.0,
                    last_update=time.time(),
                ),
            }
            
            self._states[client_id] = state
        
        return self._states[client_id]
    
    async def check_rate_limit(
        self,
        request: Request,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits.
        
        Args:
            request: FastAPI request
            user_id: Optional user ID
            role: Optional user role
            
        Returns:
            Tuple of (allowed, limit_info)
        """
        endpoint = request.url.path
        
        # Check if endpoint is exempt
        if endpoint in self.config.exempt_endpoints:
            return True, {"exempt": True}
        
        client_id = self._get_client_id(request, user_id)
        
        async with self._lock:
            state = self._get_or_create_state(client_id, role)
            
            self._metrics["total_requests"] += 1
            state.request_count += 1
            
            # Check if blocked
            if state.is_blocked():
                self._metrics["blocked"] += 1
                remaining = state.blocked_until - time.time()
                return False, {
                    "blocked": True,
                    "retry_after": int(remaining),
                    "violations": state.violations,
                }
            
            # Check endpoint-specific limit
            endpoint_limit = self.config.endpoint_limits.get(endpoint)
            if endpoint_limit:
                # Create endpoint-specific bucket if needed
                bucket_key = f"endpoint:{endpoint}"
                if bucket_key not in state.buckets:
                    state.buckets[bucket_key] = TokenBucket(
                        capacity=endpoint_limit,
                        tokens=endpoint_limit,
                        refill_rate=endpoint_limit / 60.0,
                        last_update=time.time(),
                    )
                
                if not state.buckets[bucket_key].consume():
                    self._metrics["rate_limited"] += 1
                    wait_time = state.buckets[bucket_key].get_wait_time()
                    return False, {
                        "limit": "endpoint",
                        "endpoint": endpoint,
                        "retry_after": int(wait_time) + 1,
                    }
            
            # Check standard limits
            limit_info = {}
            
            for window, bucket in state.buckets.items():
                if window.startswith("endpoint:"):
                    continue
                
                if not bucket.consume():
                    self._metrics["rate_limited"] += 1
                    wait_time = bucket.get_wait_time()
                    
                    # Block client if too many violations
                    if state.violations >= 3:
                        state.block(self.config.block_duration * state.violations)
                        return False, {
                            "blocked": True,
                            "retry_after": self.config.block_duration * state.violations,
                            "violations": state.violations,
                        }
                    
                    state.violations += 1
                    
                    return False, {
                        "limit": window,
                        "retry_after": int(wait_time) + 1,
                    }
                
                # Calculate remaining tokens for headers
                limit_info[f"remaining_{window}"] = int(bucket.tokens)
            
            # Request allowed
            return True, limit_info
    
    def get_headers(self, limit_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate rate limit headers.
        
        Args:
            limit_info: Limit information from check_rate_limit
            
        Returns:
            Dictionary of headers
        """
        headers = {}
        
        if "remaining_minute" in limit_info:
            headers["X-RateLimit-Remaining"] = str(limit_info["remaining_minute"])
            headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)
        
        if "retry_after" in limit_info:
            headers["Retry-After"] = str(limit_info["retry_after"])
            headers["X-RateLimit-Reset"] = str(int(time.time()) + limit_info["retry_after"])
        
        return headers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "total_requests": self._metrics["total_requests"],
            "rate_limited": self._metrics["rate_limited"],
            "blocked": self._metrics["blocked"],
            "active_clients": len(self._states),
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "burst_size": self.config.burst_size,
            },
        }
    
    def reset_client(self, client_id: str):
        """Reset rate limit state for a client."""
        if client_id in self._states:
            del self._states[client_id]
    
    def cleanup_old_states(self, max_age_hours: int = 24):
        """Clean up old rate limit states."""
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = []
        
        for client_id, state in self._states.items():
            if state.first_request < cutoff and not state.is_blocked():
                to_remove.append(client_id)
        
        for client_id in to_remove:
            del self._states[client_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old rate limit states")


# =============================================================================
# Global Rate Limiter
# =============================================================================

_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config)
    
    return _rate_limiter


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def rate_limit(
    request: Request,
) -> Dict[str, Any]:
    """FastAPI dependency for rate limiting.
    
    Usage:
        @app.get("/endpoint")
        async def endpoint(limit_info: Dict = Depends(rate_limit)):
            ...
    """
    limiter = get_rate_limiter()
    
    # Get user info if authenticated
    user_id = None
    role = None
    
    # Check for authenticated user in request state
    if hasattr(request.state, "user") and request.state.user:
        user_id = request.state.user.user_id
        role = request.state.user.role.value if hasattr(request.state.user.role, "value") else str(request.state.user.role)
    
    allowed, limit_info = await limiter.check_rate_limit(request, user_id, role)
    
    if not allowed:
        headers = limiter.get_headers(limit_info)
        
        if limit_info.get("blocked"):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Blocked for {limit_info['retry_after']} seconds.",
                headers=headers,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({limit_info.get('limit', 'unknown')}). Retry after {limit_info['retry_after']} seconds.",
                headers=headers,
            )
    
    return limit_info


def rate_limit_decorator(
    requests_per_minute: Optional[int] = None,
    requests_per_hour: Optional[int] = None,
):
    """Decorator for custom rate limits on specific endpoints.
    
    Usage:
        @app.get("/heavy")
        @rate_limit_decorator(requests_per_minute=10)
        async def heavy_endpoint():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request: Request = kwargs.get("request")
            if not request:
                # Try to find request in args
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if request:
                limiter = get_rate_limiter()
                
                # Create custom config for this endpoint
                custom_config = RateLimitConfig(
                    requests_per_minute=requests_per_minute or 60,
                    requests_per_hour=requests_per_hour or 1000,
                )
                
                # Use endpoint-specific key
                client_id = f"custom:{request.url.path}:{request.client.host if request.client else 'unknown'}"
                
                state = limiter._get_or_create_state(client_id)
                
                # Check minute bucket
                if requests_per_minute:
                    if "custom_minute" not in state.buckets:
                        state.buckets["custom_minute"] = TokenBucket(
                            capacity=requests_per_minute,
                            tokens=requests_per_minute,
                            refill_rate=requests_per_minute / 60.0,
                            last_update=time.time(),
                        )
                    
                    if not state.buckets["custom_minute"].consume():
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Custom rate limit exceeded",
                        )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# Middleware Integration
# =============================================================================

class RateLimitMiddleware:
    """ASGI middleware for rate limiting all requests."""
    
    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        self.app = app
        self.limiter = get_rate_limiter(config)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create request object for rate limiting
        from starlette.requests import Request
        request = Request(scope, receive)
        
        allowed, limit_info = await self.limiter.check_rate_limit(request)
        
        if not allowed:
            # Send 429 response
            headers = self.limiter.get_headers(limit_info)
            headers_list = [(k.encode(), v.encode()) for k, v in headers.items()]
            headers_list.append((b"content-type", b"application/json"))
            
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": headers_list,
            })
            
            import json
            body = json.dumps({
                "detail": f"Rate limit exceeded. Retry after {limit_info.get('retry_after', 60)} seconds.",
                "retry_after": limit_info.get("retry_after", 60),
            })
            
            await send({
                "type": "http.response.body",
                "body": body.encode(),
            })
            return
        
        # Add rate limit info to scope for headers
        scope["rate_limit_info"] = limit_info
        
        await self.app(scope, receive, send)
