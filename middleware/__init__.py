"""
Middleware Module - API Security and Rate Limiting

Provides:
- API Key authentication
- JWT token authentication
- Rate limiting
- Request validation

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from middleware.auth import (
    APIKeyAuth,
    JWTAuth,
    get_api_key,
    get_current_user,
    create_access_token,
    verify_token,
    require_api_key,
    require_jwt,
    optional_auth,
)

from middleware.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    rate_limit,
    get_rate_limiter,
)

__all__ = [
    # Auth
    "APIKeyAuth",
    "JWTAuth",
    "get_api_key",
    "get_current_user",
    "create_access_token",
    "verify_token",
    "require_api_key",
    "require_jwt",
    "optional_auth",
    # Rate Limiter
    "RateLimiter",
    "RateLimitConfig",
    "rate_limit",
    "get_rate_limiter",
]
