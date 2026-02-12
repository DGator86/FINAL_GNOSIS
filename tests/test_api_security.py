"""
Tests for API Security - Authentication and Rate Limiting

Tests:
- API Key authentication
- JWT token authentication
- Rate limiting
- Permission checking

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request
from starlette.testclient import TestClient


# =============================================================================
# Authentication Tests
# =============================================================================

class TestAPIKeyAuthentication:
    """Test API Key authentication."""
    
    def test_api_key_store_initialization(self):
        """Test API key store initialization."""
        from middleware.auth import APIKeyStore, UserRole
        
        store = APIKeyStore()
        
        # Should have created any env-based keys
        assert isinstance(store._keys, dict)
    
    def test_create_api_key(self):
        """Test API key creation."""
        from middleware.auth import APIKeyStore, UserRole, Permission
        
        store = APIKeyStore()
        
        api_key = store.create_key(
            key_value="test-key-123",
            user_id="test_user",
            name="Test Key",
            role=UserRole.TRADER,
        )
        
        assert api_key.key_id is not None
        assert api_key.user_id == "test_user"
        assert api_key.role == UserRole.TRADER
        assert api_key.is_active is True
        assert Permission.TRADE in api_key.permissions
    
    def test_validate_api_key(self):
        """Test API key validation."""
        from middleware.auth import APIKeyStore, UserRole
        
        store = APIKeyStore()
        
        # Create key
        store.create_key(
            key_value="valid-key-456",
            user_id="user1",
            name="Valid Key",
            role=UserRole.VIEWER,
        )
        
        # Validate correct key
        result = store.validate_key("valid-key-456")
        assert result is not None
        assert result.user_id == "user1"
        
        # Validate incorrect key
        result = store.validate_key("invalid-key")
        assert result is None
    
    def test_api_key_expiration(self):
        """Test API key expiration."""
        from middleware.auth import APIKeyStore, UserRole
        
        store = APIKeyStore()
        
        # Create expired key (negative days)
        api_key = store.create_key(
            key_value="expired-key",
            user_id="user2",
            name="Expired Key",
            role=UserRole.VIEWER,
            expires_in_days=-1,  # Already expired
        )
        
        # Should not validate
        result = store.validate_key("expired-key")
        assert result is None
    
    def test_revoke_api_key(self):
        """Test API key revocation."""
        from middleware.auth import APIKeyStore, UserRole
        
        store = APIKeyStore()
        
        # Create and validate key
        api_key = store.create_key(
            key_value="revokable-key",
            user_id="user3",
            name="Revokable Key",
            role=UserRole.VIEWER,
        )
        
        # Revoke
        success = store.revoke_key(api_key.key_id)
        assert success is True
        
        # Should not validate after revocation
        result = store.validate_key("revokable-key")
        assert result is None
    
    def test_api_key_auth_class(self):
        """Test APIKeyAuth class methods."""
        from middleware.auth import APIKeyAuth, UserRole
        
        # Generate key
        plain_key, api_key = APIKeyAuth.generate_key(
            user_id="api_user",
            name="API Generated Key",
            role=UserRole.ANALYST,
        )
        
        assert plain_key.startswith("gnosis_")
        assert api_key.role == UserRole.ANALYST
        
        # List keys
        keys = APIKeyAuth.list_keys("api_user")
        assert len(keys) >= 1


class TestJWTAuthentication:
    """Test JWT token authentication."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        from middleware.auth import create_access_token
        
        token = create_access_token(
            data={"sub": "user123", "role": "trader"}
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically long
    
    def test_verify_token(self):
        """Test JWT token verification."""
        from middleware.auth import create_access_token, verify_token
        
        # Create token
        token = create_access_token(
            data={"sub": "user456", "role": "admin", "email": "test@example.com"}
        )
        
        # Verify
        payload = verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == "user456"
        assert payload["role"] == "admin"
    
    def test_verify_expired_token(self):
        """Test expired token verification."""
        from middleware.auth import create_access_token, verify_token
        from datetime import timedelta
        
        # Create token that expires immediately
        token = create_access_token(
            data={"sub": "user789"},
            expires_delta=timedelta(seconds=-1),  # Already expired
        )
        
        # Should fail verification
        payload = verify_token(token)
        assert payload is None
    
    def test_verify_invalid_token(self):
        """Test invalid token verification."""
        from middleware.auth import verify_token
        
        payload = verify_token("invalid.token.here")
        assert payload is None
        
        payload = verify_token("")
        assert payload is None
    
    def test_jwt_auth_class(self):
        """Test JWTAuth class methods."""
        from middleware.auth import JWTAuth, UserRole
        
        # Create token
        token = JWTAuth.create_token(
            user_id="jwt_user",
            role=UserRole.TRADER,
            email="jwt@example.com",
            name="JWT User",
            expires_minutes=30,
        )
        
        assert token is not None
        
        # Verify
        payload = JWTAuth.verify(token)
        assert payload["sub"] == "jwt_user"
        assert payload["role"] == "trader"


class TestUserPermissions:
    """Test user permissions and roles."""
    
    def test_api_user_permissions(self):
        """Test APIUser permission checking."""
        from middleware.auth import APIUser, UserRole, Permission
        
        user = APIUser(
            user_id="perm_user",
            role=UserRole.TRADER,
            permissions=[Permission.READ, Permission.TRADE],
        )
        
        assert user.has_permission(Permission.READ) is True
        assert user.has_permission(Permission.TRADE) is True
        assert user.has_permission(Permission.ADMIN) is False
    
    def test_admin_has_all_permissions(self):
        """Test admin has all permissions."""
        from middleware.auth import APIUser, UserRole, Permission
        
        admin = APIUser(
            user_id="admin_user",
            role=UserRole.ADMIN,
            permissions=[Permission.ADMIN],
        )
        
        # Admin should have access to everything
        assert admin.has_permission(Permission.READ) is True
        assert admin.has_permission(Permission.TRADE) is True
        assert admin.has_permission(Permission.ANALYTICS) is True
    
    def test_role_permissions_mapping(self):
        """Test role to permissions mapping."""
        from middleware.auth import ROLE_PERMISSIONS, UserRole, Permission
        
        # Viewer should only have READ
        assert Permission.READ in ROLE_PERMISSIONS[UserRole.VIEWER]
        assert Permission.TRADE not in ROLE_PERMISSIONS[UserRole.VIEWER]
        
        # Trader should have TRADE
        assert Permission.TRADE in ROLE_PERMISSIONS[UserRole.TRADER]
        
        # Admin should have all
        assert len(ROLE_PERMISSIONS[UserRole.ADMIN]) == len(Permission)


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        from middleware.rate_limiter import RateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=2000,
        )
        
        limiter = RateLimiter(config)
        
        assert limiter.config.requests_per_minute == 100
        assert limiter.config.requests_per_hour == 2000
    
    def test_token_bucket(self):
        """Test token bucket algorithm."""
        from middleware.rate_limiter import TokenBucket
        
        bucket = TokenBucket(
            capacity=10,
            tokens=10,
            refill_rate=1.0,  # 1 token per second
            last_update=time.time(),
        )
        
        # Consume all tokens
        for i in range(10):
            assert bucket.consume() is True
        
        # Should be empty
        assert bucket.consume() is False
        
        # Wait for refill
        time.sleep(1.1)
        assert bucket.consume() is True
    
    def test_token_bucket_wait_time(self):
        """Test token bucket wait time calculation."""
        from middleware.rate_limiter import TokenBucket
        
        bucket = TokenBucket(
            capacity=10,
            tokens=0,
            refill_rate=10.0,  # 10 tokens per second
            last_update=time.time(),
        )
        
        wait_time = bucket.get_wait_time(5)
        assert wait_time == pytest.approx(0.5, rel=0.1)  # 5 tokens / 10 per sec = 0.5 sec
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_requests(self):
        """Test rate limiter allows requests within limits."""
        from middleware.rate_limiter import RateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=100,
            exempt_endpoints=["/health"],
        )
        
        limiter = RateLimiter(config)
        
        # Create mock request
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        
        # First request should be allowed
        allowed, info = await limiter.check_rate_limit(mock_request)
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excessive_requests(self):
        """Test rate limiter blocks excessive requests."""
        from middleware.rate_limiter import RateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=5,  # Very low limit
            burst_size=0,
        )
        
        limiter = RateLimiter(config)
        
        # Create mock request
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/api/test"
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}
        
        # Make requests until blocked
        blocked = False
        for i in range(10):
            allowed, info = await limiter.check_rate_limit(mock_request)
            if not allowed:
                blocked = True
                break
        
        assert blocked is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_exempt_endpoints(self):
        """Test rate limiter exempts specified endpoints."""
        from middleware.rate_limiter import RateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=1,  # Very restrictive
            exempt_endpoints=["/health", "/ready"],
        )
        
        limiter = RateLimiter(config)
        
        # Create mock request for exempt endpoint
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/health"
        mock_request.client.host = "10.0.0.1"
        mock_request.headers = {}
        
        # Should always be allowed
        for i in range(100):
            allowed, info = await limiter.check_rate_limit(mock_request)
            assert allowed is True
            assert info.get("exempt") is True
    
    def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        from middleware.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        stats = limiter.get_stats()
        
        assert "total_requests" in stats
        assert "rate_limited" in stats
        assert "active_clients" in stats
        assert "config" in stats


class TestRateLimitHeaders:
    """Test rate limit header generation."""
    
    def test_get_headers(self):
        """Test rate limit header generation."""
        from middleware.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        
        limit_info = {
            "remaining_minute": 50,
            "retry_after": 30,
        }
        
        headers = limiter.get_headers(limit_info)
        
        assert "X-RateLimit-Remaining" in headers
        assert headers["X-RateLimit-Remaining"] == "50"
        assert "Retry-After" in headers
        assert headers["Retry-After"] == "30"


# =============================================================================
# Integration Tests
# =============================================================================

class TestAuthIntegration:
    """Integration tests for authentication."""
    
    def test_api_user_to_dict(self):
        """Test APIUser serialization."""
        from middleware.auth import APIUser, UserRole, Permission
        
        user = APIUser(
            user_id="dict_user",
            role=UserRole.ANALYST,
            permissions=[Permission.READ, Permission.ANALYTICS],
            email="analyst@example.com",
            name="Analyst User",
        )
        
        data = user.to_dict()
        
        assert data["user_id"] == "dict_user"
        assert data["role"] == "analyst"
        assert "read" in data["permissions"]
        assert "analytics" in data["permissions"]
    
    def test_api_key_with_ip_restriction(self):
        """Test API key with IP restriction."""
        from middleware.auth import APIKeyStore, UserRole
        
        store = APIKeyStore()
        
        api_key = store.create_key(
            key_value="ip-restricted-key",
            user_id="ip_user",
            name="IP Restricted Key",
            role=UserRole.VIEWER,
            allowed_ips=["192.168.1.100", "10.0.0.1"],
        )
        
        assert len(api_key.allowed_ips) == 2
        assert "192.168.1.100" in api_key.allowed_ips


class TestRateLimitIntegration:
    """Integration tests for rate limiting."""
    
    @pytest.mark.asyncio
    async def test_role_based_rate_limits(self):
        """Test role-based rate limit multipliers."""
        from middleware.rate_limiter import RateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=10,
            role_multipliers={
                "viewer": 1.0,
                "admin": 10.0,
            },
        )
        
        limiter = RateLimiter(config)
        
        # Create state for viewer
        viewer_state = limiter._get_or_create_state("user:viewer", role="viewer")
        
        # Create state for admin
        admin_state = limiter._get_or_create_state("user:admin", role="admin")
        
        # Admin should have more capacity
        assert admin_state.buckets["minute"].capacity > viewer_state.buckets["minute"].capacity
    
    def test_cleanup_old_states(self):
        """Test cleanup of old rate limit states."""
        from middleware.rate_limiter import RateLimiter, RateLimitState
        import time
        
        limiter = RateLimiter()
        
        # Add old state
        old_state = RateLimitState(
            client_id="old_client",
            first_request=time.time() - (25 * 3600),  # 25 hours ago
        )
        limiter._states["old_client"] = old_state
        
        # Add recent state
        recent_state = RateLimitState(
            client_id="recent_client",
            first_request=time.time(),
        )
        limiter._states["recent_client"] = recent_state
        
        # Cleanup
        limiter.cleanup_old_states(max_age_hours=24)
        
        # Old should be removed, recent should remain
        assert "old_client" not in limiter._states
        assert "recent_client" in limiter._states


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
