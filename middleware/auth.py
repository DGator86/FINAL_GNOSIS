"""
Authentication Middleware

Provides:
- API Key authentication
- JWT token authentication
- User management
- Permission checking

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

# Try to import jose for JWT, fall back to simple implementation
try:
    from jose import JWTError, jwt
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False
    import base64
    import json


# =============================================================================
# Configuration
# =============================================================================

# Secret key for JWT signing (should be set via environment variable)
SECRET_KEY = os.getenv("API_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# API Key settings
API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_QUERY_NAME = "api_key"


class Permission(str, Enum):
    """API permissions."""
    READ = "read"
    WRITE = "write"
    TRADE = "trade"
    ADMIN = "admin"
    ANALYTICS = "analytics"
    ALERTS = "alerts"
    WEBSOCKET = "websocket"


class UserRole(str, Enum):
    """User roles."""
    VIEWER = "viewer"          # Read-only access
    TRADER = "trader"          # Can execute trades
    ANALYST = "analyst"        # Analytics access
    ADMIN = "admin"            # Full access
    SERVICE = "service"        # Service account


# Role to permissions mapping
ROLE_PERMISSIONS = {
    UserRole.VIEWER: [Permission.READ],
    UserRole.TRADER: [Permission.READ, Permission.WRITE, Permission.TRADE, Permission.WEBSOCKET],
    UserRole.ANALYST: [Permission.READ, Permission.ANALYTICS, Permission.WEBSOCKET],
    UserRole.ADMIN: list(Permission),
    UserRole.SERVICE: list(Permission),
}


@dataclass
class APIUser:
    """Authenticated API user."""
    user_id: str
    role: UserRole
    permissions: List[Permission] = field(default_factory=list)
    email: Optional[str] = None
    name: Optional[str] = None
    api_key_id: Optional[str] = None
    is_service_account: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or Permission.ADMIN in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "email": self.email,
            "name": self.name,
            "is_service_account": self.is_service_account,
        }


@dataclass
class APIKey:
    """API Key record."""
    key_id: str
    key_hash: str  # Hashed key value
    user_id: str
    name: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    rate_limit: Optional[int] = None  # Requests per minute
    allowed_ips: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# API Key Store (In-memory for demo, use database in production)
# =============================================================================

class APIKeyStore:
    """In-memory API key store. Replace with database in production."""
    
    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        self._key_hash_to_id: Dict[str, str] = {}
        self._initialize_default_keys()
    
    def _initialize_default_keys(self):
        """Initialize default API keys from environment."""
        # Admin key from environment
        admin_key = os.getenv("API_ADMIN_KEY")
        if admin_key:
            self.create_key(
                key_value=admin_key,
                user_id="admin",
                name="Admin API Key",
                role=UserRole.ADMIN,
            )
            logger.info("Admin API key configured from environment")
        
        # Service key from environment
        service_key = os.getenv("API_SERVICE_KEY")
        if service_key:
            self.create_key(
                key_value=service_key,
                user_id="service",
                name="Service Account Key",
                role=UserRole.SERVICE,
            )
            logger.info("Service API key configured from environment")
        
        # Demo key (only in development)
        if os.getenv("ENABLE_DEMO_KEY", "false").lower() == "true":
            demo_key = "gnosis-demo-key-2024"
            self.create_key(
                key_value=demo_key,
                user_id="demo",
                name="Demo API Key",
                role=UserRole.VIEWER,
            )
            logger.warning(f"Demo API key enabled: {demo_key}")
    
    def _hash_key(self, key_value: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key_value.encode()).hexdigest()
    
    def create_key(
        self,
        key_value: str,
        user_id: str,
        name: str,
        role: UserRole,
        permissions: Optional[List[Permission]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
    ) -> APIKey:
        """Create a new API key."""
        key_id = secrets.token_urlsafe(8)
        key_hash = self._hash_key(key_value)
        
        # Use role permissions if not specified
        if permissions is None:
            permissions = ROLE_PERMISSIONS.get(role, [Permission.READ])
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            role=role,
            permissions=permissions,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            allowed_ips=allowed_ips or [],
        )
        
        self._keys[key_id] = api_key
        self._key_hash_to_id[key_hash] = key_id
        
        return api_key
    
    def validate_key(self, key_value: str) -> Optional[APIKey]:
        """Validate an API key and return the key record if valid."""
        key_hash = self._hash_key(key_value)
        key_id = self._key_hash_to_id.get(key_hash)
        
        if not key_id:
            return None
        
        api_key = self._keys.get(key_id)
        if not api_key:
            return None
        
        # Check if active
        if not api_key.is_active:
            return None
        
        # Check expiration
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            return None
        
        # Update last used
        api_key.last_used = datetime.utcnow()
        
        return api_key
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            return True
        return False
    
    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self._keys.get(key_id)
    
    def list_keys(self, user_id: Optional[str] = None) -> List[APIKey]:
        """List API keys, optionally filtered by user."""
        keys = list(self._keys.values())
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        return keys


# Global API key store
api_key_store = APIKeyStore()


# =============================================================================
# Security Schemes
# =============================================================================

# API Key authentication
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY_NAME, auto_error=False)

# JWT Bearer authentication
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# JWT Functions
# =============================================================================

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire.timestamp(),
        "iat": datetime.utcnow().timestamp(),
    })
    
    if HAS_JOSE:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    else:
        # Simple fallback (not for production!)
        header = base64.urlsafe_b64encode(json.dumps({"alg": ALGORITHM, "typ": "JWT"}).encode()).decode().rstrip("=")
        payload = base64.urlsafe_b64encode(json.dumps(to_encode, default=str).encode()).decode().rstrip("=")
        signature = hmac.new(SECRET_KEY.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()[:32]
        encoded_jwt = f"{header}.{payload}.{signature}"
    
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        if HAS_JOSE:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        else:
            # Simple fallback verification
            parts = token.split(".")
            if len(parts) != 3:
                return None
            
            header, payload_b64, signature = parts
            expected_sig = hmac.new(SECRET_KEY.encode(), f"{header}.{payload_b64}".encode(), hashlib.sha256).hexdigest()[:32]
            
            if not hmac.compare_digest(signature, expected_sig):
                return None
            
            # Decode payload
            padding = 4 - len(payload_b64) % 4
            payload_b64 += "=" * padding
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            
            # Check expiration
            if payload.get("exp", 0) < datetime.utcnow().timestamp():
                return None
        
        return payload
        
    except Exception as e:
        logger.debug(f"Token verification failed: {e}")
        return None


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def get_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> Optional[APIUser]:
    """Get user from API key.
    
    Checks header first, then query parameter.
    """
    api_key = api_key_header or api_key_query
    
    if not api_key:
        return None
    
    # Validate key
    key_record = api_key_store.validate_key(api_key)
    
    if not key_record:
        return None
    
    # Check IP restriction
    if key_record.allowed_ips:
        client_ip = request.client.host if request.client else None
        if client_ip and client_ip not in key_record.allowed_ips:
            logger.warning(f"API key {key_record.key_id} used from unauthorized IP: {client_ip}")
            return None
    
    return APIUser(
        user_id=key_record.user_id,
        role=key_record.role,
        permissions=key_record.permissions,
        api_key_id=key_record.key_id,
        is_service_account=key_record.role == UserRole.SERVICE,
    )


async def get_jwt_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[APIUser]:
    """Get user from JWT token."""
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        return None
    
    # Extract user info from payload
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    role_str = payload.get("role", "viewer")
    try:
        role = UserRole(role_str)
    except ValueError:
        role = UserRole.VIEWER
    
    permissions = payload.get("permissions", [])
    try:
        permissions = [Permission(p) for p in permissions]
    except ValueError:
        permissions = ROLE_PERMISSIONS.get(role, [Permission.READ])
    
    return APIUser(
        user_id=user_id,
        role=role,
        permissions=permissions,
        email=payload.get("email"),
        name=payload.get("name"),
    )


async def get_current_user(
    request: Request,
    api_key_user: Optional[APIUser] = Depends(get_api_key),
    jwt_user: Optional[APIUser] = Depends(get_jwt_user),
) -> Optional[APIUser]:
    """Get current authenticated user from either API key or JWT."""
    return api_key_user or jwt_user


# =============================================================================
# Authentication Decorators/Dependencies
# =============================================================================

def require_api_key(
    api_key_user: Optional[APIUser] = Depends(get_api_key),
) -> APIUser:
    """Require valid API key authentication."""
    if not api_key_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": f"ApiKey realm='{API_KEY_HEADER_NAME}'"},
        )
    return api_key_user


def require_jwt(
    jwt_user: Optional[APIUser] = Depends(get_jwt_user),
) -> APIUser:
    """Require valid JWT authentication."""
    if not jwt_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing JWT token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return jwt_user


def optional_auth(
    user: Optional[APIUser] = Depends(get_current_user),
) -> Optional[APIUser]:
    """Optional authentication - returns None if not authenticated."""
    return user


def require_permission(permission: Permission):
    """Create dependency that requires specific permission.
    
    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: APIUser = Depends(require_permission(Permission.ADMIN))):
            ...
    """
    def dependency(user: APIUser = Depends(require_api_key)) -> APIUser:
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission.value}",
            )
        return user
    
    return dependency


def require_role(role: UserRole):
    """Create dependency that requires specific role.
    
    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: APIUser = Depends(require_role(UserRole.ADMIN))):
            ...
    """
    def dependency(user: APIUser = Depends(require_api_key)) -> APIUser:
        if user.role != role and user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role denied. Required: {role.value}",
            )
        return user
    
    return dependency


# =============================================================================
# Auth API Endpoints (to be included in router)
# =============================================================================

class APIKeyAuth:
    """API Key authentication handler for router."""
    
    @staticmethod
    def generate_key(
        user_id: str,
        name: str,
        role: UserRole = UserRole.VIEWER,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """Generate a new API key.
        
        Returns:
            Tuple of (plain_key, api_key_record)
        """
        plain_key = f"gnosis_{secrets.token_urlsafe(32)}"
        
        api_key = api_key_store.create_key(
            key_value=plain_key,
            user_id=user_id,
            name=name,
            role=role,
            expires_in_days=expires_in_days,
        )
        
        return plain_key, api_key
    
    @staticmethod
    def revoke_key(key_id: str) -> bool:
        """Revoke an API key."""
        return api_key_store.revoke_key(key_id)
    
    @staticmethod
    def list_keys(user_id: Optional[str] = None) -> List[APIKey]:
        """List API keys."""
        return api_key_store.list_keys(user_id)


class JWTAuth:
    """JWT authentication handler."""
    
    @staticmethod
    def create_token(
        user_id: str,
        role: UserRole = UserRole.VIEWER,
        email: Optional[str] = None,
        name: Optional[str] = None,
        expires_minutes: Optional[int] = None,
    ) -> str:
        """Create a JWT token for a user."""
        permissions = ROLE_PERMISSIONS.get(role, [Permission.READ])
        
        data = {
            "sub": user_id,
            "role": role.value,
            "permissions": [p.value for p in permissions],
        }
        
        if email:
            data["email"] = email
        if name:
            data["name"] = name
        
        expires_delta = None
        if expires_minutes:
            expires_delta = timedelta(minutes=expires_minutes)
        
        return create_access_token(data, expires_delta)
    
    @staticmethod
    def verify(token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token."""
        return verify_token(token)
