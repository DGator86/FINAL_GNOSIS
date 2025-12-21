"""
Authentication API Router

Provides endpoints for:
- API key management
- JWT token generation
- User authentication

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from middleware.auth import (
    APIKeyAuth,
    JWTAuth,
    APIUser,
    UserRole,
    Permission,
    require_api_key,
    require_permission,
    get_current_user,
    optional_auth,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


# =============================================================================
# Request/Response Models
# =============================================================================

class TokenRequest(BaseModel):
    """Request model for token generation."""
    user_id: str = Field(..., description="User ID for the token")
    role: str = Field(default="viewer", description="User role (viewer, trader, analyst, admin)")
    email: Optional[str] = Field(None, description="Optional email")
    name: Optional[str] = Field(None, description="Optional name")
    expires_minutes: Optional[int] = Field(None, description="Token expiration in minutes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "role": "trader",
                "email": "user@example.com",
                "name": "John Doe",
                "expires_minutes": 60
            }
        }


class TokenResponse(BaseModel):
    """Response model for token generation."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    role: str


class APIKeyRequest(BaseModel):
    """Request model for API key generation."""
    name: str = Field(..., description="Descriptive name for the API key")
    role: str = Field(default="viewer", description="Role for the API key")
    expires_in_days: Optional[int] = Field(None, description="Expiration in days (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Production Trading Key",
                "role": "trader",
                "expires_in_days": 365
            }
        }


class APIKeyResponse(BaseModel):
    """Response model for API key generation."""
    key: str = Field(..., description="The API key (only shown once!)")
    key_id: str
    name: str
    role: str
    expires_at: Optional[str]
    message: str = "Store this key securely - it won't be shown again!"


class APIKeyInfo(BaseModel):
    """Information about an API key (without the key itself)."""
    key_id: str
    name: str
    role: str
    created_at: str
    expires_at: Optional[str]
    last_used: Optional[str]
    is_active: bool


class UserInfo(BaseModel):
    """Current user information."""
    user_id: str
    role: str
    permissions: List[str]
    email: Optional[str]
    name: Optional[str]
    is_service_account: bool


# =============================================================================
# Public Endpoints
# =============================================================================

@router.get("/status")
async def auth_status() -> Dict[str, Any]:
    """Check authentication status.
    
    Returns information about authentication configuration.
    """
    return {
        "auth_enabled": True,
        "methods": ["api_key", "jwt"],
        "api_key_header": "X-API-Key",
        "jwt_scheme": "Bearer",
    }


# =============================================================================
# Token Endpoints
# =============================================================================

@router.post("/token", response_model=TokenResponse)
async def create_token(
    request: TokenRequest,
    user: APIUser = Depends(require_permission(Permission.ADMIN)),
) -> TokenResponse:
    """Generate a JWT token.
    
    Requires admin permission to generate tokens for other users.
    """
    try:
        role = UserRole(request.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {request.role}. Valid roles: {[r.value for r in UserRole]}",
        )
    
    expires_minutes = request.expires_minutes or 60
    
    token = JWTAuth.create_token(
        user_id=request.user_id,
        role=role,
        email=request.email,
        name=request.name,
        expires_minutes=expires_minutes,
    )
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_minutes * 60,
        user_id=request.user_id,
        role=request.role,
    )


@router.post("/token/verify")
async def verify_token(token: str) -> Dict[str, Any]:
    """Verify a JWT token and return its payload.
    
    Args:
        token: JWT token to verify
    """
    payload = JWTAuth.verify(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    return {
        "valid": True,
        "payload": payload,
    }


# =============================================================================
# API Key Endpoints
# =============================================================================

@router.post("/keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest,
    user: APIUser = Depends(require_permission(Permission.ADMIN)),
) -> APIKeyResponse:
    """Generate a new API key.
    
    Requires admin permission.
    
    **Important**: The key is only shown once! Store it securely.
    """
    try:
        role = UserRole(request.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {request.role}. Valid roles: {[r.value for r in UserRole]}",
        )
    
    plain_key, api_key = APIKeyAuth.generate_key(
        user_id=user.user_id,
        name=request.name,
        role=role,
        expires_in_days=request.expires_in_days,
    )
    
    return APIKeyResponse(
        key=plain_key,
        key_id=api_key.key_id,
        name=api_key.name,
        role=api_key.role.value,
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
    )


@router.get("/keys", response_model=List[APIKeyInfo])
async def list_api_keys(
    user: APIUser = Depends(require_api_key),
) -> List[APIKeyInfo]:
    """List API keys for the current user.
    
    Admins can see all keys.
    """
    # Admins see all keys, others see only their own
    user_id = None if user.has_permission(Permission.ADMIN) else user.user_id
    
    keys = APIKeyAuth.list_keys(user_id)
    
    return [
        APIKeyInfo(
            key_id=k.key_id,
            name=k.name,
            role=k.role.value,
            created_at=k.created_at.isoformat(),
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
            last_used=k.last_used.isoformat() if k.last_used else None,
            is_active=k.is_active,
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: APIUser = Depends(require_permission(Permission.ADMIN)),
) -> Dict[str, Any]:
    """Revoke an API key.
    
    Requires admin permission.
    """
    success = APIKeyAuth.revoke_key(key_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        )
    
    return {
        "status": "revoked",
        "key_id": key_id,
    }


# =============================================================================
# User Info Endpoints
# =============================================================================

@router.get("/me", response_model=UserInfo)
async def get_current_user_info(
    user: APIUser = Depends(require_api_key),
) -> UserInfo:
    """Get information about the currently authenticated user."""
    return UserInfo(
        user_id=user.user_id,
        role=user.role.value,
        permissions=[p.value for p in user.permissions],
        email=user.email,
        name=user.name,
        is_service_account=user.is_service_account,
    )


@router.get("/permissions")
async def list_permissions() -> Dict[str, Any]:
    """List all available permissions and roles."""
    return {
        "permissions": [p.value for p in Permission],
        "roles": {
            role.value: [p.value for p in perms]
            for role, perms in {
                UserRole.VIEWER: [Permission.READ],
                UserRole.TRADER: [Permission.READ, Permission.WRITE, Permission.TRADE, Permission.WEBSOCKET],
                UserRole.ANALYST: [Permission.READ, Permission.ANALYTICS, Permission.WEBSOCKET],
                UserRole.ADMIN: list(Permission),
                UserRole.SERVICE: list(Permission),
            }.items()
        },
    }


# =============================================================================
# Health Check (Authenticated)
# =============================================================================

@router.get("/check")
async def auth_check(
    user: Optional[APIUser] = Depends(optional_auth),
) -> Dict[str, Any]:
    """Check if current request is authenticated.
    
    Returns authentication status without requiring auth.
    """
    if user:
        return {
            "authenticated": True,
            "user_id": user.user_id,
            "role": user.role.value,
        }
    
    return {
        "authenticated": False,
        "message": "No valid credentials provided",
    }
