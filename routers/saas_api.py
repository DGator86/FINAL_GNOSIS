"""
SaaS API Router

Endpoints for the SaaS frontend to consume:
- User Profile & Config
- Dashboard Data
- Subscription Management (Placeholder)
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from db import get_db
from db_models.saas_user import SaasUser
from db_models.trade_decision import TradeDecision
from middleware.auth import APIUser, get_current_user, require_jwt
from pydantic import BaseModel

router = APIRouter(prefix="/saas", tags=["saas"])


# --- Pydantic Schemas ---

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    broker_name: Optional[str] = None
    broker_api_key: Optional[str] = None
    broker_secret_key: Optional[str] = None
    max_position_size: Optional[int] = None
    notifications_enabled: Optional[bool] = None

class UserProfileResponse(BaseModel):
    user_id: str
    email: Optional[str]
    tier: str
    broker_name: str
    max_position_size: int
    created_at: Any

    class Config:
        orm_mode = True

# --- Endpoints ---

@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    user: APIUser = Depends(require_jwt),
    db: Session = Depends(get_db)
):
    """Get current user's SaaS profile."""
    # Check if user exists in DB, if not create them (lazy registration)
    db_user = db.query(SaasUser).filter(SaasUser.user_id == user.user_id).first()
    
    if not db_user:
        # Create default profile for new user
        db_user = SaasUser(
            user_id=user.user_id,
            email=user.email,
            full_name=user.name
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
    return db_user


@router.patch("/profile")
async def update_profile(
    profile_update: UserProfileUpdate,
    user: APIUser = Depends(require_jwt),
    db: Session = Depends(get_db)
):
    """Update user configuration."""
    db_user = db.query(SaasUser).filter(SaasUser.user_id == user.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User profile not found")
        
    # Update fields if provided
    update_data = profile_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_user, key, value)
        
    db.commit()
    return {"status": "updated", "user_id": user.user_id}


@router.get("/dashboard/summary")
async def get_dashboard_summary(
    user: APIUser = Depends(require_jwt),
    db: Session = Depends(get_db)
):
    """Get summary stats for the user's dashboard."""
    
    # 1. Get recent trades
    recent_trades = db.query(TradeDecision)\
        .filter(TradeDecision.user_id == user.user_id)\
        .order_by(desc(TradeDecision.timestamp))\
        .limit(5)\
        .all()
        
    # 2. Calculate simple stats (this would be cached in production)
    total_trades = db.query(TradeDecision).filter(TradeDecision.user_id == user.user_id).count()
    
    return {
        "total_trades": total_trades,
        "recent_activity": [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "mode": t.mode,
                "time": t.timestamp
            } for t in recent_trades
        ],
        "system_status": "operational"
    }
