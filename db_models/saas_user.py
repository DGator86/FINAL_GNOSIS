"""SQLAlchemy model for SaaS user configuration.

This table links the external auth identity (Auth0/Clerk/Cognito)
to internal Gnosis settings like broker keys, risk limits, and subscription tier.
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from db import Base


class SaasUser(Base):
    """
    SaaS User Configuration.
    
    The 'user_id' here matches the 'sub' claim in the JWT token.
    """
    __tablename__ = "saas_users"

    user_id = Column(
        String,
        primary_key=True,
        index=True,
        nullable=False,
    )  # External Auth ID (e.g., 'auth0|123456')

    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    
    # Subscription / Tier
    tier = Column(String, default="free")  # free, pro, enterprise
    is_active = Column(Boolean, default=True)
    
    # Broker Configuration (Encrypted ideally, but keeping simple for framework)
    broker_name = Column(String, default="paper")  # alpaca, ibkr, paper
    broker_api_key = Column(String, nullable=True)
    broker_secret_key = Column(String, nullable=True)
    broker_endpoint = Column(String, nullable=True)
    
    # Risk Limits
    max_position_size = Column(Integer, default=1000)  # USD
    max_drawdown_limit = Column(Integer, default=10)   # Percent
    allowed_symbols = Column(JSONB, default=list)      # Specific watchlist
    
    # Preferences
    theme = Column(String, default="dark")
    notifications_enabled = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return f"<SaasUser(user_id={self.user_id}, email={self.email}, tier={self.tier})>"
