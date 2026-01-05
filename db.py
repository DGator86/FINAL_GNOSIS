"""Database configuration for GNOSIS trade decision tracking."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Database URL from environment or default to local Postgres
# Railway uses postgres:// but SQLAlchemy 2.0+ requires postgresql://
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://gnosis:gnosis@localhost:5432/gnosis"
)

# Fix Railway's postgres:// URL format for SQLAlchemy compatibility
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    # Ensure we use psycopg2 driver
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Create sessionmaker
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create declarative base for ORM models
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that yields a database session and ensures it is closed.

    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.

    This is a convenience function for development.
    In production, use Alembic migrations instead.
    """
    Base.metadata.create_all(bind=engine)
