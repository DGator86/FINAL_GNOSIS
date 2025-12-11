#!/usr/bin/env python3
"""Initialize GNOSIS database.

This script:
1. Creates all tables defined in SQLAlchemy models
2. Sets up indexes and constraints
3. Is idempotent (safe to run multiple times)

Usage:
    python init_db.py

Environment Variables:
    DATABASE_URL - Postgres connection string
                   Default: postgresql+psycopg2://gnosis:gnosis@localhost:5432/gnosis
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from db import Base, engine, DATABASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_db():
    """Initialize database by creating all tables."""
    logger.info("=" * 60)
    logger.info("GNOSIS Database Initialization")
    logger.info("=" * 60)
    logger.info(f"Database URL: {DATABASE_URL}")
    logger.info("")

    try:
        # Import all models to ensure they're registered with Base
        logger.info("Importing database models...")
        from db_models.trade_decision import TradeDecision
        logger.info(f"  ✓ TradeDecision model loaded")

        # Create all tables
        logger.info("")
        logger.info("Creating tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("  ✓ All tables created successfully")

        # List created tables
        logger.info("")
        logger.info("Created tables:")
        for table in Base.metadata.sorted_tables:
            logger.info(f"  - {table.name}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Database initialization complete!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Start the API: uvicorn api.main:app --reload")
        logger.info("2. Test trade decision logging:")
        logger.info("   curl -X POST http://localhost:8000/trades/decisions \\")
        logger.info("        -H 'Content-Type: application/json' \\")
        logger.info("        -d @test_data/sample_trade_decision.json")
        logger.info("")

        return True

    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error("Database initialization FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("1. Check DATABASE_URL is correct")
        logger.error("2. Ensure Postgres is running")
        logger.error("3. Verify database exists and user has permissions")
        logger.error("4. Check Postgres logs for detailed error messages")
        logger.error("")
        return False


if __name__ == "__main__":
    success = init_db()
    sys.exit(0 if success else 1)
