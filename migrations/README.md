# GNOSIS Database Migrations

This directory contains SQL migrations for the GNOSIS database.

## Quick Start

### Option 1: Use the Python init script (recommended)

```bash
python init_db.py
```

This will:
- Create all tables using SQLAlchemy models
- Set up indexes and constraints
- Be idempotent (safe to run multiple times)

### Option 2: Run SQL migrations manually

```bash
# Connect to your Postgres database
psql -U gnosis -d gnosis -f migrations/001_create_trade_decisions.sql
```

## Environment Variables

Set these before running:

```bash
export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/gnosis"
```

Or use the default:
```
postgresql+psycopg2://gnosis:gnosis@localhost:5432/gnosis
```

## Migrations

### 001_create_trade_decisions.sql

Creates the `trade_decisions` table, which is the cornerstone of the GNOSIS ML pipeline.

This table captures:
- Meta (timestamp, mode, symbol, direction, structure)
- Universe filter state (price, adv, iv_rank, etc.)
- Engine snapshots (dealer, liquidity, sentiment)
- Agent votes (hedge, liquidity, sentiment)
- Composer decision
- Portfolio context
- Execution outcome (order_id, fills, status)

Indexes:
- `idx_trade_decisions_symbol_time` - For symbol + time queries
- `idx_trade_decisions_mode_time` - For mode + time queries
- `idx_trade_decisions_status` - For status filtering
- `idx_trade_decisions_timestamp` - For time-series queries
- GIN indexes on JSONB fields for feature queries

## Future: Alembic Setup

To use Alembic for proper migration management:

```bash
# Install alembic
pip install alembic

# Initialize alembic
alembic init alembic

# Configure alembic.ini with your DATABASE_URL

# Create migration
alembic revision --autogenerate -m "Create trade_decisions table"

# Run migration
alembic upgrade head
```

For now, the SQL files and init_db.py script are sufficient.
