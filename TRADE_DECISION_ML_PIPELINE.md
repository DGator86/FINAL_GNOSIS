# GNOSIS Trade Decision ML Pipeline

## Overview

The Trade Decision ML Pipeline is the cornerstone of GNOSIS's machine learning infrastructure. It captures **every trade decision with full context**, enabling:

- **Performance attribution**: "Are GEX-driven trades better than sentiment trades?"
- **Regime diagnostics**: "Does the Sentiment Agent correctly force defensive structures in risk-off?"
- **Simulation vs Live validation**: Compare decision distributions between backtests and live trading
- **ML training datasets**: Turn trade decisions into labeled examples for supervised/policy learning

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GNOSIS Pipeline                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  Dealer  │  │Liquidity │  │Sentiment │  │  Agents  │           │
│  │  Engine  │  │  Engine  │  │  Engine  │  │ +Composer│           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │             │             │             │                   │
│       └─────────────┴─────────────┴─────────────┘                   │
│                          │                                           │
│                          ▼                                           │
│              ┌───────────────────────┐                              │
│              │ Trade Decision Record │                              │
│              └───────────┬───────────┘                              │
│                          │                                           │
└──────────────────────────┼───────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    POST /trades/       │
              │      decisions         │
              └────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   trade_decisions table      │
         │  (Postgres with JSONB)       │
         └────────┬────────────────────┘
                  │
                  ├─────────────────┐
                  │                 │
                  ▼                 ▼
    ┌───────────────────┐  ┌──────────────────┐
    │   Analytics       │  │   ML Dataset     │
    │   Dashboard       │  │   Builder        │
    └───────────────────┘  └──────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │  ML Training   │
                           │  (XGBoost/NN)  │
                           └────────────────┘
```

## Database Schema

### `trade_decisions` table

```sql
CREATE TABLE trade_decisions (
    -- Identity
    id UUID PRIMARY KEY,

    -- Meta
    timestamp TIMESTAMPTZ NOT NULL,
    mode TEXT NOT NULL,              -- 'live' | 'paper' | 'backtest'
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,         -- 'long' | 'short' | 'neutral'
    structure TEXT NOT NULL,         -- 'call_spread', 'iron_condor', etc.
    config_version TEXT NOT NULL,

    -- Universe / Filters
    universe_eligible BOOLEAN NOT NULL,
    universe_reasons TEXT[] NOT NULL,
    price NUMERIC(18,8) NOT NULL,
    adv NUMERIC(18,2) NOT NULL,
    iv_rank NUMERIC(6,3) NOT NULL,
    realized_vol_30d NUMERIC(8,4) NOT NULL,
    options_liq_score NUMERIC(8,4) NOT NULL,

    -- Engine Snapshots (JSONB)
    dealer_features JSONB NOT NULL,
    liquidity_features JSONB NOT NULL,
    sentiment_features JSONB NOT NULL,

    -- Agent Logic (JSONB)
    hedge_agent_vote JSONB NOT NULL,
    liquidity_agent_vote JSONB NOT NULL,
    sentiment_agent_vote JSONB NOT NULL,
    composer_decision JSONB NOT NULL,

    -- Portfolio Context (JSONB)
    portfolio_context JSONB NOT NULL,

    -- Execution Outcome
    order_id TEXT,
    entry_price NUMERIC(18,8),
    target_price NUMERIC(18,8),
    stop_price NUMERIC(18,8),
    slippage_bps NUMERIC(10,4),
    status TEXT,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

## API Endpoints

### 1. Log Trade Decision

**`POST /trades/decisions`**

Called by GNOSIS pipeline when it decides to take a trade.

**Request Body:**
```json
{
  "timestamp": "2025-12-11T14:30:00Z",
  "mode": "paper",
  "symbol": "SPY",
  "direction": "long",
  "structure": "call_spread",
  "config_version": "v1.0.0",
  "universe_eligible": true,
  "universe_reasons": ["adv_ok", "min_price_ok", "options_liq_ok"],
  "price": 550.25,
  "adv": 85000000.0,
  "iv_rank": 0.65,
  "realized_vol_30d": 0.18,
  "options_liq_score": 0.85,
  "dealer_features": { ... },
  "liquidity_features": { ... },
  "sentiment_features": { ... },
  "hedge_agent_vote": { ... },
  "liquidity_agent_vote": { ... },
  "sentiment_agent_vote": { ... },
  "composer_decision": { ... },
  "portfolio_context": { ... }
}
```

**Response:** `201 Created` with full trade decision record including auto-generated UUID.

### 2. Get Trade Decision

**`GET /trades/decisions/{trade_id}`**

Retrieve a single trade decision by UUID.

### 3. List Trade Decisions

**`GET /trades/decisions`**

List trade decisions with filters:
- `symbol`: Filter by symbol (e.g., "SPY")
- `mode`: Filter by mode ("live", "paper", "backtest")
- `direction`: Filter by direction ("long", "short", "neutral")
- `status`: Filter by execution status
- `limit`: Max results (1-1000, default 100)
- `offset`: Pagination offset

### 4. Update Execution Details

**`PATCH /trades/decisions/{trade_id}/execution`**

Update execution fields after broker response:

```json
{
  "order_id": "abc123",
  "entry_price": 550.30,
  "target_price": 556.00,
  "stop_price": 548.00,
  "slippage_bps": 2.5,
  "status": "filled"
}
```

### 5. Build ML Dataset

**`GET /ml/trades/dataset`**

Build ML-ready training examples from trade decisions.

**Query Parameters:**
- `start_time`: Filter trades after this time
- `end_time`: Filter trades before this time
- `mode`: Filter by mode
- `symbol`: Filter by symbol
- `horizon_days`: Horizon for computing labels (default 5 days)
- `limit`: Max trades to fetch (default 5000, max 50000)

**Response:** List of `TradeMLExample` objects with:
- All decision context (flattened features)
- Computed labels (realized_return, r_multiple, hit_target, etc.)

## ML Dataset Builder

### Components

1. **Trade Fetcher** (`ml/trade_fetcher.py`)
   - Queries `trade_decisions` table with time/symbol/mode filters
   - Optimized for ML use cases (train/test splits, per-symbol models)

2. **Labeler** (`ml/labeling.py`)
   - Computes outcome labels from price evolution:
     - `realized_return`: Percentage PnL
     - `r_multiple`: PnL / risk
     - `max_drawdown_pct`: Worst intratrade drawdown
     - `hit_target`: 1 if target hit before stop, 0 otherwise
     - `stopped_out`: 1 if stop hit, 0 otherwise
     - `horizon_return`: PnL at fixed horizon

3. **Feature Extractor** (`ml/feature_extractor.py`)
   - Flattens JSONB engine/agent snapshots into ML-ready features:
     - `dealer.gex_pressure_score`
     - `liq.dark_pool_activity_score`
     - `sentiment.sentiment_score`
     - `agent_hedge.bias_long_vol`
     - etc.

4. **Dataset Builder** (`ml/dataset_builder.py`)
   - Ties everything together: fetch → label → flatten → ML examples
   - Converts to pandas DataFrame for training

### Usage Example

```python
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from ml.dataset_builder import build_ml_examples_from_trades, ml_examples_to_dataframe

# Build ML examples
examples = build_ml_examples_from_trades(
    db=db_session,
    get_price_series=my_price_provider,
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 12, 1),
    mode="backtest",
    horizon=timedelta(days=5),
    limit=10000,
)

# Convert to DataFrame
df = ml_examples_to_dataframe(examples)

# Save for training
df.to_parquet("datasets/trades_backtest_2025.parquet")

# Train model
from sklearn.ensemble import GradientBoostingRegressor

feature_cols = [c for c in df.columns if c.startswith(('dealer.', 'liq.', 'sentiment.', 'agent_'))]
X = df[feature_cols]
y = df['r_multiple']

model = GradientBoostingRegressor()
model.fit(X, y)
```

## Integration with GNOSIS Pipeline

### 1. Log Decision Before Order

```python
from schemas.trade_decision import TradeDecisionCreate
from crud.trade_decision import create_trade_decision
from db import SessionLocal

# Build decision payload
decision_payload = TradeDecisionCreate(
    mode="live",
    symbol=symbol,
    direction="long",
    structure="call_spread",
    config_version="v1.0.0",
    universe_eligible=True,
    universe_reasons=["adv_ok", "min_price_ok"],
    price=current_price,
    adv=avg_daily_volume,
    iv_rank=iv_rank,
    realized_vol_30d=realized_vol,
    options_liq_score=liq_score,
    dealer_features=hedge_engine.snapshot,
    liquidity_features=liquidity_engine.snapshot,
    sentiment_features=sentiment_engine.snapshot,
    hedge_agent_vote=hedge_agent.vote,
    liquidity_agent_vote=liquidity_agent.vote,
    sentiment_agent_vote=sentiment_agent.vote,
    composer_decision=composer.decision,
    portfolio_context=portfolio.get_context(),
)

# Log to database
db = SessionLocal()
trade_record = create_trade_decision(db, decision_payload)
trade_id = trade_record.id
db.close()

# Now submit order to broker
order_response = broker.submit_order(...)
```

### 2. Update After Execution

```python
from schemas.trade_decision import TradeDecisionUpdateExecution
from crud.trade_decision import update_trade_execution

# After broker response
execution_update = TradeDecisionUpdateExecution(
    order_id=order_response.order_id,
    entry_price=order_response.filled_price,
    slippage_bps=compute_slippage(order_response),
    status="filled",
)

db = SessionLocal()
update_trade_execution(db, trade_id, execution_update)
db.close()
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install sqlalchemy psycopg2-binary pandas fastapi uvicorn
```

### 2. Set Up Database

**Option A: Use init_db.py (recommended)**

```bash
export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/gnosis"
python init_db.py
```

**Option B: Run SQL migration**

```bash
psql -U gnosis -d gnosis -f migrations/001_create_trade_decisions.sql
```

### 3. Start API

```bash
uvicorn web_api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test with Sample Data

```bash
curl -X POST http://localhost:8000/trades/decisions \
  -H 'Content-Type: application/json' \
  -d @test_data/sample_trade_decision.json
```

### 5. Query Decisions

```bash
# List all decisions
curl http://localhost:8000/trades/decisions

# Filter by symbol
curl http://localhost:8000/trades/decisions?symbol=SPY

# Get ML dataset
curl "http://localhost:8000/ml/trades/dataset?mode=backtest&limit=100"
```

## ML Use Cases

### 1. Trade Scoring Model

Predict expected R-multiple or win probability. Composer uses this to rank candidates.

```python
# Features: full GNOSIS state
X = df[feature_cols]

# Target: realized R-multiple
y = df['r_multiple']

# Train
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Use in Composer
predicted_r = model.predict(current_state)
if predicted_r > threshold:
    take_trade()
```

### 2. Structure Selector

Predict which structure (call, spread, fly) gives best risk-adjusted outcome.

```python
# Features: GNOSIS state
X = df[feature_cols]

# Target: structure with best R-multiple
y = df['structure']

# Train classifier
model = RandomForestClassifier()
model.fit(X, y)

# Use in Composer
best_structure = model.predict(current_state)
```

### 3. Regime-Aware Policy

Learn to skip trades in unfavorable environments.

```python
# Filter to only successful trades (r_multiple > 1.0)
df_winners = df[df['r_multiple'] > 1.0]

# Features include regime signals
X = df[regime_feature_cols]

# Binary classification: take_trade or skip
y = (df['r_multiple'] > 1.0).astype(int)

model = XGBClassifier()
model.fit(X, y)
```

## Maintenance

### Backup

```bash
# Backup trade decisions table
pg_dump -U gnosis -d gnosis -t trade_decisions > trade_decisions_backup.sql

# Backup to CSV
psql -U gnosis -d gnosis -c "COPY trade_decisions TO STDOUT WITH CSV HEADER" > trade_decisions.csv
```

### Archiving

For production, consider:
- Time-based partitioning (monthly or quarterly)
- Archiving old backtests to S3/cold storage
- Keeping only live/paper trades in hot database

### Monitoring

Key metrics to track:
- Decisions logged per day
- Distribution of modes (live vs paper vs backtest)
- Average decision context size (JSONB)
- ML dataset query performance

## Troubleshooting

### Database Connection Issues

```python
# Check DATABASE_URL
import os
print(os.getenv("DATABASE_URL"))

# Test connection
from db import engine
with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print("✓ Database connected")
```

### Missing Price Data for Labeling

The `/ml/trades/dataset` endpoint requires a `get_price_series` function. Wire it to your price store:

```python
# In routers/ml_trades.py
def get_price_series_adapter(symbol, start, end):
    # Example: query your bars table
    from storage.bars import query_bars
    bars = query_bars(symbol, start, end, timeframe='5m')
    return [bar.close for bar in bars]
```

### JSONB Query Optimization

If querying inside JSONB fields is slow, add GIN indexes:

```sql
CREATE INDEX idx_dealer_gex ON trade_decisions
USING GIN ((dealer_features->'gex_pressure_score'));
```

## Next Steps

1. **Wire to GNOSIS Pipeline**: Add `create_trade_decision()` calls before order submission
2. **Implement Price Provider**: Wire `get_price_series_adapter` to your historical price store
3. **Build First ML Model**: Train a trade scoring model on backtest data
4. **Dashboard Integration**: Show trade decision stats in your monitoring dashboard
5. **Automate ML Retraining**: Set up periodic jobs to rebuild datasets and retrain models

---

**This is no longer "cool architecture" — it's a real ML lab.**

Every trade is a labeled example. Every decision is auditable. Every agent's contribution is measurable.

Now you can **prove** what works.
