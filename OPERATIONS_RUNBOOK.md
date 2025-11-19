# Operations Runbook

Practical guide for running Super Gnosis V2 safely day-to-day.

## Environments

| Environment | Purpose | Broker | Risk Caps | Data Source |
|-------------|---------|--------|-----------|-------------|
| **Dev** | Development, testing, experimentation | Simulated | Unlimited (fake money) | Stub/historical |
| **Paper** | Pre-production validation with live data | Simulated | Same as live | Live market data |
| **Live** | Real trading with real capital | Live (Alpaca/IBKR) | Hard per-trade and daily limits | Live market data |

## Prerequisites

### System Requirements
- Python 3.12+
- 8GB RAM minimum (16GB recommended for optimization)
- Unix-like environment (Linux, macOS, WSL2)

### Installation

```bash
# Clone repository
git clone https://github.com/DGator86/V2---Gnosis.git
cd V2---Gnosis

# Install dependencies
pip install -r requirements.txt

# Install Optuna for optimization
pip install optuna

# Verify installation
pytest --version
optuna --version
```

### Configuration

1. **Create `config/config.yaml`** (if not exists):

```yaml
runtime:
  log_level: INFO
  checkpoints_dir: logs/checkpoints
  enable_checkpointing: true

hedge:
  gamma_squeeze_threshold: 1000000.0
  vanna_flow_threshold: 500000.0

liquidity:
  lookback: 30
  thin_threshold: 0.001

sentiment:
  bullish_threshold: 0.3
  bearish_threshold: 0.3

volatility:
  lookback_days: 252
  iv_rank_threshold: 0.7

trade_agent:
  default_capital: 100000.0

execution:
  initial_cash: 100000.0
  slippage_pct: 0.001
  commission_per_contract: 0.65
```

2. **Set environment variables**:

```bash
# Kill switch (disables live execution)
export GNOSIS_LIVE_TRADING_ENABLED=false

# Risk caps
export GNOSIS_MAX_TRADE_RISK_PCT=0.02  # 2% per trade
export GNOSIS_MAX_DAILY_LOSS=5000.0    # $5k daily loss limit

# Logging
export GNOSIS_LOG_LEVEL=INFO
```

## Testing

### Quick Test Suite (CI/Dev)

```bash
# Run fast tests (excludes regime-specific performance tests)
pytest

# Expected: 479/487 passing (98.4%), 8 skipped

# Fail-fast mode (stop on first failure)
pytest -x

# Verbose mode
pytest -v

# Run specific test file
pytest tests/agents/test_trade_agent_v2.py
```

### Extended Diagnostics (Pre-Production)

```bash
# Run all tests including slow regime-specific tests
pytest -m regime_slow

# Run full suite with coverage
pytest --cov=agents --cov=engines --cov=execution --cov=optimizer

# Expected coverage: >85% overall
```

### Critical Test Checklist (Before Live)

Before enabling live trading, ensure:

- [ ] **All core tests passing**: `pytest` shows 479+ passing
- [ ] **No critical failures**: agents, engines, optimizer, execution modules
- [ ] **Risk analyzer tests pass**: Verify PnL cone, breakeven, max loss calculations
- [ ] **Exit manager tests pass**: Verify stop loss, profit target, trailing stop logic
- [ ] **Execution tests pass**: Verify broker adapter, order routing, cost models
- [ ] **Optimizer tests pass**: Verify Optuna integration, seed reproducibility

## SPY-Only Daily Paper Pipeline

### Phase 3 Preparation (Current)

Daily paper trading workflow for SPY to validate end-to-end behavior.

#### Setup

```bash
# Create runs directory for persisted outputs
mkdir -p runs

# Create logs directory
mkdir -p logs

# Configure environment variables (copy from .env.example)
cp .env.example .env
# Edit .env and add your Alpaca API credentials
```

#### Alpaca Broker Setup (Optional)

To use Alpaca's paper trading instead of the simulated broker:

1. **Get Alpaca Paper Trading Credentials**:
   - Sign up at https://app.alpaca.markets/
   - Navigate to Paper Trading: https://app.alpaca.markets/paper/dashboard/overview
   - Generate API keys from "Your API Keys" section

2. **Configure Environment Variables**:
   ```bash
   # Add to .env file:
   ALPACA_API_KEY=PKELRVPCSCXJOWMMQIQEHH3LOJ
   ALPACA_SECRET_KEY=Ezf2dYZD1M85tYBGUbJb7374rmmyz6H7eTtWNhk6mMZx
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```

3. **Load Environment Variables**:
   ```bash
   # Option 1: Source .env in shell (requires python-dotenv)
   export $(cat .env | xargs)
   
   # Option 2: Use direnv (recommended)
   echo 'dotenv' > .envrc
   direnv allow
   ```

4. **Test Connection**:
   ```python
   from pipeline.full_pipeline import create_alpaca_broker
   
   broker = create_alpaca_broker(paper=True)
   account = broker.get_account()
   print(f"Connected! Buying power: ${account.buying_power:,.2f}")
   ```

#### Execution

```bash
# Dry-run (generate ideas, no execution)
python scripts/run_daily_spy_paper.py

# Execute in simulated broker (default)
python scripts/run_daily_spy_paper.py --execute

# Execute with Alpaca paper trading (requires ALPACA_API_KEY env)
python scripts/run_daily_spy_paper.py --execute --broker alpaca

# Custom parameters
python scripts/run_daily_spy_paper.py \
  --execute \
  --broker alpaca \
  --underlying-price 460.0 \
  --capital 50000.0 \
  --output-dir runs
```

#### Output Structure

```
runs/
└── YYYY-MM-DD/
    ├── SPY_context.jsonl    # Composer context, market state, hyperparams
    ├── SPY_trades.jsonl     # Trade ideas with ranking, risk metrics
    └── SPY_orders.jsonl     # Execution results (if --execute used)
```

Each file is JSONL (JSON Lines) format with one entry per line:

```json
{"as_of": "2025-01-15T14:30:00", "symbol": "SPY", "context": {...}}
{"as_of": "2025-01-15T14:30:00", "symbol": "SPY", "trade": {...}}
{"as_of": "2025-01-15T14:30:00", "symbol": "SPY", "order": {...}}
```

#### Automation (Optional)

```bash
# Cron job (daily at 3:30 PM ET)
30 15 * * 1-5 cd /path/to/V2---Gnosis && python scripts/run_daily_spy_paper.py --execute

# Systemd timer (example)
# /etc/systemd/system/gnosis-spy-paper.timer
[Unit]
Description=Daily SPY paper trading

[Timer]
OnCalendar=Mon-Fri 15:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

### 2-4 Week Review

After accumulating 10-20 trading days of data:

1. **Strategy Distribution Analysis**

```bash
# Count strategies by type
jq -r '.trade.strategy_type' runs/*/SPY_trades.jsonl | sort | uniq -c

# Expected distribution:
# - Bullish regimes: Long calls, call debit spreads, diagonals
# - Bearish regimes: Long puts, put debit spreads
# - Neutral regimes: Iron condors, strangles
# - Vol expansion: Straddles, reverse iron condors
```

2. **Exit Behavior Validation**

```bash
# Extract exit rules from context
jq '.context.exit_rules' runs/*/SPY_context.jsonl

# Verify:
# - High confidence → wider stops (smaller stop_loss_pct)
# - Directional strategies → 50% profit target
# - Income strategies → 50% profit target, tighter stops
# - Vol strategies → 75% profit target, wider stops
```

3. **Risk Metrics Check**

```bash
# Max loss distribution
jq '.trade.max_loss' runs/*/SPY_trades.jsonl | \
  python -c "import sys, statistics; vals=[float(x) for x in sys.stdin]; print(f'Mean: {statistics.mean(vals):.2f}, Median: {statistics.median(vals):.2f}, Max: {max(vals):.2f}')"

# All max_loss values should be negative
# Typical range: -$200 to -$2000 per trade
```

4. **Vol Routing Validation**

```bash
# IV rank when straddles selected
jq 'select(.trade.strategy_type=="straddle") | .context.iv_rank' runs/*/SPY_trades.jsonl

# Expected: IV rank > 0.7 (high vol regimes favor long premium)
```

## Persistence Strategy

### Recommended: JSONL (Current)

- **Pros**: Simple, human-readable, append-only, easy to parse
- **Cons**: No indexing, full scans for queries
- **Use case**: Small to medium data (1-2 years of daily runs)

### Alternative: DuckDB (Future)

For larger datasets or complex queries:

```python
import duckdb

# Create database
conn = duckdb.connect('gnosis_history.duckdb')

# Create tables
conn.execute("""
    CREATE TABLE contexts (
        as_of TIMESTAMP,
        symbol VARCHAR,
        direction VARCHAR,
        confidence DOUBLE,
        iv_rank DOUBLE,
        ...
    )
""")

# Bulk load from JSONL
conn.execute("COPY contexts FROM 'runs/*/SPY_context.jsonl' (FORMAT JSON)")

# Query
conn.execute("SELECT symbol, AVG(confidence) FROM contexts GROUP BY symbol").fetchall()
```

## Risk Caps

### Per-Trade Limits

Hard-coded in `optimizer/kelly_refinement.py`:

```python
# Global risk cap: 2% of capital per trade
global_risk_cap = 0.02

# Kelly fraction clamped to ±25%
def clamp_kelly_fraction(kelly: float) -> float:
    return max(-0.25, min(0.25, kelly))
```

### Daily Loss Limit

Enforced in pipeline orchestration:

```python
# Check daily P&L before each trade
if daily_pnl < -5000.0:  # $5k daily loss limit
    logger.critical("Daily loss limit reached. Halting trading.")
    raise DailyLossLimitExceeded()
```

### Circuit Breakers

1. **Kill Switch** (environment variable):

```bash
export GNOSIS_LIVE_TRADING_ENABLED=false
```

2. **Manual Override** (config file):

```yaml
execution:
  enable_live_trading: false  # Overrides env var
```

3. **Emergency Stop**:

```bash
# Kill all running processes
pkill -f "run_daily_spy_paper.py"

# Or use process manager
pm2 stop gnosis-pipeline
```

## Phase 4 Rollout Plan

### Phase 1: SPY Paper (Current)

- **Duration**: 2-4 weeks
- **Symbols**: SPY only
- **Mode**: Paper (simulated broker)
- **Execution**: Daily at 3:30 PM ET
- **Review**: Weekly strategy distribution, exit behavior, risk metrics

### Phase 2: Multi-Symbol Paper

- **Duration**: 2-4 weeks
- **Symbols**: SPY, QQQ, IWM, DIA
- **Mode**: Paper (simulated broker)
- **Focus**: Symbol-specific behavior, correlation analysis
- **Review**: Strategy performance across symbols, regime consistency

### Phase 3: Small Live Pilot

- **Duration**: 4-8 weeks
- **Symbols**: SPY only
- **Mode**: Live (real broker, real capital)
- **Capital**: $10k-$25k
- **Risk Caps**: 1% per trade, $500 daily loss
- **Review**: Fill quality, slippage, commission accuracy, real P&L vs paper

### Phase 4: Scale Up

- **Duration**: Ongoing
- **Symbols**: Expand to 10-20 symbols
- **Capital**: $50k-$250k+
- **Risk Caps**: 2% per trade, $5k daily loss
- **Focus**: Portfolio-level risk, correlation hedging, multi-symbol optimization

## Live Trading Checklist

Before enabling live trading:

- [ ] **Paper trading completed**: 2-4 weeks of SPY paper, reviewed
- [ ] **Risk caps configured**: Per-trade and daily limits set
- [ ] **Kill switches tested**: Environment variable and config override
- [ ] **Broker credentials secured**: API keys stored in secrets manager
- [ ] **Monitoring enabled**: Logs, alerts, P&L tracking
- [ ] **Tests passing**: 98%+ test coverage, no critical failures
- [ ] **Emergency contacts**: Phone numbers for manual intervention
- [ ] **Backup plan**: Procedure for closing all positions immediately

## Monitoring & Logging

### Log Locations

```
logs/
├── checkpoints/          # Engine output snapshots
├── pipeline_YYYYMMDD.log # Daily pipeline logs
├── errors_YYYYMMDD.log   # Error-only logs
└── audit_YYYYMMDD.log    # Order placement audit trail
```

### Log Levels

```python
# Development
GNOSIS_LOG_LEVEL=DEBUG

# Paper trading
GNOSIS_LOG_LEVEL=INFO

# Live trading
GNOSIS_LOG_LEVEL=WARNING  # Only warnings and errors
```

### Key Log Messages to Monitor

- `CRITICAL: Daily loss limit reached` → Halt immediately
- `ERROR: Order placement failed` → Investigate broker connection
- `WARNING: Slippage exceeds 1%` → Review execution quality
- `INFO: SPY paper trading complete` → Verify output persisted

## Troubleshooting

### Tests Failing

```bash
# Run verbose to see failure details
pytest -xvs tests/agents/test_trade_agent_v2.py

# Check for common issues:
# - Pydantic model changes: Update test data
# - Pricing model changes: Relax hard-coded assertions
# - New fields added: Update test validation
```

### Pipeline Not Producing Trade Ideas

```bash
# Check composer context
jq '.context.direction, .context.confidence' runs/*/SPY_context.jsonl

# If direction=neutral and confidence<0.5:
# - Trade agent may filter out low-confidence ideas
# - Increase confidence thresholds in agents
```

### Simulated Broker Rejecting Orders

```bash
# Check cash balance
jq '.order.error_message' runs/*/SPY_orders.jsonl | grep "Insufficient funds"

# Solution: Increase initial_cash in config
```

### Optuna Tests Flaky

```bash
# Ensure deterministic seeds
pytest tests/optimizer/test_strategy_optimizer.py::TestOptunaIntegration -v

# Check: All best_value results should be identical across runs
```

## Support & Contact

- **GitHub Issues**: https://github.com/DGator86/V2---Gnosis/issues
- **Dev Documentation**: [DEV_GUIDE.md](DEV_GUIDE.md)
- **Architecture**: [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)
