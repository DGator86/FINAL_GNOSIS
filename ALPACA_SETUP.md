# Alpaca Broker Integration Setup Guide

## Overview

This guide walks you through setting up Alpaca broker integration for paper and live trading with the Gnosis trading system.

**Alpaca Features**:
- ✅ Commission-free stock and ETF trading
- ✅ Paper trading with $100k+ virtual capital
- ✅ Real-time market data
- ✅ REST API + WebSocket streaming
- ✅ Fractional shares support

---

## Step 1: Create Alpaca Account

### 1.1 Sign Up for Paper Trading (Free)

1. Visit https://app.alpaca.markets/signup
2. Create your account (no funding required for paper trading)
3. Verify your email address
4. Navigate to **Paper Trading** dashboard

### 1.2 Generate API Keys

1. Go to https://app.alpaca.markets/paper/dashboard/overview
2. Click on **"Your API Keys"** in the left sidebar
3. Click **"Generate New Key"**
4. **Important**: Save both the **API Key** and **Secret Key** immediately
   - You won't be able to see the secret key again
   - Store them securely (password manager, `.env` file)

**Example API Keys** (yours will be different):
```
API Key:    PKELRVPCSCXJOWMMQIQEHH3LOJ
Secret Key: Ezf2dYZD1M85tYBGUbJb7374rmmyz6H7eTtWNhk6mMZx
```

---

## Step 2: Configure Environment Variables

### 2.1 Copy Environment Template

```bash
cd /path/to/V2---Gnosis
cp .env.example .env
```

### 2.2 Add Your Alpaca Credentials

Edit `.env` and add your API keys:

```bash
# ============================================================================
# ALPACA API (Broker for Live/Paper Trading)
# ============================================================================
ALPACA_API_KEY=PKELRVPCSCXJOWMMQIQEHH3LOJ  # Your API key here
ALPACA_SECRET_KEY=Ezf2dYZD1M85tYBGUbJb7374rmmyz6H7eTtWNhk6mMZx  # Your secret key here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading URL

# For live trading (⚠️ USE WITH EXTREME CAUTION):
# ALPACA_BASE_URL=https://api.alpaca.markets
```

### 2.3 Load Environment Variables

**Option 1: Manual Export (Temporary)**
```bash
export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

**Option 2: Source .env File (Bash)**
```bash
export $(cat .env | xargs)
```

**Option 3: Use direnv (Recommended)**
```bash
# Install direnv: https://direnv.net/
brew install direnv  # macOS
# or apt-get install direnv  # Linux

# Add to .envrc
echo 'dotenv' > .envrc
direnv allow
```

**Option 4: python-dotenv (Automatic)**
The system automatically loads `.env` using python-dotenv when scripts run.

---

## Step 3: Test Connection

### 3.1 Interactive Python Test

```python
from dotenv import load_dotenv
load_dotenv()

from pipeline.full_pipeline import create_alpaca_broker

# Create broker adapter
broker = create_alpaca_broker(paper=True)

# Test connection
account = broker.get_account()
print(f"✓ Connected to Alpaca Paper Trading")
print(f"  Account ID: {account.account_id}")
print(f"  Cash: ${account.cash:,.2f}")
print(f"  Buying Power: ${account.buying_power:,.2f}")
print(f"  Portfolio Value: ${account.portfolio_value:,.2f}")
print(f"  Positions: {len(account.positions)}")
```

**Expected Output**:
```
✓ Connected to Alpaca Paper Trading
  Account ID: PA3K2QXXXX
  Cash: $100,000.00
  Buying Power: $400,000.00
  Portfolio Value: $100,000.00
  Positions: 0
```

### 3.2 Test with SPY Pipeline

```bash
# Test with simulated broker (no Alpaca required)
python scripts/run_daily_spy_paper.py

# Test with Alpaca paper trading
python scripts/run_daily_spy_paper.py --execute --broker alpaca

# Full test with custom capital
python scripts/run_daily_spy_paper.py \
  --execute \
  --broker alpaca \
  --capital 50000 \
  --underlying-price 455.0
```

---

## Step 4: Paper Trading Workflow

### 4.1 Daily SPY Paper Trading

```bash
# Morning: Dry-run to preview trade ideas
python scripts/run_daily_spy_paper.py --broker alpaca

# Afternoon: Execute after market analysis
python scripts/run_daily_spy_paper.py --execute --broker alpaca
```

### 4.2 Monitoring Paper Trades

**Check Account Status**:
```python
from pipeline.full_pipeline import create_alpaca_broker

broker = create_alpaca_broker(paper=True)
account = broker.get_account()

print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
print(f"Cash: ${account.cash:,.2f}")
print(f"P&L: ${account.portfolio_value - 100000:+,.2f}")

# Check positions
for pos in account.positions:
    print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_entry_price:.2f}")
    print(f"    Current: ${pos.current_price:.2f} | P&L: ${pos.unrealized_pnl:+,.2f}")
```

**Review Orders**:
```bash
# Access Alpaca dashboard
open https://app.alpaca.markets/paper/trading/account-overview
```

### 4.3 Analysis Tools

```bash
# Daily report
python scripts/analysis/daily_report.py --days 7

# Regime correlation
python scripts/analysis/regime_correlation.py --days 14

# Exit trigger analysis
python scripts/analysis/exit_tracker.py --days 7
```

---

## Step 5: Live Trading Transition (⚠️ CAUTION)

### 5.1 Prerequisites

**Before transitioning to live trading**:

- [ ] Complete 2-4 weeks of paper trading validation
- [ ] 100% fill rate in paper trading
- [ ] Strategy selection adapts correctly to all regime buckets
- [ ] Exit rules configured for 100% of trades
- [ ] No trades exceed 2% risk cap
- [ ] Documented edge cases and unexpected behavior
- [ ] Funded Alpaca live trading account
- [ ] Risk management safeguards in place

### 5.2 Risk Management Configuration

Update `.env` for live trading:

```bash
# ============================================================================
# ALPACA API (LIVE TRADING - ⚠️ REAL MONEY)
# ============================================================================
ALPACA_API_KEY=your_live_api_key
ALPACA_SECRET_KEY=your_live_secret_key
ALPACA_BASE_URL=https://api.alpaca.markets  # Live trading URL

# ============================================================================
# RISK MANAGEMENT (CRITICAL FOR LIVE TRADING)
# ============================================================================
MAX_POSITION_SIZE_PCT=2.0  # Max 2% of capital per trade
MAX_DAILY_LOSS_USD=5000.0  # Max $5k daily loss (kill switch)
MAX_PORTFOLIO_LEVERAGE=1.0  # No leverage
```

### 5.3 Live Trading Execution

```bash
# ⚠️ LIVE TRADING - REAL MONEY
python scripts/run_daily_spy_paper.py \
  --execute \
  --broker alpaca \
  --capital 25000 \  # Start with minimum capital
  --output-dir live_runs
```

### 5.4 Rollout Plan (From OPERATIONS_RUNBOOK.md)

**Phase 1: SPY Paper Trading (2-4 weeks)**
- Single symbol (SPY) in paper mode
- Validate strategy selection and exit management
- Accumulate performance data

**Phase 2: Multi-Symbol Paper (1-2 weeks)**
- Expand to 3-5 liquid symbols (SPY, QQQ, IWM, etc.)
- Validate regime adaptation across different volatility profiles

**Phase 3: Small Live Capital (1-2 weeks)**
- Transition SPY to live trading with $25k-$50k
- Monitor execution quality and slippage
- Validate commission and cost assumptions

**Phase 4: Scale Up**
- Gradually increase capital allocation
- Add more symbols after validation
- Monitor risk metrics daily

---

## Troubleshooting

### Error: "unauthorized"

**Problem**: API connection fails with 401 Unauthorized

**Solutions**:
1. Verify API keys are correct (copy-paste from Alpaca dashboard)
2. Ensure you're using **Paper Trading** keys with `https://paper-api.alpaca.markets`
3. Check if API keys are activated (sometimes requires email verification)
4. Try regenerating API keys in Alpaca dashboard

### Error: "credentials not found"

**Problem**: Environment variables not loaded

**Solutions**:
```bash
# Check if .env file exists
ls -la .env

# Verify environment variables are set
echo $ALPACA_API_KEY

# Load .env manually
export $(cat .env | xargs)

# Or use python-dotenv
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('ALPACA_API_KEY'))"
```

### Error: "insufficient buying power"

**Problem**: Order size exceeds available buying power

**Solutions**:
1. Check account buying power: `broker.get_account().buying_power`
2. Reduce capital allocation: `--capital 50000`
3. Paper trading accounts have $400k buying power (4x leverage on $100k cash)

### Paper Trading Account Reset

**To reset your paper trading account**:
1. Go to https://app.alpaca.markets/paper/dashboard/settings
2. Click "Reset Account"
3. This will reset cash to $100k and close all positions

---

## Security Best Practices

### ⚠️ Never Commit API Keys

```bash
# .env is already in .gitignore
# Verify it's not tracked:
git status

# If accidentally staged:
git reset HEAD .env
git restore .env
```

### Store Keys Securely

- Use password manager (1Password, LastPass, etc.)
- Set restrictive file permissions: `chmod 600 .env`
- Never share API keys in Slack, email, or public forums
- Rotate keys periodically (every 90 days)

### Separate Paper and Live Keys

- Use different API keys for paper and live trading
- Store live keys in production-only environment
- Never test live keys in development

---

## API Rate Limits

### Alpaca Rate Limits (as of 2024)

- **REST API**: 200 requests per minute
- **Market Data**: 200 requests per minute per endpoint
- **WebSocket**: 300 subscriptions per connection

**Best Practices**:
- Batch quote requests: `get_quotes_batch(["SPY", "QQQ", "IWM"])`
- Cache market data when possible
- Use WebSocket for real-time data (more efficient than polling)

---

## Additional Resources

### Official Documentation
- Alpaca Docs: https://docs.alpaca.markets/
- API Reference: https://docs.alpaca.markets/reference/
- Python SDK: https://github.com/alpacahq/alpaca-py

### Gnosis Documentation
- Architecture: `ARCHITECTURE_OVERVIEW.md`
- Operations: `OPERATIONS_RUNBOOK.md`
- Development: `DEV_GUIDE.md`
- Phase 3 Analysis: `PHASE3_ANALYSIS.md`

### Support
- Alpaca Support: support@alpaca.markets
- Alpaca Community: https://forum.alpaca.markets/
- Gnosis Issues: https://github.com/DGator86/V2---Gnosis/issues
