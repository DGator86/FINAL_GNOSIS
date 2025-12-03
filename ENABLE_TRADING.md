# Enable Active Trading - Quick Start Guide

## ⚠️ Current Status: Trading DISABLED

**Reason**: Missing `.env` file with Alpaca API credentials

## ✅ Enable Trading in 3 Steps

### Step 1: Create .env File

```bash
cd /home/user/FINAL_GNOSIS
cp .env.example .env
```

### Step 2: Add Your Alpaca API Keys

1. **Get your API keys**:
   - Go to: https://app.alpaca.markets/paper/dashboard/overview
   - Navigate to "API Keys" section
   - Generate new keys if needed (keep secret safe!)

2. **Edit `.env` file**:
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Replace placeholder values**:
   ```bash
   # FROM THIS:
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_SECRET_KEY=your_alpaca_secret_key_here

   # TO THIS (with your actual keys):
   ALPACA_API_KEY=PKA...your_actual_key
   ALPACA_SECRET_KEY=...your_actual_secret
   ```

4. **Verify settings**:
   ```bash
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ALPACA_PAPER=true  # Keep as true for paper trading!
   ```

### Step 3: Run Active Trading

```bash
# Test with single run
python main.py run-once --symbol SPY

# Start live trading loop
python main.py live-loop --symbol SPY

# Multi-symbol trading (top 25)
python main.py multi-symbol-loop
```

**IMPORTANT**: Do NOT use `--dry-run` flag if you want active trading!

---

## 🛡️ Safety Features (Already Enabled)

Your recent improvements added critical safety measures:

✅ **Position Size Limits**: Max 2% of portfolio per trade
✅ **Circuit Breaker**: Stops trading at $5,000 daily loss
✅ **Risk Validation**: Checks run BEFORE order submission
✅ **Paper Trading**: Safe testing with virtual money

---

## 🔍 Verify It's Working

After setup, you should see:

```bash
✅ Connected to Alpaca Paper Trading
🔌 Broker: ENABLED
📊 Auto-Execute: TRUE
```

If you see:
```bash
⚠️  Running in dry-run mode (broker unavailable)
📊 Auto-Execute: FALSE
```

Then check your `.env` file has valid keys.

---

## 📊 Monitor Your Trading

### Dashboard (Real-time)
```bash
streamlit run dashboard.py
```

### Check Positions
```bash
python scripts/show_positions.py
```

### View Ledger
```bash
tail -f data/ledger.jsonl
```

---

## 🚨 Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'alpaca'"
**Solution**:
```bash
pip install -r requirements.txt
```

### Issue 2: "Failed to connect to Alpaca"
**Solutions**:
1. Check API keys are correct in `.env`
2. Verify Alpaca account is active
3. Check internet connection
4. Verify `ALPACA_BASE_URL` matches your account type

### Issue 3: "Trading halted - circuit breaker"
**Explanation**: Daily loss limit reached ($5,000 default)
**Solution**: This is intentional - prevents runaway losses

---

## 📝 Configuration Files

### `.env` (Required - You must create this)
- Alpaca API credentials
- Risk management settings
- Trading configuration

### `config/config.yaml` (Already configured)
- Engine settings
- Agent weights
- Strategy parameters

---

## 🎯 What Happens When You Enable Trading

1. **Connection**: System connects to Alpaca paper trading
2. **Analysis**: Engines analyze market data (hedge, liquidity, sentiment)
3. **Decision**: Agents vote, composer creates consensus
4. **Execution**: Trade agent generates ideas
5. **Risk Check**: Position size & circuit breaker validation
6. **Order Placement**: Orders submitted to Alpaca
7. **Tracking**: All trades logged to `data/ledger.jsonl`

---

## 📚 Additional Resources

- [Alpaca Setup Guide](ALPACA_SETUP.md)
- [Operations Runbook](OPERATIONS_RUNBOOK.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)

---

## ⚡ TL;DR

```bash
# 1. Copy template
cp .env.example .env

# 2. Add your Alpaca keys to .env
nano .env

# 3. Run trading
python main.py live-loop --symbol SPY
```

**That's it!** Your system will start active paper trading.

---

Last Updated: 2025-12-03
