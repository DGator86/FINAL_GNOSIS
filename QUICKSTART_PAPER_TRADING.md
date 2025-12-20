# Super Gnosis - Paper Trading Quick Start Guide

**Target: Full Dynamic Universe Paper Trading on Alpaca by Monday**

This guide gets you trading the **top 25 most active options underlyings** in **5 minutes**.

---

## What You'll Be Trading

The system automatically identifies and trades the **hottest options names** based on:
- Options volume (most important)
- Open interest
- Gamma exposure  
- Liquidity
- Unusual flow

**Default Universe**: ~100 symbols including SPY, QQQ, AAPL, TSLA, NVDA, AMD, META, COIN, SMCI, PLTR, meme stocks, sector ETFs, and more.

**Active Trading**: Top 25 symbols ranked by composite score, re-scanned each iteration.

---

## Prerequisites

- Python 3.10+ installed
- Alpaca paper trading account (free at [alpaca.markets](https://app.alpaca.markets))

---

## Quick Setup (5 minutes)

### Step 1: Get Your Alpaca Paper Trading Keys

1. Go to [Alpaca Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)
2. Click **"Generate new key"** under "Paper Trading API Keys"
3. Copy both the **API Key ID** and **Secret Key**

### Step 2: Configure Environment

```bash
# Copy the paper trading template
cp .env.paper.template .env

# Edit .env with your favorite editor
nano .env  # or vim, code, etc.
```

Update these two lines with your Alpaca keys:
```
ALPACA_API_KEY=PKxxxxx...         # Your API Key ID
ALPACA_SECRET_KEY=xxxxx...        # Your Secret Key
```

### Step 3: Run Pre-Flight Check

```bash
python scripts/preflight_check.py
```

You should see:
```
✅ PRE-FLIGHT CHECK PASSED
   System is ready for paper trading!
```

### Step 4: Start Dynamic Universe Trading

```bash
# DEFAULT: Trade top 25 from dynamic universe
python scripts/paper_trading_runner.py
```

That's it! The system will:
1. Run pre-flight checks
2. Connect to Alpaca Paper
3. Scan the universe for top 25 opportunities
4. Start trading all 25 symbols in rotation

---

## What Happens When You Start

1. **Pre-flight check** validates your configuration
2. **Market check** - waits if market is closed
3. **Universe scan** - ranks ~100 symbols, selects top 25
4. **Trading loop** runs every 60 seconds:
   - Re-scans universe for current top opportunities
   - Runs full DHPE pipeline on each symbol
   - **EliteTradeAgent** generates institutional-grade trade ideas:
     - **Multi-Timeframe Analysis**: Aggregates signals across scalp/intraday/swing/position
     - **IV-Aware Strategy Selection**: Credit spreads in high IV, debit spreads in low IV
     - **Kelly Criterion Position Sizing**: Optimal bet size with 25% fractional Kelly
     - **Risk Management**: ATR-based stops, min 1.5:1 R:R, max 4% per position
   - Executes with **Bracket Orders** (automatic stop-loss & take-profit)
   - Tracks performance and self-optimizes

Press `Ctrl+C` to stop - you'll see a session summary with all positions.

---

## Command Line Options

```bash
python scripts/paper_trading_runner.py [OPTIONS]

Options:
  --top, -t INT          Number of symbols from dynamic universe (default: 25)
  --symbol, -s TEXT      Single symbol only (overrides multi-symbol mode)
  --single               Force single-symbol mode with SPY
  --interval, -i INT     Seconds between iterations (default: 60)
  --dry-run, -d          No actual trades, just preview
  --skip-preflight       Skip pre-flight checks (not recommended)
```

### Examples

```bash
# Trade full top 25 universe (DEFAULT)
python scripts/paper_trading_runner.py

# Trade top 10 only (faster iterations)
python scripts/paper_trading_runner.py --top 10

# Trade top 50 (more coverage)
python scripts/paper_trading_runner.py --top 50

# Single symbol only (testing)
python scripts/paper_trading_runner.py --symbol TSLA

# Dry run - preview without trades
python scripts/paper_trading_runner.py --dry-run

# Faster iteration (30 seconds)
python scripts/paper_trading_runner.py --interval 30
```

---

## Dynamic Universe Details

The system uses `engines/dynamic_universe.py` to rank symbols:

### Full Universe (~100 symbols)
- **Index ETFs**: SPY, QQQ, IWM, DIA
- **Mega Cap Tech**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Semiconductors**: AMD, AVGO, QCOM, TSM, MU, AMAT
- **AI/ML Hype**: PLTR, SNOW, CRWD, ZS, DDOG
- **EV/Clean Energy**: RIVN, LCID, NIO, F, GM
- **Finance**: JPM, BAC, GS, MS, V, MA, PYPL
- **High Vol/Spec**: SMCI, COIN, MSTR, RIOT, MARA
- **Meme Stocks**: GME, AMC (when active)
- **Sector ETFs**: XLE, XLF, XLK, XLV, etc.

### Ranking Criteria (Weights)
- Options Volume: 40%
- Open Interest: 25%
- Gamma Exposure: 20%
- Liquidity: 10%
- Unusual Flow: 5%

### Re-scanning
Each iteration re-ranks the universe, so the active trading list adapts to market conditions.

---

## Monitoring Your Session

### Log Files

Logs are saved to `logs/trading_YYYYMMDD_HHMMSS.log`:
```bash
# Follow log in real-time
tail -f logs/trading_*.log
```

### Check Positions

```bash
python scripts/show_positions.py
```

### Check Account Status

```bash
python check_trading_status.py
```

---

## Alternative Entry Points

### Using main.py (CLI interface)

```bash
# Multi-symbol with scanner (similar to default)
python main.py multi-symbol-loop --top 25

# Single symbol live loop
python main.py live-loop --symbol SPY

# Scan opportunities without trading
python main.py scan-opportunities --top 25
```

---

## Troubleshooting

### "Missing ALPACA_API_KEY"
- Make sure `.env` file exists and has your keys
- Check that keys don't have quotes: `ALPACA_API_KEY=PKxxxxx` not `ALPACA_API_KEY="PKxxxxx"`

### "Alpaca connection failed"
- Verify you're using **paper trading keys** (not live)
- Check that `ALPACA_PAPER=true` in your `.env`
- Try regenerating your API keys on Alpaca dashboard

### "Market is CLOSED"
- Paper trading only works during market hours (9:30 AM - 4:00 PM ET, weekdays)
- The system will wait automatically for market open

### "Using stub adapter"
- This is normal if you don't have Unusual Whales or Massive.com API keys
- Trading will still work with Alpaca data
- For better signals, add `UNUSUAL_WHALES_API_TOKEN` to your `.env`

### Iterations are slow
- Default scans 25 symbols per iteration
- Try `--top 10` for faster iterations
- Or increase `--interval` to 120 seconds

---

## Safety Features Built-In

- **Paper mode enforced** - Refuses to run if `ALPACA_PAPER=false`
- **Circuit breaker** - Stops trading after MAX_DAILY_LOSS_USD loss
- **Market hours check** - Only trades when market is open
- **Graceful shutdown** - Ctrl+C saves state and shows summary
- **Position limits** - MAX_POSITION_SIZE_PCT caps exposure per trade
- **Portfolio heat limit** - Maximum 20% total portfolio risk exposure
- **Health monitoring** - Tracks broker, data adapters, circuit breakers

## Elite Trade Agent Features

The EliteTradeAgent implements institutional-grade execution:

### Strategy Selection (IV-Aware)
| Market View | High IV (>50) | Medium IV (30-50) | Low IV (<30) |
|------------|---------------|-------------------|--------------|
| **Bullish** | Bull Put Spread (Credit) | Vertical Spreads | Long Call / Bull Call Spread |
| **Bearish** | Bear Call Spread (Credit) | Vertical Spreads | Long Put / Bear Put Spread |
| **Neutral** | Iron Condor / Butterfly | Calendar Spreads | Long Straddle / Strangle |

### Risk Management
- **Kelly Criterion**: 25% fractional Kelly for position sizing
- **Max Position**: 4% of portfolio per trade
- **Portfolio Heat**: Maximum 20% total risk exposure
- **Stop Loss**: ATR-based dynamic stops (2x ATR default)
- **Take Profit**: Risk-multiple based (minimum 1.5:1 R:R)
- **Bracket Orders**: Automatic stop-loss and take-profit on entry

### Multi-Timeframe Signal Aggregation
Analyzes signals across 4 timeframes for institutional edge:
- **Scalp** (1-15min): Technical momentum
- **Intraday** (15min-4hr): Flow sentiment
- **Swing** (1-5 days): Hedge energy asymmetry
- **Position** (5-30 days): ML predictions

Confidence is boosted when 3+ timeframes align (+15%), reduced when divergent (-10%).

---

## Expected Output

When running, you'll see:
```
INFO     | STARTING PAPER TRADING SESSION
INFO     | Pre-flight check PASSED
INFO     | Initial equity: $100,000.00
INFO     | EliteTradeAgent initialized | max_position=4.0% | max_heat=20.0% | kelly=25%
INFO     | Health monitor started
INFO     | Mode: MULTI-SYMBOL
INFO     | Trading top 25 from dynamic universe

--- Iteration 1 ---
INFO     | Scanning for top 25 opportunities...
INFO     | Top 25: NVDA, TSLA, SPY, QQQ, AAPL, AMD, AMZN, META, COIN, SMCI...
INFO     | Trading NVDA...
INFO     | MTF Override: dir=long | alignment=75% | conf=82.5%
INFO     | 🎯 TRADE SIGNAL: bull_put_spread for NVDA | dir=long | conf=82.5% | R:R=3.0 | MTF=75%
INFO     | BRACKET ORDER: BUY 5 NVDA | TP=$152.00 | SL=$132.00
INFO     | Trading TSLA...
INFO     | 🎯 TRADE SIGNAL: bear_call_spread for TSLA | dir=short | conf=68.3% | R:R=2.1 | MTF=50%
...
INFO     | Summary: 18 ideas, 5 orders
INFO     | Session P&L: $+127.50
INFO     | Next iteration in 60s...
```

---

## Next Steps After Paper Trading

1. **Monitor for a few days** - Let the system run during market hours
2. **Review trades** in the ledger: `data/ledger.jsonl`
3. **Check positions**: `python scripts/show_positions.py`
4. **Analyze performance** with backtest tools
5. **Tune parameters** in `.env` if needed
6. **Add data sources** (Unusual Whales) for better signals

---

## Getting Help

- Check logs: `logs/trading_*.log`
- Review docs: `DOCS_INDEX.md`
- Architecture: `ARCHITECTURE.md`
- Dynamic universe: `DYNAMIC_TOP25_LIVE.md`
- Alpaca setup: `ALPACA_SETUP.md`

---

**Happy Trading!**

*Remember: This is paper trading with fake money. Use it to validate the system before considering real capital.*
