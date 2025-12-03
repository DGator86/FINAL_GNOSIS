# Dynamic Universe Guide

## ✅ YES - Your System Uses Dynamic Universe!

Your trading system is **already configured** to use the dynamic universe feature, which automatically selects the top 25 most active options underlyings in real-time.

---

## 🎯 How It Works

### Automatic Symbol Selection

Instead of trading a fixed list of symbols, your system:

1. **Scans** 150+ potential options underlyings
2. **Ranks** them by composite activity score
3. **Selects** top 25 most active names
4. **Trades** only those 25 symbols
5. **Re-ranks** every 5 minutes to stay current

---

## 📊 Ranking Algorithm

### Weighted Scoring (Total: 100%)

```
Composite Score =
  (Options Volume × 40%) +
  (Open Interest × 25%) +
  (Gamma Exposure × 20%) +
  (Liquidity Score × 10%) +
  (Unusual Flow × 5%)
```

### Why This Works

- **40% Options Volume** - Trades where there's actual activity
- **25% Open Interest** - Shows positioning and interest
- **20% Gamma Exposure** - Dealer hedging creates movement
- **10% Liquidity** - Ensures tight spreads
- **5% Unusual Flow** - Catches momentum shifts

### Minimum Requirements

- ✅ At least 500,000 daily options contracts
- ✅ Must have liquid options chain
- ✅ Minimum liquidity score of 0.3

---

## 🌍 Full Universe (150+ Symbols Scanned)

The system scans this comprehensive universe:

### Major Indices (4)
```
SPY, QQQ, IWM, DIA
```

### Mega Cap Tech (7)
```
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
```

### Tech & Growth (16)
```
AMD, NFLX, AVGO, ORCL, CRM, ADBE, INTC, CSCO,
QCOM, TXN, AMAT, ASML, MU, LRCX, KLAC, SNPS
```

### AI/ML Names (6)
```
PLTR, SNOW, CRWD, ZS, DDOG, NET
```

### Semiconductors (4)
```
TSM, MRVL, ON, MPWR
```

### EV / Clean Energy (7)
```
RIVN, LCID, NIO, XPEV, F, GM, TSLA
```

### Finance (13)
```
JPM, BAC, GS, MS, WFC, C, BLK, SCHW,
V, MA, AXP, PYPL, SQ
```

### Healthcare / Biotech (15)
```
UNH, JNJ, LLY, ABBV, MRK, PFE, TMO, ABT,
AMGN, GILD, BIIB, REGN, VRTX, MRNA, BNTX
```

### Consumer (14)
```
BABA, WMT, HD, NKE, MCD, SBUX, TGT,
COST, LOW, TJX, DG, DLTR, AMZN, etc.
```

### Energy (8)
```
XLE, XOM, CVX, COP, SLB, EOG, PSX, MPC
```

### Meme Stocks (5)
```
GME, AMC, BB, BBBY, WISH
```

### SPACs & Recent IPOs (7)
```
HOOD, COIN, RBLX, UBER, LYFT, DASH, ABNB
```

### High Volatility (5)
```
SMCI, ARM, MSTR, RIOT, MARA
```

### Sector ETFs (11)
```
XLF, XLK, XLV, XLI, XLP, XLY, XLU,
XLB, XLRE, XLC, XLE
```

### Volatility Products (3)
```
VXX, UVXY, SVXY
```

**Total: 150+ symbols continuously monitored**

---

## ⚡ Usage

### Default (Top 25, Auto-Selected)

```bash
python main.py multi-symbol-loop
```

This will:
- Scan all 150+ symbols
- Rank by composite score
- Trade top 25 automatically
- Re-rank every 5 minutes

### Customize Number of Symbols

```bash
# Trade top 10
python main.py multi-symbol-loop --top 10

# Trade top 50 (if you want more)
python main.py multi-symbol-loop --top 50
```

### Scan Only (No Trading)

```bash
python main.py scan-opportunities
```

Shows ranked list without executing trades.

---

## 📈 Configuration

Your settings in `config/config.yaml`:

```yaml
scanner:
  mode: dynamic_top_n          # ✅ Enabled
  default_top_n: 25            # Top 25 symbols

  ranking_criteria:
    options_volume_weight: 0.40    # 40%
    open_interest_weight: 0.25     # 25%
    gamma_exposure_weight: 0.20    # 20%
    liquidity_score_weight: 0.10   # 10%
    unusual_flow_weight: 0.05      # 5%

  min_daily_options_volume: 500000  # 500k minimum
  update_frequency: 300              # Re-rank every 5 min
  cache_duration: 60                 # Cache for 1 min
```

---

## 🎯 Example Output

When you run `python main.py multi-symbol-loop`:

```
📊 Ranking universe using dynamic scanner...
✅ Selected top 25 most active options underlyings

🚀 MULTI-SYMBOL AUTONOMOUS TRADING
════════════════════════════════════════════════════════════════════════
   Universe: 25 symbols
   Top N: 25
   Scan Interval: 300 seconds (5 minutes)
   Mode: PAPER TRADING
   Press Ctrl+C to stop
════════════════════════════════════════════════════════════════════════

ITERATION 1 - 2025-12-03 21:30:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Top 25 Opportunities (Ranked):

 1. SPY   - Score: 0.92 ⬆️  (Bullish, Massive gamma exposure)
 2. QQQ   - Score: 0.88 ⬆️  (Bullish, Tech sector strength)
 3. TSLA  - Score: 0.85 ➡️  (Neutral, High IV)
 4. NVDA  - Score: 0.83 ⬆️  (Bullish, AI momentum)
 5. AAPL  - Score: 0.81 ⬆️  (Bullish, Largest OI)
 6. AMD   - Score: 0.78 ⬆️  (Bullish, Semi strength)
 7. MSFT  - Score: 0.76 ⬆️  (Bullish, Cloud growth)
 8. META  - Score: 0.74 ⬆️  (Bullish, Social recovery)
 9. GOOGL - Score: 0.72 ➡️  (Neutral, Consolidating)
10. AMZN  - Score: 0.70 ⬆️  (Bullish, E-commerce)
...
25. XLF   - Score: 0.52 ➡️  (Neutral, Sector rotation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trading SPY...
  🏛️  Hedge: Elasticity 0.45, Energy +0.22
  💧 Liquidity: 0.95 (excellent)
  📰 Sentiment: +0.38 (bullish)
  🎯 Composer: BULLISH (confidence: 0.82)
  💡 Trade Idea: BUY 2 shares @ $455.20

  ✅ ORDER PLACED - Order ID: abc123...

Trading QQQ...
  [Similar analysis for QQQ]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ITERATION 2 - 2025-12-03 21:35:00 (5 minutes later)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Re-ranking universe...

Changes detected:
  ✅ COIN entered top 25 (new #23 - crypto pump)
  ❌ XLF dropped out (now #27 - low activity)

New top 25: SPY, QQQ, TSLA, NVDA, AAPL, ... COIN

[Trading continues with updated universe]
```

---

## 🔄 Dynamic Updates

### Every 5 Minutes

The system automatically:
1. Re-scans all 150+ symbols
2. Re-calculates scores
3. Updates top 25 list
4. Adds new hot names
5. Removes cooled-off names

### Real-Time Adaptation

If a name suddenly gets hot (e.g., earnings, news):
- It automatically enters the top 25
- System starts trading it
- Capital flows to where action is

If a name cools down:
- It drops out of top 25
- System stops new positions
- Existing positions can close naturally

---

## 💡 Why This Beats Static Lists

### Traditional Approach
```
❌ Trade fixed list: SPY, QQQ, AAPL, etc.
❌ Miss hot names when they emerge
❌ Waste capital on dead names
❌ Manual updates required
```

### Dynamic Universe Approach
```
✅ Always trade the 25 HOTTEST names
✅ Automatically catch new momentum
✅ Drop names when activity fades
✅ Zero manual intervention needed
✅ Capital efficiency maximized
```

---

## 📊 Performance Benefits

### Capital Efficiency
- Focuses on names with actual liquidity
- Avoids wide spreads and slippage
- Trades where dealers are active

### Opportunity Capture
- Catches breakouts early (enters top 25)
- Rides momentum while it lasts
- Exits before liquidity dries up

### Risk Management
- Only trades liquid options (500k+ daily volume)
- Tight spreads mean better fills
- Can exit positions easily if needed

---

## 🛠️ Advanced Usage

### Change Ranking Weights

Edit `config/config.yaml`:

```yaml
ranking_criteria:
  options_volume_weight: 0.50    # Increase volume priority
  open_interest_weight: 0.20     # Decrease OI
  gamma_exposure_weight: 0.15    # Decrease gamma
  liquidity_score_weight: 0.10
  unusual_flow_weight: 0.05
```

### Adjust Update Frequency

```yaml
update_frequency: 180   # 3 minutes (faster)
update_frequency: 600   # 10 minutes (slower)
```

### Change Minimum Volume

```yaml
min_daily_options_volume: 1000000  # 1M (more selective)
min_daily_options_volume: 250000   # 250k (more inclusive)
```

---

## 📝 Summary

**Status**: ✅ **ENABLED and READY**

When you run `python main.py multi-symbol-loop`, the system will:

1. ✅ Scan 150+ symbols automatically
2. ✅ Select top 25 by options activity
3. ✅ Trade those 25 symbols
4. ✅ Re-rank every 5 minutes
5. ✅ Adapt to market changes in real-time

**No manual symbol selection needed - the system handles everything!**

---

Last Updated: 2025-12-03
