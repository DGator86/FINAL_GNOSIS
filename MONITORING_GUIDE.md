# 🔍 Complete Monitoring Guide - See Everything Happening in Your System

This guide shows you how to monitor **engines, agents, tickers, timeframes, and decisions** in real-time.

---

## 🎯 Quick Start

### **1. Start Trading System** (Terminal 1)
```bash
python3 start_dynamic_trading.py
```

### **2. Monitor Engine & Agent Activity** (Terminal 2)
```bash
python3 monitor_engines_agents.py
```

This shows **per-ticker, real-time**:
- 🛡️ **Hedge Engine** - Volatility analysis, hedge ratios
- 💧 **Liquidity Engine** - Liquidity scores, market depth
- 💭 **Sentiment Engine** - Bullish/bearish sentiment
- ⚡ **Elasticity Engine** - Price elasticity, options sensitivity
- 📊 **Greek Exposure** - GEX, VEX (Vanna), Charm
- 🌑 **Dark Pool** - Institutional block trades
- 🤖 **Agent Decisions** - BUY/SELL/HOLD signals

---

## 📊 Monitoring Tools Overview

### **Tool 1: Engine & Agent Monitor** (Most Detailed)
```bash
python3 monitor_engines_agents.py
```

**Shows:**
- Current universe (active tickers)
- Per-ticker engine processing:
  - Hedge Engine status and last check time
  - Liquidity Engine status and score
  - Sentiment Engine direction (bullish/bearish)
  - Elasticity Engine activity
  - Greek exposure data (GEX/VEX/Charm)
  - Dark pool detection
- Agent decisions with timestamps
- Opportunity scores for each ticker

**Best For:** Deep dive into what each engine is thinking about each ticker

---

### **Tool 2: Live Activity Monitor** (Medium Detail)
```bash
python3 monitor_live_activity.py
```

**Shows:**
- 🎯 Ticker analysis events
- 📊 Greek exposure retrieval
- 🌑 Dark pool trades
- 💰 Trade execution
- 🌐 Universe updates

**Best For:** Seeing data flow and API calls in real-time

---

### **Tool 3: Terminal Dashboard** (High-Level)
```bash
python3 scripts/terminal_dashboard.py
```

**Shows:**
- Account balance & buying power
- Current positions with P&L
- Portfolio metrics

**Best For:** Quick status check and P&L monitoring

---

### **Tool 4: Trading Status** (Quick Check)
```bash
python3 check_trading_status.py
```

**Shows:**
- Market open/closed
- Account status
- Position summary

**Best For:** One-time status check without continuous monitoring

---

## 🔬 Understanding Engine Output

### **Hedge Engine**
```
🛡️ Hedge Engine: ACTIVE | Last: 10:23:45
```
- **Status**: ACTIVE = Currently calculating hedge ratios
- **Output**: Volatility analysis, hedge recommendations
- **Purpose**: Minimize directional risk through delta hedging

### **Liquidity Engine**
```
💧 Liquidity Engine: ACTIVE | Last: 10:23:46
```
- **Status**: ACTIVE = Analyzing market depth
- **Output**: Liquidity scores (0-1), bid-ask spread analysis
- **Purpose**: Ensure positions can be entered/exited efficiently

### **Sentiment Engine**
```
💭 Sentiment Engine: BULLISH | Last: 10:23:47
```
- **Direction**: BULLISH/BEARISH/NEUTRAL
- **Output**: Aggregate sentiment from news, social, flow
- **Purpose**: Gauge market psychology and momentum

### **Elasticity Engine**
```
⚡ Elasticity Engine: ACTIVE | Last: 10:23:48
```
- **Status**: ACTIVE = Calculating price elasticity
- **Output**: How price responds to order flow
- **Purpose**: Predict price movement from options activity

### **Greek Exposure**
```
📊 Greek Exposure: GEX, VEX, Charm
```
- **GEX (Gamma)**: Volatility suppression/amplification
  - Positive = Market makers suppress volatility (range-bound)
  - Negative = Market makers amplify volatility (explosive moves)
- **VEX (Vanna)**: Price sensitivity to IV changes
  - Positive vanna + rising IV = upward price pressure
- **Charm**: Delta decay over time

### **Dark Pool**
```
🌑 Dark Pool: DETECTED | Time: 10:24:00
```
- **Detected**: Large off-exchange institutional trades found
- **Purpose**: Track smart money positioning

---

## 🤖 Understanding Agent Decisions

### **Agent Output**
```
┌──────────────────────────────────────┐
│         AGENT DECISION               │
├──────────────────────────────────────┤
│ 🟢 Action: BUY | Time: 10:25:00     │
└──────────────────────────────────────┘
```

**Actions:**
- 🟢 **BUY**: Agent recommends entering long position
- 🔴 **SELL**: Agent recommends entering short position or closing long
- 🟡 **HOLD**: Agent recommends no action

**Agent Types:**
1. **Hedge Agent**: Manages delta/gamma hedging
2. **Liquidity Agent**: Ensures adequate market depth
3. **Sentiment Agent**: Acts on momentum signals
4. **Composer Agent**: Synthesizes all inputs into final decision

---

## 📈 Typical Workflow

### **Setup** (One Time)
```bash
# Terminal 1: Start trading system
python3 start_dynamic_trading.py

# Terminal 2: Monitor engines & agents
python3 monitor_engines_agents.py

# Terminal 3: Check account status
python3 check_trading_status.py
```

### **During Trading Hours**
Watch **Terminal 2** to see:
1. Scanner identifies top 25 opportunities
2. Each ticker is evaluated by all 4 engines
3. Greek exposure and dark pool data is pulled
4. Agents synthesize engine output into decisions
5. Trades are executed based on agent recommendations

### **Key Indicators to Watch**

**Positive GEX (Volatility Suppression):**
```
Net GEX: +2,343,906 (positive = volatility suppression)
```
- Market makers will dampen price moves
- Good for selling premium (iron condors, strangles)
- Expect range-bound trading

**Negative GEX (Volatility Amplification):**
```
Net GEX: -1,500,000 (negative = volatility amplification)
```
- Market makers will amplify price moves
- Good for buying premium (long straddles, spreads)
- Expect explosive breakouts/breakdowns

**High Dark Pool Activity:**
```
💰 Total Premium: $56.12M | 📦 Total Size: 82,246 shares
```
- Institutional positioning
- Smart money accumulation/distribution
- May indicate upcoming move

**Bullish Sentiment + Positive Vanna:**
```
💭 Sentiment: BULLISH | 📊 VEX: +270M
```
- If IV rises, price pressure upward
- Confirms bullish momentum

---

## 🎯 What You Can See

### **Per Ticker:**
✅ All 4 engine outputs (Hedge, Liquidity, Sentiment, Elasticity)
✅ Greek exposure (GEX, VEX, Charm)
✅ Dark pool institutional trades
✅ Agent buy/sell/hold decisions
✅ Opportunity scores (0-1 scale)
✅ Last update timestamps

### **Per Timeframe:**
The engines analyze multiple timeframes:
- **1-minute**: Intraday scalping signals
- **5-minute**: Short-term momentum
- **15-minute**: Swing trade setups
- **30-minute**: Position trade entries
- **1-hour**: Trend confirmation

Each timeframe's analysis contributes to the final opportunity score.

### **Per Universe:**
See which tickers are in the active top 25:
- Tickers entering universe (new opportunities)
- Tickers exiting universe (declining opportunities)
- Current rankings (1-25)

---

## 💡 Pro Tips

### **Tip 1: Multi-Terminal Setup**
```
Terminal 1: Trading system (start_dynamic_trading.py)
Terminal 2: Engine monitor (monitor_engines_agents.py)
Terminal 3: Activity log (tail -f logs/dynamic_trading_*.log)
Terminal 4: Account status (check_trading_status.py)
```

### **Tip 2: Watch for Alignment**
Strong signals occur when engines align:
```
✅ Hedge Engine: Low volatility (sell premium)
✅ Liquidity Engine: High score (easy entry/exit)
✅ Sentiment Engine: Bullish (momentum)
✅ Greek Exposure: Positive GEX (range-bound)
→ Signal: Sell bull put spreads
```

### **Tip 3: Dark Pool Confirmation**
```
📊 SPY Opportunity Score: 0.85
🌑 Dark Pool: $56M in large trades
→ Institutional confirmation of retail signal
```

### **Tip 4: Vanna/GEX Combo**
```
GEX: +2.3M (vol suppression)
VEX: +50M (positive vanna)
→ If IV rises, expect slow grind higher (not explosive)
```

---

## 🔧 Troubleshooting

### **"No log files found"**
```bash
# Start the trading system first
python3 start_dynamic_trading.py
```

### **"Waiting for activity"**
- System is starting up
- Wait 30-60 seconds for first scan
- Scanner runs every 15 minutes

### **"Monitor not updating"**
```bash
# Check if trading system is running
ps aux | grep start_dynamic_trading

# Check if logs are being written
ls -lah logs/
tail logs/dynamic_trading_*.log
```

---

## 📚 Advanced Usage

### **Filter Specific Engine**
```bash
# See only hedge engine activity
tail -f logs/dynamic_trading_*.log | grep "HedgeEngine"

# See only agent decisions
tail -f logs/dynamic_trading_*.log | grep -i "agent.*decision\|buy\|sell"
```

### **Export Engine Data**
```bash
# Export greek exposure data
grep "greek exposure\|GEX\|VEX" logs/dynamic_trading_*.log > greek_analysis.txt

# Export dark pool trades
grep "dark pool" logs/dynamic_trading_*.log > dark_pool_activity.txt
```

### **Create Custom Alerts**
```bash
# Alert on high opportunity scores
tail -f logs/dynamic_trading_*.log | grep "opportunity.*score" | grep -E "0\.[89]|1\.0"

# Alert on agent BUY signals
tail -f logs/dynamic_trading_*.log | grep -i "agent.*BUY"
```

---

## 🎓 Learning Resources

**Understanding GEX:**
- Positive GEX = Dealers short gamma = Stabilizing
- Negative GEX = Dealers long gamma = Destabilizing

**Understanding Vanna:**
- Positive Vanna: dDelta/dIV is positive
  - IV ↑ → Positive delta increases → Upward pressure
- Negative Vanna: dDelta/dIV is negative
  - IV ↑ → Negative delta increases → Downward pressure

**Understanding Charm:**
- Measures delta decay over time
- Positive charm: Delta increases as expiration approaches
- Negative charm: Delta decreases as expiration approaches

---

## 🚀 Summary

**For Real-Time Engine & Agent Visibility:**
```bash
python3 monitor_engines_agents.py
```

**For Data Flow & API Activity:**
```bash
python3 monitor_live_activity.py
```

**For Account & Positions:**
```bash
python3 scripts/terminal_dashboard.py
```

You now have **complete visibility** into:
- ✅ What engines are processing
- ✅ What agents are thinking
- ✅ All timeframes being analyzed
- ✅ Every ticker in the universe
- ✅ Real-time greek exposure
- ✅ Dark pool activity
- ✅ Trade decisions

Happy monitoring! 🎯
