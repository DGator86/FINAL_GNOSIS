# Live Paper Trading System

This document describes the live paper trading system with real-time data streaming, regime detection, and web dashboard.

## 🎯 Overview

The live trading system consists of three integrated components:

1. **Live Trading Bot** - WebSocket-based execution engine with position management
2. **Regime Detection** - Multi-method market classification with conditional agent activation
3. **Web Dashboard** - Real-time monitoring UI with WebSocket updates

## 📦 Components

### 1. Live Trading Bot (`gnosis/trading/`)

**Files:**
- `live_bot.py` - Main trading bot with Alpaca WebSocket integration
- `position_manager.py` - Position lifecycle management (entry/update/exit)
- `risk_manager.py` - Portfolio-level risk controls and position sizing

**Features:**
- Real-time 1-minute bar aggregation from Alpaca
- Position management with TP/SL/time stops
- Multi-layer risk management:
  - Position-level: Stop-loss, take-profit, max bars held
  - Portfolio-level: Max 3 positions, 30% total risk, -5% daily loss limit
  - Circuit breakers: -10% max drawdown protection
- Volatility-adjusted position sizing (inverse relationship)
- Kelly Criterion-inspired sizing based on confidence
- State persistence for crash recovery
- Episodic memory integration for learning

**Risk Parameters:**
- Max positions: 3
- Max position size: 15% of capital
- Default stop-loss: 2% (dynamic based on confidence/volatility)
- Default take-profit: 4% (2:1 R:R ratio)
- Max drawdown: 10%
- Daily loss limit: 5%

### 2. Regime Detection (`gnosis/regime/`)

**Files:**
- `detector.py` - Multi-method regime classifier
- `__init__.py` - Module exports

**Detection Methods:**
1. **Volatility Regime** - Realized vol classification (calm/normal/volatile)
2. **Trend Regime** - Momentum + regression slope (trending_up/down/ranging)
3. **Range Detection** - ATR + Bollinger band analysis
4. **HMM Regime** - 5-state Hidden Markov Model (from markov_regime_v0)

**Regime States:**
- `trending_up` - Strong upward momentum
- `trending_down` - Strong downward momentum
- `ranging` - Sideways/choppy market
- `volatile` - High volatility regime
- `calm` - Low volatility regime
- `accumulation` - HMM detected accumulation phase
- `distribution` - HMM detected distribution phase

**Conditional Agent Activation:**

| Agent | Active When | Reasoning |
|-------|-------------|-----------|
| **Hedge** | Always | Core portfolio protection |
| **Liquidity** | Always | Essential market microstructure |
| **Sentiment** | Always | Behavioral signals |
| **Wyckoff** | Trends, strong momentum | Catches accumulation/distribution in trending markets |
| **Markov** | Clear regimes, moderate vol | Works best with stable state transitions |

**Wyckoff Activation:**
- Trending markets (primary = trending_up/down)
- Strong trend strength (|trend_strength| > 0.4)
- Accumulation/distribution phases

**Markov Activation:**
- High regime confidence (> 0.7)
- Clear trending regimes with strong momentum
- NOT in ranging markets (too noisy)
- NOT in extreme volatility (> 30% annualized)

### 3. Web Dashboard (`gnosis/dashboard/`)

**Files:**
- `dashboard_server.py` - FastAPI backend + HTML frontend
- `__init__.py` - Module exports

**Features:**
- **Real-time Updates** - WebSocket streaming of all trading events
- **Portfolio Stats** - Capital, equity, PnL, win rate
- **Regime Display** - Current regime with confidence and characteristics
- **Position Monitor** - Open positions with live unrealized PnL
- **Agent Votes** - Real-time display of all agent signals and confidence
- **Trade History** - Last 20 trades with entry/exit/PnL
- **Memory Recalls** - Recent episodic memory retrievals (if enabled)

**REST API Endpoints:**
- `GET /` - Dashboard HTML
- `GET /api/state` - Full system state
- `GET /api/positions` - Open positions
- `GET /api/trades` - Recent trades
- `GET /api/portfolio` - Portfolio statistics
- `GET /api/memory` - Memory recalls
- `GET /api/regime` - Current regime
- `WS /ws` - WebSocket for real-time updates

**Update Types:**
- `positions` - Position changes
- `trade` - New trade closed
- `agent_votes` - Agent decision update
- `regime` - Regime change
- `bar` - New bar data
- `portfolio_stats` - Portfolio update
- `memory_recall` - Memory retrieval

## 🚀 Usage

### Option 1: Bot Only (No Dashboard)

```bash
python start_paper_trading.py
```

This runs the trading bot without the web dashboard. Output to console only.

### Option 2: Bot + Dashboard (Recommended)

```bash
python start_with_dashboard.py
```

This runs both the bot and dashboard server concurrently.

Then open: **http://localhost:8080**

### Configuration

Edit the bot initialization in the launcher scripts:

```python
bot = LiveTradingBot(
    symbol="SPY",           # Trading symbol
    bar_interval="1Min",    # Bar aggregation interval
    enable_memory=True,     # Use episodic memory system
    enable_trading=False,   # Set True to actually place orders
    paper_mode=True         # Use Alpaca paper account
)
```

**⚠️ IMPORTANT:** 
- `enable_trading=False` is DRY RUN mode (no actual orders placed)
- Set `enable_trading=True` only when ready to paper trade
- Always use `paper_mode=True` for testing

### Environment Variables

Create `.env` file with Alpaca credentials:

```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

Get free paper trading keys at: https://alpaca.markets/

## 📊 Agent Consensus Logic

**Base Agents (Always Active):** 3 agents
- Hedge, Liquidity, Sentiment

**Conditional Agents:** 0-2 agents
- Wyckoff (if regime suitable for trend analysis)
- Markov (if regime has clear state)

**Voting:**
- Minimum 2 votes required for signal
- Dynamic threshold: `max(2, num_agents // 2)`
- Confidence averaged from agreeing agents

**Example Scenarios:**

| Regime | Active Agents | Min Votes | Notes |
|--------|---------------|-----------|-------|
| Ranging | 3 (base) | 2 | Wyckoff/Markov inactive, need 2-of-3 |
| Trending Up | 5 (base + both) | 2 | All agents active, need 2-of-5 |
| Volatile | 4 (base + Wyckoff) | 2 | Markov inactive (high vol), need 2-of-4 |
| Calm Trend | 5 (base + both) | 2 | Ideal regime, all agents active |

## 🔍 Example Output

### Console (Bot)
```
============================================================
🤖 Live Trading Bot Initialized
   Symbol: SPY
   Interval: 1Min
   Memory: ✅
   Trading: ❌ (dry run)
   Mode: 📄 PAPER
============================================================

============================================================
⏰ 2024-11-03 14:32:00 | SPY @ $580.23
📊 Regime: trending_up (conf=0.85, trend=+0.62)
   ✅ Wyckoff active: markup
   ✅ Markov active: trending_up
🗳️  Votes: {-1: 0, 0: 2, 1: 3}
   Decision: 1 (conf=0.72, size=0.12)
```

### Dashboard (Browser)

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Dashboard+Preview)

**Features visible:**
- Live portfolio stats with color-coded PnL
- Current regime badge with confidence
- Open positions list with unrealized PnL
- Agent vote cards showing signals
- Trade history table with reason codes

## 🧪 Testing

### Test Regime Detector
```bash
python gnosis/regime/detector.py
```

### Test Position Manager
```bash
python gnosis/trading/position_manager.py
```

### Test Risk Manager
```bash
python gnosis/trading/risk_manager.py
```

### Test Dashboard Server
```bash
uvicorn gnosis.dashboard.dashboard_server:app --reload --port 8080
```

## 📈 Production Readiness

**Completed:**
- ✅ Live data streaming (Alpaca WebSocket)
- ✅ Position management with state persistence
- ✅ Multi-layer risk controls
- ✅ Regime detection with conditional agents
- ✅ Episodic memory integration
- ✅ Web dashboard for monitoring
- ✅ Paper trading execution

**Remaining for Production:**
- ⬜ Order execution confirmation and reconciliation
- ⬜ Slippage modeling and tracking
- ⬜ Advanced portfolio optimization
- ⬜ Multi-symbol support
- ⬜ Backtesting infrastructure for validation
- ⬜ Deployment automation (Docker/K8s)
- ⬜ Alerting and monitoring (Sentry/Datadog)
- ⬜ Performance analytics dashboard
- ⬜ Trade review workflow
- ⬜ Risk reporting and compliance
- ⬜ Historical data replay for testing
- ⬜ Strategy parameter optimization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (8080)                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Portfolio  │  │  Regime    │  │ Positions  │        │
│  └────────────┘  └────────────┘  └────────────┘        │
│  ┌────────────────────────────────────────────┐        │
│  │          Agent Votes (3-5 agents)          │        │
│  └────────────────────────────────────────────┘        │
│  ┌────────────────────────────────────────────┐        │
│  │        Trade History (last 20 trades)      │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                          ↕ WebSocket
┌─────────────────────────────────────────────────────────┐
│                   Live Trading Bot                       │
│  ┌────────────────────────────────────────────┐        │
│  │  Alpaca WebSocket → 1-Min Bar Aggregation  │        │
│  └────────────────────────────────────────────┘        │
│                          ↓                               │
│  ┌────────────────────────────────────────────┐        │
│  │  Regime Detector (4 methods, conditional)  │        │
│  └────────────────────────────────────────────┘        │
│                          ↓                               │
│  ┌────────────────────────────────────────────┐        │
│  │   Agent Evaluation (3-5 agents, dynamic)   │        │
│  │   • Hedge (always)                         │        │
│  │   • Liquidity (always)                     │        │
│  │   • Sentiment (always)                     │        │
│  │   • Wyckoff (conditional)                  │        │
│  │   • Markov (conditional)                   │        │
│  └────────────────────────────────────────────┘        │
│                          ↓                               │
│  ┌────────────────────────────────────────────┐        │
│  │   Consensus Voting (2-vote min, memory)    │        │
│  └────────────────────────────────────────────┘        │
│                          ↓                               │
│  ┌────────────────────────────────────────────┐        │
│  │   Risk Management (position, portfolio)    │        │
│  └────────────────────────────────────────────┘        │
│                          ↓                               │
│  ┌────────────────────────────────────────────┐        │
│  │   Position Manager (entry/update/exit)     │        │
│  └────────────────────────────────────────────┘        │
│                          ↓                               │
│  ┌────────────────────────────────────────────┐        │
│  │   Order Execution (Alpaca Paper Account)   │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                  Episodic Memory System                  │
│  • Write episodes on entry                              │
│  • Reflect on episodes on exit                          │
│  • Augment decisions with similar cases                 │
│  • Track win rate and adjustments                       │
└─────────────────────────────────────────────────────────┘
```

## 📝 Notes

- **Paper Trading Only**: This system is configured for paper trading. Always test thoroughly before live trading.
- **Risk Management**: Multiple layers of protection, but no system is foolproof. Monitor carefully.
- **Memory System**: Improves over time as more episodes are collected.
- **Regime Detection**: Works better with more data (requires 50+ bars for warm-up).
- **Agent Selection**: Conditional activation prevents agent confusion in unsuitable regimes.

## 🐛 Troubleshooting

**Bot won't start:**
- Check Alpaca API keys in `.env` file
- Ensure `alpaca-py` is installed: `pip install alpaca-py`
- Verify market hours (US stock market 9:30am-4pm ET)

**Dashboard not updating:**
- Check WebSocket connection indicator (green = connected)
- Open browser console to see WebSocket messages
- Fallback REST API polling every 5 seconds if WebSocket fails

**No agent votes showing:**
- Wait for 50+ bars to accumulate (warm-up period)
- Check regime detection output in console
- Ensure agents are being activated based on regime

**Memory not working:**
- Check if `gnosis/memory/` module exists
- Verify DuckDB database is writable
- Set `enable_memory=True` in bot initialization

## 📚 References

- [Alpaca Markets API](https://alpaca.markets/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Wyckoff Method](https://school.stockcharts.com/doku.php?id=market_analysis:the_wyckoff_method)
- [Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model)
