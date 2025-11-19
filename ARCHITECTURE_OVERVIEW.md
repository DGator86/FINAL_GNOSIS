# Architecture Overview

**Super Gnosis V2** - Institutional-grade options trading system with probabilistic decision-making, volatility intelligence, and adaptive position sizing.

## System Philosophy

- **Function-first, DI-friendly**: Minimize global state, maximize testability
- **Typed contracts**: Pydantic v2 for all data models
- **Layered intelligence**: Raw engines → fusion → strategy → execution
- **Deterministic where possible**: Seed-controlled optimization, reproducible tests

## 3-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EXECUTION LAYER                          │
│  • Broker Adapters (Simulated, Live)                        │
│  • Smart Order Router                                        │
│  • Cost Models (slippage, commissions, market impact)       │
│  • Order Controller                                          │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                  STRATEGY & OPTIMIZATION                     │
│  • Trade Agent v2 (14 strategy builders)                    │
│  • Risk Analyzer (PnL cones, breakevens, Greeks)            │
│  • Exit Manager (profit targets, stops, trailing)           │
│  • Optimizer (Optuna-based hyperparameter tuning)           │
│  • ML Integration (TensorFlow, PyTorch, scikit-learn)       │
│  • Kelly Refinement (dynamic position sizing)               │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                     ENGINES & FUSION                         │
│  • Hedge Agent v3 (dealer positioning, gamma/vanna/charm)   │
│  • Liquidity Agent v1 (spread, depth, flow)                 │
│  • Sentiment Agent v1 (news, social, analyst)               │
│  • Vol Surface Engine (IV rank, skew, term structure)       │
│  • Composer Agent (directive fusion, conflict resolution)   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                            │
│  • Market Data Adapter                                       │
│  • Options Chain Adapter                                     │
│  • News Adapter                                              │
└─────────────────────────────────────────────────────────────┘
```

## Component Map

### Engines & Fusion

| Component | Purpose | Key Outputs |
|-----------|---------|-------------|
| **Hedge Agent v3** | Interpret dealer positioning (gamma, vanna, charm exposures) | `EngineDirective` with dealer bias, elasticity, energy barriers |
| **Liquidity Agent v1** | Assess market depth, spread quality, flow imbalances | `EngineDirective` with liquidity score, flow direction |
| **Sentiment Agent v1** | Aggregate news, social, analyst sentiment | `EngineDirective` with bullish/bearish/neutral classification |
| **Vol Surface Engine** | IV rank, IV percentile, skew, term structure slopes | `VolSurfaceAnalysis` with expansion/crush probabilities |
| **Composer Agent** | Fuse engine directives into unified market context | `ComposerTradeContext` with direction, confidence, regime |

### Strategy & Optimization

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Trade Agent v2** | Generate ranked trade ideas from composer context | 14 strategies (stock, calls, puts, spreads, calendars, diagonals, broken wings, synthetics, straddles, strangles, iron condors, reverse IC) |
| **Risk Analyzer** | Compute PnL cones, breakevens, max profit/loss | 50-point P/L profile, linear interpolation for zero-crossings |
| **Exit Manager** | Strategy-specific exit rules with confidence scaling | Profit targets, stop loss, trailing stops, time-based, Greeks-based, breakeven adjustment |
| **Optimizer** | Optuna-based hyperparameter tuning | Profit %, stop %, DTE, strike offset, return on risk |
| **ML Integration** | Framework-agnostic prediction interface | Win rate, avg win/loss, prob_up for Kelly sizing |
| **Kelly Refinement** | Dynamic position sizing with empirical edge | Clamps between -25% to +25%, confidence-scaled, global risk cap |

### Execution Layer

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Broker Adapters** | Pluggable execution backends | Simulated (paper trading), Live (future: Alpaca, IBKR, Tradier) |
| **Smart Order Router** | Order type selection, order splitting | Spread-based routing, peg-to-mid pricing, market impact limits |
| **Cost Models** | Realistic execution cost estimation | Bid-ask spread, slippage (0.1% default), commission ($0.65/contract), square root market impact |
| **Order Controller** | Order lifecycle management | Placement, cancellation, modification tracking |

## Data Flow

```
1. Market Data Ingestion
   └─> Market Data Adapter, Options Chain Adapter

2. Engine Execution
   └─> Hedge Agent (dealer positioning)
   └─> Liquidity Agent (spread quality)
   └─> Sentiment Agent (news/social)
   └─> Vol Surface Engine (IV analytics)

3. Composer Fusion
   └─> Weighted aggregation of engine directives
   └─> Conflict resolution (e.g., sentiment vs liquidity)
   └─> Regime classification (VIX, trend, dealer, liquidity)

4. Trade Generation
   └─> Trade Agent selects strategies based on direction, confidence, vol regime
   └─> Risk Analyzer computes PnL cones, breakevens, Greeks
   └─> Exit Manager creates strategy-specific exit rules

5. Optimization Enhancement (optional)
   └─> Hyperparameter Provider retrieves regime-optimized params
   └─> ML Model predicts win rate, avg win/loss
   └─> Kelly Refinement adjusts position sizing

6. Execution (optional)
   └─> Smart Order Router selects order type, splits large orders
   └─> Broker Adapter places orders (paper or live)
   └─> Order Controller tracks fill status

7. Persistence
   └─> Composer context → runs/YYYY-MM-DD/spy_context.jsonl
   └─> Trade ideas → runs/YYYY-MM-DD/spy_trades.jsonl
   └─> Order results → runs/YYYY-MM-DD/spy_orders.jsonl
```

## Technology Stack

- **Language**: Python 3.12+
- **Data Validation**: Pydantic v2
- **Testing**: pytest (479/487 passing, 98.4%)
- **Optimization**: Optuna 4.6.0
- **ML Integration**: Framework-agnostic (TensorFlow, PyTorch, scikit-learn, LightGBM)
- **Broker Support**: Simulated (paper), planned: Alpaca, Interactive Brokers, Tradier

## Key Metrics (As of PR #23)

| Metric | Value |
|--------|-------|
| **Test Coverage** | 98.4% (479/487 passing) |
| **Strategy Count** | 14 (stock + 13 options strategies) |
| **Risk Metrics** | 6 (max profit, max loss, breakevens, risk/reward, RoR, capital efficiency) |
| **Exit Triggers** | 6 (profit target, stop loss, trailing stop, time decay, Greeks, breakeven) |
| **Vol Analytics** | 8 (IV rank, IV percentile, put/call skew, term structure, expansion/crush probability) |

## Evolution Timeline

| Version | Status | Key Features |
|---------|--------|--------------|
| **v1.0** | Baseline | Hedge/Liquidity/Sentiment engines, basic composer |
| **v2.0** | **Current** | 14 strategies, risk analyzer, exit manager, vol intelligence |
| **v2.5** | **Current** | Vol surface analytics, IV rank/percentile, skew detection |
| **v3.0** | **Current** | Execution layer, smart router, cost models, simulated broker |
| **v3.1** | Next | Orchestration pipeline, SPY paper trading, persistence |
| **v4.0** | Future | Multi-symbol support, live broker integration, real-time streaming |

## Before/After: Major Milestones

### PR #19: Trade Agent v2 (v1 → v2)
- **Before**: 7 strategies, basic Greeks, no risk analysis
- **After**: 14 strategies, PnL cones, breakevens, exit manager
- **LOC**: +2,181 (10 files)
- **Tests**: +49

### PR #20: Vol Intelligence v2.5
- **Before**: No volatility surface analytics
- **After**: IV rank/percentile, skew, term structure, expansion/crush signals
- **LOC**: +1,487 (4 files)
- **Tests**: +37

### PR #21: Execution Layer v3
- **Before**: No execution infrastructure
- **After**: Smart router, cost models, simulated broker, order tracking
- **LOC**: +1,633 (9 files)
- **Tests**: +17

### PR #22: Strategy Optimizer
- **Before**: Manual hyperparameter tuning
- **After**: Optuna optimization, regime-based performance, Kelly refinement, ML integration
- **LOC**: +1,557 (10 files)
- **Tests**: +63

### PR #23: Test Hardening Phase 1
- **Before**: 440/487 passing (90.3%), 33 failing
- **After**: 479/487 passing (98.4%), 0 failing
- **Fixes**: Direction enum bug, PnL cone pricing, Optuna seed, exit manager logic
- **LOC**: +42 (5 files)

## Extension Points

See [DEV_GUIDE.md](DEV_GUIDE.md) for:
- Adding new strategies
- Plugging in ML models
- Adding broker adapters
- Extending engines
- Defining custom optimization targets

## Operational Readiness

See [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) for:
- Daily SPY paper trading workflow
- Persistence and review procedures
- Risk caps and kill switches
- Multi-symbol rollout plan
- Live trading checklist
