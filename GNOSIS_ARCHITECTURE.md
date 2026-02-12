# GNOSIS Trading System - Complete Architecture

## Overview

The **GNOSIS (Great Neural Optimization System for Intelligent Speculation)** is a comprehensive algorithmic trading system designed to profit in any market condition through intelligent options strategies. It combines multiple analysis methodologies, physics-based price modeling, and machine learning to generate high-confidence trading signals.

---

## System Architecture Diagram

```
                                    GNOSIS TRADING SYSTEM
    ══════════════════════════════════════════════════════════════════════════════

                                   ┌─────────────────────────┐
                                   │     DATA SOURCES        │
                                   └─────────────────────────┘
                                              │
            ┌─────────────────────────────────┼─────────────────────────────────┐
            │                                 │                                 │
            ▼                                 ▼                                 ▼
    ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
    │  Massive.com  │              │    Alpaca     │              │ Unusual Whales│
    │   (Polygon)   │              │  Market Data  │              │  Options Flow │
    │               │              │               │              │               │
    │ • OHLCV Bars  │              │ • OHLCV Bars  │              │ • Flow Data   │
    │ • Options     │              │ • Real-time   │              │ • Alerts      │
    │ • Snapshots   │              │ • Paper Trade │              │ • Greeks      │
    └───────┬───────┘              └───────┬───────┘              └───────┬───────┘
            │                                 │                                 │
            └─────────────────────────────────┼─────────────────────────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────────┐
                           │        INPUT ADAPTERS LAYER          │
                           │  /engines/inputs/                    │
                           │                                      │
                           │  • MassiveMarketAdapter              │
                           │  • MassiveOptionsAdapter             │
                           │  • AlpacaMarketDataAdapter           │
                           │  • UnusualWhalesAdapter              │
                           │  • PolygonOptionsAdapter             │
                           └──────────────────┬───────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────────┐
                    │                         │                             │
                    ▼                         ▼                             ▼
    ┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
    │   SENTIMENT ENGINE     │ │   LIQUIDITY ENGINE     │ │   PHYSICS ENGINE       │
    │   /engines/sentiment/  │ │   /engines/liquidity/  │ │   (Price-as-Particle)  │
    │                        │ │                        │ │                        │
    │ ┌────────────────────┐ │ │ ┌────────────────────┐ │ │ ┌────────────────────┐ │
    │ │   RSI Analysis     │ │ │ │ PENTA Methodology │ │ │ │  Mass (Mkt Cap)    │ │
    │ │   • Overbought     │ │ │ │                   │ │ │ │  • Mega: 10        │ │
    │ │   • Oversold       │ │ │ │ 1. Wyckoff VSA    │ │ │ │  • Large: 5        │ │
    │ │   • Divergences    │ │ │ │ 2. ICT Concepts   │ │ │ │  • Mid: 2          │ │
    │ └────────────────────┘ │ │ │ 3. Order Flow     │ │ │ │  • Small: 0.5      │ │
    │ ┌────────────────────┐ │ │ │ 4. Supply/Demand  │ │ │ └────────────────────┘ │
    │ │   MACD Analysis    │ │ │ │ 5. Liquidity Pool │ │ │ ┌────────────────────┐ │
    │ │   • Cross signals  │ │ │ └────────────────────┘ │ │ │  Velocity          │ │
    │ │   • Histogram      │ │ │ ┌────────────────────┐ │ │ │  (Rate of Change)  │ │
    │ │   • Trend          │ │ │ │  Market Quality    │ │ │ │  p = m × v         │ │
    │ └────────────────────┘ │ │ │  • Bid-Ask Spread  │ │ │ └────────────────────┘ │
    │ ┌────────────────────┐ │ │ │  • Volume          │ │ │ ┌────────────────────┐ │
    │ │   Momentum         │ │ │ │  • Depth           │ │ │ │  Energy (Volume)   │ │
    │ │   • 5/10/20 period │ │ │ │  • Impact Cost     │ │ │ │  KE = 0.5mv²       │ │
    │ │   • Acceleration   │ │ │ └────────────────────┘ │ │ │  Force = Energy/m  │ │
    │ └────────────────────┘ │ │ ┌────────────────────┐ │ │ └────────────────────┘ │
    │ ┌────────────────────┐ │ │ │  Bollinger Bands   │ │ │ ┌────────────────────┐ │
    │ │   Stochastic/      │ │ │ │  • BB Width        │ │ │ │  Momentum          │ │
    │ │   Williams %R      │ │ │ │  • BB Position     │ │ │ │  • Direction       │ │
    │ └────────────────────┘ │ │ │  • Squeeze Detect  │ │ │ │  • Continuation    │ │
    │ ┌────────────────────┐ │ │ └────────────────────┘ │ │ └────────────────────┘ │
    │ │  Social Media      │ │ │ ┌────────────────────┐ │ │ ┌────────────────────┐ │
    │ │  • Twitter/X       │ │ │ │  Accum/Distrib     │ │ │ │  Uncertainty       │ │
    │ │  • Reddit          │ │ │ │  • A/D Line        │ │ │ │  (Volatility)      │ │
    │ │  • WSB Sentiment   │ │ │ │  • OBV Trend       │ │ │ │  • Position range  │ │
    │ └────────────────────┘ │ │ │  • Money Flow      │ │ │ │  • Momentum range  │ │
    │                        │ │ └────────────────────┘ │ │ └────────────────────┘ │
    └───────────┬────────────┘ └───────────┬────────────┘ └───────────┬────────────┘
                │                          │                          │
                │    ┌─────────────────────┼──────────────────────────┘
                │    │                     │
                ▼    ▼                     ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                          PREDICTION ENGINE                                  │
    │                                                                             │
    │  ┌─────────────────────────┐    ┌─────────────────────────────┐            │
    │  │   PREDICTIVE CONES     │    │   SUPPORT / RESISTANCE      │            │
    │  │                        │    │                             │            │
    │  │   Based on volatility  │    │   • Swing Highs/Lows        │            │
    │  │   (Geometric Brownian) │    │   • Pivot Points (R1/R2/S1) │            │
    │  │                        │    │   • Level Clustering        │            │
    │  │   • 1σ bounds (68%)    │    │   • Strength Scoring        │            │
    │  │   • 2σ bounds (95%)    │    │                             │            │
    │  │   • Expected path      │    │   Distance to nearest:      │            │
    │  │                        │    │   • Resistance %            │            │
    │  │   Horizons:            │    │   • Support %               │            │
    │  │   [1, 5, 10, 21 days]  │    │                             │            │
    │  └─────────────────────────┘    └─────────────────────────────┘            │
    └────────────────────────────────────────┬───────────────────────────────────┘
                                             │
                                             ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                        MARKET REGIME CLASSIFIER                             │
    │                                                                             │
    │   Combined Score = (Sentiment × 0.4) + (Liquidity × 0.3) + (Physics × 0.3) │
    │                                                                             │
    │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
    │   │STRONG_BULL │  │   BULL     │  │  NEUTRAL   │  │   BEAR     │           │
    │   │  ≥ 0.6     │  │  ≥ 0.3     │  │ -0.3~+0.3  │  │  ≤ -0.3    │           │
    │   └────────────┘  └────────────┘  └────────────┘  └────────────┘           │
    │                                                                             │
    │   ┌────────────┐  ┌────────────┐                                           │
    │   │  HIGH_VOL  │  │  LOW_VOL   │  (Based on Bollinger Squeeze/Width)       │
    │   │ BB Width>10%│ │ BB Squeeze │                                           │
    │   └────────────┘  └────────────┘                                           │
    └────────────────────────────────────────┬───────────────────────────────────┘
                                             │
                                             ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                      OPTIONS STRATEGY SELECTOR                              │
    │                                                                             │
    │   ┌─────────────────────────────────────────────────────────────────────┐  │
    │   │                    STRATEGY SELECTION MATRIX                         │  │
    │   ├────────────────┬────────────┬────────┬─────────────────────────────┤  │
    │   │ REGIME         │ SENTIMENT  │ IV     │ STRATEGY                    │  │
    │   ├────────────────┼────────────┼────────┼─────────────────────────────┤  │
    │   │ Strong Bull    │ Bullish    │ High   │ Bull Call Spread            │  │
    │   │ Strong Bull    │ Bullish    │ Low    │ Long Call                   │  │
    │   │ Strong Bear    │ Bearish    │ High   │ Bear Put Spread             │  │
    │   │ Strong Bear    │ Bearish    │ Low    │ Long Put                    │  │
    │   │ Neutral        │ Mixed      │ High   │ Iron Condor                 │  │
    │   │ Neutral        │ Mixed      │ Low    │ Long Straddle               │  │
    │   │ High Vol       │ Unclear    │ High   │ Iron Condor / Strangle      │  │
    │   │ Low Vol        │ Any        │ Low    │ Long Straddle / Strangle    │  │
    │   └────────────────┴────────────┴────────┴─────────────────────────────┘  │
    │                                                                             │
    │   Black-Scholes Pricing: C = S·N(d₁) - K·e^(-rT)·N(d₂)                     │
    │   Greeks: Delta, Gamma, Theta, Vega                                        │
    └────────────────────────────────────────┬───────────────────────────────────┘
                                             │
                                             ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                          AGENT LAYER                                        │
    │                          /agents/                                           │
    │                                                                             │
    │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
    │   │ Liquidity Agent │  │ Sentiment Agent │  │   Hedge Agent   │            │
    │   │      V5         │  │      V3         │  │      V4         │            │
    │   │                 │  │                 │  │                 │            │
    │   │ PENTA Analysis  │  │ Multi-source    │  │ Risk Management │            │
    │   │ • Confluence    │  │ • News          │  │ • Delta Hedge   │            │
    │   │ • Trading Bias  │  │ • Flow          │  │ • Portfolio     │            │
    │   │ • Quality Grade │  │ • Technical     │  │ • Greeks Mgmt   │            │
    │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
    │            │                    │                    │                     │
    │            └────────────────────┼────────────────────┘                     │
    │                                 │                                          │
    │                                 ▼                                          │
    │                    ┌─────────────────────────┐                             │
    │                    │       COMPOSER          │                             │
    │                    │   /agents/composer/     │                             │
    │                    │                         │                             │
    │                    │ Combines all signals    │                             │
    │                    │ into trade decisions    │                             │
    │                    └────────────┬────────────┘                             │
    │                                 │                                          │
    └─────────────────────────────────┼──────────────────────────────────────────┘
                                      │
                                      ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                       TRADE EXECUTION LAYER                                 │
    │                                                                             │
    │   ┌─────────────────────────────────────────────────────────────────────┐  │
    │   │                    RISK MANAGEMENT                                   │  │
    │   │                                                                      │  │
    │   │   • Max Position Size: 5% of capital per trade                      │  │
    │   │   • Max Open Positions: 5 concurrent                                │  │
    │   │   • Stop Loss: 50% of premium                                       │  │
    │   │   • Take Profit: 100% gain                                          │  │
    │   │   • DTE Exit: Close at 7 days before expiration                     │  │
    │   │   • Signal Reversal Exit                                            │  │
    │   └─────────────────────────────────────────────────────────────────────┘  │
    │                                                                             │
    │   ┌─────────────────────────────────────────────────────────────────────┐  │
    │   │                    BROKERS                                           │  │
    │   │                                                                      │  │
    │   │   • Alpaca (Paper + Live Trading)                                   │  │
    │   │   • Interactive Brokers (Integration ready)                         │  │
    │   └─────────────────────────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
/home/root/webapp/
│
├── agents/                          # Intelligent Agents
│   ├── __init__.py
│   ├── base.py                      # Base agent class
│   ├── liquidity_agent_v5.py        # PENTA methodology agent
│   ├── sentiment_agent_v3.py        # Multi-source sentiment
│   ├── hedge_agent_v4.py            # Hedging strategies
│   ├── risk_management_agent.py     # Risk controls
│   ├── regime_detection_agent.py    # Market regime detection
│   └── composer/                    # Signal composition
│
├── engines/                         # Core Processing Engines
│   ├── inputs/                      # Data Adapters
│   │   ├── massive_market_adapter.py
│   │   ├── massive_options_adapter.py
│   │   ├── alpaca_market_adapter.py
│   │   ├── unusual_whales_adapter.py
│   │   └── polygon_options_adapter.py
│   │
│   ├── liquidity/                   # Liquidity Analysis
│   │   ├── liquidity_engine_v5.py   # Main engine (PENTA)
│   │   ├── ict_engine.py            # ICT concepts
│   │   ├── order_flow_engine.py     # Order flow analysis
│   │   ├── supply_demand_engine.py  # S/D zones
│   │   └── liquidity_concepts_engine.py
│   │
│   ├── sentiment/                   # Sentiment Analysis
│   │   ├── sentiment_engine_v3.py   # Main engine
│   │   └── social_media_adapter.py  # Twitter/Reddit
│   │
│   └── hedge/                       # Hedging Engine
│
├── backtesting/                     # Backtesting Framework
│   ├── gnosis_options_backtest.py   # Options backtest engine
│   ├── mtf_backtest_engine.py       # Multi-timeframe backtest
│   ├── liquidity_sentiment_backtest.py
│   └── walk_forward_engine.py       # Walk-forward optimization
│
├── models/                          # ML Models
│   ├── time_series/
│   │   ├── lstm_forecaster.py
│   │   └── transformer_forecaster.py
│   ├── ensemble/
│   │   └── xgboost_model.py
│   └── rl_agents/                   # Reinforcement Learning
│
├── alpha/                           # Alpha Generation
│   ├── signal_generator.py
│   ├── technical_analyzer.py
│   └── options_trader.py
│
├── config/                          # Configuration
│   ├── gnosis_config_v2.py
│   └── credentials.py
│
└── gnosis/                          # Core GNOSIS Module
    ├── trading/
    │   ├── live_trading_engine.py
    │   └── live_bot.py
    └── unified_trading_bot.py
```

---

## Engine Details

### 1. Price-as-Particle Physics Engine

The physics model treats price as a particle with physical properties:

```python
class PriceParticle:
    price: float              # Current price
    velocity: float           # Rate of change (momentum)
    acceleration: float       # Change in velocity
    mass: float               # Market cap relative (0.5 - 10)
    energy: float             # Volume-weighted energy
    kinetic_energy: float     # 0.5 * mass * velocity²
    potential_energy: float   # Distance from equilibrium (MA)
    momentum: float           # mass × velocity
    force: float              # Volume pressure
    friction: float           # Liquidity resistance
    position_uncertainty: float  # Price volatility range
```

**Key Insights:**
- Mega-cap stocks (AAPL, MSFT) have high mass → Need huge volume to move
- Small-caps have low mass → Can move violently on small volume
- Volume = Energy input → Moves price based on mass
- Breakouts require energy > potential barrier

### 2. Sentiment Engine V3

Multi-source sentiment analysis:

| Source | Weight | Indicators |
|--------|--------|------------|
| News | 25% | Headlines, Filings |
| Flow | 35% | Options flow, Unusual Whales |
| Technical | 25% | RSI, MACD, Momentum |
| Social | 15% | Twitter, Reddit, WSB |

**Technical Indicators:**
- **RSI (14)**: Overbought (>70), Oversold (<30), Divergences
- **MACD**: Line, Signal, Histogram, Crosses
- **Momentum**: 5/10/20 period rate of change
- **Stochastic**: %K, %D, Overbought/Oversold
- **Williams %R**: Momentum confirmation

### 3. Liquidity Engine V5 (PENTA Methodology)

Five sub-engines for comprehensive market analysis:

| Engine | Analysis | Key Outputs |
|--------|----------|-------------|
| **Wyckoff** | VSA, Phases, Events | Phase, Event, Bias |
| **ICT** | FVGs, Order Blocks, OTE | Discount/Premium, FVG count |
| **Order Flow** | Footprint, CVD, Volume Profile | CVD trend, Absorption, Exhaustion |
| **Supply/Demand** | Zones, Strength | Fresh zones, Nearest levels |
| **Liquidity Concepts** | Pools, Voids, Inducements | Buy/Sell pools, Weak highs/lows |

**Confluence Levels:**
- PENTA (5/5) → 30% confidence bonus
- QUAD (4/5) → 25% confidence bonus
- TRIPLE (3/5) → 15% confidence bonus
- DOUBLE (2/5) → 8% confidence bonus

### 4. Prediction Engine

**Predictive Cones (Geometric Brownian Motion):**
```
Price(t) = Price(0) × exp(drift × t ± σ × √t)

Horizons: [1, 5, 10, 21 days]
1σ bounds: 68% probability
2σ bounds: 95% probability
```

**Support/Resistance Detection:**
- Swing highs/lows identification
- Pivot points (R1, R2, S1, S2)
- Level clustering (2% threshold)
- Distance-to-level calculations

---

## Options Strategy Matrix

| Regime | Direction | IV | Strategy | Win Condition |
|--------|-----------|-----|----------|---------------|
| Strong Bull | Bullish | High | Bull Call Spread | Price rises |
| Strong Bull | Bullish | Low | Long Call | Price rises |
| Strong Bear | Bearish | High | Bear Put Spread | Price falls |
| Strong Bear | Bearish | Low | Long Put | Price falls |
| Neutral | Mixed | High | Iron Condor | Price stays range |
| Neutral | Mixed | Low | Long Straddle | Big move either way |
| High Vol | Any | High | Iron Condor | Volatility crush |
| Low Vol (Squeeze) | Any | Low | Long Straddle | Breakout expected |

---

## Backtest Results Summary

### GNOSIS Options Backtest (2020-2024)

**Configuration:**
- Symbols: SPY, QQQ, AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOGL
- Initial Capital: $100,000
- Period: 2020-01-01 to 2024-12-01

**Results by Strategy:**

| Strategy | Trades | Win Rate | P&L |
|----------|--------|----------|-----|
| Long Straddle | 225 | 41.3% | +$102,263 |
| Long Strangle | - | - | Profitable |
| Bull Call Spread | - | 22% | -$157,224 |
| Bear Put Spread | - | - | Loss |

**Key Findings:**

1. **Straddles/Strangles dominate** in uncertain markets
2. **High Volatility regime** most profitable (+$154/trade)
3. **Neutral regime** second best (47.9% win rate)
4. **Directional spreads** underperform in choppy markets

### MTF Backtest Results (2020-2024)

**Configuration:**
- Timeframes: 1W, 1D, 4H, 1H
- Initial Capital: $100,000

**Results:**
- Total Return: +11.01% ($11,008.92)
- Total Trades: 335
- Win Rate: 32.8%
- Profit Factor: 1.13
- Max Drawdown: 11.05%

**Alignment Analysis:**

| Alignment | Win Rate | P&L |
|-----------|----------|-----|
| 4/4 TF | 33.6% | +$7,541 |
| 3/4 TF | 30.9% | +$3,467 |

**HTF Bias Performance:**
- Bullish HTF: 40.7% win rate, +$13,867
- Bearish HTF: 24.1% win rate, -$2,858

---

## API Credentials

```python
# Massive.com (Polygon) API
Primary Key: Jm_fqc_gtSTSXG78P67dpBpO3LX_4P6D
Secondary Key: 22265906-ec01-4a42-928a-0037ccadbde3

# Alpaca API - Configured in environment
# Unusual Whales - Configured in environment
```

---

## Entry Points

| Use Case | Entry Point | Command |
|----------|-------------|---------|
| Live Trading | `gnosis_live_trading_quickstart.py` | `python gnosis_live_trading_quickstart.py` |
| Paper Trading | `gnosis/trading/live_bot.py` | `python -m gnosis.trading.live_bot --paper` |
| Options Backtest | `backtesting/gnosis_options_backtest.py` | `python -m backtesting.gnosis_options_backtest` |
| MTF Backtest | `backtesting/mtf_backtest_engine.py` | `python -m backtesting.mtf_backtest_engine` |
| Dashboard | `dashboard.py` | `python dashboard.py` |
| API Server | `saas/app.py` | `python -m saas.app` |

---

## Signal Flow

```
1. DATA COLLECTION
   Market Data → Adapters → OHLCV Bars, Options Chain, Flow Data

2. ENGINE PROCESSING
   ├── Physics Engine → Mass, Velocity, Momentum, Energy
   ├── Sentiment Engine → RSI, MACD, Momentum, Social Score
   ├── Liquidity Engine → PENTA Confluence, Quality Grade
   └── Prediction Engine → Cones, Support/Resistance

3. REGIME CLASSIFICATION
   Combined Score → Market Regime (Bull/Bear/Neutral/Vol)

4. STRATEGY SELECTION
   Regime + IV → Optimal Options Strategy

5. POSITION CONSTRUCTION
   Strategy + Greeks → Theoretical Position

6. RISK MANAGEMENT
   Position Size, Stop Loss, Take Profit, DTE Rules

7. EXECUTION
   Signal → Broker API → Order Execution
```

---

## Recommendations

Based on backtest results:

1. **Focus on Straddles/Strangles** in uncertain markets
2. **Use 4/4 Timeframe Alignment** for higher confidence
3. **Filter for High Volatility Regime** for volatility plays
4. **Prefer Bullish HTF Bias** for directional trades
5. **Apply Bollinger Squeeze** filter for breakout entries
6. **Use Physics Model** to identify energy required for moves

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| V1.0 | 2024-01 | Initial GNOSIS framework |
| V2.0 | 2024-06 | Added PENTA methodology |
| V3.0 | 2024-09 | Physics engine, MTF analysis |
| V3.1 | 2024-12 | Options backtest, Social sentiment |

---

*GNOSIS Trading System - Profit in Any Market Condition*
