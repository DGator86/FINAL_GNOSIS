# GNOSIS Trading System Architecture

## Overview

The GNOSIS Trading System follows a layered architecture where data flows upward through increasingly intelligent processing layers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MONITORING AGENT LAYER                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   Full Gnosis Monitor       │    │        Alpha Monitor                │ │
│  │   - Position tracking       │    │   - Signal performance tracking     │ │
│  │   - Risk monitoring         │    │   - Accuracy metrics                │ │
│  │   - P&L feedback            │    │   - User feedback loop              │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TRADE AGENT LAYER                                │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   Full Gnosis Agent         │    │        Alpha Agent                  │ │
│  │   - Full automated trading  │    │   - Signals only                    │ │
│  │   - Entry/exit execution    │    │   - Directional signals             │ │
│  │   - Stop loss management    │    │   - Simple options                  │ │
│  │   - Position sizing         │    │   - Retail-focused (Robinhood)      │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPOSER AGENT LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Composer Agent                                   ││
│  │   - Receives signals from all primary agents                            ││
│  │   - Builds consensus across methodologies                               ││
│  │   - Weights and combines signals                                        ││
│  │   - Produces unified direction + confidence                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRIMARY AGENT LAYER                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   Hedge Agent   │  │ Sentiment Agent │  │     Liquidity Agent         │  │
│  │                 │  │                 │  │                             │  │
│  │ Fed by:         │  │ Fed by:         │  │ Fed by:                     │  │
│  │ - Hedge Engine  │  │ - Sentiment Eng │  │ - Liquidity Engine          │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             ENGINE LAYER                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   Hedge Engine  │  │Sentiment Engine │  │     Liquidity Engine        │  │
│  │                 │  │                 │  │                             │  │
│  │ - Dealer flow   │  │ - News          │  │ - Market Quality            │  │
│  │ - Gamma/Vanna   │  │ - Social Media  │  │ - PENTA Methodology:        │  │
│  │ - Options Greeks│  │   (Twitter,     │  │   • Wyckoff (VSA, Phases)   │  │
│  │ - Energy flow   │  │    Reddit)      │  │   • ICT (FVGs, OB, OTE)     │  │
│  │ - Regime detect │  │ - Technical     │  │   • Order Flow (CVD, VP)    │  │
│  │                 │  │   Indicators    │  │   • Supply & Demand         │  │
│  │                 │  │                 │  │   • Liquidity Concepts      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA ADAPTERS LAYER                                │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐│
│  │Market Adapter │ │Options Adapter│ │ News Adapter  │ │Social Media Adapt ││
│  │(Alpaca/IBKR)  │ │(UW/CBOE)      │ │               │ │(Twitter/Reddit)   ││
│  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Layer Details

### 1. ENGINE LAYER (Data Processing)

#### Hedge Engine (HedgeEngineV3)
- **Purpose**: Analyze dealer positioning and options flow
- **Inputs**: Options chain data, market data
- **Outputs**: HedgeSnapshot with gamma, vanna, energy asymmetry, regime

#### Sentiment Engine (SentimentEngineV1/V3)
- **Purpose**: Multi-source sentiment analysis
- **Components**:
  - News sentiment processor
  - Social media processor (Twitter, Reddit)
  - Technical indicators processor
- **Outputs**: SentimentSnapshot with composite score

#### Liquidity Engine (LiquidityEngineV5 - NEW)
- **Purpose**: Comprehensive market structure and liquidity analysis
- **Components**:
  - **Market Quality**: Bid-ask spreads, depth, tradability
  - **PENTA Methodology** (5 sub-engines):
    1. **Wyckoff** (LiquidityEngineV4): VSA, phases, events
    2. **ICT** (ICTEngine): FVGs, order blocks, OTE
    3. **Order Flow** (OrderFlowEngine): Footprint, CVD, volume profile
    4. **Supply & Demand** (SupplyDemandEngine): Zones, strength
    5. **Liquidity Concepts** (LiquidityConceptsEngine): Pools, voids, inducements
- **Outputs**: LiquiditySnapshot with PENTA state

### 2. PRIMARY AGENT LAYER (Signal Generation)

Each primary agent:
- Receives data from its corresponding engine
- Generates directional signals with confidence
- Provides reasoning and risk factors

#### Hedge Agent
- Interprets dealer flow and gamma positioning
- Generates signals based on energy asymmetry

#### Sentiment Agent  
- Interprets news and social sentiment
- Generates signals based on market mood

#### Liquidity Agent (LiquidityAgentV5 - Updated)
- Interprets market structure and PENTA methodology
- Combines signals from all 5 sub-methodologies
- Applies confluence bonuses (PENTA +30%, QUAD +25%, etc.)

### 3. COMPOSER AGENT LAYER (Consensus Building)

The Composer Agent:
- Receives suggestions from all primary agents
- Weighs signals based on confidence and methodology
- Builds consensus across different views
- Produces unified trading direction and confidence

### 4. TRADE AGENT LAYER (Execution)

#### Full Gnosis Agent
- **Purpose**: Fully automated trading
- **Capabilities**:
  - Trade entry and exit
  - Stop loss management
  - Take profit targets
  - Position sizing
  - Risk management

#### Alpha Agent
- **Purpose**: Signal generation for retail traders
- **Capabilities**:
  - Directional signals (BUY/SELL/HOLD)
  - Simple options recommendations
  - PDT-compliant suggestions
  - Robinhood/Webull compatible

### 5. MONITORING AGENT LAYER (Feedback)

#### Full Gnosis Monitor
- Tracks all open positions
- Monitors risk exposure
- Provides P&L feedback
- Alerts on risk thresholds

#### Alpha Monitor
- Tracks signal accuracy
- Monitors win rate and R:R
- Provides performance metrics
- User feedback integration

### 6. MACHINE LEARNING SCAFFOLD

ML components that enhance all layers:
- Regime detection models
- Price prediction models
- Signal optimization
- Reinforcement learning for trade sizing

## Data Flow

```
Market Data → Adapters → Engines → Primary Agents → Composer → Trade Agents → Monitors
                                                                    │
                                                                    ▼
                                                              ML Feedback Loop
```

## Key Principles

1. **Single Responsibility**: Each engine/agent has one job
2. **Hierarchical Flow**: Data flows upward through layers
3. **Loose Coupling**: Layers communicate through defined interfaces
4. **Composability**: Trade agents can use different combinations
5. **Observability**: Monitors provide feedback at each layer

## Component Versions

### Current (Recommended)
| Layer | Component | Version | Description |
|-------|-----------|---------|-------------|
| Engine | LiquidityEngineV5 | v5.0.0 | Unified PENTA engine |
| Agent | LiquidityAgentV5 | v5.0.0 | PENTA methodology integration |
| Composer | ComposerAgentV4 | v4.0.0 | Full architecture integration |
| Trade | FullGnosisTradeAgentV2 | v2.0.0 | Automated trading with PENTA |
| Trade | AlphaTradeAgentV2 | v2.0.0 | Retail signals with confluence |
| Monitor | GnosisMonitor | v1.0.0 | Position/P&L tracking |
| Monitor | AlphaMonitor | v1.0.0 | Signal accuracy tracking |

### Legacy (Backward Compatible)
| Layer | Component | Version | Notes |
|-------|-----------|---------|-------|
| Engine | LiquidityEngineV1-V4 | Various | Still supported |
| Agent | LiquidityAgentV1-V4 | Various | Use V5 for new code |
| Composer | ComposerAgentV1-V3 | Various | V4 recommended |
| Trade | TradeAgentV1-V3 | Various | Use V2 for new code |

## PENTA Methodology

The PENTA methodology combines 5 trading approaches:

```
PENTA Methodology
├── Wyckoff (18% weight)
│   ├── Volume Spread Analysis (VSA)
│   ├── Seven Logical Events
│   └── Five Phases
├── ICT (18% weight)
│   ├── Fair Value Gaps (FVG)
│   ├── Order Blocks
│   └── Optimal Trade Entry (OTE)
├── Order Flow (18% weight)
│   ├── Footprint Charts
│   ├── Cumulative Volume Delta
│   └── Volume Profile
├── Supply & Demand (18% weight)
│   ├── Zone Detection
│   ├── Zone Strength
│   └── Risk/Reward
└── Liquidity Concepts (18% weight)
    ├── Latent Liquidity Pools
    ├── Strong/Weak Swing Classification
    ├── Liquidity Voids
    ├── Fractal Market Structure
    └── Inducement Detection
```

### Confluence Bonuses
| Confluence | Agreeing Methods | Confidence Bonus |
|------------|------------------|------------------|
| PENTA | 5/5 | +30% |
| QUAD | 4/5 | +25% |
| TRIPLE | 3/5 | +20% |
| DOUBLE | 2/5 | +10% |

## Usage Example

```python
from engines.engine_factory import create_unified_analysis_engines
from agents.liquidity_agent_v5 import LiquidityAgentV5
from agents.composer.composer_agent_v4 import ComposerAgentV4
from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
from agents.monitoring import AlphaMonitor

# Create engines
engines = create_unified_analysis_engines(use_unified_v5=True)

# Create agent with unified engine
liquidity_agent = LiquidityAgentV5(
    config={"min_confidence": 0.6},
    liquidity_engine_v5=engines["liquidity_engine_v5"],
)

# Create composer
composer = ComposerAgentV4(
    weights={"hedge": 0.4, "sentiment": 0.4, "liquidity": 0.2}
)

# Create trade agent
monitor = AlphaMonitor({})
alpha_agent = AlphaTradeAgentV2(
    config={"min_confidence": 0.6},
    composer=composer,
    monitor=monitor,
)

# Generate signal
composer_output = composer.compose(
    hedge_signal={"direction": "bullish", "confidence": 0.7},
    sentiment_signal={"direction": "bullish", "confidence": 0.6},
    liquidity_signal={"direction": "neutral", "confidence": 0.5},
    penta_confluence="QUAD",
)

signal = alpha_agent.process_composer_output(
    composer_output, symbol="AAPL", current_price=230.0
)

print(signal.to_robinhood_format())
```
