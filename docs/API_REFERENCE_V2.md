# GNOSIS Trading System - API Reference V2

## Overview

The GNOSIS Trading System is a multi-layer architecture for intelligent trading signal generation and execution. This document provides API reference for all V2 components.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING AGENT LAYER                       │
│   GnosisMonitor (positions, P&L) | AlphaMonitor (signals)      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     TRADE AGENT LAYER                           │
│   FullGnosisTradeAgentV2 (auto) | AlphaTradeAgentV2 (signals)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   COMPOSER AGENT LAYER                          │
│              ComposerAgentV4 (consensus builder)               │
│         PENTA Confluence | Express Modes | Weighted Avg        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                  PRIMARY AGENT LAYER                            │
│   HedgeAgentV3 | SentimentAgentV1/V3 | LiquidityAgentV5        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     ENGINE LAYER                                │
│  HedgeEngineV3 | SentimentEngineV1 | LiquidityEngineV5 (PENTA) │
│                                                                 │
│  PENTA Sub-Engines: Wyckoff | ICT | Order Flow | S&D | LiqCon  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ComposerAgentV4

The consensus builder that aggregates signals from primary agents.

### Import

```python
from agents.composer.composer_agent_v4 import ComposerAgentV4, ComposerOutput, ComposerMode
```

### Initialization

```python
composer = ComposerAgentV4(
    weights={"hedge": 0.4, "sentiment": 0.4, "liquidity": 0.2},
    min_consensus_score=0.5,
)
```

### Methods

#### compose()

Compose signals from primary agents into a consensus.

```python
output = composer.compose(
    hedge_signal={"direction": "bullish", "confidence": 0.75},
    sentiment_signal={"direction": "bullish", "confidence": 0.70},
    liquidity_signal={"direction": "bullish", "confidence": 0.65},
    penta_confluence="QUAD",  # Optional: PENTA, QUAD, TRIPLE, DUAL
    mode=ComposerMode.STANDARD,  # Optional: EXPRESS_0DTE, EXPRESS_CHEAP
)
```

### ComposerOutput

```python
@dataclass
class ComposerOutput:
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    consensus_score: float  # Raw weighted score
    reasoning: str  # Human-readable explanation
    penta_confluence: Optional[str]  # PENTA level if detected
    penta_confidence_bonus: float  # Bonus applied (0.05-0.30)
    mode: str  # "standard", "0dte", "cheap_calls"
```

---

## AlphaTradeAgentV2

Signal-only trade agent for retail traders (Robinhood/Webull).

### Import

```python
from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2, AlphaSignalV2
```

### Initialization

```python
agent = AlphaTradeAgentV2(config={
    "min_confidence": 0.5,
    "strong_confidence_threshold": 0.8,
    "default_stop_pct": 0.03,
    "default_target_pct": 0.05,
})
```

### Methods

#### process_composer_output()

Process ComposerOutput into an actionable signal.

```python
signal = agent.process_composer_output(
    composer_output=composer_output,
    symbol="AAPL",
    current_price=230.0,
)
```

### AlphaSignalV2

```python
@dataclass
class AlphaSignalV2:
    symbol: str
    signal_type: SignalType  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    strength: SignalStrength  # STRONG, MODERATE, WEAK
    direction: str  # "BUY", "SELL", "HOLD"
    confidence: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    penta_confluence: Optional[str]
    options_play: Optional[Dict]  # Options suggestion
```

---

## FullGnosisTradeAgentV2

Automated trading agent with position management.

### Import

```python
from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2, TradeAction
```

### Initialization

```python
agent = FullGnosisTradeAgentV2(config={
    "min_confidence": 0.6,
    "max_position_size": 0.10,
    "portfolio_value": 100000,
})
```

### Methods

#### process_composer_output()

Process ComposerOutput into a trade action.

```python
action = agent.process_composer_output(
    composer_output=composer_output,
    symbol="NVDA",
    current_price=145.0,
)
```

### TradeAction

```python
@dataclass
class TradeAction:
    action_type: TradeActionType  # ENTER_LONG, ENTER_SHORT, EXIT, NO_ACTION
    symbol: str
    direction: str
    quantity: int
    price: float
    stop_loss: float
    take_profit: float
    penta_bonus: float  # Position size bonus from PENTA
```

---

## Monitoring Agents

### GnosisMonitor

Monitors full trading agent positions and P&L.

```python
from agents.monitoring import GnosisMonitor

monitor = GnosisMonitor(config={"initial_equity": 100000})

# Update with positions
monitor.update(
    positions={"AAPL": {"direction": "LONG", "quantity": 10, "entry_price": 230}},
    current_prices={"AAPL": 235.0}
)

# Get metrics
metrics = monitor.get_metrics()
print(f"Win Rate: {metrics.win_rate}")
print(f"P&L: {metrics.total_pnl}")
```

### AlphaMonitor

Monitors signal accuracy for retail signals.

```python
from agents.monitoring import AlphaMonitor

monitor = AlphaMonitor(config={})

# Track a signal
monitor.update(signal={
    "symbol": "AAPL",
    "signal_type": "BUY",
    "confidence": 0.75,
})

# Record outcome
monitor.update(outcome={
    "symbol": "AAPL",
    "correct": True,
    "pnl": 150.0,
})

print(f"Signal Accuracy: {monitor.metrics.signal_accuracy}")
```

---

## LiquidityEngineV5

Unified market quality + PENTA methodology engine.

### Import

```python
from engines.liquidity import LiquidityEngineV5
```

### Initialization

```python
engine = LiquidityEngineV5(
    market_adapter=market_adapter,  # Optional
    config={
        "min_volume_threshold": 1_000_000,
        "spread_threshold": 0.5,
    }
)
```

### Methods

#### get_penta_engines()

Get references to all PENTA sub-engines.

```python
engines = engine.get_penta_engines()
# Returns: {
#   "wyckoff": WyckoffEngine or None,
#   "ict": ICTEngine or None,
#   "order_flow": OrderFlowEngine or None,
#   "supply_demand": SupplyDemandEngine or None,
#   "liquidity_concepts": LiquidityConceptsEngine or None,
# }
```

---

## LiquidityAgentV5

Primary agent for PENTA methodology signals.

### Import

```python
from agents.liquidity_agent_v5 import LiquidityAgentV5
```

### Initialization

```python
agent = LiquidityAgentV5(
    config={"min_confidence": 0.5},
    liquidity_engine_v5=engine,  # Optional unified engine
)
```

### Methods

#### suggest()

Generate a suggestion from pipeline result.

```python
suggestion = agent.suggest(pipeline_result, timestamp)
```

#### get_confluence_analysis()

Get detailed PENTA confluence analysis.

```python
analysis = agent.get_confluence_analysis(symbol, timestamp)
```

---

## Configuration

Use centralized configuration for all components.

### Import

```python
from config.gnosis_config_v2 import (
    GnosisConfigV2,
    TradingMode,
    RiskLevel,
    get_config,
    set_config,
)
```

### Usage

```python
# Default config
config = GnosisConfigV2()

# Risk-level preset
config = GnosisConfigV2.for_risk_level(RiskLevel.CONSERVATIVE)

# Trading mode preset
config = GnosisConfigV2.for_trading_mode(TradingMode.BACKTEST)

# Set global config
set_config(config)

# Get global config
current_config = get_config()
```

### Configuration Structure

```python
config = GnosisConfigV2(
    trading_mode=TradingMode.ALPHA_SIGNALS,
    risk_level=RiskLevel.MODERATE,
    engines=EngineConfig(...),
    agents=AgentConfig(...),
    composer=ComposerConfig(...),
    trade_agent=TradeAgentConfig(...),
    monitor=MonitorConfig(...),
    backtest=BacktestConfig(...),
)
```

---

## PENTA Confluence Levels

| Level  | Methodologies Aligned | Confidence Bonus |
|--------|----------------------|------------------|
| PENTA  | 5 (All)              | +30%             |
| QUAD   | 4                    | +25%             |
| TRIPLE | 3                    | +15%             |
| DUAL   | 2                    | +5%              |

---

## Quick Start Example

```python
from agents.composer.composer_agent_v4 import ComposerAgentV4
from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
from agents.monitoring import AlphaMonitor

# Setup
composer = ComposerAgentV4()
trade_agent = AlphaTradeAgentV2(config={})
monitor = AlphaMonitor({})

# Generate consensus
output = composer.compose(
    hedge_signal={"direction": "bullish", "confidence": 0.8},
    sentiment_signal={"direction": "bullish", "confidence": 0.75},
    liquidity_signal={"direction": "bullish", "confidence": 0.7},
    penta_confluence="QUAD",
)

# Generate signal
signal = trade_agent.process_composer_output(output, "AAPL", 230.0)

# Track signal
monitor.update(signal={
    "symbol": signal.symbol,
    "signal_type": signal.direction,
    "confidence": signal.confidence,
})

# Output signal for user
print(signal.to_robinhood_format())
```

---

## Version History

- **V2.0.0** - Complete architecture refactoring with PENTA methodology
- **V1.0.0** - Initial release
