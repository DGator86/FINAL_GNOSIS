# Super Gnosis Documentation Index

This document provides a consolidated index of all documentation in the repository.
Many historical documents exist for reference but are not actively maintained.

---

## 📚 Core Documentation (Start Here)

| Document | Purpose | Status |
|----------|---------|--------|
| [README.md](./README.md) | Project overview, quick start, architecture | ✅ Active |
| [QUICKSTART.md](./QUICKSTART.md) | Getting started guide | ✅ Active |
| [ARCHITECTURE_OVERVIEW.md](./ARCHITECTURE_OVERVIEW.md) | System architecture, component map | ✅ Active |
| [DEV_GUIDE.md](./DEV_GUIDE.md) | Developer guide, extending the system | ✅ Active |

---

## 🔧 Configuration & Setup

| Document | Purpose | Status |
|----------|---------|--------|
| [ALPACA_SETUP.md](./ALPACA_SETUP.md) | Alpaca broker configuration | ✅ Active |
| [.env.example](./.env.example) | Environment variable template | ✅ Active |
| [CLOUD_DEPLOYMENT.md](./CLOUD_DEPLOYMENT.md) | Cloud deployment guide | ✅ Active |

---

## 📊 Feature Documentation

| Document | Purpose | Status |
|----------|---------|--------|
| [DASHBOARD_GUIDE.md](./DASHBOARD_GUIDE.md) | Streamlit dashboard usage | ✅ Active |
| [LIVE_TRADING.md](./LIVE_TRADING.md) | Live trading configuration | ✅ Active |
| [MEMORY_SYSTEM.md](./MEMORY_SYSTEM.md) | Memory/state management | ✅ Active |
| [OPTIONS_STRATEGY_BOOK.md](./OPTIONS_STRATEGY_BOOK.md) | Options strategies reference | ✅ Active |

---

## 🔬 Technical Deep-Dives

| Document | Purpose | Status |
|----------|---------|--------|
| [TRADE_DECISION_ML_PIPELINE.md](./TRADE_DECISION_ML_PIPELINE.md) | ML pipeline architecture | ✅ Active |
| [ML_FEATURE_MATRIX.md](./ML_FEATURE_MATRIX.md) | Feature engineering details | ✅ Active |
| [DATA_REQUIREMENTS.md](./DATA_REQUIREMENTS.md) | Data sources and requirements | ✅ Active |

---

## 📜 Historical / Reference Documents

These documents capture historical development decisions and may be outdated.
They are retained for context but should not be relied upon for current behavior.

### Implementation Status (Historical)
- `IMPLEMENTATION_STATUS.md` - Historical implementation tracking
- `IMPLEMENTATION_TASKS.md` - Historical task tracking
- `COMPLETE_INTEGRATION_SUMMARY.md` - Integration milestone notes
- `MIGRATION_COMPLETE.md` - Migration notes
- `FINAL_SUMMARY.md` - Historical summary

### Integration Notes (Historical)
- `ALPACA_INTEGRATION.md` - Alpaca integration notes
- `UNUSUAL_WHALES_INTEGRATION.md` - UW integration notes
- `MEMORY_INTEGRATION_GUIDE.md` - Memory system integration
- `MEMORY_INTEGRATION_PLAN.md` - Memory planning doc

### Status Reports (Historical)
- `SYSTEM_STATUS.md` - System status snapshot
- `TRADING_SYSTEM_STATUS.md` - Trading status snapshot
- `PUBLIC_API_FINAL_STATUS.md` - API status
- `ROADMAP_EXECUTION_STATUS.md` - Roadmap tracking

### Fix/Debug Notes (Historical)
- `UNUSUAL_WHALES_FIX_SUMMARY.md` - Bug fix notes
- `CI_WORKFLOW_MANUAL_FIX.md` - CI fix notes
- `MERGE_CONFLICT_RESOLUTION.md` - Merge notes
- `REFACTORING_NOTES.md` - Refactoring notes

---

## 🏗️ Component Version Guide

The system has multiple versions of some components. Here's which to use:

### Agents (use v1 for production)

| Component | Canonical | Experimental |
|-----------|-----------|--------------|
| Hedge Agent | `HedgeAgentV3` | `HedgeAgentV3Enhanced`, `HedgeAgentV4` |
| Liquidity Agent | `LiquidityAgentV1` | `LiquidityAgentV2`, `LiquidityAgentV3` |
| Sentiment Agent | `SentimentAgentV1` | `SentimentAgentV2`, `SentimentAgentV3` |
| Trade Agent | `TradeAgentV1` | `TradeAgentV2`, `TradeAgentV3` |
| Composer | `ComposerAgentV1` | `ComposerAgentV2` |

### When to use experimental versions:

- **V2 agents**: Multi-timeframe analysis experiments
- **V3 agents**: Advanced features (0DTE scalping, flow conviction, etc.)
- **TradeAgentV2**: Universe-wide portfolio optimization
- **TradeAgentV3**: Multi-timeframe strategy generation

---

## 🛠️ Utilities Reference

### New Utilities (v3.0+)

| Module | Purpose |
|--------|---------|
| `utils/cache.py` | TTL caching, rate limiting |
| `utils/circuit_breaker.py` | Circuit breaker pattern |
| `execution/broker_adapters/resilient_adapter.py` | Resilient broker wrapper |

### Adapter Factory

```python
from engines.inputs.adapter_factory import (
    create_market_data_adapter,      # Market data with fallback
    create_options_adapter,          # Options data with fallback
    create_broker_adapter,           # Broker with fallback
    create_cached_options_adapter,   # Options with TTL cache
    create_cached_market_data_adapter,  # Market data with TTL cache
)

from execution.broker_adapters.resilient_adapter import (
    ResilientBrokerAdapter,          # Circuit breaker + rate limiting
    create_resilient_broker_adapter, # Factory function
)
```

---

## 📁 Directory Structure

```
super-gnosis/
├── agents/           # Trading agents (hedge, liquidity, sentiment)
├── config/           # Configuration models and loaders
├── engines/          # Analysis engines (hedge, liquidity, sentiment, elasticity)
│   ├── inputs/       # Data adapters (Alpaca, Unusual Whales, stubs)
│   ├── ml/           # ML enhancement engines
│   └── orchestration/# Pipeline runner
├── execution/        # Order execution and broker adapters
├── feedback/         # Adaptation and tracking agents
├── gnosis/           # Extended trading features
├── ledger/           # Trade ledger and metrics
├── models/           # ML models (LSTM, transformers, etc.)
├── schemas/          # Pydantic data models
├── trade/            # Trade agents and execution mapping
├── utils/            # Utilities (caching, circuit breakers)
└── tests/            # Test suite
```

---

## 🔗 Quick Links

- **Run pipeline**: `python main.py run-once --symbol SPY`
- **Start dashboard**: `streamlit run dashboard.py`
- **Run tests**: `pytest tests/`
- **Scan opportunities**: `python main.py scan-opportunities --top 25`

---

*Last updated: 2024*
