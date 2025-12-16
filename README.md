# Super Gnosis / DHPE v3

Super Gnosis is a modular multi-engine, multi-agent trading research framework. The project aligns with the Dealer Hedge Positioning Engine (DHPE) v3 architecture and provides a production-grade implementation with advanced dealer flow analytics.

## 🎉 **NEW: Hedge Engine v3.0 Fully Implemented**

The **Hedge Engine v3.0** is now production-ready with full elasticity theory, movement energy calculation, and multi-dimensional regime detection. See [`HEDGE_ENGINE_V3_IMPLEMENTATION.md`](./HEDGE_ENGINE_V3_IMPLEMENTATION.md) for complete documentation.

## Architecture Overview

- **Schemas** – Canonical Pydantic models describing engine outputs, agent suggestions, trades, and ledger entries.
- **Engines** – Hedge, Liquidity, Sentiment, and Elasticity analytics with a shared `Engine` protocol.
  - **Hedge Engine v3.0** ⭐ – **FULLY IMPLEMENTED** with 8 modular processors, elasticity theory, movement energy, and multi-dimensional regime detection
- **Agents** – Primary agents per engine, a composer for consensus, and a trade agent translating policy into trade ideas.
  - **Hedge Agent v3.0** – Energy-aware interpretation using elasticity and movement_energy
- **Orchestration** – `PipelineRunner` coordinates engines → snapshot → agents → ledger.
- **Ledger & Feedback** – JSONL ledger store with metrics and configuration feedback hooks.
- **Models** – Feature builder and lookahead model placeholders for ML driven signals.
- **Execution** – Broker adapter protocol and order simulator stub.
- **Backtesting** – Lightweight runner that replays a pipeline across a historical window.
- **CLI & UI** – Typer CLI entry point (`main.py`) plus a dashboard stub.

## Hedge Engine v3.0 Highlights

The **Hedge Engine v3.0** represents the first production-grade implementation in the Super Gnosis framework:

### Core Features
- ✅ **Modular Processor Architecture**: 8 specialized processors (dealer sign, gamma/vanna/charm fields, elasticity, movement energy, regime detection, MTF fusion)
- ✅ **Elasticity Theory**: Market stiffness calculated from Greek fields, OI distribution, and liquidity friction
- ✅ **Movement Energy**: Quantifiable "cost" to move price = Pressure / Elasticity
- ✅ **Multi-Dimensional Regime Detection**: 6+ regime dimensions with jump-diffusion handling
- ✅ **SWOT Fixes Integrated**: Vanna shock absorber, jump-diffusion term, adaptive smoothing
- ✅ **Energy-Aware Agent**: Hedge agent uses elasticity/energy for directional bias and confidence
- ✅ **Comprehensive Tests**: 18 processor + integration tests (all passing)

### Key Outputs
```python
{
    "elasticity": float,              # Market stiffness (always > 0)
    "movement_energy": float,         # Energy required to move price
    "energy_asymmetry": float,        # Directional bias (up/down)
    "pressure_up/down/net": float,    # Dealer hedge pressure vectors
    "gamma/vanna/charm_pressure": float,
    "dealer_gamma_sign": float,       # Stabilizing/destabilizing
}
```

For complete documentation, see [`docs/HEDGE_ENGINE_V3_IMPLEMENTATION.md`](./docs/HEDGE_ENGINE_V3_IMPLEMENTATION.md).

---

## Directory Structure

The refactored directory tree emphasizes modularity and separation of concerns:

```
FINAL_GNOSIS/
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── config.yaml
│   ├── config_models.py
│   └── loader.py
├── schemas/
│   ├── __init__.py
│   └── core_schemas.py
├── adapters/                          # ⭐ Shared data input layer
│   ├── __init__.py
│   ├── adapter_factory.py
│   ├── market_data_adapter.py
│   ├── news_adapter.py
│   ├── options_chain_adapter.py
│   ├── stub_adapters.py
│   ├── alpaca_market_adapter.py
│   ├── unusual_whales_adapter.py
│   └── massive_*.py                   # Massive.io adapters
├── engines/
│   ├── __init__.py
│   ├── base.py
│   ├── hedge/
│   │   ├── __init__.py
│   │   └── hedge_engine_v3.py
│   ├── liquidity/
│   │   ├── __init__.py
│   │   └── liquidity_engine_v*.py
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── processors.py
│   │   └── sentiment_engine_v*.py
│   ├── elasticity/
│   │   ├── __init__.py
│   │   └── elasticity_engine_v1.py
│   └── ml/
│       ├── __init__.py
│       ├── forecasting.py
│       └── enhancement_engine.py
├── core/                              # ⭐ Control plane orchestration
│   ├── __init__.py
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── pipeline_runner.py
│   │   └── unified_orchestrator.py
│   ├── feedback/
│   │   ├── __init__.py
│   │   └── adaptation_agent.py
│   └── ledger/
│       ├── __init__.py
│       └── ledger_store.py
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── hedge_agent_v3.py
│   ├── liquidity_agent_v1.py
│   └── sentiment_agent_v1.py
├── trade/
│   ├── __init__.py
│   └── trade_agent_v*.py
├── models/                            # ⭐ Enhanced ML structure
│   ├── __init__.py
│   ├── base.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_builder.py
│   ├── predictors/                    # All prediction models
│   │   ├── __init__.py
│   │   ├── lookahead_model.py
│   │   ├── lstm_lookahead.py
│   │   ├── time_series/
│   │   │   └── lstm_forecaster.py
│   │   ├── ensemble/
│   │   │   └── xgboost_model.py
│   │   └── rl_agents/
│   │       └── dqn_agent.py
│   └── trainers/                      # Training utilities
│       ├── __init__.py
│       └── hyperparameter_optimizer.py
├── backtest/
│   ├── __init__.py
│   └── runner.py
├── execution/
│   ├── __init__.py
│   ├── broker_adapters/
│   │   └── alpaca_*.py
│   └── order_simulator.py
├── interfaces/                        # ⭐ User-facing layer
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── commands/
│   │   └── pipeline_builder.py
│   └── ui/
│       ├── __init__.py
│       └── dashboard.py
├── examples/                          # ⭐ Merged scripts/examples
│   ├── __init__.py
│   ├── run_gnosis_backtest.py
│   ├── terminal_dashboard.py
│   └── *.py                          # Utility scripts
├── tests/
│   ├── __init__.py
│   └── test_*.py
├── docs/                              # ⭐ Consolidated documentation
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── HEDGE_ENGINE_V3_IMPLEMENTATION.md
│   ├── DASHBOARD_GUIDE.md
│   ├── guides/
│   ├── implementation/
│   └── *.md                          # All other documentation
└── data/
    └── ledger.jsonl (created at runtime)
```

### Key Structural Improvements

1. **`adapters/`** - Top-level shared data adapters serving all components (engines, agents, models)
2. **`core/`** - Centralized control plane with orchestration, feedback, and ledger
3. **`models/predictors/`** - Unified prediction models (LSTM, XGBoost, RL agents)
4. **`models/trainers/`** - Training and hyperparameter optimization utilities
5. **`interfaces/`** - Clean separation of CLI and UI from core logic
6. **`examples/`** - Consolidated utility scripts and examples
7. **`docs/`** - All documentation in one place (except main README)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DGator86/FINAL_GNOSIS.git
cd FINAL_GNOSIS

# Install dependencies
pip install -r requirements.txt
pip install -e .[dev]
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API credentials
# - ALPACA_API_KEY and ALPACA_SECRET_KEY (required for trading)
# - UNUSUAL_WHALES_API_KEY (optional for options data)
nano .env
```

### Run the System

**Option 1: Enhanced Dashboard (Recommended)** 🎯
```bash
# Start the live trading dashboard
./start_dashboard.sh
# OR
streamlit run dashboard.py
```
Opens a beautiful web interface with:
- Real-time position tracking
- Live P&L monitoring
- Engine analytics visualization
- Agent suggestions display
- Trade history viewer

**Option 2: Command Line Interface**
```bash
# Single pipeline run
python main.py run-once --symbol SPY

# Live trading loop
python main.py live-loop --symbol SPY

# Scan for opportunities
python main.py scan-opportunities --top 25

# Multi-symbol autonomous trading
python main.py multi-symbol-loop --top 5
```

**Option 3: API Connection Test**
```bash
# Verify your API credentials
python test_api_connections.py
```

**Option 4: Live Demo**
```bash
# Beautiful terminal output
python demo_live_trading.py
```

### 📊 Dashboard Features

The enhanced dashboard provides:
- 💰 **Account Overview**: Portfolio value, cash, buying power, daily P&L
- 💼 **Position Tracking**: All open positions with live P&L
- 📈 **Live Analytics**: Hedge Engine v3.0 metrics, elasticity, movement energy
- 🤖 **Agent Intelligence**: Individual suggestions and consensus
- 📜 **Trade History**: Historical pipeline executions
- ⚙️ **Engine Metrics**: Performance tracking over time

See [`docs/DASHBOARD_GUIDE.md`](./docs/DASHBOARD_GUIDE.md) for complete documentation.

---

## 🔌 Live API Integrations

The system connects to real trading APIs:

### Alpaca Markets (Required)
- Paper and live trading support
- Real-time market data
- Position and order management
- Account tracking

### Unusual Whales (Optional)
- Options chain data with Greeks
- Unusual activity alerts
- Options flow analysis
- Implied volatility tracking

The adapter factory (in `adapters/`) automatically falls back to stub data if APIs are unavailable, ensuring the system always runs.

## Testing

```bash
pytest
```

## Extending the Framework

- **Adapters**: Implement new data sources in `adapters/` that conform to the provided protocols (MarketDataAdapter, OptionsChainAdapter, NewsAdapter).
- **Engines**: Add analytics in `engines/` while maintaining output schemas defined in `schemas/`.
- **Models**: Add ML models in `models/predictors/` and training scripts in `models/trainers/`.
- **Orchestration**: Extend pipeline logic in `core/orchestration/` for multi-symbol or advanced workflows.
- **Trade Logic**: Enhance strategy selection in `trade/` with broker integration via `execution/broker_adapters/`.
- **UI**: Build dashboards in `interfaces/ui/` and CLI commands in `interfaces/cli/commands/`.
- **Examples**: Add utility scripts to `examples/` for backtesting, analysis, and demonstrations.

The repository serves as the authoritative reference for Super Gnosis / DHPE v3. Update both the documentation (in `docs/`) and implementation together to keep them in sync.
