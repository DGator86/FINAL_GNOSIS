# Super Gnosis / DHPE v3

Super Gnosis is a modular multi-engine, multi-agent trading research framework. The project aligns with the Dealer Hedge Positioning Engine (DHPE) v3 architecture and provides a production-grade implementation with advanced dealer flow analytics.

## 🎯 **Project Status: Production Ready**

| Component | Status | Tests |
|-----------|--------|-------|
| Trading Engines | ✅ Complete | 200+ |
| ML Integration | ✅ Complete | 71 |
| Multi-Leg Options | ✅ Complete | 36 |
| Safety Controls | ✅ Complete | 36 |
| Model Registry | ✅ Complete | 40 |
| **Total Tests** | **✅ All Passing** | **920+** |

## 🚀 **Recent Updates (2025-12-21)**

### ML Model Registry
- Version control for LSTM and other ML models
- A/B testing support and rollback capabilities
- Performance tracking per version
- Promotion workflows (dev → staging → production)

### Trading Safety
- Circuit breakers and position limits
- Daily loss limits and max drawdown protection
- Comprehensive pre-trade validation

### Multi-Leg Options
- Atomic execution via Alpaca mleg order class
- Support for spreads, iron condors, butterflies
- Position intent tracking

### Social Media Sentiment
- Twitter/X integration with sentiment analysis
- Reddit (WSB, stocks, options) monitoring
- Engagement-weighted scoring

See [`TODO_TRACKER.md`](./TODO_TRACKER.md) for detailed changelog.

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

For complete documentation, see [`HEDGE_ENGINE_V3_IMPLEMENTATION.md`](./HEDGE_ENGINE_V3_IMPLEMENTATION.md).

---

## Directory Structure

The canonical directory tree implemented here:

```
V2---Gnosis/
├── README.md
├── QUICKSTART.md
├── QUICK_REFERENCE.md
├── FINAL_SUMMARY.md
├── IMPLEMENTATION_COMPLETE.md
├── INDEX.md
├── DELIVERABLES.txt
├── pyproject.toml
├── requirements.txt
├── main.py
├── config/
│   ├── config.yaml
│   ├── config_models.py
│   └── loader.py
├── schemas/
│   ├── __init__.py
│   └── core_schemas.py
├── engines/
│   ├── __init__.py
│   ├── base.py
│   ├── inputs/
│   │   ├── __init__.py
│   │   ├── market_data_adapter.py
│   │   ├── news_adapter.py
│   │   ├── options_chain_adapter.py
│   │   └── stub_adapters.py
│   ├── hedge/
│   │   ├── __init__.py
│   │   └── hedge_engine_v3.py
│   ├── liquidity/
│   │   ├── __init__.py
│   │   └── liquidity_engine_v1.py
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── processors.py
│   │   └── sentiment_engine_v1.py
│   ├── elasticity/
│   │   ├── __init__.py
│   │   └── elasticity_engine_v1.py
│   └── orchestration/
│       ├── __init__.py
│       └── pipeline_runner.py
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── hedge_agent_v3.py
│   ├── liquidity_agent_v1.py
│   └── sentiment_agent_v1.py
├── trade/
│   ├── __init__.py
│   └── trade_agent_v1.py
├── models/
│   ├── __init__.py
│   ├── feature_builder.py
│   └── lookahead_model.py
├── ledger/
│   ├── __init__.py
│   ├── ledger_store.py
│   └── ledger_metrics.py
├── feedback/
│   ├── __init__.py
│   └── feedback_engine.py
├── backtest/
│   ├── __init__.py
│   └── runner.py
├── execution/
│   ├── __init__.py
│   ├── broker_adapter.py
│   └── order_simulator.py
├── ui/
│   ├── __init__.py
│   └── dashboard.py
├── scripts/
│   ├── example_usage.py
│   └── verify_integration.py
├── tests/
│   ├── __init__.py
│   ├── test_elasticity_engine_v1.py
│   ├── test_hedge_engine_v3.py
│   ├── test_liquidity_engine_v1.py
│   ├── test_pipeline_smoke.py
│   ├── test_sentiment_engine_v1.py
│   └── test_schemas.py
└── data/
    └── ledger.jsonl (created at runtime)
```

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

See [`DASHBOARD_GUIDE.md`](./DASHBOARD_GUIDE.md) for complete documentation.

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

The adapter factory automatically falls back to stub data if APIs are unavailable, ensuring the system always runs.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_model_registry.py -v  # ML model registry
pytest tests/test_trading_safety.py -v  # Safety controls
pytest tests/test_paper_trading_integration.py -v  # Integration tests
pytest tests/test_alpaca_options_adapter.py -v  # Multi-leg options
```

### Test Coverage Summary
- **909+ tests passing**
- ML Integration: 71 tests
- Trading Safety: 36 tests
- Multi-Leg Options: 36 tests
- Scanner Integration: 20 tests
- Paper Trading Integration: 30 tests

## Extending the Skeleton

- Implement real adapters under `engines/inputs/` that conform to the provided protocols.
- Replace analytics inside each engine with your production models while keeping output schemas intact.
- Extend the trade agent with richer strategy selection or broker integration via `execution/broker_adapter.py`.
- Plug an ML model into `models/lookahead_model.py` and feed predictions into agents/composer.
- Integrate UI requirements inside `ui/dashboard.py` and expose metrics in real time.

The repository serves as the authoritative reference for Super Gnosis / DHPE v3. Update both the documentation and implementation together to keep them in sync.

---

## 📚 Documentation Index

| Document | Description |
|----------|-------------|
| [`README.md`](./README.md) | This file - project overview |
| [`QUICKSTART.md`](./QUICKSTART.md) | Getting started guide |
| [`TODO_TRACKER.md`](./TODO_TRACKER.md) | TODO items and changelog |
| [`ARCHITECTURE_OVERVIEW.md`](./ARCHITECTURE_OVERVIEW.md) | System architecture |
| [`ALPACA_INTEGRATION.md`](./ALPACA_INTEGRATION.md) | Alpaca API integration |
| [`OPTIONS_STRATEGY_BOOK.md`](./OPTIONS_STRATEGY_BOOK.md) | Options trading strategies |
| [`ML_FEATURE_MATRIX.md`](./ML_FEATURE_MATRIX.md) | ML features documentation |
| [`OPERATIONS_RUNBOOK.md`](./OPERATIONS_RUNBOOK.md) | Production operations guide |

---

## 🔧 CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

```yaml
# Triggered on push to main/master and PRs
- Linting (Black, isort, flake8)
- Unit tests (Python 3.10, 3.11)
- Integration tests
- Security scanning (Bandit, Safety)
- Build verification
- Coverage reporting (Codecov)
```

See [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) for details.

---

## 📦 Key Components

### ML Components (`ml/`)
- `model_registry.py` - Version control for ML models
- `hyperparameter_manager.py` - Hyperparameter optimization
- `pipeline_integration.py` - Full engine integration
- `adaptive_pipeline.py` - ML-driven trading decisions

### Trading Components (`trade/`)
- `ml_trading_engine.py` - ML-driven trade execution
- `trading_safety.py` - Circuit breakers and safety controls

### Market Utilities (`gnosis/`)
- `market_utils.py` - Price fetching, P&L, market hours
- `scanner/__init__.py` - Multi-timeframe opportunity scanner

### Data Providers (`data/`)
- `price_provider.py` - Unified price data with Alpaca/yfinance fallback

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure all tests pass before submitting a PR.

---

## 📄 License

This project is proprietary. All rights reserved.

---

**Last Updated:** 2025-12-21
