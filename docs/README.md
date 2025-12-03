# Super Gnosis Documentation

Welcome to the Super Gnosis / DHPE v3 documentation.

## Quick Links

- **[Quick Reference](../QUICK_REFERENCE.md)** - Command cheat sheet
- **[Quickstart Guide](../QUICKSTART.md)** - Get started in 5 minutes
- **[Architecture Overview](../ARCHITECTURE_OVERVIEW.md)** - System design
- **[Operations Runbook](../OPERATIONS_RUNBOOK.md)** - Production operations

## Documentation Structure

### 📖 Guides
- [Development Guide](../DEV_GUIDE.md) - Development workflow and best practices
- [Dashboard Guide](../DASHBOARD_GUIDE.md) - Using the Streamlit dashboards
- [Live Trading Guide](../LIVE_TRADING.md) - Production trading setup
- [Multi-Leg Options Guide](MULTI_LEG_OPTIONS_GUIDE.md) - Advanced options strategies

### 🏗️ Implementation Details
- [Hedge Engine v3](implementation/HEDGE_ENGINE_V3.md) - Elasticity and dealer positioning
- [Sentiment Engine v3](implementation/SENTIMENT_ENGINE_V3.md) - Multi-source sentiment fusion
- [Liquidity Engine v3](implementation/LIQUIDITY_ENGINE_V3.md) - Market depth analysis
- [Trade Execution v3](implementation/TRADE_EXECUTION_V3.md) - Order management
- [LSTM Forecaster](../ML_FEATURE_MATRIX.md) - ML-powered predictions

### 🔌 Integration Guides
- [Alpaca Setup](../ALPACA_SETUP.md) - Broker integration
- [Unusual Whales Integration](../UNUSUAL_WHALES_INTEGRATION.md) - Options data
- [Memory System](../MEMORY_SYSTEM.md) - Semantic memory and learning

### 📚 API Reference
- [Core Schemas](api/SCHEMAS.md) - Pydantic data models
- [Engine API](api/ENGINES.md) - Engine interfaces
- [Agent API](api/AGENTS.md) - Agent interfaces

### 📜 History
Session logs and progress tracking have been moved to `docs/history/`:
- Session summaries
- Implementation milestones
- Migration notes

## Getting Started

### Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Run Your First Analysis
```bash
python main.py run-once --symbol SPY
```

### Launch Dashboard
```bash
streamlit run dashboard.py
```

### Run Tests
```bash
pytest tests/
```

## System Overview

Super Gnosis is an institutional-grade options trading system powered by:

1. **4 Analytical Engines**
   - Hedge Engine: Dealer positioning & elasticity
   - Liquidity Engine: Market depth & spreads
   - Sentiment Engine: News, flow, and technical sentiment
   - Elasticity Engine: Volatility regimes

2. **Multi-Agent Decision System**
   - Primary agents process engine outputs
   - Composer fuses signals with weighted consensus
   - Trade agent generates executable strategies

3. **Risk Management**
   - Position size limits (configurable % of portfolio)
   - Daily loss circuit breakers
   - Stop-loss and take-profit automation

4. **Machine Learning**
   - LSTM forecaster with multi-head attention
   - Anomaly detection
   - Regime similarity search
   - Reinforcement learning evaluation

## Architecture

```
Data Sources → Engines → Agents → Composer → Trade Execution
     ↓           ↓         ↓          ↓             ↓
  Adapters   Snapshots  Signals  Consensus      Orders
                                                   ↓
                                              Ledger/Tracking
```

See [ARCHITECTURE_OVERVIEW.md](../ARCHITECTURE_OVERVIEW.md) for details.

## Configuration

Main configuration: `config/config.yaml`

Key settings:
- Engine parameters (lookback windows, thresholds)
- Agent weights (composer consensus: 40% hedge, 40% sentiment, 20% liquidity)
- Trading rules (position sizing, risk limits)
- Scanner mode (dynamic top 25)

## Support

- **Issues**: [GitHub Issues](https://github.com/DGator86/FINAL_GNOSIS/issues)
- **Research Log**: [docs/research_log.md](research_log.md)

## License

Proprietary - All Rights Reserved
