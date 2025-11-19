# Super Gnosis Quick Start

This guide walks through installing the Super Gnosis / DHPE v3 skeleton, running the pipeline, and exploring the core components.

## 1. Install Dependencies

```bash
pip install -e .[dev]
```

## 2. Run a Single Pipeline Pass

```bash
python main.py run-once --symbol SPY
```

The CLI uses stub adapters that generate deterministic synthetic data for options, market, and news feeds. Replace them with production adapters by implementing the protocols in `engines/inputs/` and passing them to `build_pipeline`.

## 3. Project Map

- `schemas/core_schemas.py` – Canonical Pydantic models shared across engines and agents.
- `engines/*` – Hedge, Liquidity, Sentiment, and Elasticity engines plus orchestration.
- `agents/*` – Primary agents and the composer that produce `Suggestion` objects.
- `trade/trade_agent_v1.py` – Converts composed suggestions into executable `TradeIdea`s.
- `ledger/` – JSONL ledger store and metrics utilities.
- `feedback/feedback_engine.py` – Example feedback loop returning metric-based adjustments.
- `models/` – Feature builder and lookahead model placeholders for ML driven forecasts.
- `backtest/runner.py` – Replays the pipeline over a time window for quick simulations.
- `scripts/` – Usage and verification helpers.

## 4. Configuration

The default configuration lives at `config/config.yaml`. Load it programmatically:

```python
from config import load_config

app_config = load_config()
print(app_config.engines.hedge.gamma_squeeze_threshold)
```

## 5. Tests

Run the bundled smoke and engine tests to confirm everything is wired up:

```bash
pytest
```

## Next Steps

1. Swap stub adapters for real data sources (broker APIs, market data vendors, news feeds).
2. Enrich each engine with your proprietary analytics while preserving the output schema contract.
3. Extend the trade agent with broker integration or advanced strategy playbooks.
4. Feed ledger metrics into the feedback engine to inform dynamic parameter tuning.

The skeleton is intentionally modular—focus on one subsystem at a time and iterate until your production requirements are satisfied.
