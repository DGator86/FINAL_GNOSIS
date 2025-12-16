# Super Gnosis – Quick Reference

## 🚀 Run It

```bash
pip install -e .[dev]
python main.py run-once --symbol SPY
```

## 📦 Key Modules

| File | Purpose |
| --- | --- |
| `schemas/core_schemas.py` | Canonical Pydantic data contracts |
| `engines/hedge/hedge_engine_v3.py` | Dealer hedge pressure analytics |
| `engines/liquidity/liquidity_engine_v1.py` | Liquidity and order flow metrics |
| `engines/sentiment/sentiment_engine_v1.py` | News / flow / technical sentiment fusion |
| `engines/elasticity/elasticity_engine_v1.py` | Price elasticity & energy model |
| `agents/*.py` | Primary agents and composer |
| `trade/trade_agent_v1.py` | Suggestion → TradeIdea translation |
| `engines/orchestration/pipeline_runner.py` | Orchestrates one full pipeline pass |
| `ledger/ledger_store.py` | JSONL ledger writer |
| `feedback/feedback_engine.py` | Metrics driven feedback adjustments |
| `backtest/runner.py` | Simple event-driven backtester |

## ⚙️ Configuration Snippets (`config/config.yaml`)

```yaml
engines:
  hedge:
    gamma_squeeze_threshold: 1_500_000
  liquidity:
    lookback: 30
  sentiment:
    bullish_threshold: 0.3
  elasticity:
    baseline_move_cost: 1.0
agents:
  trade:
    max_capital_per_trade: 10000
tracking:
  ledger_path: data/ledger.jsonl
```

Load config in code:

```python
from config import load_config
cfg = load_config()
print(cfg.agents.trade.max_capital_per_trade)
```

## 🧪 Testing

```bash
pytest
```

## 🔁 Backtesting

```python
from datetime import datetime, timedelta
from backtest.runner import BacktestRunner
from main import build_pipeline
from config import load_config

config = load_config()
factory = lambda symbol: build_pipeline(symbol, config)
runner = BacktestRunner(factory, data_source=None, config={"step": timedelta(days=1)})
metrics = runner.run_backtest("SPY", datetime(2024, 1, 1), datetime(2024, 1, 5))
print(metrics)
```

## 🛠 Customisation Hooks

- Implement adapters under `engines/inputs/` to replace stub data sources.
- Extend engines while keeping `EngineOutput` fields intact.
- Adjust agent behaviour via the config models (`config/config_models.py`).
- Add broker execution by implementing `execution/broker_adapter.BrokerAdapter`.

## 📂 Logs & Ledger

Ledger entries append to `data/ledger.jsonl`. Stream them using `LedgerStore.stream()` or `cat data/ledger.jsonl`.

Stay within these contracts to ensure compatibility with future Super Gnosis modules.
