# Super Gnosis Documentation Index

Use this index to navigate the canonical Super Gnosis / DHPE v3 docs.

## Getting Started
- **QUICK_REFERENCE.md** – Command cheatsheet and configuration snippets.
- **QUICKSTART.md** – Installation and first pipeline run.
- **README.md** – Full architecture overview, quick start, and extension guidance.

## Implementation Summaries
- **FINAL_SUMMARY.md** – High-level status and file map.
- **IMPLEMENTATION_COMPLETE.md** – Detailed component checklist and next steps.
- **DELIVERABLES.txt** – Concise deliverable breakdown.

## Configuration & Tooling
- **config/config.yaml** – Default runtime configuration.
- **config/config_models.py** – Typed configuration schema.
- **scripts/verify_integration.py** – Minimal integration checklist.
- **scripts/example_usage.py** – Programmatic pipeline example.

## Code Reference
- **schemas/core_schemas.py** – Canonical types used across the system.
- **engines/** – Hedge, Liquidity, Sentiment, Elasticity, inputs, orchestration.
- **agents/** – Primary agents and composer.
- **trade/trade_agent_v1.py** – Suggestion → TradeIdea mapping.
- **ledger/** – Ledger store and metrics utilities.
- **feedback/feedback_engine.py** – Metrics-driven feedback loop.
- **backtest/runner.py** – Simple event-driven backtester.

## Running & Testing
```bash
pip install -e .[dev]
python main.py run-once --symbol SPY
pytest
```

Review docs in the order above to gain context, then dive into the modules most relevant to your integration or extension work.
