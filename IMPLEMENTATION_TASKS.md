# IMPLEMENTATION_TASKS.md

*Final Gnosis – Execution Layer & Repo Fixes*

This document captures the required code changes and additions to stabilize the execution layer, ensure Alpaca trading works end-to-end, and provide tests plus helper scripts for validation. Treat this as the authoritative implementation spec.

---

## 1. Repository Context

* Repo: `https://github.com/DGator86/FINAL_GNOSIS`
* Branch: `main`
* Known good commit with working execution: `582ed494c429e3e4606196321b047c5d00ad9a4e` (message: `feat: Complete execution layer - system can actually trade`)

Core components involved:

* CLI entrypoint: `main.py`
* Trade logic: `trade/trade_agent_v1.py`
* Broker adapter: `execution/broker_adapters/alpaca_adapter.py`
* Orchestration: `engines/orchestration/pipeline_runner.py`
* Base agent utilities: `agents/base_agent.py`
* Execution wrapper: `trade/order_executor.py`
* Watchlist / adaptive logic: `watchlist/*`
* Schemas: `schemas/core_schemas.py`

---

## 2. File-by-File Implementation Tasks

### 2.1 `agents/base_agent.py`

**Issue:** `from __future__ import annotations` must be at the top of the file to avoid `SyntaxError`.

**Required top-of-file layout:**

```python
from __future__ import annotations

"""Lightweight base classes for agent implementations."""
"""Shared agent utilities for advanced trading agents."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import logging
import numpy as np

# ... existing BaseAgent / dataclasses / utilities remain unchanged ...
```

**Tasks:**

* Ensure `from __future__ import annotations` is the first import line.
* Remove any duplicate future imports further down the file.
* Run `pytest` to confirm no syntax errors originate here.

---

### 2.2 `execution/broker_adapters/alpaca_adapter.py`

#### 2.2.1 Fix `place_order` return type

**Issue:** `place_order` returned a UUID object; `OrderResult` expects a `str`.

**Required behavior:**

```python
logger.info(
    f"Order submitted: {side.upper()} {quantity} {symbol} - Order ID: {order.id}"
)

return str(order.id)
```

**Tasks:**

* Update `AlpacaBrokerAdapter.place_order(...)` so successful calls return `str(order.id)`.
* Keep failure paths cleanly logged and return a defined failure (`None` or similar) without throwing.

---

### 2.3 `trade/trade_agent_v1.py`

Key areas: idea gating, fallback price logic, quantity sizing, and broker delegation.

#### 2.3.1 Idea gating logic

* Convert consensus direction to enum; treat missing as `NEUTRAL`.
* Apply a confidence threshold (e.g., `MIN_CONFIDENCE = 0.5`, ideally config-driven).
* If `direction == NEUTRAL` **or** `confidence < MIN_CONFIDENCE`, return no ideas.
* Remove temporary testing hacks like `confidence < 0.1`.

#### 2.3.2 Fallback price logic

Replace any `return 1.0` behavior with:

```python
try:
    bars = self.market_data_adapter.get_recent_bars(symbol, limit=1)
    if bars:
        return float(bars[-1].close)
except Exception:
    logger.debug("Fallback price lookup failed", exc_info=True)

fallback_prices = {
    "SPY": 600.0,
    "QQQ": 500.0,
    "IWM": 230.0,
    "NVDA": 145.0,
    "TSLA": 350.0,
    "AAPL": 230.0,
    "MSFT": 430.0,
    "GOOGL": 175.0,
    "AMZN": 210.0,
    "META": 560.0,
}
return fallback_prices.get(symbol, 100.0)
```

Prefer skipping the trade (with a warning) if no reliable price is available rather than using a nonsense fallback.

#### 2.3.3 Quantity and size calculation

* Treat `TradeIdea.size` as **dollar notional**.
* Compute quantity as `max(1, round(dollars / price, 2))` (or integer shares if fractional not supported).
* If no valid price is available, log and reject the idea instead of executing with a placeholder price.

#### 2.3.4 Execution method

* `execute_trades` should call the broker (or injected `OrderExecutor`) and return `List[OrderResult]` with string `order_id`, symbol, status, and message.
* On exceptions, log and return `OrderResult` entries with `status=REJECTED` and error details.
* Longer term, delegate to `OrderExecutor` once risk logic is finalized.

---

### 2.4 `trade/order_executor.py`

Implement a complete executor that enforces risk controls and routes orders via the broker adapter.

**Spec:**

* Initialize with `broker`, `max_position_size_pct`, and `max_daily_loss_usd`.
* `execute_ideas(ideas, timestamp) -> List[OrderResult]`:
  * Optionally block trading if daily P&L is below `-max_daily_loss_usd` (stub method ok).
  * Fetch account equity and enforce per-trade max notional: `equity * (max_position_size_pct / 100)`.
  * For each idea:
    * If requested size exceeds limit → log warning, return `OrderResult` with `REJECTED`.
    * Otherwise call `broker.place_order(...)` with computed quantity.
    * Append `OrderResult` with `SUBMITTED` (or `REJECTED` on failure/no ID).
  * Catch exceptions per idea and return `OrderResult` with `REJECTED` plus error message.
* Provide `_compute_quantity` helper (can call shared price logic) converting dollar size to order quantity.

Wire `TradeAgentV1` to use `OrderExecutor` when available.

---

### 2.5 `engines/orchestration/pipeline_runner.py`

**Problem:** Watchlist gating previously crashed with `'HedgeSnapshot' object has no attribute 'data'` and was temporarily disabled.

**Tasks:**

* Inspect `HedgeSnapshot` and `watchlist` APIs to fix attribute mismatch.
* Reinstate gating:

```python
if self.watchlist and not self.watchlist.is_symbol_active(self.symbol):
    logger.info("Skipping trade idea generation for %s — not on adaptive watchlist", self.symbol)
    result.trade_ideas = []
else:
    trade_ideas = self.trade_agent.generate_ideas(result, timestamp)
    result.trade_ideas = trade_ideas or []
```

* Ensure trade execution uses `execute_trades` when `auto_execute` is enabled:

```python
if self.auto_execute and result.trade_ideas:
    result.order_results = self.trade_agent.execute_trades(result.trade_ideas, timestamp)
```

* Add tests so inactive symbols skip idea generation and active symbols call the trade agent once per tick.

---

## 3. Test Skeletons to Implement

Codex should add/fix pytest coverage:

### 3.1 `tests/test_alpaca_adapter.py`

```python
import uuid
from unittest.mock import MagicMock, patch

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter

def test_place_order_returns_string_id(monkeypatch):
    adapter = AlpacaBrokerAdapter(api_key="TEST", secret_key="TEST", base_url="https://paper-api.alpaca.markets")

    mock_order = MagicMock()
    mock_order.id = uuid.uuid4()

    with patch.object(adapter.trading_client, "submit_order", return_value=mock_order):
        order_id = adapter.place_order(symbol="SPY", quantity=1, side="buy", order_type="market", time_in_force="day")

    assert isinstance(order_id, str)
    assert order_id == str(mock_order.id)
```

### 3.2 `tests/test_order_executor.py`

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock

from trade.order_executor import OrderExecutor
from schemas.core_schemas import TradeIdea, DirectionEnum, StrategyType, OrderStatus

def make_idea(symbol="SPY", size=500.0):
    return TradeIdea(timestamp=datetime.now(timezone.utc), symbol=symbol, strategy_type=StrategyType.DIRECTIONAL, direction=DirectionEnum.LONG, confidence=0.8, size=size, reasoning="test")

def test_executor_calls_broker_for_valid_idea():
    broker = MagicMock()
    broker.place_order.return_value = "ORDER123"
    broker.get_account.return_value = MagicMock(equity=30000.0)

    executor = OrderExecutor(broker=broker, max_position_size_pct=2.0, max_daily_loss_usd=5000.0)

    now = datetime.now(timezone.utc)
    results = executor.execute_ideas([make_idea(size=500.0)], now)

    broker.place_order.assert_called_once()
    assert len(results) == 1
    assert results[0].order_id == "ORDER123"
    assert results[0].symbol == "SPY"
    assert results[0].status in {OrderStatus.SUBMITTED, OrderStatus.FILLED}

def test_executor_blocks_when_size_exceeds_max_pct():
    broker = MagicMock()
    broker.place_order.return_value = "ORDER123"
    broker.get_account.return_value = MagicMock(equity=10000.0)  # 2% => $200 max

    executor = OrderExecutor(broker=broker, max_position_size_pct=2.0, max_daily_loss_usd=5000.0)

    now = datetime.now(timezone.utc)
    results = executor.execute_ideas([make_idea(size=1000.0)], now)

    broker.place_order.assert_not_called()
    assert len(results) == 1
    assert results[0].status == OrderStatus.REJECTED
```

### 3.3 `tests/test_trade_agent_v1.py`

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock

from trade.trade_agent_v1 import TradeAgentV1
from schemas.core_schemas import PipelineResult, TradeIdea, DirectionEnum, StrategyType

def make_pipeline_result(symbol="SPY", direction="long", confidence=0.8):
    return PipelineResult(symbol=symbol, timestamp=datetime.now(timezone.utc), consensus={"direction": direction, "confidence": confidence})

def test_generate_ideas_blocks_neutral_direction():
    agent = TradeAgentV1(MagicMock(), MagicMock(), {"max_position_size": 500.0, "risk_per_trade": 0.02}, broker=None)
    neutral_result = make_pipeline_result(direction="neutral", confidence=0.9)
    ideas = agent.generate_ideas(neutral_result, datetime.now(timezone.utc))
    assert ideas == []

def test_generate_ideas_allows_high_confidence_directional():
    agent = TradeAgentV1(MagicMock(), MagicMock(), {"max_position_size": 500.0, "risk_per_trade": 0.02}, broker=None)
    bullish_result = make_pipeline_result(direction="long", confidence=0.8)
    ideas = agent.generate_ideas(bullish_result, datetime.now(timezone.utc))
    assert len(ideas) >= 1
    assert isinstance(ideas[0], TradeIdea)
    assert ideas[0].direction == DirectionEnum.LONG
    assert ideas[0].strategy_type in {StrategyType.DIRECTIONAL, StrategyType.MEAN_REVERSION, StrategyType.BREAKOUT}
```

### 3.4 `tests/test_pipeline_runner_watchlist.py`

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock

from engines.orchestration.pipeline_runner import PipelineRunner
from schemas.core_schemas import PipelineResult

def test_pipeline_skips_ideas_when_symbol_not_on_watchlist():
    trade_agent = MagicMock()
    watchlist = MagicMock(); watchlist.is_symbol_active.return_value = False
    runner = PipelineRunner(symbol="SPY", hedge_engine=MagicMock(), elasticity_engine=None, liquidity_engine=MagicMock(), sentiment_engine=MagicMock(), composer=MagicMock(), trade_agent=trade_agent, watchlist=watchlist, auto_execute=False, ledger=None, tracking_agent=None, ml_agent=None)
    result: PipelineResult = runner.run_once(timestamp=datetime.now(timezone.utc))
    trade_agent.generate_ideas.assert_not_called()
    assert result.trade_ideas == []

def test_pipeline_calls_trade_agent_when_symbol_active():
    trade_agent = MagicMock(); trade_agent.generate_ideas.return_value = ["dummy"]
    watchlist = MagicMock(); watchlist.is_symbol_active.return_value = True
    runner = PipelineRunner(symbol="SPY", hedge_engine=MagicMock(), elasticity_engine=None, liquidity_engine=MagicMock(), sentiment_engine=MagicMock(), composer=MagicMock(), trade_agent=trade_agent, watchlist=watchlist, auto_execute=False, ledger=None, tracking_agent=None, ml_agent=None)
    result: PipelineResult = runner.run_once(timestamp=datetime.now(timezone.utc))
    trade_agent.generate_ideas.assert_called_once()
    assert result.trade_ideas == ["dummy"]
```

---

## 4. Helper Scripts (Optional but Useful)

Add or validate scripts under `scripts/` for manual checks:

### 4.1 `scripts/test_alpaca_connection.py`

A simple broker connectivity check that loads `.env`, creates the broker adapter, and prints account stats and positions.

### 4.2 `scripts/manual_execution_test.py`

A manual SPY 1-share (≈$600) execution test via `TradeAgentV1.execute_trades`, printing order IDs and statuses from Alpaca Paper Trading.

---

## 5. Expected End State

* Repo imports cleanly and `pytest` passes (with the tests above).
* Alpaca adapter returns string order IDs and handles errors without crashes.
* `TradeAgentV1` applies sensible confidence gating and skips trades without reliable prices.
* `OrderExecutor` enforces risk limits and uses the broker adapter as the single execution path.
* `pipeline_runner` watchlist gating works without attribute errors and only calls the trade agent for active symbols.
* Manual scripts confirm live connectivity and order placement in Alpaca Paper Trading.
