# Developer Guide

How to extend Super Gnosis V2 without breaking it.

## Principles

1. **Function-first, DI-friendly**: Minimize global state, inject dependencies
2. **Typed contracts**: All data models use Pydantic v2
3. **Test invariants, not magic numbers**: Assert properties, not exact values
4. **Deterministic where possible**: Use seeds for reproducibility

## Folder Structure

```
V2---Gnosis/
├── agents/                    # High-level decision-making agents
│   ├── composer/             # Directive fusion and conflict resolution
│   ├── trade_agent/          # Strategy generation and risk analysis
│   └── *.py                  # Individual agents (hedge, liquidity, sentiment)
├── engines/                   # Low-level feature extraction
│   ├── hedge/                # Dealer positioning (gamma, vanna, charm)
│   ├── liquidity/            # Market depth and flow
│   └── sentiment/            # News and social signals
├── execution/                 # Order routing and broker integration
│   └── broker_adapters/      # Pluggable broker backends
├── optimizer/                 # Hyperparameter tuning and ML
│   ├── historical_evaluator.py
│   ├── kelly_refinement.py
│   ├── ml_integration.py
│   └── strategy_optimizer.py
├── pipeline/                  # Orchestration and workflows
│   └── full_pipeline.py
├── scripts/                   # Operational scripts
│   └── run_daily_spy_paper.py
├── tests/                     # Test suite (98.4% passing)
└── config/                    # Configuration models
```

## How To: Add a New Engine

Engines extract low-level features from market data (e.g., Macro, Factor, Seasonality).

### 1. Create Engine Module

```python
# engines/macro/macro_engine.py

from typing import Dict, Any
from datetime import datetime

class MacroEngine:
    """
    Extract macro indicators (Fed policy, GDP, unemployment, inflation).
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def run(
        self,
        symbol: str,
        as_of: datetime,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute macro features.
        
        Returns:
            Dict with macro indicators (fed_policy, gdp_growth, etc.)
        """
        # Fetch macro data (FRED API, Bloomberg, etc.)
        fed_funds_rate = self._fetch_fed_funds_rate(as_of)
        gdp_growth = self._fetch_gdp_growth(as_of)
        unemployment = self._fetch_unemployment(as_of)
        
        return {
            "fed_funds_rate": fed_funds_rate,
            "gdp_growth_yoy": gdp_growth,
            "unemployment_rate": unemployment,
            "macro_regime": self._classify_regime(fed_funds_rate, gdp_growth),
        }
    
    def _classify_regime(self, fed_rate: float, gdp_growth: float) -> str:
        """Classify macro regime (expansion, contraction, stagflation)."""
        if gdp_growth > 2.0 and fed_rate < 3.0:
            return "expansion"
        elif gdp_growth < 0.0:
            return "contraction"
        else:
            return "stagflation"
```

### 2. Create Agent (if needed)

```python
# agents/macro_agent.py

from agents.base import PrimaryAgent
from schemas.core_schemas import StandardSnapshot, Suggestion
from engines.macro.macro_engine import MacroEngine

class MacroAgent(PrimaryAgent):
    """Interpret macro engine output into policy suggestions."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.engine = MacroEngine(config)
    
    def step(self, snapshot: StandardSnapshot) -> Suggestion:
        """
        Interpret macro regime into directional bias.
        
        Expansion → bullish
        Contraction → bearish
        Stagflation → neutral/defensive
        """
        macro = snapshot.macro  # Assume snapshot has macro field
        regime = macro.get("macro_regime", "expansion")
        
        if regime == "expansion":
            return Suggestion(
                action="increase_risk",
                confidence=0.7,
                reasoning="Expansion regime: bullish GDP, low rates",
            )
        elif regime == "contraction":
            return Suggestion(
                action="decrease_risk",
                confidence=0.8,
                reasoning="Contraction regime: negative GDP growth",
            )
        else:
            return Suggestion(
                action="hold",
                confidence=0.5,
                reasoning="Stagflation regime: defensive stance",
            )
```

### 3. Integrate into Pipeline

```python
# pipeline/full_pipeline.py

from engines.macro.macro_engine import MacroEngine

@dataclass
class PipelineComponents:
    # ... existing engines ...
    macro_engine: Optional[MacroEngine] = None

# In run_full_pipeline_for_symbol():
if components.macro_engine is not None:
    macro_output = components.macro_engine.run(
        symbol=symbol,
        as_of=as_of,
        market_data=market_data,
    )
    
    # Pass to composer
    composer_context = components.composer_agent.compose(
        # ... existing inputs ...
        macro_output=macro_output,
    )
```

### 4. Write Tests

```python
# tests/engines/test_macro_engine.py

def test_macro_engine_expansion_regime():
    engine = MacroEngine(config={})
    
    # Mock data with expansion indicators
    market_data = {
        "fed_funds_rate": 2.5,
        "gdp_growth": 3.0,
        "unemployment": 4.0,
    }
    
    output = engine.run(
        symbol="SPY",
        as_of=datetime(2025, 1, 15),
        market_data=market_data,
    )
    
    # Assert invariants (not magic numbers)
    assert output["macro_regime"] in ["expansion", "contraction", "stagflation"]
    assert output["fed_funds_rate"] > 0
    assert -5.0 < output["gdp_growth_yoy"] < 10.0
```

## How To: Add a New Options Strategy

The Trade Agent supports 14 strategies. To add a 15th (e.g., Collar):

### 1. Define Strategy in Schemas

```python
# agents/trade_agent/schemas.py

class StrategyType(str, Enum):
    # ... existing strategies ...
    COLLAR = "collar"  # Long stock + long put + short call
```

### 2. Implement Builder Function

```python
# agents/trade_agent/options_strategies.py

def build_collar(
    ctx: ComposerTradeContext,
    underlying_price: float,
) -> TradeIdea:
    """
    Collar: Long stock + long protective put + short covered call.
    
    Downside protected, upside capped, minimal cost (put paid by call premium).
    Defensive strategy for existing stock holdings.
    """
    expiry = select_expiry(ctx)
    
    # Stock leg
    stock_leg = OptionLeg(
        side="long",
        option_type="stock",  # Special case
        strike=underlying_price,
        expiry=expiry,
        quantity=100,  # 100 shares
    )
    
    # Protective put (5% OTM)
    put_strike = underlying_price * 0.95
    put_leg = OptionLeg(
        side="long",
        option_type="put",
        strike=put_strike,
        expiry=expiry,
        quantity=1,
    )
    
    # Covered call (5% OTM)
    call_strike = underlying_price * 1.05
    call_leg = OptionLeg(
        side="short",
        option_type="call",
        strike=call_strike,
        expiry=expiry,
        quantity=1,
    )
    
    desc = (
        f"Collar: Long {underlying_price:.2f} stock + "
        f"Long {put_strike:.2f}P + Short {call_strike:.2f}C, expiry={expiry}. "
        "Downside protected, upside capped."
    )
    
    return TradeIdea(
        asset=ctx.asset,
        strategy_type=StrategyType.COLLAR,
        description=desc,
        legs=[stock_leg, put_leg, call_leg],
        timeframe=ctx.timeframe,
        confidence=ctx.confidence,
        greeks_profile=StrategyGreeks(
            delta=0.5,  # Reduced from 1.0 due to short call
            gamma=0.1,
            theta=0.2,  # Positive from short call
            vega=-0.1,  # Negative from net short premium
        ),
    )
```

### 3. Register in Strategy Selector

```python
# agents/trade_agent/strategy_selector.py

def select_strategies_for_context(ctx: ComposerTradeContext) -> List[StrategyBuilder]:
    builders: List[StrategyBuilder] = []
    
    # ... existing selection logic ...
    
    # Collar for defensive bullish (existing holdings)
    if ctx.direction == Direction.BULLISH and ctx.confidence < 0.6:
        builders.append(build_collar)
    
    return builders
```

### 4. Add Exit Rules

```python
# agents/trade_agent/exit_manager.py

def create_default_exit_rules(strategy_type: StrategyType, confidence: float) -> ExitRule:
    # ... existing rules ...
    
    elif strategy_type == StrategyType.COLLAR:
        # Defensive strategy: tighter profit target, moderate stop
        base_profit_target = 0.30  # 30% of max profit
        base_stop_loss = 0.70  # 70% of max loss
        max_dte = 7  # Exit 1 week before expiry
    
    # ... rest of function ...
```

### 5. Write Tests

```python
# tests/agents/test_collar_strategy.py

def test_collar_structure(bullish_defensive_context):
    """Collar should have 3 legs: stock, put, call."""
    idea = build_collar(bullish_defensive_context, underlying_price=450.0)
    
    assert idea.strategy_type == StrategyType.COLLAR
    assert len(idea.legs) == 3
    
    # Find each leg type
    stock_legs = [leg for leg in idea.legs if leg.option_type == "stock"]
    put_legs = [leg for leg in idea.legs if leg.option_type == "put"]
    call_legs = [leg for leg in idea.legs if leg.option_type == "call"]
    
    assert len(stock_legs) == 1
    assert len(put_legs) == 1
    assert len(call_legs) == 1
    
    # Put is protective (long)
    assert put_legs[0].side == "long"
    assert put_legs[0].strike < underlying_price
    
    # Call is covered (short)
    assert call_legs[0].side == "short"
    assert call_legs[0].strike > underlying_price

def test_collar_greeks(bullish_defensive_context):
    """Collar should have reduced delta, positive theta."""
    idea = build_collar(bullish_defensive_context, underlying_price=450.0)
    
    assert 0.3 < idea.greeks_profile.delta < 0.7  # Reduced from 1.0
    assert idea.greeks_profile.theta > 0  # Positive from short call
    assert idea.greeks_profile.vega < 0  # Net short premium
```

## How To: Add a New Broker Adapter

To integrate a new broker (e.g., Alpaca, Interactive Brokers):

### 1. Implement BrokerAdapter Interface

```python
# execution/broker_adapters/alpaca_adapter.py

from typing import Dict, Optional
from execution.schemas import OrderRequest, OrderResult, OrderStatus, Quote

class AlpacaBrokerAdapter:
    """
    Alpaca broker adapter for paper and live trading.
    
    Docs: https://alpaca.markets/docs/api-documentation/
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.client = self._create_client()
    
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place order via Alpaca API."""
        try:
            response = self.client.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side.value,
                type=order.order_type.value,
                time_in_force="day",
                limit_price=order.limit_price,
            )
            
            return OrderResult(
                order_id=response.id,
                status=OrderStatus.PENDING,
                filled_quantity=0,
                average_fill_price=0.0,
                timestamp=datetime.utcnow(),
            )
        
        except Exception as e:
            return OrderResult(
                order_id=str(uuid4()),
                status=OrderStatus.REJECTED,
                error_message=str(e),
                timestamp=datetime.utcnow(),
            )
    
    def get_quote(self, symbol: str) -> Quote:
        """Fetch current quote from Alpaca."""
        quote = self.client.get_latest_quote(symbol)
        return Quote(
            symbol=symbol,
            bid=quote.bid_price,
            ask=quote.ask_price,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            timestamp=datetime.utcnow(),
        )
    
    def get_position(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current position from Alpaca."""
        try:
            position = self.client.get_position(symbol)
            return {
                "quantity": float(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "market_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
            }
        except:
            return None
```

### 2. Add to Pipeline Components

```python
# pipeline/full_pipeline.py

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter

def build_default_pipeline_components(
    config: Optional[Dict[str, Any]] = None,
    broker_type: Literal["simulated", "alpaca", "ibkr"] = "simulated",
    **kwargs,
) -> PipelineComponents:
    # ... existing setup ...
    
    if broker_type == "alpaca":
        broker = AlpacaBrokerAdapter(
            api_key=config.get("alpaca_api_key"),
            api_secret=config.get("alpaca_api_secret"),
            base_url=config.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
        )
    elif broker_type == "simulated":
        broker = SimulatedBrokerAdapter(...)
    # ... etc ...
```

### 3. Write Tests

```python
# tests/execution/test_alpaca_adapter.py

@pytest.fixture
def mock_alpaca_client(mocker):
    """Mock Alpaca API client."""
    mock = mocker.Mock()
    mock.submit_order.return_value = mocker.Mock(id="order_123")
    mock.get_latest_quote.return_value = mocker.Mock(
        bid_price=449.50,
        ask_price=450.50,
        bid_size=100,
        ask_size=100,
    )
    return mock

def test_alpaca_place_order_success(mock_alpaca_client):
    adapter = AlpacaBrokerAdapter(
        api_key="test_key",
        api_secret="test_secret",
    )
    adapter.client = mock_alpaca_client
    
    order = OrderRequest(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        asset_class=AssetClass.STOCK,
    )
    
    result = adapter.place_order(order)
    
    assert result.status == OrderStatus.PENDING
    assert result.order_id == "order_123"
    mock_alpaca_client.submit_order.assert_called_once()
```

## How To: Plug In an ML Model

ML models must implement the `PredictionModel` protocol:

```python
from typing import Protocol, Dict

class PredictionModel(Protocol):
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict win rate, avg win/loss, prob_up from features.
        
        Args:
            features: Dict with composer context fields (confidence, elastic_energy, etc.)
        
        Returns:
            Dict with:
                - win_rate: Empirical win rate (0.0-1.0)
                - avg_win: Average winning trade return (e.g., 1.5)
                - avg_loss: Average losing trade return (e.g., 1.0)
                - prob_up: Probability of price increase (0.0-1.0)
        """
        ...
```

### Example: scikit-learn Model

```python
# optimizer/ml_models/sklearn_model.py

import joblib
import numpy as np
from typing import Dict

class SklearnPredictionModel:
    """
    scikit-learn model wrapper for prediction.
    
    Trains a classifier to predict win rate and directional probability.
    """
    
    def __init__(self, model_path: str) -> None:
        self.model = joblib.load(model_path)
    
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict using trained scikit-learn model."""
        # Convert features to array
        feature_array = np.array([
            features.get("confidence", 0.5),
            features.get("elastic_energy", 1.0),
            features.get("gamma_exposure", 0.0),
            features.get("vanna_exposure", 0.0),
            features.get("liquidity_score", 0.8),
        ]).reshape(1, -1)
        
        # Predict
        proba = self.model.predict_proba(feature_array)[0]
        win_rate = proba[1]  # Probability of winning trade
        prob_up = proba[1]   # Probability of price increase
        
        # Estimate avg win/loss (could come from separate regressor)
        avg_win = 1.5 if win_rate > 0.6 else 1.2
        avg_loss = 1.0
        
        return {
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "prob_up": float(prob_up),
        }
```

### Training Script

```python
# scripts/train_sklearn_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load historical data
df = pd.read_csv("runs/historical_trades.csv")

# Features
X = df[["confidence", "elastic_energy", "gamma_exposure", "vanna_exposure", "liquidity_score"]]

# Target (1 = winning trade, 0 = losing trade)
y = (df["final_pnl"] > 0).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")

# Save
joblib.dump(model, "models/sklearn_win_rate_model.pkl")
```

### Integration

```python
# pipeline/full_pipeline.py

from optimizer.ml_models.sklearn_model import SklearnPredictionModel

components = build_default_pipeline_components(
    prediction_model=SklearnPredictionModel("models/sklearn_win_rate_model.pkl"),
)
```

## How To: Run Offline Hyperparameter Optimization

Optuna-based offline tuning to find best profit_target_pct, stop_loss_pct, etc. per regime.

### 1. Define Evaluator

```python
# optimizer/evaluators/backtest_evaluator.py

from typing import Dict, Tuple, Any
import pandas as pd

class BacktestEvaluator:
    """
    Evaluate hyperparameters via backtest.
    
    Returns objective value (e.g., Sharpe ratio, return on risk).
    """
    
    def __init__(self, historical_data: pd.DataFrame) -> None:
        self.data = historical_data
    
    def __call__(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate params on historical data.
        
        Args:
            params: Dict with profit_target_pct, stop_loss_pct, dte, etc.
        
        Returns:
            Tuple of (objective_value, metrics_dict)
        """
        profit_target_pct = params["profit_target_pct"]
        stop_loss_pct = params["stop_loss_pct"]
        dte = params["dte"]
        
        # Simulate trades with these params
        trades = self._simulate_trades(profit_target_pct, stop_loss_pct, dte)
        
        # Compute metrics
        total_return = trades["pnl"].sum()
        sharpe = trades["pnl"].mean() / trades["pnl"].std() if len(trades) > 1 else 0.0
        win_rate = (trades["pnl"] > 0).mean()
        
        metrics = {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "win_rate": float(win_rate),
        }
        
        return sharpe, metrics  # Maximize Sharpe ratio
```

### 2. Run Optimization

```python
# scripts/run_optimization.py

from optimizer.strategy_optimizer import run_optuna_optimization, OptimizationConfig, OptimizationBounds
from optimizer.evaluators.backtest_evaluator import BacktestEvaluator
import pandas as pd

# Load historical data
df = pd.read_csv("runs/historical_trades.csv")

# Create evaluator
evaluator = BacktestEvaluator(df)

# Define search space
config = OptimizationConfig(
    study_name="spy_bull_regime",
    direction="maximize",
    n_trials=100,
    bounds=[
        OptimizationBounds(name="profit_target_pct", low=0.2, high=1.0, step=0.1),
        OptimizationBounds(name="stop_loss_pct", low=0.2, high=0.8, step=0.1),
        OptimizationBounds(name="dte", low=7, high=60, step=7),
    ],
    seed=42,
)

# Run optimization
result = run_optuna_optimization(evaluator, config)

print(f"Best params: {result['best_params']}")
print(f"Best Sharpe: {result['best_value']:.3f}")

# Save to JSON for pipeline
import json
with open("config/hyperparams_spy_bull.json", "w") as f:
    json.dump(result["best_params"], f, indent=2)
```

### 3. Create Hyperparameter Provider

```python
# optimizer/hyperparameter_store.py

import json
from pathlib import Path
from typing import Dict, Any

class FileBasedHyperparameterProvider:
    """
    Load hyperparameters from JSON files.
    
    Files named: config/hyperparams_{symbol}_{regime}.json
    """
    
    def __init__(self, config_dir: str = "config") -> None:
        self.config_dir = Path(config_dir)
    
    def get_best_params(self, symbol: str, regime: Dict[str, str]) -> Dict[str, Any]:
        """
        Load hyperparameters for symbol/regime.
        
        Args:
            symbol: Trading symbol (e.g., "SPY")
            regime: Dict with vix_bucket, trend_bucket, etc.
        
        Returns:
            Dict with profit_target_pct, stop_loss_pct, dte, etc.
        """
        # Build filename
        regime_key = f"{regime['vix_bucket']}_{regime['trend_bucket']}"
        filename = self.config_dir / f"hyperparams_{symbol.lower()}_{regime_key}.json"
        
        # Load or return defaults
        if filename.exists():
            with open(filename) as f:
                return json.load(f)
        else:
            return {
                "profit_target_pct": 0.5,
                "stop_loss_pct": 0.5,
                "dte": 30,
            }
```

## Test Guidelines

### Assert Invariants, Not Magic Numbers

❌ **Bad**: Hard-coded values break when pricing model improves

```python
def test_iron_condor_risk_reward():
    metrics = analyze_trade_risk(idea, underlying_price=450.0)
    assert metrics.risk_reward_ratio == 0.35  # Brittle!
```

✅ **Good**: Test properties that should always hold

```python
def test_iron_condor_risk_reward():
    metrics = analyze_trade_risk(idea, underlying_price=450.0)
    assert metrics.risk_reward_ratio > 0  # Positive
    assert metrics.max_profit > 0  # Profitable upside
    assert metrics.max_loss < 0  # Defined downside
```

### Deterministic Optuna Seeds

```python
def test_optuna_reproducibility():
    config = OptimizationConfig(
        study_name="test",
        n_trials=5,
        bounds=[...],
        seed=42,  # Always set seed!
    )
    
    result1 = run_optuna_optimization(evaluator, config)
    result2 = run_optuna_optimization(evaluator, config)
    
    assert result1["best_value"] == result2["best_value"]
```

### No Network Calls in Unit Tests

Use mocks or fixtures:

```python
@pytest.fixture
def mock_market_data():
    return {
        "symbol": "SPY",
        "price": 450.0,
        "volume": 100_000_000,
    }

def test_hedge_agent(mock_market_data):
    agent = HedgeAgentV3(config={})
    snapshot = StandardSnapshot(market=mock_market_data)
    suggestion = agent.step(snapshot)
    
    assert suggestion.confidence > 0
```

## Common Pitfalls

### 1. Forgetting to Call `.model_dump()`

Pydantic v2 replaced `.dict()` with `.model_dump()`:

```python
# Old (v1)
data = idea.dict()

# New (v2)
data = idea.model_dump()
```

### 2. Direction Enum Comparison

Never compare enums with integers:

```python
# Bad
if ctx.direction > 0:  # TypeError!

# Good
if ctx.direction == Direction.BULLISH:
```

### 3. DateTime Without Timezone

Always use timezone-aware datetimes:

```python
# Bad
now = datetime.utcnow()  # Deprecated!

# Good
from datetime import timezone
now = datetime.now(timezone.utc)
```

### 4. Mutable Default Arguments

Never use mutable defaults:

```python
# Bad
def func(data: List[int] = []):  # Shared across calls!

# Good
def func(data: Optional[List[int]] = None):
    if data is None:
        data = []
```

## Additional Resources

- **Pydantic v2 Docs**: https://docs.pydantic.dev/
- **Optuna Docs**: https://optuna.readthedocs.io/
- **pytest Docs**: https://docs.pytest.org/
- **Architecture**: [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)
- **Operations**: [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md)
