# Trade + Execution Engine v3.0 Implementation

## Overview

The **Universal Policy Composer** is the master orchestration system that integrates signals from all v3 engines (Elasticity, Liquidity, Sentiment) into cohesive, executable trade ideas with comprehensive risk management.

**Status:** ✅ **COMPLETE** - 28/28 tests passing (100%)

---

## Architecture

### Multi-Engine Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                  UNIVERSAL POLICY COMPOSER                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Elasticity  │  │  Liquidity   │  │  Sentiment   │        │
│  │  Engine v3   │  │  Engine v3   │  │  Engine v3   │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│  ┌──────────────────────────────────────────────────┐         │
│  │         Signal Extraction & Weighting            │         │
│  │  Energy: 40% | Liquidity: 30% | Sentiment: 30%  │         │
│  └──────────────────┬───────────────────────────────┘         │
│                     ▼                                          │
│  ┌──────────────────────────────────────────────────┐         │
│  │         Direction Determination                  │         │
│  │  LONG | SHORT | NEUTRAL | AVOID                 │         │
│  └──────────────────┬───────────────────────────────┘         │
│                     ▼                                          │
│  ┌──────────────────────────────────────────────────┐         │
│  │         Multi-Method Position Sizing             │         │
│  │  Kelly | Vol-Target | Energy-Aware | Composite  │         │
│  └──────────────────┬───────────────────────────────┘         │
│                     ▼                                          │
│  ┌──────────────────────────────────────────────────┐         │
│  │         Monte Carlo Risk Simulation              │         │
│  │  1000 simulations | VaR | Sharpe | Win Rate     │         │
│  └──────────────────┬───────────────────────────────┘         │
│                     ▼                                          │
│  ┌──────────────────────────────────────────────────┐         │
│  │         Trade Idea Validation                    │         │
│  │  Risk Limits | Regime Checks | Cost Analysis    │         │
│  └──────────────────┬───────────────────────────────┘         │
│                     ▼                                          │
│              📋 TRADE IDEA                                     │
│   Direction | Size | Entry | Stop | Target | Costs           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Signal Extraction

**Purpose:** Extract directional signals from each engine's state.

#### Energy Signal

```python
def _extract_energy_signal(energy_state) -> float:
    """
    Logic:
    - Energy asymmetry: positive = easier up (bullish)
    - Elasticity asymmetry: positive = more resistance up (bearish)
    - Dampen with stability score
    
    Returns: Signal in [-1, 1]
    """
    asymmetry_signal = tanh(energy_asymmetry / 100)
    elasticity_signal = -tanh(elasticity_asymmetry / 10)
    
    signal = 0.7 * asymmetry_signal + 0.3 * elasticity_signal
    signal *= stability
    
    return clip(signal, -1, 1)
```

**Interpretation:**
- `+1.0`: Strong bullish (low energy to move up)
- `0.0`: Neutral
- `-1.0`: Strong bearish (low energy to move down)

#### Liquidity Signal

```python
def _extract_liquidity_signal(liquidity_state) -> float:
    """
    Logic:
    - Depth imbalance: positive = more bids (bullish)
    - Adjust for impact cost (high impact dampens)
    - Dampen with stability
    
    Returns: Signal in [-1, 1]
    """
    imbalance_signal = depth_imbalance
    impact_adjustment = 1.0 - min(impact_cost / 100, 1.0)
    
    signal = imbalance_signal * impact_adjustment
    signal *= stability
    
    return clip(signal, -1, 1)
```

**Interpretation:**
- `+1.0`: Strong buy depth with low impact
- `0.0`: Balanced
- `-1.0`: Strong sell depth with low impact

#### Sentiment Signal

```python
def _extract_sentiment_signal(sentiment_state) -> float:
    """
    Logic:
    - Primary: Contrarian signal (fade extremes)
    - Secondary: Momentum (for trends)
    - Amplify with crowd conviction
    - Dampen with stability
    
    Returns: Signal in [-1, 1]
    """
    contrarian = contrarian_signal
    momentum = tanh(sentiment_momentum / 0.5)
    
    # Blend: 70% contrarian, 30% momentum
    signal = 0.7 * contrarian + 0.3 * momentum
    
    # Amplify with conviction
    signal *= (1.0 + crowd_conviction) / 2.0
    signal *= stability
    
    return clip(signal, -1, 1)
```

**Interpretation:**
- `+1.0`: Extreme bearish sentiment (contrarian bullish)
- `0.0`: Neutral sentiment
- `-1.0`: Extreme bullish sentiment (contrarian bearish)

#### Composite Signal

```python
composite_signal = (
    0.4 * energy_signal +      # 40% weight
    0.3 * liquidity_signal +   # 30% weight
    0.3 * sentiment_signal     # 30% weight
)
```

**Configurable Weights:** Adjust weights based on market conditions or strategy preference.

---

### 2. Direction Determination

**Rules:**

| Composite Signal | Energy Regime | Liquidity Regime | Direction |
|-----------------|---------------|------------------|-----------|
| \|signal\| < 0.2 | Any | Any | **NEUTRAL** |
| signal > 0.2 | elastic/super_elastic | liquid/deep | **LONG** |
| signal < -0.2 | elastic/brittle | liquid/thin | **SHORT** |
| Any | plastic | Any | **AVOID** |
| Any | Any | frozen | **AVOID** |

**Bad Regimes:**
- **Energy = plastic:** High resistance to movement (avoid trading)
- **Liquidity = frozen:** No liquidity available (avoid trading)

---

### 3. Position Sizing

#### Method 1: Kelly Criterion

**Formula:**
```
Kelly % = (p × W - (1-p) × L) / W

where:
- p = win probability (from historical returns)
- W = average win size
- L = average loss size
```

**Implementation:**
```python
def _kelly_criterion_size(historical_returns, account_value, price):
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    p = len(wins) / len(returns)
    avg_win = mean(wins)
    avg_loss = abs(mean(losses))
    
    kelly_pct = (p * avg_win - (1-p) * avg_loss) / avg_win
    kelly_pct *= kelly_fraction  # Use fractional Kelly (default: 0.25)
    
    return account_value * kelly_pct / price
```

**When to Use:**
- Sufficient historical data (20+ trades)
- Clear edge (expected return > 5%)
- Conservative with fractional Kelly (1/4 Kelly recommended)

#### Method 2: Volatility Targeting

**Formula:**
```
Position = (Target_Vol / Asset_Vol) × Account_Value / Price
```

**Implementation:**
```python
def _vol_target_size(volatility, account_value, price):
    target_vol = 0.15  # Target 15% annualized
    vol_ratio = target_vol / volatility
    position_value = account_value * vol_ratio
    
    return position_value / price
```

**When to Use:**
- Want consistent portfolio volatility
- Asset volatility is known
- No historical edge data

#### Method 3: Energy-Aware Sizing

**Formula:**
```
Adjusted_Size = Base_Size × Energy_Factor × Elasticity_Factor

where:
- Energy_Factor = min(Max_Energy / Movement_Energy, 1.0)
- Elasticity_Factor = min(Elasticity / Min_Elasticity, 1.0)
```

**Implementation:**
```python
def _energy_aware_size(base_size, movement_energy, elasticity):
    # Dampen if energy too high
    energy_factor = min(max_energy / movement_energy, 1.0)
    
    # Dampen if elasticity too low
    elasticity_factor = min(elasticity / min_elasticity, 1.0)
    
    return base_size * energy_factor * elasticity_factor
```

**Logic:**
- **High movement energy** → Reduce size (avoid forcing moves)
- **Low elasticity** → Reduce size (weak restoring forces)

#### Method 4: Composite (Conservative)

**Strategy:** Take **minimum** of all methods
```python
position_size = min(
    fixed_size,           # Hard cap
    kelly_size,           # Edge-based
    vol_target_size,      # Vol-based
    energy_aware_size     # Physics-based
)
```

**Why Minimum?**
- Most conservative approach
- Respects all constraints
- Prevents over-sizing

---

### 4. Entry & Exit Levels

#### Entry Range

**For LONG:**
```python
entry_price = current_price
entry_min = current_price
entry_max = current_price + spread_adjustment
```

**For SHORT:**
```python
entry_price = current_price
entry_max = current_price
entry_min = current_price - spread_adjustment
```

#### Stop Loss & Take Profit

**Default Parameters:**
- **Stop Loss:** 2% from entry
- **Take Profit:** 6% from entry
- **Risk/Reward Ratio:** 3:1

**For LONG:**
```python
stop_loss = entry_price × (1 - 0.02)     # -2%
take_profit = entry_price × (1 + 0.06)   # +6%
```

**For SHORT:**
```python
stop_loss = entry_price × (1 + 0.02)     # +2%
take_profit = entry_price × (1 - 0.06)   # -6%
```

---

### 5. Execution Cost Estimation

**Components:**

1. **Slippage:** Price movement while executing
2. **Impact Cost:** Cost to walk the order book

**Formula:**
```python
# Base costs from liquidity engine
slippage_bps = liquidity_state.slippage
impact_bps = liquidity_state.impact_cost

# Adjust for position size
size_adjustment = 1.0 + (position_value / 100000)

adjusted_slippage = slippage_bps × size_adjustment
adjusted_impact = impact_bps × size_adjustment

total_cost_bps = adjusted_slippage + adjusted_impact
```

**Example:**
- Base slippage: 5 bps
- Base impact: 10 bps
- Position: $50,000
- Size adjustment: 1.5×
- Total cost: (5 + 10) × 1.5 = 22.5 bps

---

### 6. Monte Carlo Simulation

**Purpose:** Simulate 1000 possible trade outcomes using geometric Brownian motion.

**Model:**
```
dS = μ × S × dt + σ × S × dW

where:
- dS = price change
- μ = drift (assume 0 for conservatism)
- σ = volatility (annualized)
- S = current price
- dt = time step (1/252 for daily)
- dW = Wiener process (random)
```

**Implementation:**
```python
def _run_monte_carlo(entry, target, stop, volatility, num_sim=1000):
    pnl_results = []
    
    for _ in range(num_sim):
        price = entry
        
        for step in range(20):  # Simulate 20 days
            dW = random.normal(0, sqrt(dt))
            dS = volatility * price * dW
            price += dS
            
            if price <= stop:
                pnl_results.append(stop - entry)
                break
            elif price >= target:
                pnl_results.append(target - entry)
                break
        else:
            pnl_results.append(price - entry)
    
    return analyze_results(pnl_results)
```

**Output Metrics:**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Win Rate** | % of profitable outcomes | Higher = better edge |
| **Avg P&L** | Mean profit/loss | Expected value |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 is good |
| **VaR (95%)** | 5th percentile loss | Worst-case planning |
| **CVaR (95%)** | Expected loss beyond VaR | Tail risk |
| **Profit Factor** | Gross profit / Gross loss | > 1.0 = profitable |

**Example Results:**
```
Win Rate:       55%
Avg P&L:        $150
Sharpe Ratio:   1.8
VaR (95%):      -$200
CVaR (95%):     -$250
Profit Factor:  2.1
```

---

### 7. Trade Idea Validation

**Validation Checks:**

#### Position Limits
```python
# Check 1: Absolute position size
if position_value > max_position_size:
    ERROR: "Position exceeds max size"

# Check 2: Portfolio percentage
if position_value / account_value > max_position_pct:
    ERROR: "Position exceeds max portfolio %"
```

#### Regime Checks
```python
# Check 3: Energy regime
if energy_regime == "plastic":
    ERROR: "Energy regime unfavorable"

# Check 4: Liquidity regime
if liquidity_regime == "frozen":
    ERROR: "Liquidity regime unfavorable"
```

#### Execution Costs
```python
# Check 5: Impact cost
if impact_bps > max_impact_bps:
    WARNING: "High impact cost"

# Check 6: Slippage
if slippage_bps > max_slippage_bps:
    WARNING: "High slippage expected"
```

#### Edge Requirements
```python
# Check 7: Minimum edge (Kelly)
if kelly_edge < min_edge:
    WARNING: "Insufficient edge"

# Check 8: Movement energy
if movement_energy > max_movement_energy:
    WARNING: "High movement energy"
```

**Validation Outcome:**
- **is_valid = True:** All critical checks passed (may have warnings)
- **is_valid = False:** One or more errors present (do not trade)

---

## Usage Examples

### Example 1: Basic Trade Idea

```python
from engines.composer.universal_policy_composer import (
    UniversalPolicyComposer,
    RiskParameters
)
from engines.hedge.universal_energy_interpreter import UniversalEnergyInterpreter
from engines.liquidity.universal_liquidity_interpreter import UniversalLiquidityInterpreter
from engines.sentiment.universal_sentiment_interpreter import UniversalSentimentInterpreter

# Initialize engines
energy_engine = UniversalEnergyInterpreter()
liquidity_engine = UniversalLiquidityInterpreter()
sentiment_engine = UniversalSentimentInterpreter()

# Initialize composer
composer = UniversalPolicyComposer(
    risk_params=RiskParameters(
        max_position_size=10000.0,
        max_portfolio_exposure=100000.0,
        target_volatility=0.15,
        kelly_fraction=0.25
    ),
    energy_weight=0.4,
    liquidity_weight=0.3,
    sentiment_weight=0.3,
    enable_monte_carlo=True,
    mc_simulations=1000
)

# Get engine states (from your data pipeline)
energy_state = energy_engine.interpret(...)
liquidity_state = liquidity_engine.interpret(...)
sentiment_state = sentiment_engine.interpret(...)

# Compose trade idea
trade_idea = composer.compose_trade_idea(
    symbol="AAPL",
    current_price=150.0,
    energy_state=energy_state,
    liquidity_state=liquidity_state,
    sentiment_state=sentiment_state,
    account_value=100000.0,
    current_volatility=0.25,
    historical_returns=[0.01, -0.005, 0.02, ...]  # Optional
)

# Inspect results
print(f"Direction: {trade_idea.direction.value}")
print(f"Position Size: {trade_idea.position_size} shares")
print(f"Entry: ${trade_idea.entry_price:.2f}")
print(f"Stop Loss: ${trade_idea.stop_loss:.2f}")
print(f"Take Profit: ${trade_idea.take_profit:.2f}")
print(f"Composite Signal: {trade_idea.composite_signal:+.3f}")
print(f"Valid: {trade_idea.is_valid}")
print(f"Confidence: {trade_idea.confidence:.1%}")
```

### Example 2: Custom Risk Parameters

```python
# Conservative parameters
conservative_params = RiskParameters(
    max_position_size=5000.0,          # Smaller positions
    max_portfolio_exposure=50000.0,     # Lower exposure
    max_position_pct=0.05,              # Max 5% per position
    target_volatility=0.10,             # Target 10% vol
    kelly_fraction=0.10,                # 1/10 Kelly (very conservative)
    max_movement_energy=500.0,          # Lower energy threshold
    max_impact_bps=25.0,                # Lower impact tolerance
    stop_loss_pct=0.015,                # 1.5% stop
    take_profit_pct=0.045               # 4.5% target (3:1 R/R)
)

composer_conservative = UniversalPolicyComposer(
    risk_params=conservative_params
)
```

### Example 3: Aggressive Parameters

```python
# Aggressive parameters
aggressive_params = RiskParameters(
    max_position_size=25000.0,          # Larger positions
    max_portfolio_exposure=200000.0,    # Higher exposure
    max_position_pct=0.20,              # Max 20% per position
    target_volatility=0.25,             # Target 25% vol
    kelly_fraction=0.50,                # 1/2 Kelly (aggressive)
    max_movement_energy=2000.0,         # Higher energy tolerance
    max_impact_bps=100.0,               # Higher impact tolerance
    stop_loss_pct=0.03,                 # 3% stop
    take_profit_pct=0.09                # 9% target (3:1 R/R)
)

composer_aggressive = UniversalPolicyComposer(
    risk_params=aggressive_params,
    energy_weight=0.5,                  # Higher energy weight
    liquidity_weight=0.25,
    sentiment_weight=0.25
)
```

### Example 4: Signal Extraction Only

```python
# Extract individual signals without full composition
energy_signal = composer._extract_energy_signal(energy_state)
liquidity_signal = composer._extract_liquidity_signal(liquidity_state)
sentiment_signal = composer._extract_sentiment_signal(sentiment_state)

composite = composer._compute_composite_signal(
    energy_signal, liquidity_signal, sentiment_signal
)

print(f"Energy Signal: {energy_signal:+.3f}")
print(f"Liquidity Signal: {liquidity_signal:+.3f}")
print(f"Sentiment Signal: {sentiment_signal:+.3f}")
print(f"Composite: {composite:+.3f}")
```

### Example 5: Position Sizing Methods

```python
# Compare all sizing methods
direction = TradeDirection.LONG
current_price = 100.0
account_value = 100000.0
volatility = 0.20
historical_returns = [0.01, -0.005, 0.015, ...]

# Kelly
kelly_size, kelly_frac = composer._kelly_criterion_size(
    historical_returns, account_value, current_price
)

# Vol target
vol_size = composer._vol_target_size(
    volatility, account_value, current_price
)

# Energy-aware
energy_size = composer._energy_aware_size(
    base_size=vol_size,
    movement_energy=energy_state.movement_energy,
    elasticity=energy_state.elasticity
)

print(f"Kelly: {kelly_size:.0f} shares ({kelly_frac:.2%})")
print(f"Vol Target: {vol_size:.0f} shares")
print(f"Energy-Aware: {energy_size:.0f} shares")
print(f"Final (min): {min(kelly_size, vol_size, energy_size):.0f} shares")
```

### Example 6: Monte Carlo Analysis

```python
# Run standalone Monte Carlo
mc_result = composer._run_monte_carlo(
    entry_price=100.0,
    target_price=106.0,
    stop_price=98.0,
    volatility=0.25,
    num_simulations=1000
)

print(f"Win Rate: {mc_result.win_rate:.1%}")
print(f"Avg P&L: ${mc_result.avg_pnl:.2f}")
print(f"Median P&L: ${mc_result.median_pnl:.2f}")
print(f"Std Dev: ${mc_result.std_pnl:.2f}")
print(f"Sharpe: {mc_result.sharpe_ratio:.2f}")
print(f"Profit Factor: {mc_result.profit_factor:.2f}")
print(f"VaR (95%): ${mc_result.var_95:.2f}")
print(f"CVaR (95%): ${mc_result.cvar_95:.2f}")
```

---

## Integration with Existing Systems

### 1. With Alpaca Executor

```python
from engines.execution.alpaca_executor import AlpacaExecutor

# Create trade idea
trade_idea = composer.compose_trade_idea(...)

# Validate
if not trade_idea.is_valid:
    print("Trade idea failed validation")
    for error in trade_idea.validation_errors:
        print(f"  ERROR: {error}")
    return

# Execute with Alpaca
executor = AlpacaExecutor(
    api_key="your_key",
    secret_key="your_secret",
    paper=True
)

if trade_idea.direction == TradeDirection.LONG:
    order = executor.submit_market_order(
        symbol=trade_idea.symbol,
        qty=trade_idea.position_size,
        side=OrderSideEnum.BUY
    )
elif trade_idea.direction == TradeDirection.SHORT:
    order = executor.submit_market_order(
        symbol=trade_idea.symbol,
        qty=trade_idea.position_size,
        side=OrderSideEnum.SELL
    )

print(f"Order submitted: {order.order_id}")
```

### 2. With LangChain Agent

```python
from engines.composer.langchain_agent import TradingAgent

# Create agent with composer
agent = TradingAgent(policy_composer=composer)

# Ask agent to analyze
response = agent.analyze_trade_opportunity(
    symbol="TSLA",
    energy_state=energy_state,
    liquidity_state=liquidity_state,
    sentiment_state=sentiment_state
)

print(response)
```

### 3. With Backtesting (Future: Week 5-6)

```python
# This is the planned integration for Weeks 5-6
from engines.backtest.backtrader_engine import BacktraderEngine

backtest = BacktraderEngine(
    policy_composer=composer,
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_cash=100000.0
)

results = backtest.run()
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

---

## Performance Benchmarks

**Hardware:** Standard sandbox environment

| Operation | Time | Notes |
|-----------|------|-------|
| Signal extraction | <1ms | All 3 engines |
| Position sizing | <2ms | All methods |
| Monte Carlo (100 sims) | <10ms | Fast testing |
| Monte Carlo (1000 sims) | <100ms | Production |
| Full composition | <200ms | Complete trade idea |

**Test Results:** 28/28 passing (100%)

---

## Risk Management Framework

### Position Limits

**Default Limits:**
```python
max_position_size = 10000.0         # $10k per position
max_portfolio_exposure = 100000.0   # $100k total
max_position_pct = 0.10             # 10% of portfolio
```

### Volatility Controls

**Default Settings:**
```python
target_volatility = 0.15            # Target 15% annualized
vol_lookback_days = 30              # 30-day vol estimate
```

### Kelly Criterion

**Default Settings:**
```python
kelly_fraction = 0.25               # 1/4 Kelly (conservative)
min_edge = 0.05                     # Minimum 5% edge
```

### Energy Constraints

**Default Settings:**
```python
max_movement_energy = 1000.0        # Avoid high-energy moves
min_elasticity = 0.1                # Minimum elasticity required
```

### Liquidity Constraints

**Default Settings:**
```python
max_impact_bps = 50.0               # Max 50 bps impact
max_slippage_bps = 30.0             # Max 30 bps slippage
```

### Stop Loss / Take Profit

**Default Settings:**
```python
stop_loss_pct = 0.02                # 2% stop loss
take_profit_pct = 0.06              # 6% take profit
# Risk/Reward Ratio = 3:1
```

---

## Troubleshooting

### Issue 1: All Trades Return NEUTRAL

**Cause:** Composite signal too weak (|signal| < 0.2)

**Solution:**
1. Check engine states are producing non-zero signals
2. Adjust signal weights to emphasize stronger engine
3. Lower neutral threshold (risky - use caution)

### Issue 2: All Trades Return AVOID

**Cause:** Bad regimes detected

**Solution:**
1. Check energy regime (avoid if "plastic")
2. Check liquidity regime (avoid if "frozen")
3. Wait for better market conditions
4. Or override regime checks (not recommended)

### Issue 3: Position Size Always Zero

**Cause:** Kelly criterion insufficient edge

**Solution:**
1. Check historical returns show positive edge
2. Lower `min_edge` parameter
3. Use alternative sizing method (vol target)
4. Ensure `kelly_fraction > 0`

### Issue 4: Monte Carlo Shows Low Win Rate

**Cause:** Stop loss too tight or target too far

**Solution:**
1. Adjust `stop_loss_pct` and `take_profit_pct`
2. Check volatility input is realistic
3. Consider wider stops for volatile assets

### Issue 5: High Execution Costs

**Cause:** Large position size or low liquidity

**Solution:**
1. Reduce position size
2. Trade more liquid instruments
3. Split orders over time
4. Adjust `max_impact_bps` tolerance

---

## Future Enhancements (Post v3.0)

### Week 5-6: Backtest Integration
- Historical simulation with backtrader
- Walk-forward optimization
- Parallel processing with ray/joblib

### UI Features
- Real-time trade idea monitoring
- Interactive Monte Carlo visualization
- Position sizing calculator
- Risk dashboard

### Advanced Features
- Multi-asset portfolio optimization
- Dynamic regime detection
- Adaptive signal weights
- Machine learning edge prediction

---

## References

### Internal Documentation
- [Elasticity Engine v3](ELASTICITY_ENGINE_V3_IMPLEMENTATION.md)
- Liquidity Engine v3 (documentation pending)
- Sentiment Engine v3 (documentation pending)

### Academic References
1. Kelly Criterion: Kelly, J. L. (1956). "A New Interpretation of Information Rate"
2. Geometric Brownian Motion: Black-Scholes-Merton option pricing model
3. Value at Risk: Jorion, P. (2007). "Value at Risk"
4. Position Sizing: Tharp, V. K. (1998). "Trade Your Way to Financial Freedom"

### Code Location
```
engines/composer/universal_policy_composer.py  # Main implementation (39,803 chars)
tests/test_trade_execution_v3.py               # Test suite (21,246 chars)
```

---

## Appendix: Complete Data Model

```python
@dataclass
class TradeIdea:
    """Complete trade idea output."""
    
    # Basic info
    symbol: str
    direction: TradeDirection  # LONG | SHORT | NEUTRAL | AVOID
    
    # Position sizing
    position_size: float
    position_value: float
    sizing_method: PositionSizeMethod
    
    # Entry/exit
    entry_price: float
    entry_price_min: float
    entry_price_max: float
    stop_loss: float
    take_profit: float
    
    # Costs
    expected_slippage_bps: float
    expected_impact_bps: float
    total_expected_cost_bps: float
    
    # Signals
    energy_signal: float           # -1 to 1
    liquidity_signal: float        # -1 to 1
    sentiment_signal: float        # -1 to 1
    composite_signal: float        # -1 to 1
    
    # Risk
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    kelly_fraction: float
    
    # Regimes
    energy_regime: str
    liquidity_regime: str
    sentiment_regime: str
    regime_consistency: float      # 0 to 1
    
    # Monte Carlo
    mc_win_rate: float
    mc_avg_pnl: float
    mc_sharpe: float
    mc_var_95: float
    
    # Validation
    is_valid: bool
    validation_warnings: List[str]
    validation_errors: List[str]
    
    # Meta
    timestamp: datetime
    confidence: float              # 0 to 1
```

---

**Status:** ✅ **COMPLETE**  
**Tests:** 28/28 passing (100%)  
**Performance:** <200ms per trade idea  
**Documentation:** Complete

🎉 **WEEK 4 - TRADE + EXECUTION V3 COMPLETE!**
