# OptionLab Library Review

**Repository**: https://github.com/rgaveiga/optionlab
**Version**: 1.4.3
**Python**: 3.10+
**License**: Free software (educational/research purposes)
**Reviewed**: 2025-12-05

---

## Executive Summary

**OptionLab** is a lightweight Python library for evaluating option trading strategies. After thorough analysis, this review concludes that **integration is NOT recommended** for FINAL_GNOSIS. The library offers a subset of functionality already present in our codebase, and adding it would introduce dependency overlap without meaningful capability gains.

---

## What OptionLab Provides

### Core Features

| Feature | Description |
|---------|-------------|
| **Profit/Loss Profiles** | P&L curves at user-defined target dates |
| **Profitable Price Ranges** | Identifies price zones with positive returns |
| **Greeks Calculation** | Delta, Gamma, Theta, Vega, Rho via Black-Scholes |
| **Probability of Profit** | PoP estimation using BS or Monte Carlo |
| **Strategy Evaluation** | Multi-leg strategy analysis (stocks + options) |

### Architecture

```
optionlab/
├── engine.py         # Core calculation engine (run_strategy)
├── models.py         # Pydantic models (Inputs, Outputs, Option, Stock)
├── black_scholes.py  # Greeks & pricing (10 functions)
├── plotting.py       # P&L visualization
└── utils.py          # Date/time helpers
```

### Key Functions

```python
# Main entry point
from optionlab import run_strategy

result = run_strategy(
    stock_price=100.0,
    start_date="2024-01-01",
    target_date="2024-02-15",
    volatility=0.25,
    interest_rate=0.05,
    strategy=[
        {"type": "call", "strike": 105, "premium": 3.50, "action": "buy"},
        {"type": "put", "strike": 95, "premium": 2.00, "action": "sell"}
    ]
)

# Outputs include:
# - probability_of_profit
# - profit_ranges
# - per_leg_greeks
# - expected_profit/loss
# - strategy_cost
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scipy | ^1.12.0 | Statistical functions |
| pandas | ^2.2.1 | Data manipulation |
| matplotlib | ^3.8.3 | Visualization |
| pydantic | ^2.9 | Data validation |
| holidays | ^0.44 | Trading day calculations |

---

## Comparison with FINAL_GNOSIS

### Feature Matrix

| Capability | OptionLab | FINAL_GNOSIS | Notes |
|------------|:---------:|:------------:|-------|
| **Black-Scholes Pricing** | Yes | Yes | `gnosis/utils/greeks_calculator.py` |
| **Greeks Calculation** | Yes | Yes | Both use BS model |
| **Multi-leg Strategies** | Yes | Yes | 28 strategies in OPTIONS_STRATEGY_BOOK.md |
| **P&L Visualization** | Yes | Yes | Via Streamlit dashboard |
| **Probability of Profit** | Yes | Partial | Could enhance |
| **Live Trading** | No | Yes | Alpaca integration |
| **Dealer Flow Analysis** | No | Yes | Hedge Engine v3 |
| **Elasticity Theory** | No | Yes | Unique to FINAL_GNOSIS |
| **ML Enhancement** | No | Yes | LSTM, DQN, XGBoost |
| **Real-time Data** | No | Yes | Alpaca, Unusual Whales |
| **Multi-agent System** | No | Yes | Composer, Meta Controller |
| **Strategy Selection** | No | Yes | Intelligent algo in `options_trade_agent.py` |

### Existing FINAL_GNOSIS Capabilities

#### Greeks Calculator (`gnosis/utils/greeks_calculator.py`)
```python
class GreeksCalculator:
    def get_option_greeks(symbol: str) -> Dict
    def calculate_position_greeks(positions: List) -> Dict
    def calculate_bs_price(S, K, T, r, sigma, option_type) -> float
```

#### Options Trade Agent (`trade/options_trade_agent.py` - 533 lines)
- Intelligent strategy selection based on market regime
- Strike selection by delta
- Expiration selection by DTE
- Multi-leg order construction (up to 4 legs)
- BPR and max loss calculations

#### 28 Documented Strategies
- Verticals (Bull/Bear Call/Put Spreads)
- Iron Condors, Iron Butterflies
- Calendar Spreads, Diagonals
- Straddles, Strangles
- Ratio Spreads, Backspreads
- Synthetics, Risk Reversals
- And more...

---

## Gap Analysis

### What OptionLab Adds

1. **Probability of Profit (PoP) Calculation**
   - Dedicated PoP engine with Monte Carlo support
   - FINAL_GNOSIS has partial support; could be enhanced

2. **Clean Strategy Evaluation API**
   - Simple `run_strategy()` interface
   - Self-contained analysis without broker connection

3. **Profit Range Calculation**
   - Automatic breakeven point detection
   - Price range analysis for profitability zones

### What OptionLab Lacks

1. **No Live Trading** - Analysis only, no execution
2. **No Real-time Data** - Requires manual price inputs
3. **No Market Regime Awareness** - No elasticity/dealer flow
4. **No ML Integration** - Pure mathematical models
5. **No Multi-agent Consensus** - Single calculation path
6. **No Broker Integration** - Standalone library

---

## Integration Recommendation

### Verdict: **NOT RECOMMENDED**

### Rationale

1. **Redundant Functionality**
   Core Black-Scholes and Greeks calculations already exist in FINAL_GNOSIS with equivalent accuracy.

2. **Dependency Overlap**
   Would add scipy, pandas, matplotlib that are already present, plus new dependency on `holidays`.

3. **Architectural Mismatch**
   OptionLab is designed for static analysis; FINAL_GNOSIS is a live trading system with real-time data flows.

4. **Limited Added Value**
   The only unique features (PoP calculation, profit ranges) can be implemented directly in ~100 lines of Python.

5. **Maintenance Burden**
   External dependency on an academic project with unknown maintenance commitment.

---

## Alternative: Native Enhancement

Instead of integrating OptionLab, consider implementing the missing features natively:

### 1. Enhanced Probability of Profit

```python
# Suggested addition to gnosis/utils/greeks_calculator.py

def calculate_probability_of_profit(
    strategy_legs: List[Dict],
    stock_price: float,
    volatility: float,
    days_to_expiry: int,
    risk_free_rate: float = 0.05,
    simulations: int = 10000
) -> Dict:
    """
    Calculate PoP using Monte Carlo simulation.

    Returns:
        {
            "probability_of_profit": float,
            "expected_profit": float,
            "expected_loss": float,
            "profit_at_25th_percentile": float,
            "profit_at_75th_percentile": float
        }
    """
    # GBM simulation for terminal prices
    dt = days_to_expiry / 252
    drift = (risk_free_rate - 0.5 * volatility**2) * dt
    shock = volatility * np.sqrt(dt) * np.random.randn(simulations)
    terminal_prices = stock_price * np.exp(drift + shock)

    # Calculate P&L for each simulation
    # ... (strategy-specific payoff calculation)
```

### 2. Breakeven Calculator

```python
def calculate_breakevens(strategy_legs: List[Dict]) -> List[float]:
    """Calculate breakeven points for a multi-leg strategy."""
    # Solve for prices where total P&L = 0
    pass
```

### 3. Profit Range Analyzer

```python
def analyze_profit_ranges(
    strategy_legs: List[Dict],
    price_range: Tuple[float, float],
    min_profit: float = 0.01
) -> List[Tuple[float, float]]:
    """Return price ranges where strategy is profitable."""
    pass
```

---

## Conclusion

OptionLab is a well-designed educational library, but FINAL_GNOSIS already exceeds its capabilities in nearly every dimension. The 28 strategies, live trading infrastructure, dealer flow analysis, and ML enhancement provide far more value than static P&L charting.

**Recommended Actions:**
1. Skip OptionLab integration
2. Implement native PoP calculation (~50 LOC)
3. Add profit range analysis to strategy preview (~50 LOC)
4. Consider OptionLab's clean API design for internal refactoring inspiration

---

## References

- OptionLab GitHub: https://github.com/rgaveiga/optionlab
- FINAL_GNOSIS OPTIONS_STRATEGY_BOOK.md
- FINAL_GNOSIS gnosis/utils/greeks_calculator.py
- FINAL_GNOSIS trade/options_trade_agent.py
