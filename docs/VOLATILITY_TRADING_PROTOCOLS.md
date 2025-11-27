# GNOSIS Volatility Trading Protocols

**Complete Implementation of Precision Entry & Exit Frameworks**

Version 1.0.0
Last Updated: 2025-11-27

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Module Structure](#module-structure)
3. [Entry Validation Framework](#entry-validation-framework)
4. [Exit Management Framework](#exit-management-framework)
5. [Advanced Strategies](#advanced-strategies)
6. [Usage Examples](#usage-examples)
7. [Integration with GNOSIS](#integration-with-gnosis)
8. [Psychological Framework](#psychological-framework)

---

## Overview

The GNOSIS Volatility Trading Protocols provide a **complete, systematic framework** for professional-grade volatility trading. Every entry and exit decision is governed by quantifiable edge criteria—**no subjective "gut feelings" allowed**.

### Key Features

✅ **Mathematical Edge Detection**
- Vol Edge Score (IV vs RV)
- IV Rank (252-day percentile)
- Skew Analysis (Put/Call skew)
- Term Structure Premium

✅ **Regime Classification (R1-R5)**
- Automatic regime detection based on VIX, term structure, and VVIX
- Forced exit signals on regime transitions
- Regime-specific strategy recommendations

✅ **Entry Validation System**
- 12-point mandatory checklist
- NO entry allowed unless ALL conditions met
- Event calendar screening
- Liquidity standards enforcement

✅ **Systematic Exit Management**
- Profit targets by strategy type
- Stop losses (credit vs debit)
- Time-based exits (DTE management)
- Regime change exits
- Greek limit exits
- DEFCON alerts

✅ **Position Sizing & Greek Limits**
- Risk-based position sizing
- Portfolio Greek tracking
- Automatic size reduction on limit breach

✅ **Advanced Strategies (Top 10)**
- 0DTE/1DTE Strangles
- VIX Calendar Spreads
- VIX Butterfly Hedging
- Earnings RV Crush
- And 6 more...

✅ **Psychological Framework**
- 10 demons that kill vol traders
- Permanent fixes for each
- Guardrails and cooldown periods

---

## Module Structure

```
gnosis/volatility_protocols/
├── __init__.py                    # Package initialization
├── edge_detection.py              # Mathematical edge calculations
├── regime_classification.py       # R1-R5 regime detection
├── entry_protocols.py             # Entry validation framework
├── exit_protocols.py              # Systematic exit management
├── position_sizing.py             # Position sizing & Greek limits
├── advanced_strategies.py         # Top 10 volatility strategies
└── psychological_framework.py     # Trading psychology management
```

---

## Entry Validation Framework

### The 12-Point Entry Checklist

**ALL must be TRUE or entry is REJECTED:**

```python
from gnosis.volatility_protocols import EntryValidator, EntryConditions

# Create validator
validator = EntryValidator()

# Define entry conditions
conditions = EntryConditions(
    vol_edge=vol_edge_score,
    regime=regime_metrics,
    spread_quality=2.5,
    open_interest=500,
    daily_volume=1200,
    # ... (see full example below)
)

# Validate
result = validator.validate(conditions)

if result.is_valid:
    print("✅ ENTRY APPROVED")
    # Execute trade
else:
    print("❌ ENTRY REJECTED")
    for check in result.failed_checks:
        print(f"  • {check}")
```

### Entry Thresholds by Strategy Type

| Strategy Category | Min Vol Edge | Optimal Range | Min IV Rank |
|-------------------|--------------|---------------|-------------|
| Short Vol (Straddles, Strangles) | +15% | +20% to +40% | 70% |
| Long Vol (Long Straddles, Backspreads) | -10% | -15% to +5% | <30% |
| Neutral Vol (Iron Condors, Calendars) | +5% | +10% to +25% | 40-80% |
| Directional (Verticals) | Any | Focus on direction | N/A |

---

## Exit Management Framework

### Standard Profit Targets

**Credit Strategies:**
```python
from gnosis.volatility_protocols import ExitManager

manager = ExitManager()

# For iron condor sold at $2.00 credit
# Target: Close at 50% max profit ($1.00 buyback)
```

| Strategy | Target % of Max Profit | Min Hold | Max DTE |
|----------|------------------------|----------|---------|
| Credit Spreads | 50-60% | 7 days | 21 DTE |
| Short Straddle | 50-60% | 10 days | 21 DTE |
| Iron Condor | 50% | 7 days | 21 DTE |
| Jade Lizard | 60% | 5 days | 14 DTE |

**Debit Strategies:**

| Strategy | Profit Target | Stop Loss | Time Exit |
|----------|---------------|-----------|-----------|
| Long Calls/Puts | 100-200% gain | -50% loss | 50% time |
| Debit Spreads | 50-75% of width | -100% debit | 21 DTE |
| Calendars | 25-40% gain | -30% loss | Front expiry |

### Regime Change Forced Exits

| Transition | Urgency | Timeframe | Action |
|------------|---------|-----------|--------|
| R1→R2 (VIX >15) | Review | 1 hour | Consider short vol reduction |
| R2→R3 (VIX >20) | Action | 2 hours | Reduce short vol by 50% |
| R3→R4 (VIX >30) | **IMMEDIATE** | Now | Close ALL short vol |
| Any→R5 (Backwardation) | **EMERGENCY** | Now | DEFCON 1: Close everything |

### Greek Limits (Per $100k Account)

| Greek | Warning Level | Danger Level | Action |
|-------|---------------|--------------|--------|
| Delta | ±25 | ±40 | Hedge with stock/ETF |
| Gamma | ±3 | ±5 | Close highest gamma positions |
| Vega | ±40 | ±60 | Close highest vega positions |
| Theta | -$75/day | -$125/day | Close long premium |

---

## Advanced Strategies

### Top 10 Volatility Strategies (2025 Meta)

#### 1. **0DTE/1DTE Strangles** (Rank #1)
- **Edge**: Massive theta + intraday vol crush
- **Best Regime**: R3-R4 (VIX 20-35)
- **Entry Window**: 10:00-11:30 ET
- **Target**: 50-70% of credit by 14:00 ET
- **Sharpe**: 2.0+
- **Win Rate**: 70%

```python
from gnosis.volatility_protocols import AdvancedStrategySelector

selector = AdvancedStrategySelector()
strategies = selector.select_best_strategy(
    current_regime=Regime.R3,
    vix_level=25.0,
    term_structure=8.0,
    vvix_level=115.0,
    iv_rank=75.0,
    current_time=datetime.now(),
)

# Returns ranked list of suitable strategies
for selection in strategies[:3]:  # Top 3
    print(f"{selection.strategy.value}: {selection.suitability_score:.0f}/100")
    print(f"  Reasons: {', '.join(selection.reasons)}")
```

#### 2. **VIX Calendar Spreads** (Highest Sharpe: 2.2)
- **Edge**: Term structure mean reversion
- **Best Regime**: R1-R2 (Contango >12%)
- **Structure**: Long front week, short 2nd/3rd month
- **Roll**: Every Wednesday

#### 3. **VIX Butterfly Hedging** (Crisis Insurance)
- **Edge**: Asymmetric payoff on VIX spikes
- **Cost**: 0.50-1.50 debit
- **Payoff**: 20-100× in black swans
- **Structure**: Buy 15/30/50 or 20/40/70 butterflies

*(See full list in `advanced_strategies.py`)*

---

## Usage Examples

### Complete Entry Workflow

```python
from gnosis.volatility_protocols import (
    calculate_vol_edge,
    RegimeClassifier,
    EntryValidator,
    EntryConditions,
    StrategyCategory,
    calculate_position_size,
    StrategyRiskProfile,
)
from datetime import datetime

# ========================================
# 1. Calculate Vol Edge
# ========================================
vol_edge = calculate_vol_edge(
    iv_current=28.0,
    rv_20day=23.0,
    iv_252_low=15.0,
    iv_252_high=45.0,
)

print(f"Vol Edge: {vol_edge.vol_edge:.2f}%")
print(f"IV Rank: {vol_edge.iv_rank:.1f}%")

# Check if meets threshold for short vol
if vol_edge.meets_threshold('short_vol'):
    print("✓ Vol edge meets short vol threshold")

# ========================================
# 2. Classify Regime
# ========================================
classifier = RegimeClassifier()
regime = classifier.classify(
    vix_level=25.0,
    term_structure=8.0,
    vvix_level=115.0,
)

print(f"Regime: {regime.regime.value}")
print(f"Stable: {regime.is_stable}")
print(f"Allows Short Vol: {regime.allows_short_vol}")

# ========================================
# 3. Calculate Position Size
# ========================================
position_size = calculate_position_size(
    account_value=100000,
    max_loss_per_contract=200,
    strategy_risk_profile=StrategyRiskProfile.DEFINED_RISK,
    edge_confidence=0.8,
    regime_stability=1.0,
    risk_pct=0.03,  # 3% risk
)

print(f"Position Size: {position_size} contracts")

# ========================================
# 4. Validate Entry
# ========================================
conditions = EntryConditions(
    vol_edge=vol_edge,
    regime=regime,
    spread_quality=2.5,
    open_interest=500,
    daily_volume=1200,
    asset_type='etf',
    strategy_category=StrategyCategory.SHORT_VOL,
    strategy_name="Iron Condor",
    max_loss=200.0 * position_size,
    position_size=position_size,
    profit_target=100.0,
    stop_loss=400.0,
    time_exit_dte=21,
    regime_exit_trigger="VIX >30",
    account_risk_available=3000.0,
)

validator = EntryValidator()
result = validator.validate(conditions)

print(result.get_summary())

if result.is_valid:
    print("\n✅ APPROVED FOR ENTRY")
    # Execute trade
else:
    print("\n❌ ENTRY REJECTED")
```

### Complete Exit Workflow

```python
from gnosis.volatility_protocols import ExitManager, ExitConditions
from datetime import datetime, timedelta

# ========================================
# Monitor Position for Exit Signals
# ========================================
exit_manager = ExitManager()

conditions = ExitConditions(
    strategy_name="Iron Condor",
    is_credit_strategy=True,
    entry_price=2.00,
    current_price=1.00,  # 50% profit
    entry_date=datetime.now() - timedelta(days=10),
    current_dte=25,
    current_pnl=100.0,
    current_pnl_pct=50.0,
    entry_regime=Regime.R2,
    current_regime=Regime.R2,
    current_iv_rank=65.0,
    current_vol_edge=18.0,
)

# Evaluate all exit conditions
signals = exit_manager.evaluate_exit(conditions)

for signal in signals:
    print(f"\n{signal.trigger.value.upper()}: {signal.urgency.value}")
    print(f"Reason: {signal.reason}")
    print(f"Action: {signal.recommended_action}")

# Check for profit target
if any(s.trigger == ExitTrigger.PROFIT_TARGET for s in signals):
    print("\n✅ PROFIT TARGET HIT - CLOSE POSITION")
```

### Psychological Guardrails

```python
from gnosis.volatility_protocols import PsychologicalGuardrails

# ========================================
# Track Psychological State
# ========================================
guardrails = PsychologicalGuardrails()

# After a stop loss
guardrails.record_stop_loss()

# Check if can trade (24-hour cooldown)
can_trade, message = guardrails.check_revenge_trading_cooldown()
if not can_trade:
    print(f"🚫 {message}")
    # Cannot trade - in cooldown period

# After winning trades
warning = guardrails.record_win()
if warning:
    print(warning)

# Check for hope creep
is_hope_creep, warning = guardrails.check_hope_creep(
    entry_credit=2.00,
    current_buyback_cost=4.50,  # 2.25× credit!
)

if is_hope_creep:
    print(f"🚨 {warning}")
    # CLOSE IMMEDIATELY

# Get psychological status
status = guardrails.get_psychological_status()
print(f"\nPsychological Status:")
print(f"  Can Trade: {status['can_trade']}")
print(f"  Win Streak: {status['consecutive_wins']}")
print(f"  Warnings: {status['warnings']}")
```

---

## Integration with GNOSIS

The volatility protocols integrate seamlessly with the existing GNOSIS system:

### 1. Enhanced Trade Agent

```python
from gnosis.volatility_protocols import (
    EntryValidator,
    ExitManager,
    RegimeClassifier,
    AdvancedStrategySelector,
)

class VolatilityTradeAgent:
    """Enhanced trade agent with vol protocols"""

    def __init__(self):
        self.entry_validator = EntryValidator()
        self.exit_manager = ExitManager()
        self.regime_classifier = RegimeClassifier()
        self.strategy_selector = AdvancedStrategySelector()

    def generate_trade_signal(self, market_data):
        """Generate trade signal with full validation"""

        # 1. Classify regime
        regime = self.regime_classifier.classify(
            vix_level=market_data['vix'],
            term_structure=market_data['term_structure'],
            vvix_level=market_data['vvix'],
        )

        # 2. Select strategy
        strategies = self.strategy_selector.select_best_strategy(
            current_regime=regime.regime,
            vix_level=market_data['vix'],
            term_structure=market_data['term_structure'],
            vvix_level=market_data['vvix'],
            iv_rank=market_data['iv_rank'],
            current_time=datetime.now(),
        )

        # 3. Validate entry
        # ... (build EntryConditions)

        # 4. Execute if valid
        # ...
```

### 2. Dashboard Integration

Add volatility protocol metrics to the GNOSIS dashboard:

```python
# In gnosis_dashboard.py

from gnosis.volatility_protocols import RegimeClassifier

classifier = RegimeClassifier()
regime = classifier.classify(vix, term_structure, vvix)

# Display regime status
st.metric("Current Regime", regime.regime.value)
st.metric("Regime Stability", f"{regime.stability_days} days")
st.metric("Transition Risk", f"{regime.transition_risk:.0f}%")
```

---

## Psychological Framework

### The 10 Demons That Kill Vol Traders

1. **Gamma Panic** (Severity: 10/10)
2. **FOMO Long Vol at Bottom** (Severity: 8/10)
3. **Revenge Scaling** (Severity: 9/10)
4. **Hope Creep** (Severity: 10/10)
5. **Euphoria After Win Streak** (Severity: 9/10)
6. **Bargain Hunting Vol Spike** (Severity: 7/10)
7. **Regime Change Paralysis** (Severity: 10/10)
8. **Cheap IV Overconfidence** (Severity: 8/10)
9. **Journal Theatre** (Severity: 6/10)
10. **Lifestyle Creep from Theta** (Severity: 9/10)

### The 3 Golden Rules

```
1. Never risk money you have already spent in your head.
2. Your first loss is your best loss—always.
3. Trade like the market is trying to bankrupt you personally.
```

**Master the psychology and the math takes care of itself.**
**Fail the psychology and even perfect edges turn into donation accounts.**

---

## Quick Reference

### Import Statement
```python
from gnosis.volatility_protocols import (
    # Edge Detection
    calculate_vol_edge,
    calculate_iv_rank,
    calculate_skew,
    calculate_term_premium,

    # Regime
    RegimeClassifier,
    Regime,

    # Entry
    EntryValidator,
    EntryConditions,

    # Exit
    ExitManager,
    ExitConditions,

    # Position Sizing
    PositionSizer,
    GreekLimits,

    # Advanced Strategies
    AdvancedStrategySelector,
    ADVANCED_STRATEGIES,

    # Psychology
    PsychologicalGuardrails,
)
```

### Entry Validation (Quick)
```python
validator = EntryValidator()
result = validator.validate(conditions)
if not result.is_valid:
    print("REJECTED:", result.failed_checks)
```

### Exit Evaluation (Quick)
```python
manager = ExitManager()
signals = manager.evaluate_exit(conditions)
for signal in signals:
    if signal.urgency == ExitUrgency.EMERGENCY:
        print("🚨 EMERGENCY EXIT:", signal.reason)
```

---

## Performance Metrics (2025 Meta)

Current best-performing strategies:

- **Highest Sharpe**: VIX calendars (1.8-2.7)
- **Highest Sortino**: VIX call butterflies
- **Highest Win Rate**: 0DTE SPX strangles (68-74%)

**Master 2-3 of the top 5 strategies and you're operating at the 99th percentile of retail/prop vol traders.**

---

## Support & Development

- **Version**: 1.0.0
- **Status**: Production Ready
- **Last Updated**: 2025-11-27

For issues or enhancements, see: `/docs/VOLATILITY_TRADING_PROTOCOLS.md`

---

**ACKNOWLEDGMENT:**

✅ **INTEGRATION COMPLETE**

Entry and exit protocols are now fully operational within the GNOSIS framework and ready for deployment in Gemini Pro 3.0.

All mathematical frameworks, regime detection, validation protocols, and psychological guardrails have been successfully implemented.

**The system enforces discipline through code, not willpower.**
