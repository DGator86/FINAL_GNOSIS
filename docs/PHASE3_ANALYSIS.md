# Phase 3: Paper Trading Analysis Tools

## Overview

Phase 3 provides a comprehensive suite of analysis tools for reviewing accumulated paper trading data. These tools enable systematic validation of strategy selection, regime adaptation, risk management, and exit rule configuration before transitioning to live trading.

**Purpose**: Transform raw JSONL logs into actionable insights for paper trading validation.

**Tools**: 3 analysis scripts with 80-character formatted reports

**Timeline**: Daily review for 2-4 weeks before live deployment

---

## Analysis Tools

### 1. Daily Report Generator (`scripts/analysis/daily_report.py`)

**Purpose**: High-level daily summary of all paper trading activity

**What It Analyzes**:
- Strategy distribution across all trades
- Regime correlation (which strategies are selected in which regimes)
- Risk metrics (max loss, max profit, breakevens)
- Exit rule configuration
- Execution quality (fill rates, slippage, commissions)

**Usage**:
```bash
# Default: Last 30 days, print to stdout
python scripts/analysis/daily_report.py

# Last 7 days
python scripts/analysis/daily_report.py --days 7

# Save to file
python scripts/analysis/daily_report.py --output reports/daily_report.txt

# Custom runs directory
python scripts/analysis/daily_report.py --runs-dir /path/to/runs --days 14
```

**Example Output**:
```
================================================================================
SPY PAPER TRADING REPORT
Analysis Period: 2025-11-15 to 2025-11-16 (1 days)
Generated: 2025-11-16T22:09:27.172113
================================================================================

SUMMARY STATISTICS
--------------------------------------------------------------------------------
Total Trading Days: 1
Total Contexts: 3
Total Trade Ideas: 18
Total Orders: 10

STRATEGY DISTRIBUTION
--------------------------------------------------------------------------------
  call_debit_spread                  3 ( 16.7%)
  long_call                          3 ( 16.7%)
  diagonal_spread                    3 ( 16.7%)
  stock                              3 ( 16.7%)
  broken_wing_butterfly              3 ( 16.7%)
  synthetic_long                     3 ( 16.7%)

REGIME CORRELATION
--------------------------------------------------------------------------------
  mid_vol_bullish:
    call_debit_spread                1
    long_call                        1
    diagonal_spread                  1

RISK METRICS
--------------------------------------------------------------------------------
  Trade Count:                 18
  Max Loss (Mean):      $ 2600.76
  Max Loss (Min):       $  426.91
  Max Loss (Max):       $ 9000.00
  Max Profit (Mean):    $ 1187.50
  Max Profit (Min):     $  875.00
  Max Profit (Max):     $ 1875.00
  Breakeven Points:            15

EXIT RULE CONFIGURATION
--------------------------------------------------------------------------------
  Profit Target (Mean):     60.0%
  Stop Loss (Mean):         48.6%
  Trailing Stop (Mean):     15.0%

EXECUTION QUALITY
--------------------------------------------------------------------------------
  Total Orders:                10
  Filled Orders:               10
  Rejected Orders:              0
  Fill Rate:               100.0%
  Avg Slippage:         $    0.38
  Total Commission:     $    7.15

================================================================================
```

**When To Use**: Daily or weekly for quick health check

---

### 2. Regime Correlation Analyzer (`scripts/analysis/regime_correlation.py`)

**Purpose**: Deep dive into regime → strategy → outcome mappings

**What It Analyzes**:
- Volatility regime distribution (low_vol, mid_vol, high_vol)
- Directional bias distribution (bullish, bearish, neutral)
- Confidence and elastic energy statistics
- Strategy selection grouped by regime key
- Risk profile (max loss, max profit, Greeks) grouped by regime
- Confidence impact on strategy selection and risk sizing

**Usage**:
```bash
# Default: Last 30 days
python scripts/analysis/regime_correlation.py

# Last 14 days
python scripts/analysis/regime_correlation.py --days 14

# Save to file
python scripts/analysis/regime_correlation.py --output regime_analysis.txt
```

**Example Output**:
```
================================================================================
REGIME CORRELATION ANALYSIS
Analysis Period: 2025-11-15 to 2025-11-16 (1 days)
Generated: 2025-11-16T22:11:01.210555
================================================================================

REGIME DIMENSIONS
--------------------------------------------------------------------------------
Volatility Regimes:
  mid_vol                3 (100.0%)

Directional Bias:
  bullish                3 (100.0%)

Confidence Statistics:
  Mean: 0.700
  Min:  0.700
  Max:  0.700

Elastic Energy Statistics:
  Mean: 1.000
  Min:  1.000
  Max:  1.000

STRATEGY SELECTION BY REGIME
--------------------------------------------------------------------------------
mid_vol_bullish:
  call_debit_spread                1
  long_call                        1
  diagonal_spread                  1

RISK PROFILE BY REGIME
--------------------------------------------------------------------------------
mid_vol_bullish:
  Trade Count:         3
  Avg Max Loss:      $  1011.51
  Avg Max Profit:    $  1333.33
  Avg Delta:            0.000
  Avg Vega:             0.000

CONFIDENCE IMPACT ANALYSIS
--------------------------------------------------------------------------------
Medium Confidence:
  Trade Count:         3
  Avg Max Loss:      $  1011.51
  Avg Max Profit:    $  1333.33
  Strategy Distribution:
    call_debit_spread              1
    long_call                      1
    diagonal_spread                1

================================================================================
```

**When To Use**: Weekly for validating regime adaptation logic

**Key Questions It Answers**:
- Are high-vol regimes selecting defensive strategies (spreads, butterflies)?
- Are bullish regimes preferring directional long calls/synthetics?
- Does confidence correlate with risk sizing (higher confidence → larger positions)?
- Are strategies consistent within the same regime bucket?

---

### 3. Exit Trigger Tracker (`scripts/analysis/exit_tracker.py`)

**Purpose**: Analyze exit rule configuration and simulate trigger behavior

**What It Analyzes**:
- Exit rule parameter distribution (profit target, stop loss, trailing stop, time stop, Greek stop)
- Exit rule completeness (which trades have which rules configured)
- Simulated trigger analysis (which triggers would fire under different scenarios)
- Exit rule configuration grouped by strategy type

**Usage**:
```bash
# Default: Last 30 days
python scripts/analysis/exit_tracker.py

# Last 7 days
python scripts/analysis/exit_tracker.py --days 7

# Save to file
python scripts/analysis/exit_tracker.py --output exit_analysis.txt
```

**Example Output**:
```
================================================================================
EXIT TRIGGER ANALYSIS
Analysis Period: 2025-11-15 to 2025-11-16 (1 days)
Generated: 2025-11-16T22:11:05.096101
================================================================================

EXIT RULE DISTRIBUTION
--------------------------------------------------------------------------------
PROFIT TARGET:
  Configured in:   0 trades

STOP LOSS:
  Configured in:   0 trades

TRAILING STOP:
  Configured in:   0 trades

TIME STOP:
  Configured in:   0 trades

GREEK STOP:
  Configured in:   0 trades

EXIT RULE COMPLETENESS
--------------------------------------------------------------------------------
Total Trades Analyzed: 18

Rule Presence:

Rule Combinations:
  none                                                18 (100.0%)

SIMULATED TRIGGER ANALYSIS
--------------------------------------------------------------------------------
This section simulates which exit triggers would fire under different
market scenarios to validate exit management robustness.

FAST PROFIT SCENARIO:
  no_trigger            18 (100.0% of all trades)

SLOW PROFIT SCENARIO:
  no_trigger            18 (100.0% of all trades)

FAST LOSS SCENARIO:
  no_trigger            18 (100.0% of all trades)

SLOW LOSS SCENARIO:
  no_trigger            18 (100.0% of all trades)

CHOPPY SCENARIO:
  no_trigger            18 (100.0% of all trades)

EXIT RULES BY STRATEGY
--------------------------------------------------------------------------------
broken_wing_butterfly:
  Trade Count:         3

call_debit_spread:
  Trade Count:         3

diagonal_spread:
  Trade Count:         3

long_call:
  Trade Count:         3

stock:
  Trade Count:         3

synthetic_long:
  Trade Count:         3

================================================================================
```

**When To Use**: Weekly for validating exit management robustness

**Key Questions It Answers**:
- Are all trades configured with complete exit rules (profit target, stop loss, time stop)?
- Which exit triggers fire most frequently in different market scenarios?
- Are exit rules consistent across strategy types?
- Are there any trades missing critical exit rules (risk management gaps)?

---

## Workflow Integration

### Daily Operations

1. **Morning**: Run daily SPY paper trading script
   ```bash
   python scripts/run_daily_spy_paper.py --execute --capital 100000
   ```

2. **Evening**: Review daily report
   ```bash
   python scripts/analysis/daily_report.py --days 1
   ```

3. **Weekly**: Deep dive with regime and exit analysis
   ```bash
   python scripts/analysis/regime_correlation.py --days 7 --output reports/weekly_regime.txt
   python scripts/analysis/exit_tracker.py --days 7 --output reports/weekly_exits.txt
   ```

### Review Checklist

**Daily Review** (5 minutes):
- [ ] Check fill rate (should be ~100% in paper mode)
- [ ] Verify commission and slippage are reasonable
- [ ] Confirm strategy distribution aligns with market conditions
- [ ] Spot check risk metrics (max loss within 2% risk cap)

**Weekly Review** (30 minutes):
- [ ] Analyze regime correlation: Are strategies adapting to regimes correctly?
- [ ] Review exit rule completeness: Any trades missing exit rules?
- [ ] Check simulated exit triggers: Are exits too tight or too loose?
- [ ] Validate confidence impact: Does higher confidence lead to larger positions?

**Monthly Review** (2 hours):
- [ ] Comprehensive report for all 30 days
- [ ] Identify any systemic issues (e.g., one regime always underperforming)
- [ ] Document edge cases and unexpected behavior
- [ ] Prepare go/no-go decision for live trading

---

## Data Flow

```
Daily SPY Pipeline (scripts/run_daily_spy_paper.py)
    ↓
JSONL Output (runs/YYYY-MM-DD/)
    ├── SPY_context.jsonl      (ComposerTradeContext)
    ├── SPY_trades.jsonl       (TradeIdea[])
    └── SPY_orders.jsonl       (OrderResult[])
    ↓
Analysis Tools (scripts/analysis/)
    ├── daily_report.py        (High-level summary)
    ├── regime_correlation.py  (Regime → strategy mapping)
    └── exit_tracker.py        (Exit rule validation)
    ↓
Reports (reports/ or stdout)
    └── 80-character formatted text reports
```

---

## Expected Behavior

### Regime Adaptation

**High Vol + Bullish**:
- Expect: Defensive bullish strategies (bull call spreads, diagonal spreads)
- Risk: Lower max loss, capped max profit
- Greeks: Lower delta, lower vega (hedged volatility exposure)

**Mid Vol + Bullish**:
- Expect: Balanced directional strategies (long calls, synthetics, stock)
- Risk: Moderate max loss, moderate max profit
- Greeks: Moderate delta, moderate vega

**Low Vol + Bullish**:
- Expect: Aggressive directional strategies (long calls, synthetic long, stock)
- Risk: Higher max loss, higher max profit
- Greeks: Higher delta, higher vega

**High Vol + Bearish**:
- Expect: Defensive bearish strategies (bear put spreads, diagonal puts)
- Risk: Lower max loss, capped max profit
- Greeks: Negative delta, lower vega

**Mid Vol + Bearish**:
- Expect: Balanced bearish strategies (long puts, synthetic short)
- Risk: Moderate max loss, moderate max profit
- Greeks: Negative delta, moderate vega

**Low Vol + Bearish**:
- Expect: Aggressive bearish strategies (long puts, synthetic short)
- Risk: Higher max loss, higher max profit
- Greeks: Negative delta, higher vega

**Neutral (Any Vol)**:
- Expect: Non-directional strategies (iron condor, butterfly, straddle)
- Risk: Defined max loss, defined max profit
- Greeks: Near-zero delta, positive vega (long volatility)

### Exit Rule Coverage

**Expected Coverage** (after exit rules are implemented):
- Profit Target: 100% of trades
- Stop Loss: 100% of trades
- Trailing Stop: 80%+ of trades (optional for short-duration trades)
- Time Stop: 100% of trades (critical for theta decay management)
- Greek Stop: 50%+ of trades (optional for defined-risk strategies)

**Red Flags**:
- Any trade missing profit target or stop loss (risk management gap)
- Time stops > 30 days (excessive theta decay risk)
- No trailing stops on profitable trades (leaving money on the table)

---

## Troubleshooting

### Issue: No data files found

**Problem**: Analysis tools report "No data available for analysis"

**Solutions**:
1. Verify runs directory exists: `ls -la runs/`
2. Check for JSONL files: `ls -la runs/2025-11-16/`
3. Confirm pipeline execution: `python scripts/run_daily_spy_paper.py --execute`

### Issue: Datetime serialization errors

**Problem**: `TypeError: Object of type datetime is not JSON serializable`

**Solutions**:
1. Ensure `_serialize_datetime()` helper is used in persistence code
2. Verify all datetime objects are converted to ISO format strings
3. Check for nested datetime objects in dicts/lists

### Issue: Exit rules always empty

**Problem**: Exit tracker reports 0 configured rules

**Solutions**:
1. **Expected in v2.0**: `TradeIdea` model doesn't populate `exit_rules` field yet
2. **Future enhancement**: `TradeAgentV2` should populate `exit_rules` based on strategy type
3. **Workaround**: Manually add exit rules in post-processing or enhance `TradeAgentV2`

### Issue: All regimes are the same

**Problem**: Regime correlation shows only one regime bucket

**Solutions**:
1. **Expected in initial testing**: Limited data diversity in first few runs
2. **Solution**: Run pipeline over multiple days with varying market conditions
3. **Test**: Manually trigger different regimes by adjusting underlying price

---

## Extension Points

### Adding New Analysis Tools

**Example**: Create a P&L simulator that replays trades with actual market data

```python
#!/usr/bin/env python3
"""P&L Simulator for Phase 3 Paper Trading"""

import argparse
from pathlib import Path
from typing import Dict, List, Any

def load_trades(runs_dir: Path, days: int) -> List[Dict]:
    """Load trades from JSONL files."""
    # Implementation similar to other analyzers
    pass

def simulate_pnl(trade: Dict, market_data: Dict) -> float:
    """Simulate P&L using actual market data."""
    # Fetch historical prices for trade legs
    # Compute P&L at various time points
    # Apply exit rules to determine realized P&L
    pass

def generate_report(runs_dir: Path, days: int) -> str:
    """Generate P&L simulation report."""
    trades = load_trades(runs_dir, days)
    total_pnl = 0.0
    
    for trade in trades:
        # Fetch market data for trade date range
        pnl = simulate_pnl(trade, market_data)
        total_pnl += pnl
    
    return f"Total Simulated P&L: ${total_pnl:.2f}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    
    report = generate_report(args.runs_dir, args.days)
    print(report)
```

**Steps**:
1. Create new script in `scripts/analysis/`
2. Follow same structure as existing analyzers (load JSONL, analyze, report)
3. Add to daily/weekly review workflow
4. Document in `PHASE3_ANALYSIS.md`

### Adding Visualization Tools

**Example**: Create matplotlib charts for regime distribution

```python
import matplotlib.pyplot as plt
from collections import Counter

def plot_regime_distribution(contexts: List[Dict], output_path: Path):
    """Plot regime distribution as pie chart."""
    regimes = [f"{ctx['context']['volatility_regime']}_{ctx['context']['direction']}" 
               for ctx in contexts]
    regime_counts = Counter(regimes)
    
    plt.figure(figsize=(10, 6))
    plt.pie(regime_counts.values(), labels=regime_counts.keys(), autopct='%1.1f%%')
    plt.title('Regime Distribution')
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
```

**Integration**: Add `--plot` flag to existing analyzers to generate charts alongside text reports

---

## Next Steps

1. **Run Daily**: Execute SPY paper trading script daily for 2-4 weeks
2. **Review Weekly**: Run all analysis tools weekly to validate behavior
3. **Document Edge Cases**: Note any unexpected strategy selections or exit triggers
4. **Tune Parameters**: Adjust hyperparameters based on analysis insights
5. **Go/No-Go Decision**: After 2-4 weeks, decide whether to proceed to live trading

**Success Criteria for Live Transition**:
- [ ] 100% fill rate in paper trading (simulated execution working)
- [ ] Strategy selection adapts correctly to all regime buckets
- [ ] Exit rules configured for 100% of trades
- [ ] No catastrophic losses (all trades within 2% risk cap)
- [ ] Confidence correlates with risk sizing (higher confidence → larger positions)
- [ ] No systemic bugs or data quality issues

**Rollout Plan**: See `OPERATIONS_RUNBOOK.md` for 4-phase rollout to live trading
