# ICT (Inner Circle Trader) Methodology Implementation

## Overview

This document describes the implementation of ICT trading concepts in the Gnosis trading system. The ICT methodology is implemented in `/home/root/webapp/engines/liquidity/ict_engine.py`.

## Implemented Concepts

### 1. Swing Points & Liquidity Levels

**Swing Point Detection:**
- **Swing High**: Center candle has a lower high to the left AND a lower high to the right
- **Swing Low**: Center candle has a higher low to the left AND a higher low to the right

**Liquidity Classification:**
- **Buy-side Liquidity**: Located above swing highs (stop hunts for shorts, buy stops for breakout longs)
- **Sell-side Liquidity**: Located below swing lows (stop hunts for longs, sell stops for breakdown shorts)

**Equal/Old Highs and Lows:**
- **Equal Highs/Lows**: Multiple swing points clustering at similar price levels (stronger liquidity)
- **Old Highs/Lows**: Isolated swing points (single liquidity target)

```python
from engines.liquidity import ICTEngine, SwingPointDetector

detector = SwingPointDetector({})
swing_highs, swing_lows = detector.detect_swing_points(bars)

# Classify into liquidity levels
buy_side_liquidity, sell_side_liquidity = detector.classify_highs_lows(
    swing_highs, swing_lows, current_price
)
```

### 2. Premium/Discount Zones & OTE

**Zone Classification:**
- **Premium Zone**: Upper 50% of range - look for SHORT entries
- **Discount Zone**: Lower 50% of range - look for LONG entries
- **Equilibrium**: 50% retracement level

**OTE (Optimal Trade Entry):**
- Fibonacci zone from 0.62 to 0.79 retracement
- Sweet spot at 0.705 (midpoint)
- For longs: OTE is in discount zone
- For shorts: OTE is in premium zone

```python
from engines.liquidity import PremiumDiscountCalculator

calc = PremiumDiscountCalculator({})
zones = calc.calculate_zones(swing_high=110.0, swing_low=100.0, is_upward_range=True)

print(f"Equilibrium: {zones.equilibrium}")  # 105.0
print(f"OTE Zone: {zones.ote_low} - {zones.ote_high}")  # 102.1 - 103.8
print(f"Current zone: {zones.get_zone(103)}")  # discount
print(f"In OTE: {zones.is_in_ote(103)}")  # True
```

### 3. Fair Value Gaps (FVG)

**BISI (Bullish FVG):**
- Buy-side Imbalance, Sell-side Inefficiency
- Gap between bar1's high and bar3's low (no overlap)
- Acts as support, price often returns to fill

**SIBI (Bearish FVG):**
- Sell-side Imbalance, Buy-side Inefficiency
- Gap between bar1's low and bar3's high (no overlap)
- Acts as resistance, price often returns to fill

**Consequent Encroachment:**
- Midpoint (50%) of the FVG
- Key reaction level within the gap

**FVG Status Tracking:**
- `UNFILLED`: Gap not yet tested
- `PARTIALLY_FILLED`: Price reached consequent encroachment
- `FILLED`: Price completely filled the gap
- `INVERTED`: Price tested gap from opposite side

```python
from engines.liquidity import FairValueGapDetector, FVGType

detector = FairValueGapDetector({'min_gap_size_pct': 0.001})
bullish_fvgs, bearish_fvgs = detector.detect_fvgs(bars)

for fvg in bullish_fvgs:
    print(f"FVG: {fvg.low} - {fvg.high}")
    print(f"Consequent Encroachment: {fvg.consequent_encroachment}")
    print(f"Status: {fvg.status.value}")
```

### 4. Volume Imbalances

Similar to FVGs but with shadow overlap:
- Gap between open and close of adjacent candles
- Trading activity exists within the gap (shadows overlap)
- Can act as support/resistance similar to FVGs

```python
volume_imbalances = detector.detect_volume_imbalances(bars)
for vi in volume_imbalances:
    print(f"VI: {vi.gap_low} - {vi.gap_high}, Bullish: {vi.is_bullish}")
```

### 5. Order Blocks

**High Probability Order Blocks:**
1. **Bullish HP OB**:
   - Large bearish candle sweeps sell-side liquidity
   - Price then breaks structure upward (breaks a swing high)
   - Order block = opening price of the sweep candle

2. **Bearish HP OB**:
   - Large bullish candle sweeps buy-side liquidity
   - Price then breaks structure downward (breaks a swing low)
   - Order block = opening price of the sweep candle

**Low Probability Order Blocks:**
1. **Bullish LP OB**: Small bearish candle within bullish move
2. **Bearish LP OB**: Small bullish candle within bearish move

**Mean Threshold:**
- 50% Fibonacci retracement of the order block candle
- Alternative entry level to the opening price

```python
from engines.liquidity import OrderBlockDetector

detector = OrderBlockDetector({})
bullish_obs, bearish_obs = detector.detect_order_blocks(bars, swing_highs, swing_lows)

for ob in bullish_obs:
    print(f"OB Type: {ob.type.value}")
    print(f"Zone: {ob.low} - {ob.high}")
    print(f"Entry (open): {ob.open_price}")
    print(f"Mean Threshold: {ob.mean_threshold}")
```

### 6. Daily Bias

**Bullish Bias Conditions:**
- Price broke AND closed above previous day's high
- Price broke previous day's low but FAILED to close below it (reversal signal)

**Bearish Bias Conditions:**
- Price broke AND closed below previous day's low
- Price broke previous day's high but FAILED to close above it (reversal signal)

```python
from engines.liquidity import DailyBiasCalculator, DailyBias

calc = DailyBiasCalculator({})
result = calc.calculate_bias(
    previous_day_high=105.0,
    previous_day_low=100.0,
    current_high=108.0,
    current_low=103.0,
    current_close=107.0
)

print(f"Bias: {result.bias.value}")  # bullish
print(f"Confidence: {result.confidence}")  # 0.8
print(f"Reasoning: {result.reasoning}")
```

### 7. Liquidity Sweeps

Detection of price sweeping liquidity levels:
- **Buy-side Sweep**: Price goes above swing high
- **Sell-side Sweep**: Price goes below swing low

**Reversal Detection:**
- Swept liquidity but failed to hold beyond the level
- Closed back inside range = potential reversal setup

```python
from engines.liquidity import LiquiditySweepDetector

detector = LiquiditySweepDetector({'reversal_bars': 3})
sweeps = detector.detect_sweeps(bars, liquidity_levels)

for sweep in sweeps:
    print(f"Swept: {sweep.liquidity_level.type.value} at {sweep.sweep_price}")
    print(f"Failed to hold: {sweep.failed_to_hold}")
    print(f"Reversal detected: {sweep.reversal_detected}")
```

## Full Engine Usage

```python
from engines.liquidity import ICTEngine
from engines.liquidity.ict_engine import OHLCV

# Initialize engine
engine = ICTEngine({
    'swing': {'cluster_threshold_pct': 0.005},
    'fvg': {'min_gap_size_pct': 0.001},
    'order_blocks': {'lookback': 20}
})

# Run analysis
snapshot = engine.analyze(
    symbol='AAPL',
    bars=intraday_bars,  # List[OHLCV]
    daily_bars=daily_bars,  # Optional for daily bias
    timestamp=datetime.now()
)

# Access results
print(f"Swing Highs: {len(snapshot.swing_highs)}")
print(f"Buy-side Liquidity: {len(snapshot.buy_side_liquidity)}")
print(f"Sell-side Liquidity: {len(snapshot.sell_side_liquidity)}")
print(f"Current Zone: {snapshot.current_zone.value if snapshot.current_zone else 'N/A'}")
print(f"In OTE: {snapshot.in_ote}")
print(f"Bullish FVGs: {len(snapshot.bullish_fvgs)}")
print(f"Bearish FVGs: {len(snapshot.bearish_fvgs)}")
print(f"Bullish Order Blocks: {len(snapshot.bullish_order_blocks)}")
print(f"Bearish Order Blocks: {len(snapshot.bearish_order_blocks)}")
print(f"Daily Bias: {snapshot.daily_bias.bias.value if snapshot.daily_bias else 'N/A'}")
print(f"Entry Signal: {snapshot.entry_signal}")
print(f"Entry Confidence: {snapshot.entry_confidence:.0%}")

# Get specific data
state = engine.get_state('AAPL')
signal, confidence = engine.get_signal('AAPL')
bias = engine.get_daily_bias('AAPL')
nearest_fvg = engine.get_nearest_fvg('AAPL')
```

## Signal Generation Logic

The ICT Engine generates entry signals based on confluence of factors:

| Factor | Weight | Long Score | Short Score |
|--------|--------|------------|-------------|
| Daily Bias | 25% | Bullish bias | Bearish bias |
| Zone | 20% | In discount | In premium |
| OTE | 15% | OTE in discount | OTE in premium |
| FVG | 20% | Price in bullish FVG | Price in bearish FVG |
| Order Block | 15% | At bullish OB | At bearish OB |
| Liquidity Sweep | 20% | Sell-side sweep + reversal | Buy-side sweep + reversal |

**Signal threshold**: Score >= 40% generates entry signal

## Integration with Wyckoff

The ICT methodology complements the existing Wyckoff implementation:

| ICT Concept | Wyckoff Equivalent |
|-------------|-------------------|
| Liquidity Sweep | Spring/Upthrust |
| FVG | Fair Value / Inefficiency |
| Order Block | Climax Candle |
| Premium/Discount | Overbought/Oversold |
| Daily Bias | Phase Direction |

Both systems can be used together:
- Use Wyckoff for phase identification (accumulation/distribution)
- Use ICT for precise entry points (FVG, OB, OTE)

```python
from engines.liquidity import LiquidityEngineV4, ICTEngine

# Wyckoff analysis
wyckoff_engine = LiquidityEngineV4(market_adapter, options_adapter, {})
wyckoff_snapshot = wyckoff_engine.run(symbol, timestamp)

# ICT analysis
ict_engine = ICTEngine({})
ict_snapshot = ict_engine.analyze(symbol, bars, daily_bars)

# Combine signals
if wyckoff_snapshot.state.phase == WyckoffPhase.PHASE_C:
    if ict_snapshot.entry_signal == "long" and ict_snapshot.in_ote:
        # High confluence long setup
        print("Strong long setup: Wyckoff Phase C + ICT OTE in discount")
```

## Exported Components

```python
from engines.liquidity import (
    # Main Engine
    ICTEngine,
    
    # Data Structures
    SwingPoint,
    LiquidityLevel,
    PremiumDiscountZone,
    FairValueGap,
    VolumeImbalance,
    OrderBlock,
    DailyBiasResult,
    LiquiditySweep,
    ICTSnapshot,
    
    # Enums
    LiquidityType,      # buy_side, sell_side
    SwingType,          # swing_high, swing_low
    HighLowType,        # equal_highs, equal_lows, old_high, old_low, etc.
    FVGType,            # bisi, sibi
    FVGStatus,          # unfilled, partially_filled, filled, inverted
    OrderBlockType,     # bullish_high_probability, etc.
    DailyBias,          # bullish, bearish, neutral
    ZoneType,           # premium, discount, equilibrium
    
    # Component Classes
    SwingPointDetector,
    PremiumDiscountCalculator,
    FairValueGapDetector,
    OrderBlockDetector,
    DailyBiasCalculator,
    LiquiditySweepDetector,
)
```

## Configuration Options

```python
config = {
    'swing': {
        'lookback': 50,
        'cluster_threshold_pct': 0.005  # 0.5% for equal highs/lows
    },
    'premium_discount': {
        'ote_high_fib': 0.62,
        'ote_low_fib': 0.79
    },
    'fvg': {
        'min_gap_size_pct': 0.001  # Minimum 0.1% gap
    },
    'order_blocks': {
        'lookback': 20
    },
    'sweeps': {
        'reversal_bars': 3
    },
    'max_bar_history': 200
}
```
