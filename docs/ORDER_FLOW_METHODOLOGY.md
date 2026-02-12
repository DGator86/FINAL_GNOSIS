# Order Flow Methodology Implementation

## Overview

The Order Flow Engine (`/engines/liquidity/order_flow_engine.py`) implements comprehensive order flow analysis based on Auction Market Theory, Market Microstructure, and professional trading tools.

## Core Components

### 1. Footprint Analysis

The Footprint Analyzer provides bid/ask volume data at each price level within a bar.

**Key Features:**
- **Bid/Ask Volume Split**: Shows buying vs selling aggression at each price
- **Delta Calculation**: `ask_volume - bid_volume` per price level
- **Imbalance Detection**: Identifies 2:1+ ratios (200% threshold)
- **Stacked Imbalances**: Multiple consecutive imbalances (diagonal pattern)

**Patterns Detected:**
- `ABSORPTION_AT_HIGH`: Buying absorbed at highs (bearish)
- `ABSORPTION_AT_LOW`: Selling absorbed at lows (bullish)
- `INITIATIVE_BUYING`: Aggressive buying lifting offers
- `INITIATIVE_SELLING`: Aggressive selling hitting bids
- `BREAKOUT_CONFIRMATION`: Volume confirms breakout
- `FAILED_AUCTION`: Price rejected, auction incomplete
- `STOP_HUNT_COMPLETE`: Stops hit, reversal likely

### 2. Cumulative Volume Delta (CVD)

CVD tracks the running total of delta across all bars.

**Key Signals:**
- **Exhaustion**: CVD flattening while price continues trending
- **Divergence**: Price and CVD moving in opposite directions
- **Delta Flip**: CVD crosses from positive to negative or vice versa

**Usage:**
```python
from engines.liquidity import CVDAnalyzer

cvd = CVDAnalyzer(smoothing_period=14)
points = cvd.calculate_cvd(footprint_bars)

# Check for exhaustion
if cvd.detect_exhaustion():
    print("Trend exhaustion detected")

# Get current bias
bias, confidence = cvd.get_current_bias()
```

### 3. Volume Profile

Volume Profile distributes volume across price levels to identify key areas.

**Key Concepts:**
- **POC (Point of Control)**: Price with highest volume - "fair value"
- **Value Area**: Range containing 70% of volume
- **HVN (High Volume Node)**: Strong support/resistance
- **LVN (Low Volume Node)**: Fast price movement zones

**Auction States:**
- `BALANCED`: Trading within value area
- `INITIATING_LONG`: Breaking out upward
- `INITIATING_SHORT`: Breaking out downward
- `RESPONSIVE_LONG`: Responsive buying at lows
- `RESPONSIVE_SHORT`: Responsive selling at highs
- `ROTATIONAL`: Price rotating without direction

## Signal Types

```python
class OrderFlowSignal(Enum):
    ABSORPTION         # Large orders absorbed without price movement
    INITIATION         # Aggressive orders moving price
    EXHAUSTION         # Buying/selling pressure weakening
    DIVERGENCE         # Price vs delta divergence
    IMBALANCE          # Significant bid/ask imbalance
    DELTA_FLIP         # Delta changes direction
    POC_TEST           # Price testing Point of Control
    VALUE_AREA_BREAK   # Breaking out of value area
    HVN_REJECTION      # Rejection at high volume node
    LVN_ACCELERATION   # Acceleration through low volume node
    STACKED_IMBALANCE  # Multiple consecutive imbalances
    UNFINISHED_AUCTION # Single prints indicating incomplete auction
```

## Usage Examples

### Basic Order Flow Analysis

```python
from engines.liquidity import create_order_flow_engine

# Create engine
engine = create_order_flow_engine(
    imbalance_threshold=2.0,      # 200% for imbalance
    stacked_min_count=3,          # Min for stacked pattern
    cvd_smoothing=14,             # CVD smoothing period
    value_area_percent=0.70,      # 70% value area
)

# Analyze price data
bars = [
    {'open': 100, 'high': 102, 'low': 99, 'close': 101, 'volume': 10000},
    # ... more bars
]

state = engine.analyze("SPY", bars, current_price=101.5)

# Check results
print(f"Auction State: {state.auction_state.name}")
print(f"Current CVD: {state.current_cvd:+.0f}")
print(f"Signals: {[s.name for s in state.signals]}")
print(f"Signal Strength: {state.signal_strength:.0%}")
```

### With Footprint Data

```python
from engines.liquidity import FootprintAnalyzer
from datetime import datetime

analyzer = FootprintAnalyzer()

# Create footprint bar with price level data
bar = analyzer.create_footprint_bar(
    timestamp=datetime.now(),
    ohlc={'open': 100, 'high': 105, 'low': 98, 'close': 103},
    price_levels=[
        {'price': 98, 'bid_volume': 5000, 'ask_volume': 2000},
        {'price': 100, 'bid_volume': 3000, 'ask_volume': 8000},
        {'price': 102, 'bid_volume': 2000, 'ask_volume': 6000},
        {'price': 104, 'bid_volume': 1000, 'ask_volume': 4000},
    ]
)

print(f"Total Delta: {bar.total_delta:+.0f}")
print(f"POC: {bar.poc_price}")
print(f"Patterns: {[p.name for p in bar.patterns]}")
```

### Getting Entry Signals

```python
# Get entry signals
entries = engine.get_entry_signals("SPY")

for entry in entries:
    print(f"Direction: {entry.direction}")
    print(f"Confidence: {entry.confidence:.0%}")
    print(f"Entry: ${entry.entry_price}")
    print(f"Stop: ${entry.stop_price}")
    print(f"Target: ${entry.target_price}")
    print(f"Signal: {entry.signal_type.name}")
    print(f"Reasoning: {entry.reasoning}")
```

### Getting Key Levels

```python
levels = engine.get_key_levels("SPY")

print(f"POC: ${levels['poc']}")
print(f"Value Area: ${levels['value_area'][0]} - ${levels['value_area'][1]}")
print(f"Support levels: {levels['support']}")
print(f"Resistance levels: {levels['resistance']}")
```

## Integration with Wyckoff and ICT

The Order Flow Engine is designed to work alongside the Wyckoff (LiquidityEngineV4) and ICT (ICTEngine) methodologies through the unified LiquidityAgentV5.

### Triple Methodology Integration

```python
from agents import LiquidityAgentV5
from engines.engine_factory import create_unified_analysis_engines

# Create all engines
engines = create_unified_analysis_engines()

# Create unified agent
agent = LiquidityAgentV5(
    config={
        "wyckoff_weight": 0.30,
        "ict_weight": 0.30,
        "order_flow_weight": 0.30,
    },
    wyckoff_engine=engines['wyckoff_engine'],
    ict_engine=engines['ict_engine'],
    order_flow_engine=engines['order_flow_engine'],
)

# Get confluence analysis
confluence = agent.get_confluence_analysis("SPY")
print(f"Confluence Score: {confluence['confluence']['score']:.0%}")
print(f"Order Flow Confirms: {confluence['confluence']['order_flow_confirms']}")

# Get entry setups with Order Flow confirmation
setups = agent.get_entry_setups("SPY")
for setup in setups:
    if setup.get("order_flow_confirmed"):
        print(f"CONFIRMED: {setup['type']} - {setup['direction']}")
```

## Order Flow Confirmation Patterns

When Order Flow confirms other methodology signals:

| Wyckoff Signal | Order Flow Confirmation | Confidence Boost |
|----------------|------------------------|------------------|
| Spring | Absorption at lows | +20% |
| Upthrust | Absorption at highs | +20% |
| SOS/SOW | Initiative buying/selling | +15% |
| Phase C entry | CVD divergence | +25% |

| ICT Signal | Order Flow Confirmation | Confidence Boost |
|------------|------------------------|------------------|
| FVG entry | Matching auction state | +20% |
| Order Block | Absorption pattern | +25% |
| Liquidity Sweep | CVD exhaustion/divergence | +30% |
| OTE entry | Initiative pattern | +15% |

## Best Practices

1. **Multiple Timeframe Confirmation**: Use Order Flow on lower timeframes to confirm higher timeframe signals

2. **Volume Quality**: Real footprint data provides better signals than synthetic data

3. **Context Matters**: Order Flow signals are stronger when aligned with market structure

4. **CVD Divergence**: One of the most reliable signals - price making new highs/lows while CVD fails to confirm

5. **Value Area**: Trading within value area is rotational; breaks with volume confirm direction

## Data Requirements

For optimal analysis, the Order Flow Engine supports:

1. **OHLCV Data** (minimum): Standard bar data with volume
2. **Footprint Data** (recommended): Bid/ask volume per price level
3. **Time & Sales** (advanced): Individual trade data for precise delta

The engine will create synthetic footprint data from OHLCV if detailed data is not available.

## Configuration Options

```python
engine = OrderFlowEngine(
    footprint_config={
        'imbalance_threshold': 2.0,    # 200% for bid/ask imbalance
        'stacked_min_count': 3,         # Minimum for stacked imbalance
        'absorption_volume_mult': 2.0,  # Volume multiplier for absorption
    },
    cvd_config={
        'smoothing_period': 14,         # CVD smoothing
        'divergence_threshold': 0.3,    # Divergence detection
    },
    profile_config={
        'value_area_percent': 0.70,     # 70% value area
        'hvn_threshold': 1.5,           # HVN volume multiplier
        'lvn_threshold': 0.5,           # LVN volume multiplier
        'price_resolution': 0.01,       # Price bucket size
    },
)
```

## File Locations

- **Order Flow Engine**: `/engines/liquidity/order_flow_engine.py`
- **Engine Factory**: `/engines/engine_factory.py`
- **LiquidityAgentV5**: `/agents/liquidity_agent_v5.py`
- **Package Exports**: `/engines/liquidity/__init__.py`

## Version History

- **v1.0.0**: Initial implementation
  - Footprint Analysis (bid/ask aggression, imbalance, absorption)
  - CVD Analysis (cumulative delta, exhaustion, divergence)
  - Volume Profile (POC, Value Area, HVN/LVN)
  - Auction Market Theory integration
  - Entry signal generation
  - Integration with Wyckoff and ICT methodologies
