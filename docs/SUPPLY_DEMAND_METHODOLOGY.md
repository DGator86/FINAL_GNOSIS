# Supply and Demand Methodology Implementation

## Overview

The Supply and Demand Engine (`/engines/liquidity/supply_demand_engine.py`) implements a trading methodology based on fundamental economic principles - the Law of Supply and Demand.

## Core Economic Principles

### Law of Demand
- **Higher price = Lower quantity demanded**
- Buyers seek the lowest possible price
- More expensive items sell in lower quantities

### Law of Supply
- **Higher price = Higher quantity supplied**
- Sellers want to maximize profit
- Higher prices incentivize more selling

### Market Equilibrium
- Price naturally seeks the point where supply equals demand
- At equilibrium, buyers and sellers are satisfied
- Price remains stable until supply/demand shifts occur

### Supply and Demand Shifts
When market perception changes:
- **Demand Increase**: Curve shifts right → Price rises
- **Demand Decrease**: Curve shifts left → Price falls
- **Supply Increase**: Curve shifts right → Price falls
- **Supply Decrease**: Curve shifts left → Price rises

## Zone Formation Rules

### Demand Zones
A Demand Zone forms at the **LOW between TWO HIGHS** where:
- The **second high MUST be higher than the first** (momentum confirmation)
- This validates that buyers had enough strength to push price higher
- The zone represents where a significant demand shift occurred

```
      High₂ (Higher)
       /\
      /  \
     /    \
High₁      \
 /\         \
/  \         \
    \       /
     \_____/  ← DEMAND ZONE (Low between highs)
```

**Zone Boundaries:**
- **Lower boundary**: Absolute low between the two highs
- **Upper boundary**: Point where volatility shifted (buyers decided to push up)

### Supply Zones
A Supply Zone forms at the **HIGH between TWO LOWS** where:
- The **second low MUST be lower than the first** (momentum confirmation)
- This validates that sellers had enough strength to push price lower
- The zone represents where a significant supply shift occurred

```
     _____   ← SUPPLY ZONE (High between lows)
    /     \
   /       \
  /         \
Low₁        \
             \  /\
              \/  \
            Low₂ (Lower)
```

**Zone Boundaries:**
- **Upper boundary**: Absolute high between the two lows
- **Lower boundary**: Point where volatility shifted (sellers decided to push down)

## Zone Properties

### Zone Strength
```python
class ZoneStrength(Enum):
    STRONG = auto()      # Properly validated (HH for demand, LL for supply)
    MODERATE = auto()    # Partially validated
    WEAK = auto()        # Not validated - use with caution
    BROKEN = auto()      # Zone has been violated
```

### Zone Status
```python
class ZoneStatus(Enum):
    FRESH = auto()       # Never tested - highest probability
    TESTED = auto()      # Tested once - still valid
    RETESTED = auto()    # Tested multiple times - weakening
    BROKEN = auto()      # Price broke through - invalidated
```

**Key Insight**: Fresh zones are the most reliable. Each test weakens the zone.

## Risk Management Integration

Supply and Demand zones provide natural risk management:

### Built-in Stop Loss
- **Demand Zone**: Stop loss below the zone's lower boundary
- **Supply Zone**: Stop loss above the zone's upper boundary

### Take Profit Targets (R:R ratios)
```python
# For a long trade from a demand zone:
risk = zone.boundary.upper - zone.stop_loss
take_profit_1 = zone.boundary.upper + risk * 1.0   # 1:1 R:R
take_profit_2 = zone.boundary.upper + risk * 2.0   # 1:2 R:R
take_profit_3 = zone.boundary.upper + risk * 3.0   # 1:3 R:R
take_profit_4 = zone.boundary.upper + risk * 4.0   # 1:4 R:R
```

### Why This Works
With a 1:3 Risk:Reward ratio:
- You can lose 75% of the time and still break even
- Small, logical stop losses near the entry
- Large potential profits from zone-to-zone moves

## Usage Examples

### Basic Zone Detection

```python
from engines.liquidity import create_supply_demand_engine

# Create engine
engine = create_supply_demand_engine(
    swing_lookback=3,           # Bars to confirm swings
    min_swing_distance=3,       # Minimum bars between swings
    max_zones=10,               # Maximum zones per type
    default_risk_reward=3.0,    # Default 1:3 R:R
)

# Analyze price data
bars = [
    {'open': 100, 'high': 102, 'low': 99, 'close': 101, 'volume': 10000},
    # ... more bars
]

state = engine.analyze("SPY", bars, current_price=105)

# Check zones
print(f"Demand Zones: {len(state.demand_zones)}")
print(f"Supply Zones: {len(state.supply_zones)}")
print(f"Equilibrium: {state.equilibrium_state.name}")
```

### Getting Entry Signals

```python
# Get entry signals
entries = engine.get_entry_signals("SPY")

for entry in entries:
    print(f"Direction: {entry.direction}")
    print(f"Entry: ${entry.entry_price:.2f}")
    print(f"Stop: ${entry.stop_loss:.2f}")
    print(f"TP1: ${entry.take_profit_1:.2f}")
    print(f"TP3: ${entry.take_profit_3:.2f}")
    print(f"Confidence: {entry.confidence:.0%}")
    print(f"Zone Status: {entry.zone.status.name}")
```

### Getting Key Levels

```python
levels = engine.get_key_levels("SPY")

print("Demand Zones:")
for zone in levels['demand_zones']:
    print(f"  {zone['lower']:.2f} - {zone['upper']:.2f}")
    print(f"  Strength: {zone['strength']}, Reliability: {zone['reliability']:.0%}")

print("Supply Zones:")
for zone in levels['supply_zones']:
    print(f"  {zone['lower']:.2f} - {zone['upper']:.2f}")
```

### Getting Bias

```python
bias, confidence, reasoning = engine.get_bias("SPY")

print(f"Bias: {bias}")
print(f"Confidence: {confidence:.0%}")
print(f"Reasoning: {reasoning}")
```

## Integration with Other Methodologies

The Supply and Demand Engine is designed to work alongside:
- **Wyckoff** (LiquidityEngineV4): Phase and structure context
- **ICT** (ICTEngine): Fair Value Gaps and Order Blocks
- **Order Flow** (OrderFlowEngine): Volume confirmation

### Using All Engines Together

```python
from engines.engine_factory import create_unified_analysis_engines

# Create all engines
engines = create_unified_analysis_engines()

wyckoff = engines['wyckoff_engine']
ict = engines['ict_engine']
order_flow = engines['order_flow_engine']
supply_demand = engines['supply_demand_engine']

# Analyze with all methodologies
sd_state = supply_demand.analyze("SPY", bars)
of_state = order_flow.analyze("SPY", bars)

# Look for confluence
if sd_state.nearest_demand and sd_state.nearest_demand.status == ZoneStatus.FRESH:
    # Fresh demand zone
    of_bias, of_conf, _ = order_flow.get_bias("SPY")
    if of_bias == "bullish":
        print("HIGH CONFLUENCE: Fresh demand zone + bullish order flow")
```

## Entry Signal Types

```python
class EntrySignal(Enum):
    DEMAND_ZONE_TOUCH = auto()      # Price touching demand zone
    SUPPLY_ZONE_TOUCH = auto()      # Price touching supply zone
    DEMAND_ZONE_BOUNCE = auto()     # Price bouncing from demand zone
    SUPPLY_ZONE_BOUNCE = auto()     # Price bouncing from supply zone
    ZONE_BREAK = auto()             # Zone broken - potential reversal
    NO_SIGNAL = auto()              # No actionable signal
```

## Best Practices

### 1. Fresh Zones are Best
- Prioritize zones that have never been tested
- Each test weakens the zone

### 2. Momentum Confirmation is Required
- Demand zones: Second high must be higher than first
- Supply zones: Second low must be lower than first

### 3. Use Appropriate Risk:Reward
- Minimum 1:3 R:R recommended
- This allows for a 75% loss rate while breaking even

### 4. Zone Size Matters
- Too small: Easily broken
- Too large: Poor risk:reward
- The engine auto-validates zone height (0.1% - 3.0% of price)

### 5. Combine with Other Analysis
- Use Order Flow for volume confirmation
- Use Wyckoff for phase context
- Use ICT for additional entry precision

## Configuration Options

```python
engine = SupplyDemandEngine(
    swing_lookback=3,           # Bars to confirm swing points
    min_swing_distance=3,       # Minimum bars between swings
    max_zones=10,               # Maximum zones to track per type
    volatility_multiplier=1.5,  # For boundary calculation
    default_risk_reward=3.0,    # Default R:R ratio
)
```

## Data Structures

### SupplyDemandZone
```python
@dataclass
class SupplyDemandZone:
    zone_type: ZoneType          # DEMAND or SUPPLY
    boundary: ZoneBoundary       # upper and lower prices
    formation_time: datetime
    strength: ZoneStrength
    status: ZoneStatus
    
    # Formation details
    origin_price: float          # The extreme that created the zone
    first_swing_price: float     # First high/low before the zone
    second_swing_price: float    # Second high/low after the zone
    momentum_confirmed: bool     # HH for demand, LL for supply
    
    # Risk management
    stop_loss: float
    take_profit_1: float         # 1:1 R:R
    take_profit_2: float         # 1:2 R:R
    take_profit_3: float         # 1:3 R:R
    take_profit_4: float         # 1:4 R:R
```

### ZoneEntry
```python
@dataclass
class ZoneEntry:
    direction: str               # 'long' or 'short'
    confidence: float            # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_3: float
    zone: SupplyDemandZone
    signal_type: EntrySignal
    reasoning: str
```

## Why Supply and Demand Works

1. **Based on Economic Law**: Not just pattern recognition, but fundamental market mechanics

2. **No Lag**: Unlike indicators, zones form from actual price action

3. **Built-in Risk Management**: Natural stop loss and target levels

4. **Works on All Timeframes**: The economic principle applies universally

5. **Clear Rules**: Objective zone formation criteria

## Limitations

1. **Not 100% Reliable**: Sometimes zones fail - use risk management

2. **Requires Context**: Works best with confluence from other analysis

3. **Subjectivity in Boundaries**: Upper boundary calculation can vary

4. **Market Conditions**: Major news can override technical zones

## File Locations

- **Engine**: `/engines/liquidity/supply_demand_engine.py`
- **Factory**: `/engines/engine_factory.py`
- **Package**: `/engines/liquidity/__init__.py`

## Integration with LiquidityAgentV5

The Supply & Demand methodology is fully integrated into `LiquidityAgentV5`, creating a **QUAD methodology** trading system:

### LiquidityAgentV5 Features

```python
from agents import LiquidityAgentV5
from engines.engine_factory import create_unified_analysis_engines

# Create all 4 methodology engines
engines = create_unified_analysis_engines()

# Create quad-methodology agent
config = {
    'enable_wyckoff': True,
    'enable_ict': True,
    'enable_order_flow': True,
    'enable_supply_demand': True,
}

agent = LiquidityAgentV5(
    config,
    wyckoff_engine=engines['wyckoff_engine'],
    ict_engine=engines['ict_engine'],
    order_flow_engine=engines['order_flow_engine'],
    supply_demand_engine=engines['supply_demand_engine']
)

# Get entry setups with built-in risk management
setups = agent.get_entry_setups('SPY')

for setup in setups:
    if 'supply_demand' in setup['type']:
        print(f"S&D Entry: {setup['direction']}")
        print(f"  Entry: {setup['entry_price']:.2f}")
        print(f"  Stop Loss: {setup['stop_loss']:.2f}")
        print(f"  TP3 (1:3 R:R): {setup['take_profit_3']:.2f}")
        print(f"  Zone Status: {setup['zone_status']}")
        print(f"  Confidence: {setup['confidence']:.0%}")
        if setup.get('quad_confluence'):
            print("  [QUAD CONFLUENCE DETECTED]")
```

### Confluence Scoring with S&D

The agent applies multipliers based on zone quality:

**Zone Strength Multipliers:**
- STRONG: 1.2x confidence boost
- MODERATE: 1.0x (no change)
- WEAK: 0.8x confidence penalty
- BROKEN: 0.0x (zone ignored)

**Zone Status Multipliers:**
- FRESH: 1.25x (highest probability)
- TESTED: 1.0x (still valid)
- RETESTED: 0.85x (weakening)
- BROKEN: 0.0x (zone invalidated)

### QUAD Confluence Bonus

When all 4 methodologies align:
- **4 methods aligned**: 25% confidence bonus + "QUAD CONFLUENCE" flag
- **3 methods aligned**: 15% confidence bonus + "TRIPLE CONFLUENCE" flag
- **2 methods aligned**: 8% confidence bonus + "Double confluence" flag

### Signal Combination Weights

Default weights (configurable):
- Wyckoff: 23%
- ICT: 23%
- Order Flow: 23%
- Supply/Demand: 23%
- Base (liquidity snapshot): 8%

### Example Analysis Output

```python
analysis = agent.get_confluence_analysis('SPY')

print(f"Symbol: {analysis['symbol']}")
print(f"Supply/Demand:")
print(f"  Equilibrium: {analysis['supply_demand']['equilibrium']}")
print(f"  Entry Signal: {analysis['supply_demand']['entry_signal']}")
print(f"  Demand Zones: {analysis['supply_demand']['demand_zones_count']}")
print(f"  Supply Zones: {analysis['supply_demand']['supply_zones_count']}")

if analysis['supply_demand']['nearest_demand']:
    nd = analysis['supply_demand']['nearest_demand']
    print(f"  Nearest Demand: {nd['lower']:.2f} - {nd['upper']:.2f}")
    print(f"    Status: {nd['status']}, Strength: {nd['strength']}")
    print(f"    Stop Loss: {nd['stop_loss']:.2f}")
    print(f"    TP3: {nd['take_profit_3']:.2f}")

print(f"Confluence:")
print(f"  Score: {analysis['confluence']['score']:.0%}")
print(f"  Direction: {analysis['confluence']['direction']}")
print(f"  Methods Aligned: {analysis['confluence']['methods_aligned']}/4")
print(f"  S&D Confirms: {analysis['confluence']['supply_demand_confirms']}")
```

## Version History

- **v1.0.0**: Initial implementation
  - Demand zone detection (low between two highs)
  - Supply zone detection (high between two lows)
  - Zone strength validation (momentum confirmation)
  - Zone boundary calculation (volatility shift)
  - Zone status tracking (fresh, tested, broken)
  - Risk management integration (built-in R:R levels)
  - Entry signal generation
  - Engine factory integration

- **v1.1.0**: LiquidityAgentV5 Integration
  - Full integration with quad-methodology agent
  - Zone strength/status multipliers in confluence scoring
  - QUAD confluence bonus (25% for 4-method alignment)
  - Fresh zone prioritization in entry type selection
  - Built-in risk management in entry setups
  - `get_confluence_analysis()` includes S&D data
  - `get_entry_setups()` includes S&D entries with TP levels
