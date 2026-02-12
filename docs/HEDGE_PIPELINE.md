# GNOSIS Hedge Pipeline - Complete Documentation

## Overview

The **Hedge Pipeline** is one of the three primary agent pipelines in GNOSIS, responsible for analyzing dealer positioning, options Greeks, and market elasticity to detect institutional flow and predict price movement.

---

## Hedge Pipeline Architecture

```
═══════════════════════════════════════════════════════════════════════════════════════════
                              GNOSIS HEDGE PIPELINE
═══════════════════════════════════════════════════════════════════════════════════════════

                              ┌─────────────────────────────┐
                              │      OPTIONS CHAIN DATA     │
                              │                             │
                              │  From: Massive.com/Polygon  │
                              │  • Strike prices            │
                              │  • Expirations              │
                              │  • Open Interest            │
                              │  • Volume                   │
                              │  • Greeks (Δ,Γ,Θ,ν)        │
                              │  • Implied Volatility       │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │   OPTIONS CHAIN ADAPTER     │
                              │   /engines/inputs/          │
                              │                             │
                              │  Normalizes raw options     │
                              │  data into OptionsContract  │
                              │  dataclass objects          │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                              HEDGE ENGINE V3                                             │
│                              /engines/hedge/hedge_engine_v3.py                           │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          DEALER GAMMA ANALYSIS                                      │ │
│  │                                                                                     │ │
│  │  dealer_gamma_sign = (call_OI - put_OI) / (call_OI + put_OI)                       │ │
│  │                                                                                     │ │
│  │  Interpretation:                                                                    │ │
│  │  • Positive = Dealers are NET SHORT gamma (long calls dominate)                    │ │
│  │    → Dealers must BUY when price rises, SELL when price falls                      │ │
│  │    → AMPLIFIES moves (volatility expansion)                                        │ │
│  │                                                                                     │ │
│  │  • Negative = Dealers are NET LONG gamma (long puts dominate)                      │ │
│  │    → Dealers must SELL when price rises, BUY when price falls                      │ │
│  │    → DAMPENS moves (volatility compression)                                        │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          GREEK PRESSURE FIELDS                                      │ │
│  │                                                                                     │ │
│  │  GAMMA PRESSURE = Σ |Γᵢ| × OIᵢ / n                                                 │ │
│  │    → Measures total directional hedging pressure                                   │ │
│  │    → High = More violent moves expected                                            │ │
│  │                                                                                     │ │
│  │  VANNA PRESSURE = Σ |νᵢ × Δᵢ| × OIᵢ / n                                            │ │
│  │    → Volatility-Delta cross sensitivity                                            │ │
│  │    → Measures vol smile impact on hedging                                          │ │
│  │                                                                                     │ │
│  │  CHARM PRESSURE = Σ |Θᵢ × Δᵢ| × OIᵢ / n                                            │ │
│  │    → Time decay impact on delta                                                    │ │
│  │    → Increases near expiration                                                     │ │
│  │                                                                                     │ │
│  │  VANNA SHOCK ABSORBER:                                                              │ │
│  │    vol_spike = max(0, IV_mean - IV_median)                                         │ │
│  │    shock_factor = e^(-1.2 × vol_spike)                                             │ │
│  │    vanna_pressure = vanna_pressure × shock_factor                                  │ │
│  │    → Dampens vanna during volatility spikes                                        │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          DIRECTIONAL PRESSURES                                      │ │
│  │                                                                                     │ │
│  │  PRESSURE_UP = Σ Γᵢ × OIᵢ  (where option=call AND Δ > 0.3)                         │ │
│  │    → Upward hedging pressure from ITM/ATM calls                                    │ │
│  │                                                                                     │ │
│  │  PRESSURE_DOWN = Σ |Γᵢ × OIᵢ|  (where option=put AND Δ < -0.3)                     │ │
│  │    → Downward hedging pressure from ITM/ATM puts                                   │ │
│  │                                                                                     │ │
│  │  PRESSURE_NET = PRESSURE_UP - PRESSURE_DOWN                                         │ │
│  │    → Net directional bias from options positioning                                 │ │
│  │    → Positive = Bullish pressure, Negative = Bearish pressure                      │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          ELASTICITY THEORY                                          │ │
│  │                          (Inelastic Markets Hypothesis)                             │ │
│  │                                                                                     │ │
│  │  BASE ELASTICITY:                                                                   │ │
│  │    base = γ_pressure × w_gamma + ν_pressure × w_vanna                              │ │
│  │    where w_gamma = 0.6, w_vanna = 0.4 (adaptive)                                   │ │
│  │                                                                                     │ │
│  │  FLOW MULTIPLIER (IMH - Gabaix & Koijen):                                          │ │
│  │    Regress: prices ~ flows (OLS on ledger history)                                 │ │
│  │    flow_multiplier = regression_coefficient                                        │ │
│  │    → Captures how flows impact prices                                              │ │
│  │                                                                                     │ │
│  │  OI-WEIGHTED ELASTICITY:                                                            │ │
│  │    For each contract:                                                               │ │
│  │      oi_weight = OIᵢ / Σ OI                                                        │ │
│  │      greek_pressure = |Γᵢ × Δᵢ|                                                    │ │
│  │      weighted_elasticity += oi_weight × greek_pressure                             │ │
│  │                                                                                     │ │
│  │  FINAL ELASTICITY:                                                                  │ │
│  │    elasticity = base × (1 + flow_multiplier) + weighted_elasticity                 │ │
│  │    → Low elasticity = Price moves easily (inelastic)                               │ │
│  │    → High elasticity = Price resists moves (elastic)                               │ │
│  │                                                                                     │ │
│  │  DIRECTIONAL ELASTICITY:                                                            │ │
│  │    up_elasticity = OI-weighted Γ×Δ for CALLS                                       │ │
│  │    down_elasticity = OI-weighted Γ×Δ for PUTS                                      │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          MOVEMENT ENERGY                                            │ │
│  │                                                                                     │ │
│  │  MOVEMENT_ENERGY = |PRESSURE_NET| / ELASTICITY                                      │ │
│  │                                                                                     │ │
│  │  Interpretation:                                                                    │ │
│  │  • High energy + Low elasticity = Big move imminent                                │ │
│  │  • Low energy + High elasticity = Range-bound/Stable                               │ │
│  │                                                                                     │ │
│  │  ENERGY_ASYMMETRY = (PRESSURE_UP - PRESSURE_DOWN) / (PRESSURE_UP + PRESSURE_DOWN)  │ │
│  │                                                                                     │ │
│  │  Interpretation:                                                                    │ │
│  │  • > +0.3 = Bullish bias (call pressure dominates)                                 │ │
│  │  • < -0.3 = Bearish bias (put pressure dominates)                                  │ │
│  │  • -0.3 to +0.3 = Neutral                                                          │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          RISK METRICS                                               │ │
│  │                                                                                     │ │
│  │  JUMP_INTENSITY = count(|Δ| < 0.05) / total_contracts                              │ │
│  │    → Ratio of deep OTM options                                                     │ │
│  │    → High = Tail risk hedging active (fear of jumps)                               │ │
│  │                                                                                     │ │
│  │  LIQUIDITY_FRICTION = top_10%_OI / total_OI                                        │ │
│  │    → OI concentration in top strikes                                               │ │
│  │    → High = Liquidity concentrated, harder to move through                         │ │
│  │    → Low = Distributed liquidity, easier movement                                  │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          REGIME DETECTION                                           │ │
│  │                          (Gaussian Mixture Model)                                   │ │
│  │                                                                                     │ │
│  │  Feature Vector (8 dimensions):                                                     │ │
│  │  [dealer_gamma_sign, gamma_pressure, vanna_pressure, charm_pressure,               │ │
│  │   movement_energy, energy_asymmetry, jump_intensity, liquidity_friction]           │ │
│  │                                                                                     │ │
│  │  GMM Clustering (n_components=3):                                                   │ │
│  │  • STABILITY  - Low energy, balanced pressures                                     │ │
│  │  • EXPANSION  - High energy, directional pressure                                  │ │
│  │  • SQUEEZE    - Building pressure, low realized vol                                │ │
│  │                                                                                     │ │
│  │  Rolling history: 256 samples, min 32 for fit                                      │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  OUTPUT: HedgeSnapshot                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  timestamp: datetime                                                                │ │
│  │  symbol: str                                                                        │ │
│  │  elasticity: float                   # Market elasticity                            │ │
│  │  movement_energy: float              # Potential for movement                       │ │
│  │  energy_asymmetry: float             # Directional bias (-1 to 1)                  │ │
│  │  pressure_up: float                  # Bullish pressure                            │ │
│  │  pressure_down: float                # Bearish pressure                            │ │
│  │  pressure_net: float                 # Net pressure                                │ │
│  │  gamma_pressure: float               # Total gamma pressure                        │ │
│  │  vanna_pressure: float               # Vol-delta sensitivity                       │ │
│  │  charm_pressure: float               # Time decay impact                           │ │
│  │  dealer_gamma_sign: float            # Dealer positioning                          │ │
│  │  regime: str                         # stability/expansion/squeeze                 │ │
│  │  regime_probabilities: Dict          # GMM probabilities                           │ │
│  │  jump_intensity: float               # Tail risk indicator                         │ │
│  │  liquidity_friction: float           # OI concentration                            │ │
│  │  confidence: float                   # Data quality score                          │ │
│  │  directional_elasticity: Dict        # {up: float, down: float}                    │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
└──────────────────────────────────────────────┬──────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                              HEDGE AGENT V4                                              │
│                              /agents/hedge_agent_v4.py                                   │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          SINGLE-TIMEFRAME PROCESSING                                │ │
│  │                                                                                     │ │
│  │  Input: HedgeSnapshot from HedgeEngineV3                                           │ │
│  │                                                                                     │ │
│  │  Direction Logic:                                                                   │ │
│  │    if energy_asymmetry > +0.3:                                                     │ │
│  │        direction = LONG                                                            │ │
│  │        reasoning = "Positive energy asymmetry"                                     │ │
│  │    elif energy_asymmetry < -0.3:                                                   │ │
│  │        direction = SHORT                                                           │ │
│  │        reasoning = "Negative energy asymmetry"                                     │ │
│  │    else:                                                                           │ │
│  │        direction = NEUTRAL                                                         │ │
│  │                                                                                     │ │
│  │  Confidence Adjustment:                                                             │ │
│  │    confidence = snapshot.confidence × (1 + min(0.5, movement_energy/100))          │ │
│  │    → Boosts confidence when movement energy is high                                │ │
│  │                                                                                     │ │
│  │  Output: AgentSuggestion                                                            │ │
│  │    {agent_name, timestamp, symbol, direction, confidence, reasoning}               │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          MULTI-TIMEFRAME PROCESSING                                 │ │
│  │                                                                                     │ │
│  │  Input: Dict[timeframe, HedgeSnapshot]                                             │ │
│  │    e.g., {"1H": snapshot_1h, "4H": snapshot_4h, "1D": snapshot_1d}                 │ │
│  │                                                                                     │ │
│  │  For each timeframe:                                                                │ │
│  │    direction = 1.0 (bullish) / -1.0 (bearish) / 0.0 (neutral)                      │ │
│  │    strength = min(1.0, movement_energy / 100)                                      │ │
│  │    confidence = snapshot.confidence                                                │ │
│  │                                                                                     │ │
│  │  Output: List[TimeframeSignal] for ConfidenceBuilder                               │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          CONFLUENCE DETECTION                                       │ │
│  │                                                                                     │ │
│  │  For each timeframe, check energy_asymmetry direction                              │ │
│  │                                                                                     │ │
│  │  bullish_count = count(asymmetry > +0.3)                                           │ │
│  │  bearish_count = count(asymmetry < -0.3)                                           │ │
│  │                                                                                     │ │
│  │  Confluence if 80%+ timeframes agree:                                               │ │
│  │    has_confluence = (bullish_count/total ≥ 0.8) OR (bearish_count/total ≥ 0.8)    │ │
│  │                                                                                     │ │
│  │  Output: {has_confluence, direction, agreement_ratio, confidence}                  │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
└──────────────────────────────────────────────┬──────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                              ENHANCED HEDGE AGENT V3                                     │
│                              /agents/hedge_agent_v3_enhanced.py                          │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          ML ENHANCEMENT LAYER                                       │ │
│  │                                                                                     │ │
│  │  EPISODIC MEMORY:                                                                   │ │
│  │    • Stores past trading episodes with outcomes                                    │ │
│  │    • Retrieves similar past experiences                                            │ │
│  │    • Learns from successes and failures                                            │ │
│  │                                                                                     │ │
│  │  SEMANTIC MEMORY:                                                                   │ │
│  │    • Knowledge graph of market rules                                               │ │
│  │    • Pattern-action associations                                                   │ │
│  │    • Market regime rules                                                           │ │
│  │                                                                                     │ │
│  │  ML MODELS:                                                                         │ │
│  │    • XGBoost Ensemble for regime classification                                    │ │
│  │    • LSTM Forecaster for price prediction                                          │ │
│  │                                                                                     │ │
│  │  TOOL CALLING:                                                                      │ │
│  │    • web_search: External market research                                          │ │
│  │    • risk_calculator: Monte Carlo VaR                                              │ │
│  │    • historical_query: Pattern matching                                            │ │
│  │    • news_search: Sentiment context                                                │ │
│  │    • options_analyzer: Flow analysis                                               │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          ENHANCED DECISION FLOW                                     │ │
│  │                                                                                     │ │
│  │  1. Get base suggestion from HedgeEngineV3 logic                                   │ │
│  │  2. Prepare current state for memory queries                                       │ │
│  │  3. Retrieve similar past experiences (top 5, 90-day window)                       │ │
│  │  4. Get success rate for this action in similar situations                         │ │
│  │  5. Query semantic memory for applicable rules                                     │ │
│  │  6. Get ML model predictions (XGBoost, LSTM)                                       │ │
│  │  7. Call tools if high uncertainty/novelty                                         │ │
│  │  8. Integrate all information sources                                              │ │
│  │  9. Store experience for future learning                                           │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
└──────────────────────────────────────────────┬──────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                              ELASTICITY ENGINE V1                                        │
│                              /engines/elasticity/elasticity_engine_v1.py                 │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  VOLATILITY ANALYSIS (When HedgeEngine not available)                               │ │
│  │                                                                                     │ │
│  │  Input: OHLCV bars from MarketDataAdapter                                          │ │
│  │                                                                                     │ │
│  │  Calculations:                                                                      │ │
│  │    returns = (closeᵢ - closeᵢ₋₁) / closeᵢ₋₁                                        │ │
│  │    mean_return = Σ returns / n                                                     │ │
│  │    variance = Σ (rᵢ - mean)² / n                                                   │ │
│  │    volatility = √variance × √252  (annualized)                                     │ │
│  │                                                                                     │ │
│  │  Regime Classification:                                                             │ │
│  │    vol < 15% → "low"                                                               │ │
│  │    vol > 30% → "high"                                                              │ │
│  │    else → "moderate"                                                               │ │
│  │                                                                                     │ │
│  │  Trend Strength:                                                                    │ │
│  │    price_range = max(prices) - min(prices)                                         │ │
│  │    recent_move = |price_last - price_first|                                        │ │
│  │    trend_strength = recent_move / price_range                                      │ │
│  │                                                                                     │ │
│  │  DELEGATION TO HEDGE ENGINE:                                                        │ │
│  │    If HedgeEngineV3 is available, delegates to avoid divergence:                   │ │
│  │      elasticity = hedge_snapshot.elasticity                                        │ │
│  │      volatility_regime = hedge_snapshot.regime                                     │ │
│  │      trend_strength = hedge_snapshot.energy_asymmetry                              │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  OUTPUT: ElasticitySnapshot                                                              │
│    {timestamp, symbol, volatility, volatility_regime, trend_strength}                   │
│                                                                                          │
└──────────────────────────────────────────────┬──────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                              COMPOSER AGENT V4                                           │
│                              /agents/composer/composer_agent_v4.py                       │
│                                                                                          │
│  Receives hedge_signal from HedgeAgent and combines with:                               │
│  • sentiment_signal from SentimentAgent                                                 │
│  • liquidity_signal from LiquidityAgent                                                 │
│                                                                                          │
│  Default Weights:                                                                        │
│    HEDGE: 40%  |  SENTIMENT: 40%  |  LIQUIDITY: 20%                                     │
│                                                                                          │
│  Express Mode Weights:                                                                   │
│    0DTE:        HEDGE: 30%  |  LIQUIDITY: 50%  |  SENTIMENT: 20%                        │
│    CHEAP_CALL:  HEDGE: 20%  |  LIQUIDITY: 20%  |  SENTIMENT: 60%                        │
│                                                                                          │
│  Output: ComposerOutput                                                                  │
│    {direction, confidence, consensus_score, hedge_contribution, ...}                    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════
```

---

## Data Structures

### HedgeSnapshot (Engine Output)

```python
@dataclass
class HedgeSnapshot:
    timestamp: datetime
    symbol: str
    
    # Core Metrics
    elasticity: float              # Market elasticity (IMH)
    movement_energy: float         # Potential for movement
    energy_asymmetry: float        # Directional bias (-1 to 1)
    
    # Directional Pressures
    pressure_up: float             # Bullish gamma pressure
    pressure_down: float           # Bearish gamma pressure
    pressure_net: float            # Net directional pressure
    
    # Greek Pressures
    gamma_pressure: float          # Total gamma exposure
    vanna_pressure: float          # Vol-delta sensitivity
    charm_pressure: float          # Time-delta decay
    
    # Dealer Positioning
    dealer_gamma_sign: float       # -1 (long gamma) to +1 (short gamma)
    
    # Regime Detection
    regime: str                    # "stability" | "expansion" | "squeeze"
    regime_probabilities: Dict[str, float]
    regime_features: Dict[str, float]
    
    # Risk Metrics
    jump_intensity: float          # Tail risk indicator
    liquidity_friction: float      # OI concentration
    
    # Quality
    confidence: float              # Data quality score
    adaptive_weights: Dict[str, float]
    directional_elasticity: Dict[str, float]  # {up, down}
```

### AgentSuggestion (Agent Output)

```python
@dataclass
class AgentSuggestion:
    agent_name: str         # "hedge_agent_v4"
    timestamp: datetime
    symbol: str
    direction: DirectionEnum  # LONG | SHORT | NEUTRAL
    confidence: float       # 0.0 to 1.0
    reasoning: str
    target_allocation: float
```

---

## Key Formulas

### Dealer Gamma Sign
```
dealer_gamma_sign = (Σ call_OI - Σ put_OI) / (Σ call_OI + Σ put_OI)
```

### Greek Pressures
```
gamma_pressure = Σ |Γᵢ × OIᵢ| / n
vanna_pressure = Σ |νᵢ × Δᵢ × OIᵢ| / n  
charm_pressure = Σ |Θᵢ × Δᵢ × OIᵢ| / n
```

### Directional Pressures
```
pressure_up   = Σ Γᵢ × OIᵢ   (where type=call AND Δ > 0.3)
pressure_down = Σ |Γᵢ × OIᵢ| (where type=put AND Δ < -0.3)
pressure_net  = pressure_up - pressure_down
```

### Elasticity (IMH)
```
base_elasticity = γ_pressure × 0.6 + ν_pressure × 0.4

# Flow regression multiplier
flows, prices = ledger_history
flow_multiplier = OLS(prices ~ flows).β₁

# OI-weighted elasticity
weighted_elasticity = Σ (OIᵢ/Σ OI) × |Γᵢ × Δᵢ|

elasticity = base × (1 + flow_multiplier) + weighted_elasticity
```

### Movement Energy
```
movement_energy = |pressure_net| / elasticity

energy_asymmetry = (pressure_up - pressure_down) / (pressure_up + pressure_down)
```

### Regime Detection (GMM)
```
features = [dealer_gamma, γ_pressure, ν_pressure, charm_pressure,
            movement_energy, energy_asymmetry, jump_intensity, liquidity_friction]

GMM(n_components=3) → {stability, expansion, squeeze}
```

---

## Signal Interpretation

| Metric | Value | Interpretation | Action |
|--------|-------|----------------|--------|
| energy_asymmetry | > +0.3 | Bullish call pressure | LONG bias |
| energy_asymmetry | < -0.3 | Bearish put pressure | SHORT bias |
| movement_energy | High | Big move imminent | Increase size |
| elasticity | Low | Price moves easily | Expect volatility |
| dealer_gamma_sign | Positive | Dealers short gamma | Amplified moves |
| dealer_gamma_sign | Negative | Dealers long gamma | Dampened moves |
| regime | expansion | High energy state | Directional trades |
| regime | squeeze | Building pressure | Breakout expected |
| regime | stability | Low volatility | Range strategies |

---

## Example Flow

```
1. OPTIONS DATA: SPY chain with 500 contracts
   └── Calls: 60% OI, Puts: 40% OI

2. HEDGE ENGINE:
   └── dealer_gamma_sign = (0.6 - 0.4) / 1.0 = +0.2 (slightly short gamma)
   └── gamma_pressure = 0.05
   └── vanna_pressure = 0.03
   └── pressure_up = 150, pressure_down = 80
   └── pressure_net = +70 (bullish)
   └── elasticity = 0.08
   └── movement_energy = 70 / 0.08 = 875 (high)
   └── energy_asymmetry = (150-80)/(150+80) = +0.30 (bullish)
   └── regime = "expansion"
   └── confidence = 0.75

3. HEDGE AGENT:
   └── energy_asymmetry (+0.30) > threshold (+0.3)
   └── direction = LONG
   └── confidence = 0.75 × (1 + min(0.5, 875/100)) = 0.75 × 1.5 = 1.0
   └── reasoning = "Positive energy asymmetry (0.30)"

4. COMPOSER:
   └── hedge_signal = {direction: "bullish", confidence: 1.0}
   └── Combined with sentiment, liquidity signals
   └── Output: {direction: "LONG", confidence: 0.85}
```

---

## Integration Points

### With Sentiment Engine
- Hedge provides **institutional flow** direction
- Sentiment provides **retail/news** direction
- Combined for **consensus**

### With Liquidity Engine
- Hedge provides **options pressure** analysis
- Liquidity provides **market quality** and **PENTA confluence**
- Combined for **execution quality** assessment

### With Physics Engine
- Hedge **movement_energy** ≈ Physics **kinetic_energy**
- Hedge **elasticity** ≈ Physics **mass** (resistance to movement)
- Both measure **potential for price movement**

---

*GNOSIS Hedge Pipeline - Dealer Flow & Market Elasticity Analysis*
