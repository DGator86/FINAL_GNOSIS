# GNOSIS Trading System - Complete Architecture

## Full System Overview

```
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    GNOSIS TRADING SYSTEM
                        Great Neural Optimization System for Intelligent Speculation
═══════════════════════════════════════════════════════════════════════════════════════════════════════

                                        DATA SOURCES
══════════════════════════════════════════════════════════════════════════════════════════════════════

┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│   MASSIVE.COM     │  │      ALPACA       │  │  UNUSUAL WHALES   │  │   SOCIAL MEDIA    │
│    (Polygon)      │  │                   │  │                   │  │                   │
├───────────────────┤  ├───────────────────┤  ├───────────────────┤  ├───────────────────┤
│ • OHLCV Bars      │  │ • Real-time Quotes│  │ • Options Flow    │  │ • Twitter/X       │
│ • Options Chain   │  │ • Order Execution │  │ • Smart Money     │  │ • Reddit (WSB)    │
│ • Greeks          │  │ • Paper Trading   │  │ • Unusual Activity│  │ • News Feeds      │
│ • Snapshots       │  │ • Account Mgmt    │  │ • Flow Alerts     │  │ • Earnings        │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │                      │
          └──────────────────────┴──────────────────────┴──────────────────────┘
                                           │
                                           ▼
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    INPUT ADAPTERS LAYER
                                    /engines/inputs/
══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │MassiveMarketAdapter │  │MassiveOptionsAdapter│  │AlpacaMarketAdapter  │  │UnusualWhalesAdapter │ │
│  │                     │  │                     │  │                     │  │                     │ │
│  │ → Bar dataclass     │  │ → OptionsChain      │  │ → Bar dataclass     │  │ → FlowData          │ │
│  │ → Quote dataclass   │  │ → OptionsContract   │  │ → Quote dataclass   │  │ → UnusualActivity   │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘ │
│             │                        │                        │                        │            │
│  ┌─────────────────────┐  ┌─────────────────────┐                                                   │
│  │PolygonOptionsAdapter│  │SocialMediaAggregator│                                                   │
│  │                     │  │                     │                                                   │
│  │ → OptionsSnapshot   │  │ → SocialSentiment   │                                                   │
│  └──────────┬──────────┘  └──────────┬──────────┘                                                   │
│             │                        │                                                              │
└─────────────┼────────────────────────┼──────────────────────────────────────────────────────────────┘
              │                        │
              └────────────────────────┼─────────────────────────────────────┐
                                       │                                     │
                                       ▼                                     ▼
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    THREE PRIMARY PIPELINES
══════════════════════════════════════════════════════════════════════════════════════════════════════

  ╔═══════════════════════════════╗    ╔═══════════════════════════════╗    ╔═══════════════════════════════╗
  ║                               ║    ║                               ║    ║                               ║
  ║      HEDGE PIPELINE           ║    ║    SENTIMENT PIPELINE         ║    ║    LIQUIDITY PIPELINE         ║
  ║      Weight: 40%              ║    ║    Weight: 40%                ║    ║    Weight: 20%                ║
  ║                               ║    ║                               ║    ║                               ║
  ╠═══════════════════════════════╣    ╠═══════════════════════════════╣    ╠═══════════════════════════════╣
  ║                               ║    ║                               ║    ║                               ║
  ║  ┌─────────────────────────┐  ║    ║  ┌─────────────────────────┐  ║    ║  ┌─────────────────────────┐  ║
  ║  │    HEDGE ENGINE V3      │  ║    ║  │  SENTIMENT ENGINE V3    │  ║    ║  │  LIQUIDITY ENGINE V5    │  ║
  ║  │                         │  ║    ║  │                         │  ║    ║  │                         │  ║
  ║  │ • Dealer Gamma Analysis │  ║    ║  │ • RSI (14-period)       │  ║    ║  │ • PENTA Methodology:    │  ║
  ║  │ • Greek Pressures:      │  ║    ║  │ • MACD (12,26,9)        │  ║    ║  │   1. Wyckoff VSA        │  ║
  ║  │   - Gamma (Γ)           │  ║    ║  │ • Momentum (5,10,20)    │  ║    ║  │   2. ICT Concepts       │  ║
  ║  │   - Vanna (∂Δ/∂σ)       │  ║    ║  │ • Stochastic            │  ║    ║  │   3. Order Flow         │  ║
  ║  │   - Charm (∂Δ/∂t)       │  ║    ║  │ • Williams %R           │  ║    ║  │   4. Supply/Demand      │  ║
  ║  │ • Directional Pressures │  ║    ║  │ • Divergence Detection  │  ║    ║  │   5. Liquidity Pools    │  ║
  ║  │ • Elasticity (IMH)      │  ║    ║  │                         │  ║    ║  │                         │  ║
  ║  │ • Movement Energy       │  ║    ║  │ Multi-Source Blend:     │  ║    ║  │ • Bollinger Bands       │  ║
  ║  │ • Energy Asymmetry      │  ║    ║  │ • News: 25%             │  ║    ║  │ • A/D Line              │  ║
  ║  │                         │  ║    ║  │ • Flow: 35%             │  ║    ║  │ • OBV                   │  ║
  ║  │ Regime Detection (GMM): │  ║    ║  │ • Technical: 25%        │  ║    ║  │ • Money Flow Index      │  ║
  ║  │ • Stability             │  ║    ║  │ • Social: 15%           │  ║    ║  │ • VWAP                  │  ║
  ║  │ • Expansion             │  ║    ║  │                         │  ║    ║  │ • Squeeze Detection     │  ║
  ║  │ • Squeeze               │  ║    ║  │ Unusual Whales:         │  ║    ║  │                         │  ║
  ║  │                         │  ║    ║  │ • Flow Sentiment        │  ║    ║  │ Market Quality:         │  ║
  ║  │ Jump/Liquidity Risk     │  ║    ║  │ • Large Trades          │  ║    ║  │ • Bid-Ask Spread        │  ║
  ║  └───────────┬─────────────┘  ║    ║  │ • Put/Call Ratio        │  ║    ║  │ • Volume                │  ║
  ║              │                ║    ║  │                         │  ║    ║  │ • Depth                 │  ║
  ║              ▼                ║    ║  │ Social Media:           │  ║    ║  │ • Impact Cost           │  ║
  ║  ┌─────────────────────────┐  ║    ║  │ • Twitter Sentiment     │  ║    ║  └───────────┬─────────────┘  ║
  ║  │    HEDGE AGENT V4       │  ║    ║  │ • Reddit Sentiment      │  ║    ║              │                ║
  ║  │                         │  ║    ║  │ • WSB Sentiment         │  ║    ║              ▼                ║
  ║  │ Direction Logic:        │  ║    ║  └───────────┬─────────────┘  ║    ║  ┌─────────────────────────┐  ║
  ║  │ • asymmetry > +0.3 →    │  ║    ║              │                ║    ║  │  LIQUIDITY AGENT V5     │  ║
  ║  │   LONG                  │  ║    ║              ▼                ║    ║  │                         │  ║
  ║  │ • asymmetry < -0.3 →    │  ║    ║  ┌─────────────────────────┐  ║    ║  │ PENTA Confluence:       │  ║
  ║  │   SHORT                 │  ║    ║  │  SENTIMENT AGENT V3     │  ║    ║  │ • 5/5 = PENTA (+30%)    │  ║
  ║  │ • else → NEUTRAL        │  ║    ║  │                         │  ║    ║  │ • 4/5 = QUAD (+25%)     │  ║
  ║  │                         │  ║    ║  │ Direction Logic:        │  ║    ║  │ • 3/5 = TRIPLE (+20%)   │  ║
  ║  │ Confidence Boost:       │  ║    ║  │ • score > +0.2 →        │  ║    ║  │ • 2/5 = DOUBLE (+10%)   │  ║
  ║  │ conf × (1 + energy/100) │  ║    ║  │   BULLISH               │  ║    ║  │                         │  ║
  ║  │                         │  ║    ║  │ • score < -0.2 →        │  ║    ║  │ Combined Bias:          │  ║
  ║  │ Multi-TF Confluence:    │  ║    ║  │   BEARISH               │  ║    ║  │ • Bullish/Bearish/      │  ║
  ║  │ • 80%+ TFs agree →      │  ║    ║  │ • else → NEUTRAL        │  ║    ║  │   Neutral               │  ║
  ║  │   has_confluence        │  ║    ║  │                         │  ║    ║  │                         │  ║
  ║  └───────────┬─────────────┘  ║    ║  └───────────┬─────────────┘  ║    ║  │ Quality Grade:          │  ║
  ║              │                ║    ║              │                ║    ║  │ A+ → F                  │  ║
  ║              │                ║    ║              │                ║    ║  └───────────┬─────────────┘  ║
  ║              │                ║    ║              │                ║    ║              │                ║
  ╚══════════════╪════════════════╝    ╚══════════════╪════════════════╝    ╚══════════════╪════════════════╝
                 │                                    │                                    │
                 │         ┌──────────────────────────┴────────────────────────────┐       │
                 │         │                                                       │       │
                 └─────────┴───────────────────────┬───────────────────────────────┴───────┘
                                                   │
                                                   ▼
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    COMPOSER AGENT V4
                                    /agents/composer/composer_agent_v4.py
══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                              SIGNAL COMPOSITION                                                  ││
│  │                                                                                                  ││
│  │  Input Signals:                                                                                  ││
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐                         ││
│  │  │ hedge_signal       │  │ sentiment_signal   │  │ liquidity_signal   │                         ││
│  │  │ • direction        │  │ • direction        │  │ • direction        │                         ││
│  │  │ • confidence       │  │ • confidence       │  │ • confidence       │                         ││
│  │  │ • weight: 40%      │  │ • weight: 40%      │  │ • weight: 20%      │                         ││
│  │  └────────────────────┘  └────────────────────┘  └────────────────────┘                         ││
│  │                                                                                                  ││
│  │  Weighted Direction:                                                                             ││
│  │    consensus = Σ (directionᵢ × weightᵢ × confidenceᵢ) / Σ (weightᵢ × confidenceᵢ)              ││
│  │                                                                                                  ││
│  │  Direction Classification:                                                                       ││
│  │    consensus > +0.3  → LONG                                                                      ││
│  │    consensus < -0.3  → SHORT                                                                     ││
│  │    else              → NEUTRAL                                                                   ││
│  │                                                                                                  ││
│  │  Consensus Score:                                                                                ││
│  │    alignment_count = count(signals agreeing with consensus)                                      ││
│  │    consensus_score = alignment_count / total_signals                                             ││
│  │                                                                                                  ││
│  │  Confidence:                                                                                     ││
│  │    base_confidence = avg(confidences) × (0.5 + 0.5 × |consensus|)                               ││
│  │    penta_bonus = {PENTA: +30%, QUAD: +25%, TRIPLE: +20%, DOUBLE: +10%}                          ││
│  │    final_confidence = base_confidence × (1 + penta_bonus) × consensus_score                     ││
│  │                                                                                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                              OPERATING MODES                                                     ││
│  │                                                                                                  ││
│  │  STANDARD:      Hedge: 40%  |  Sentiment: 40%  |  Liquidity: 20%                                ││
│  │  0DTE:          Hedge: 30%  |  Sentiment: 20%  |  Liquidity: 50%  (execution critical)          ││
│  │  CHEAP_CALL:    Hedge: 20%  |  Sentiment: 60%  |  Liquidity: 20%  (flow conviction)             ││
│  │  PENTA_CONF:    Apply PENTA bonus for high-confluence setups                                    ││
│  │                                                                                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                      │
│  OUTPUT: ComposerOutput                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  direction: "LONG" | "SHORT" | "NEUTRAL"                                                        ││
│  │  confidence: 0.0 - 1.0                                                                          ││
│  │  consensus_score: 0.0 - 1.0                                                                     ││
│  │  hedge_contribution: float                                                                      ││
│  │  sentiment_contribution: float                                                                  ││
│  │  liquidity_contribution: float                                                                  ││
│  │  penta_confluence: "PENTA" | "QUAD" | "TRIPLE" | "DOUBLE" | None                               ││
│  │  reasoning: str                                                                                 ││
│  │  risk_factors: List[str]                                                                        ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                      │
└──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                   │
                                                   ▼
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    PHYSICS + PREDICTION LAYER
══════════════════════════════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────┐    ┌────────────────────────────────────────────┐
│                                            │    │                                            │
│         PRICE PHYSICS ENGINE               │    │         PREDICTION ENGINE                  │
│                                            │    │                                            │
│  Price as Particle:                        │    │  Predictive Cones (GBM):                   │
│  ┌──────────────────────────────────────┐  │    │  ┌──────────────────────────────────────┐  │
│  │ • mass = f(market_cap)               │  │    │  │ σ_t = daily_vol × √days              │  │
│  │   - Mega cap: 10                     │  │    │  │                                      │  │
│  │   - Large cap: 5                     │  │    │  │ 1σ bounds (68%):                     │  │
│  │   - Mid cap: 2                       │  │    │  │   upper = price × e^σ                │  │
│  │   - Small cap: 0.5                   │  │    │  │   lower = price × e^-σ               │  │
│  │                                      │  │    │  │                                      │  │
│  │ • velocity = ROC (rate of change)    │  │    │  │ 2σ bounds (95%):                     │  │
│  │                                      │  │    │  │   upper = price × e^2σ               │  │
│  │ • energy = volume above average      │  │    │  │   lower = price × e^-2σ              │  │
│  │                                      │  │    │  │                                      │  │
│  │ • momentum = mass × velocity         │  │    │  │ Horizons: [1, 5, 10, 21 days]        │  │
│  │                                      │  │    │  └──────────────────────────────────────┘  │
│  │ • kinetic_energy = 0.5 × m × v²      │  │    │                                            │
│  │                                      │  │    │  Support/Resistance:                       │
│  │ • potential_energy = distance to MA  │  │    │  ┌──────────────────────────────────────┐  │
│  │                                      │  │    │  │ • Swing highs/lows                   │  │
│  │ • force = energy × direction         │  │    │  │ • Pivot points (R1, R2, S1, S2)      │  │
│  │                                      │  │    │  │ • Level clustering (2%)              │  │
│  │ • uncertainty = volatility range     │  │    │  │ • Distance to nearest %              │  │
│  └──────────────────────────────────────┘  │    │  └──────────────────────────────────────┘  │
│                                            │    │                                            │
│  Key Insight:                              │    │  Key Insight:                              │
│  Big caps need more volume energy          │    │  Price likely stays within 2σ cone        │
│  to move than small caps                   │    │  Support/resistance levels are magnets    │
│                                            │    │                                            │
└────────────────────────────────────────────┘    └────────────────────────────────────────────┘
                         │                                              │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    MARKET REGIME CLASSIFICATION
══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                      │
│  Combined Score = (Sentiment × 0.4) + (Liquidity × 0.3) + (Physics Momentum × 0.3)                  │
│                                                                                                      │
│  VOLATILITY REGIMES (Priority):                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  bb_squeeze = True     → LOW_VOL (Bollinger inside Keltner, breakout pending)                   ││
│  │  bb_width > 10%        → HIGH_VOL (Wide bands, volatile)                                        ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                      │
│  DIRECTIONAL REGIMES:                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  score ≥ +0.6   → STRONG_BULL                                                                   ││
│  │  score ≥ +0.3   → BULL                                                                          ││
│  │  score ≤ -0.6   → STRONG_BEAR                                                                   ││
│  │  score ≤ -0.3   → BEAR                                                                          ││
│  │  else           → NEUTRAL                                                                       ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                      │
└──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                   │
                                                   ▼
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    OPTIONS STRATEGY SELECTION
══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ REGIME           │ DIRECTION  │ IV PERCENTILE │ STRATEGY                                        ││
│  ├──────────────────┼────────────┼───────────────┼─────────────────────────────────────────────────┤│
│  │ STRONG_BULL      │ Bullish    │ High (>60%)   │ BULL_CALL_SPREAD (reduce IV cost)               ││
│  │ STRONG_BULL      │ Bullish    │ Low (<40%)    │ LONG_CALL (cheap premium)                       ││
│  │ STRONG_BEAR      │ Bearish    │ High (>60%)   │ BEAR_PUT_SPREAD (reduce IV cost)                ││
│  │ STRONG_BEAR      │ Bearish    │ Low (<40%)    │ LONG_PUT (cheap premium)                        ││
│  │ NEUTRAL          │ Mixed      │ High (>60%)   │ IRON_CONDOR (sell premium)                      ││
│  │ NEUTRAL          │ Mixed      │ Low (<40%)    │ LONG_STRADDLE (expect breakout)                 ││
│  │ HIGH_VOL         │ Unclear    │ Any           │ IRON_CONDOR / STRANGLE                          ││
│  │ LOW_VOL (Squeeze)│ Any        │ Low (<40%)    │ LONG_STRADDLE (breakout imminent)               ││
│  │ BULL             │ Bullish    │ Any           │ BULL_CALL_SPREAD                                ││
│  │ BEAR             │ Bearish    │ Any           │ BEAR_PUT_SPREAD                                 ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                      │
│  BLACK-SCHOLES PRICING:                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)                                                          ││
│  │  d₂ = d₁ - σ√T                                                                                  ││
│  │  Call = S·N(d₁) - K·e^(-rT)·N(d₂)                                                              ││
│  │  Put  = K·e^(-rT)·N(-d₂) - S·N(-d₁)                                                            ││
│  │  Greeks: Delta (Δ), Gamma (Γ), Theta (Θ), Vega (ν)                                             ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                      │
└──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                   │
                                                   ▼
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    RISK MANAGEMENT & EXECUTION
══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                      │
│  ENTRY FILTERS:                              EXIT CONDITIONS:                                        │
│  ┌───────────────────────────────────────┐   ┌───────────────────────────────────────────────────┐  │
│  │ • positions.count < max_positions (5) │   │ • STOP_LOSS:      P&L ≤ -50% of premium          │  │
│  │ • confidence ≥ min_confidence (0.5)   │   │ • TAKE_PROFIT:    P&L ≥ +100% gain               │  │
│  │ • signal_strength ≥ MODERATE (3)      │   │ • DTE_EXIT:       remaining_dte ≤ 7 days         │  │
│  │ • capital available for position      │   │ • SIGNAL_REVERSAL: sentiment flipped             │  │
│  └───────────────────────────────────────┘   │ • MAX_LOSS:       position max loss exceeded      │  │
│                                              └───────────────────────────────────────────────────┘  │
│  POSITION SIZING:                                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  max_risk = capital × max_position_pct (5%)                                                     ││
│  │  contracts = max_risk / (max_loss × 100)                                                        ││
│  │  contracts = clamp(contracts, 1, 10)                                                            ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                      │
│  EXECUTION:                                                                                          │
│  ┌────────────────────────────────────────────┐    ┌────────────────────────────────────────────┐  │
│  │           BACKTEST MODE                    │    │             LIVE MODE                      │  │
│  │                                            │    │                                            │  │
│  │  • Theoretical Black-Scholes pricing      │    │  • Alpaca API for order execution          │  │
│  │  • Simulated fills                        │    │  • Real market orders                       │  │
│  │  • 2% slippage/commission estimate        │    │  • Position tracking                        │  │
│  │  • Results saved to JSON                  │    │  • Real-time P&L monitoring                │  │
│  └────────────────────────────────────────────┘    └────────────────────────────────────────────┘  │
│                                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## Pipeline Summary

| Pipeline | Engine | Agent | Weight | Key Signals |
|----------|--------|-------|--------|-------------|
| **HEDGE** | HedgeEngineV3 | HedgeAgentV4 | 40% | energy_asymmetry, movement_energy, dealer_gamma, regime |
| **SENTIMENT** | SentimentEngineV3 | SentimentAgentV3 | 40% | RSI, MACD, momentum, social, flow |
| **LIQUIDITY** | LiquidityEngineV5 | LiquidityAgentV5 | 20% | PENTA confluence, market quality, BB squeeze |

---

## Key Files

### Engines
- `/engines/hedge/hedge_engine_v3.py` - Dealer flow, Greeks, elasticity
- `/engines/sentiment/sentiment_engine_v3.py` - Multi-source sentiment
- `/engines/liquidity/liquidity_engine_v5.py` - PENTA methodology
- `/engines/elasticity/elasticity_engine_v1.py` - Volatility analysis

### Agents
- `/agents/hedge_agent_v4.py` - Hedge signal generation
- `/agents/hedge_agent_v3_enhanced.py` - ML-enhanced hedge agent
- `/agents/sentiment_agent_v3.py` - Sentiment signal generation
- `/agents/liquidity_agent_v5.py` - Liquidity/PENTA signal generation
- `/agents/composer/composer_agent_v4.py` - Signal composition

### Backtesting
- `/backtesting/gnosis_options_backtest.py` - Full options backtest
- `/backtesting/mtf_backtest_engine.py` - Multi-timeframe backtest

---

## Documentation Files

- `GNOSIS_ARCHITECTURE.md` - System overview
- `docs/INFORMATION_FLOW.md` - Data transformation layers
- `docs/HEDGE_PIPELINE.md` - Complete hedge subsystem
- `docs/FULL_GNOSIS_ARCHITECTURE.md` - This file

---

*GNOSIS Trading System - Profit in Any Market Condition*
