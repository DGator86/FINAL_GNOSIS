# GNOSIS Trading System - Information Flow

## Complete Data Flow Architecture

```
═══════════════════════════════════════════════════════════════════════════════════════════
                              GNOSIS INFORMATION FLOW DIAGRAM
═══════════════════════════════════════════════════════════════════════════════════════════

LAYER 1: EXTERNAL DATA SOURCES
══════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MASSIVE.COM   │    │     ALPACA      │    │ UNUSUAL WHALES  │    │  SOCIAL MEDIA   │
│    (Polygon)    │    │                 │    │                 │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Daily OHLCV   │    │ • Real-time     │    │ • Options Flow  │    │ • Twitter/X     │
│ • Hourly OHLCV  │    │   Quotes        │    │ • Unusual       │    │ • Reddit        │
│ • Options Chain │    │ • OHLCV Bars    │    │   Activity      │    │ • r/WSB         │
│ • Snapshots     │    │ • Account       │    │ • Smart Money   │    │ • r/stocks      │
│ • Greeks        │    │ • Orders        │    │ • Alerts        │    │ • News          │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │                      │
         │    API Keys:         │    API Keys:         │    API Key:          │
         │    Primary/Secondary │    Key/Secret        │    Token             │
         │                      │                      │                      │
         ▼                      ▼                      ▼                      ▼
═══════════════════════════════════════════════════════════════════════════════════════════

LAYER 2: INPUT ADAPTERS (Data Normalization)
══════════════════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              /engines/inputs/                                           │
│                                                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐          │
│  │ MassiveMarketAdapter │  │ AlpacaMarketAdapter  │  │ UnusualWhalesAdapter │          │
│  │                      │  │                      │  │                      │          │
│  │ Input:               │  │ Input:               │  │ Input:               │          │
│  │  • API Response      │  │  • API Response      │  │  • API Response      │          │
│  │                      │  │                      │  │                      │          │
│  │ Output:              │  │ Output:              │  │ Output:              │          │
│  │  • Bar (dataclass)   │  │  • Bar (dataclass)   │  │  • FlowData          │          │
│  │    - timestamp       │  │    - timestamp       │  │    - calls_volume    │          │
│  │    - open            │  │    - open            │  │    - puts_volume     │          │
│  │    - high            │  │    - high            │  │    - premium         │          │
│  │    - low             │  │    - low             │  │    - sentiment       │          │
│  │    - close           │  │    - close           │  │                      │          │
│  │    - volume          │  │    - volume          │  │                      │          │
│  └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘          │
│             │                         │                         │                      │
│  ┌──────────────────────┐  ┌──────────────────────┐                                    │
│  │ MassiveOptionsAdapter│  │ SocialMediaAdapter   │                                    │
│  │                      │  │                      │                                    │
│  │ Output:              │  │ Output:              │                                    │
│  │  • OptionsChain      │  │  • SocialSentiment   │                                    │
│  │    - strikes         │  │    - twitter_score   │                                    │
│  │    - expirations     │  │    - reddit_score    │                                    │
│  │    - calls/puts      │  │    - wsb_score       │                                    │
│  │    - greeks          │  │    - confidence      │                                    │
│  │    - IV              │  │                      │                                    │
│  └──────────┬───────────┘  └──────────┬───────────┘                                    │
│             │                         │                                                │
└─────────────┼─────────────────────────┼────────────────────────────────────────────────┘
              │                         │
              ▼                         ▼
═══════════════════════════════════════════════════════════════════════════════════════════

LAYER 3: CORE ANALYSIS ENGINES (Signal Generation)
══════════════════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│   PRICE PHYSICS ENGINE                 SENTIMENT ENGINE V3                             │
│   ════════════════════                 ═══════════════════                             │
│                                                                                         │
│   Input:                               Input:                                           │
│   ┌─────────────────────┐              ┌─────────────────────┐                         │
│   │ • prices (Series)   │              │ • prices (Series)   │                         │
│   │ • volumes (Series)  │              │ • highs (Series)    │                         │
│   │ • market_cap        │              │ • lows (Series)     │                         │
│   └─────────────────────┘              │ • flow_data         │                         │
│                                        │ • social_data       │                         │
│   Processing:                          └─────────────────────┘                         │
│   ┌─────────────────────┐                                                              │
│   │ 1. Calculate Mass   │              Processing:                                     │
│   │    mega_cap → 10    │              ┌─────────────────────┐                         │
│   │    large_cap → 5    │              │ 1. RSI Calculation  │                         │
│   │    mid_cap → 2      │              │    gain/loss ratio  │                         │
│   │    small_cap → 0.5  │              │    14-period        │                         │
│   │                     │              │                     │                         │
│   │ 2. Calculate        │              │ 2. MACD Calculation │                         │
│   │    Velocity         │              │    EMA(12) - EMA(26)│                         │
│   │    = returns[-5:]   │              │    Signal: EMA(9)   │                         │
│   │      .mean() * 100  │              │                     │                         │
│   │                     │              │ 3. Momentum         │                         │
│   │ 3. Acceleration     │              │    5/10/20 period   │                         │
│   │    = Δvelocity      │              │    ROC              │                         │
│   │                     │              │                     │                         │
│   │ 4. Energy           │              │ 4. Stochastic       │                         │
│   │    = (vol_ratio-1)  │              │    %K, %D           │                         │
│   │      * 100          │              │                     │                         │
│   │                     │              │ 5. Social Blend     │                         │
│   │ 5. Momentum         │              │    weight: 15%      │                         │
│   │    = mass * vel     │              │                     │                         │
│   │                     │              │ 6. Flow Blend       │                         │
│   │ 6. Kinetic Energy   │              │    weight: 35%      │                         │
│   │    = 0.5*m*v²       │              └─────────────────────┘                         │
│   └─────────────────────┘                                                              │
│                                        Output:                                          │
│   Output:                              ┌─────────────────────┐                         │
│   ┌─────────────────────┐              │ SentimentState      │                         │
│   │ PriceParticle       │              │ • rsi: 0-100        │                         │
│   │ • price: float      │              │ • rsi_signal        │                         │
│   │ • velocity: float   │              │ • macd: float       │                         │
│   │ • acceleration      │              │ • macd_histogram    │                         │
│   │ • mass: 0.5-10      │              │ • macd_cross        │                         │
│   │ • energy: float     │              │ • momentum_signal   │                         │
│   │ • kinetic_energy    │              │ • stoch_signal      │                         │
│   │ • potential_energy  │              │ • overall: -1 to 1  │                         │
│   │ • momentum: float   │              │ • strength: 0-5     │                         │
│   │ • force: float      │              │ • confidence: 0-1   │                         │
│   │ • uncertainty       │              └──────────┬──────────┘                         │
│   └──────────┬──────────┘                         │                                    │
│              │                                    │                                    │
└──────────────┼────────────────────────────────────┼────────────────────────────────────┘
               │                                    │
               │    ┌───────────────────────────────┘
               │    │
               ▼    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│   LIQUIDITY ENGINE V5 (PENTA)          PREDICTION ENGINE                               │
│   ═══════════════════════════          ═════════════════                               │
│                                                                                         │
│   Input:                               Input:                                           │
│   ┌─────────────────────┐              ┌─────────────────────┐                         │
│   │ • bars (OHLCV)      │              │ • prices (Series)   │                         │
│   │ • current_price     │              │ • highs (Series)    │                         │
│   │ • quote (bid/ask)   │              │ • lows (Series)     │                         │
│   └─────────────────────┘              └─────────────────────┘                         │
│                                                                                         │
│   Processing:                          Processing:                                      │
│   ┌─────────────────────────────┐      ┌─────────────────────────────┐                 │
│   │ PENTA SUB-ENGINES:          │      │ PREDICTIVE CONES:           │                 │
│   │                             │      │                             │                 │
│   │ 1. WYCKOFF VSA              │      │ daily_vol = returns.std()   │                 │
│   │    • Phase detection        │      │ annual_vol = daily * √252   │                 │
│   │    • Event identification   │      │                             │                 │
│   │    • Accumulation/Distrib   │      │ For horizon in [1,5,10,21]: │                 │
│   │    Output: phase, event,    │      │   σ_t = daily_vol * √days   │                 │
│   │            bias, confidence │      │   upper_1σ = price * e^σ    │                 │
│   │                             │      │   lower_1σ = price * e^-σ   │                 │
│   │ 2. ICT CONCEPTS             │      │   upper_2σ = price * e^2σ   │                 │
│   │    • Fair Value Gaps        │      │   lower_2σ = price * e^-2σ  │                 │
│   │    • Order Blocks           │      │                             │                 │
│   │    • OTE (Optimal Entry)    │      │ SUPPORT/RESISTANCE:         │                 │
│   │    • Premium/Discount       │      │                             │                 │
│   │    Output: bias, fvg_count, │      │ 1. Find swing highs/lows    │                 │
│   │            ob_count, conf   │      │    (2 bars each side)       │                 │
│   │                             │      │                             │                 │
│   │ 3. ORDER FLOW               │      │ 2. Calculate pivot points   │                 │
│   │    • CVD (Cum. Vol. Delta)  │      │    pivot = (H+L+C)/3        │                 │
│   │    • Absorption detection   │      │    R1 = 2*pivot - L         │                 │
│   │    • Exhaustion patterns    │      │    S1 = 2*pivot - H         │                 │
│   │    Output: cvd_trend,       │      │                             │                 │
│   │            absorption,      │      │ 3. Cluster nearby levels    │                 │
│   │            exhaustion       │      │    (2% threshold)           │                 │
│   │                             │      │                             │                 │
│   │ 4. SUPPLY & DEMAND          │      │ 4. Calculate distances      │                 │
│   │    • Zone detection         │      │    to nearest levels        │                 │
│   │    • Zone strength          │      └─────────────────────────────┘                 │
│   │    • Fresh zones count      │                                                      │
│   │    Output: demand_zones,    │      Output:                                         │
│   │            supply_zones,    │      ┌─────────────────────────────┐                 │
│   │            nearest_levels   │      │ PredictiveCone              │                 │
│   │                             │      │ • current_price             │                 │
│   │ 5. LIQUIDITY CONCEPTS       │      │ • upper_1std[]              │                 │
│   │    • Liquidity pools        │      │ • lower_1std[]              │                 │
│   │    • Voids                  │      │ • upper_2std[]              │                 │
│   │    • Inducements            │      │ • lower_2std[]              │                 │
│   │    • Strong/Weak H/L        │      │ • expected_path[]           │                 │
│   │    Output: buy/sell_pools,  │      │ • annualized_vol            │                 │
│   │            voids, structure │      │                             │                 │
│   │                             │      │ SupportResistance           │                 │
│   │ CONFLUENCE CALCULATION:     │      │ • support_levels[]          │                 │
│   │                             │      │ • resistance_levels[]       │                 │
│   │ agreeing = count(same_bias) │      │ • nearest_support           │                 │
│   │                             │      │ • nearest_resistance        │                 │
│   │ PENTA (5/5) → +30% conf     │      │ • distance_to_support_%     │                 │
│   │ QUAD  (4/5) → +25% conf     │      │ • distance_to_resistance_%  │                 │
│   │ TRIPLE(3/5) → +15% conf     │      │ • pivot, R1, R2, S1, S2     │                 │
│   │ DOUBLE(2/5) → +8% conf      │      └──────────┬──────────────────┘                 │
│   └─────────────────────────────┘                 │                                    │
│                                                   │                                    │
│   Output:                                         │                                    │
│   ┌─────────────────────────────┐                 │                                    │
│   │ LiquidityState              │                 │                                    │
│   │ • ad_trend                  │                 │                                    │
│   │ • bb_squeeze: bool          │                 │                                    │
│   │ • bb_width: float           │                 │                                    │
│   │ • bb_position: 0-1          │                 │                                    │
│   │ • obv_trend                 │                 │                                    │
│   │ • volume_ratio              │                 │                                    │
│   │ • liquidity_score: -1 to 1  │                 │                                    │
│   │ • breakout_prob: 0-1        │                 │                                    │
│   │                             │                 │                                    │
│   │ PENTAState                  │                 │                                    │
│   │ • confluence_level          │                 │                                    │
│   │ • combined_bias             │                 │                                    │
│   │ • combined_confidence       │                 │                                    │
│   │ • agreeing_methodologies    │                 │                                    │
│   └──────────┬──────────────────┘                 │                                    │
│              │                                    │                                    │
└──────────────┼────────────────────────────────────┼────────────────────────────────────┘
               │                                    │
               └────────────────┬───────────────────┘
                                │
                                ▼
═══════════════════════════════════════════════════════════════════════════════════════════

LAYER 4: SIGNAL AGGREGATION (Regime Classification)
══════════════════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                            MARKET REGIME CLASSIFIER                                     │
│                                                                                         │
│   Input:                                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ • SentimentState.overall_sentiment    (-1 to 1)                                 │  │
│   │ • LiquidityState.liquidity_score      (-1 to 1)                                 │  │
│   │ • PriceParticle.momentum              (normalized to -1 to 1)                   │  │
│   │ • LiquidityState.bb_squeeze           (bool)                                    │  │
│   │ • LiquidityState.bb_width             (float)                                   │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   Processing:                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                  │  │
│   │   combined_score = (sentiment × 0.4) + (liquidity × 0.3) + (momentum × 0.3)    │  │
│   │                                                                                  │  │
│   │   VOLATILITY CHECK (Priority):                                                  │  │
│   │   ├── if bb_squeeze → LOW_VOL (breakout pending)                               │  │
│   │   └── if bb_width > 10% → HIGH_VOL (volatile)                                  │  │
│   │                                                                                  │  │
│   │   DIRECTIONAL CLASSIFICATION:                                                   │  │
│   │   ├── combined_score ≥ 0.6  → STRONG_BULL                                      │  │
│   │   ├── combined_score ≥ 0.3  → BULL                                             │  │
│   │   ├── combined_score ≤ -0.6 → STRONG_BEAR                                      │  │
│   │   ├── combined_score ≤ -0.3 → BEAR                                             │  │
│   │   └── else                  → NEUTRAL                                          │  │
│   │                                                                                  │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   Output:                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ MarketRegime (Enum)                                                             │  │
│   │ ├── STRONG_BULL    │ Strong uptrend with high confidence                       │  │
│   │ ├── BULL           │ Moderate uptrend                                          │  │
│   │ ├── NEUTRAL        │ Range-bound, no clear direction                           │  │
│   │ ├── BEAR           │ Moderate downtrend                                        │  │
│   │ ├── STRONG_BEAR    │ Strong downtrend with high confidence                     │  │
│   │ ├── HIGH_VOL       │ High volatility, direction unclear                        │  │
│   │ └── LOW_VOL        │ Compressed (squeeze), breakout pending                    │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
═══════════════════════════════════════════════════════════════════════════════════════════

LAYER 5: STRATEGY SELECTION (Options Strategy)
══════════════════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          OPTIONS STRATEGY SELECTOR                                      │
│                                                                                         │
│   Input:                                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ • MarketRegime                                                                  │  │
│   │ • SentimentState (direction_score, confidence)                                  │  │
│   │ • LiquidityState (bb_squeeze, breakout_probability)                            │  │
│   │ • PriceParticle (momentum)                                                      │  │
│   │ • PredictiveCone (volatility)                                                   │  │
│   │ • SupportResistance (nearest levels)                                            │  │
│   │ • IV Percentile (historical volatility rank)                                    │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   Processing:                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                  │  │
│   │   direction_score = (sentiment × 0.4) + (liquidity × 0.3) + (momentum × 0.3)   │  │
│   │   vol_expanding = bb_squeeze OR breakout_prob > 0.5                            │  │
│   │   iv_high = iv_percentile > 60                                                 │  │
│   │   iv_low = iv_percentile < 40                                                  │  │
│   │                                                                                  │  │
│   │   STRATEGY SELECTION LOGIC:                                                     │  │
│   │                                                                                  │  │
│   │   if |direction_score| ≥ 0.6 AND confidence ≥ 0.6:                             │  │
│   │       if direction > 0 (BULLISH):                                              │  │
│   │           iv_high → BULL_CALL_SPREAD (reduce cost)                             │  │
│   │           else    → LONG_CALL                                                  │  │
│   │       if direction < 0 (BEARISH):                                              │  │
│   │           iv_high → BEAR_PUT_SPREAD                                            │  │
│   │           else    → LONG_PUT                                                   │  │
│   │                                                                                  │  │
│   │   elif |direction_score| ≥ 0.3:                                                │  │
│   │       direction > 0 → BULL_CALL_SPREAD                                         │  │
│   │       direction < 0 → BEAR_PUT_SPREAD                                          │  │
│   │                                                                                  │  │
│   │   else (LOW CONVICTION - Volatility Play):                                     │  │
│   │       if vol_expanding OR bb_squeeze:                                          │  │
│   │           iv_low  → LONG_STRADDLE (cheap vol)                                  │  │
│   │           else    → LONG_STRANGLE                                              │  │
│   │       else (range-bound):                                                      │  │
│   │           iv_high → IRON_CONDOR (sell premium)                                 │  │
│   │           else    → LONG_STRADDLE (expect breakout)                            │  │
│   │                                                                                  │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   Output:                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ OptionsStrategy (Enum):                                                         │  │
│   │ ├── LONG_CALL         │ Bullish, low IV                                        │  │
│   │ ├── LONG_PUT          │ Bearish, low IV                                        │  │
│   │ ├── BULL_CALL_SPREAD  │ Bullish, high IV or moderate conviction                │  │
│   │ ├── BEAR_PUT_SPREAD   │ Bearish, high IV or moderate conviction                │  │
│   │ ├── LONG_STRADDLE     │ Expecting big move, low IV                             │  │
│   │ ├── LONG_STRANGLE     │ Expecting big move, moderate IV                        │  │
│   │ ├── IRON_CONDOR       │ Range-bound, high IV                                   │  │
│   │ ├── BUTTERFLY         │ Pinning to specific price                              │  │
│   │ └── CALENDAR_SPREAD   │ Time decay play                                        │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
═══════════════════════════════════════════════════════════════════════════════════════════

LAYER 6: POSITION CONSTRUCTION (Black-Scholes Pricing)
══════════════════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          POSITION CONSTRUCTION                                          │
│                                                                                         │
│   Input:                                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ • OptionsStrategy                                                               │  │
│   │ • underlying_price (S)                                                          │  │
│   │ • volatility (σ)                                                                │  │
│   │ • days_to_expiration (T = DTE/365)                                              │  │
│   │ • risk_free_rate (r = 0.05)                                                     │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   Processing (Black-Scholes):                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                  │  │
│   │   ATM Strike (K) = round(S / 5) × 5                                            │  │
│   │                                                                                  │  │
│   │   d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)                                         │  │
│   │   d₂ = d₁ - σ√T                                                                 │  │
│   │                                                                                  │  │
│   │   CALL Price = S·N(d₁) - K·e^(-rT)·N(d₂)                                       │  │
│   │   PUT Price  = K·e^(-rT)·N(-d₂) - S·N(-d₁)                                     │  │
│   │                                                                                  │  │
│   │   GREEKS:                                                                        │  │
│   │   ├── Delta (Δ) = N(d₁) for call, N(d₁)-1 for put                              │  │
│   │   ├── Gamma (Γ) = n(d₁) / (S·σ·√T)                                             │  │
│   │   ├── Theta (Θ) = daily time decay                                             │  │
│   │   └── Vega (ν)  = S·n(d₁)·√T / 100                                             │  │
│   │                                                                                  │  │
│   │   POSITION SIZING:                                                               │  │
│   │   max_risk = capital × max_position_pct (5%)                                   │  │
│   │   contracts = max_risk / (max_loss × 100)                                      │  │
│   │   contracts = clamp(contracts, 1, 10)                                          │  │
│   │                                                                                  │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   Output:                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ OptionsPosition                                                                 │  │
│   │ ├── strategy: OptionsStrategy                                                   │  │
│   │ ├── legs: [{type, strike, action, price}, ...]                                 │  │
│   │ ├── entry_price: float (total premium)                                         │  │
│   │ ├── underlying_price: float                                                     │  │
│   │ ├── delta, gamma, theta, vega: float                                           │  │
│   │ ├── contracts: int                                                              │  │
│   │ ├── max_profit: float                                                           │  │
│   │ ├── max_loss: float                                                             │  │
│   │ ├── breakeven: [float, ...]                                                     │  │
│   │ ├── confidence: float                                                           │  │
│   │ └── regime: MarketRegime                                                        │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
═══════════════════════════════════════════════════════════════════════════════════════════

LAYER 7: RISK MANAGEMENT & EXECUTION
══════════════════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              RISK MANAGEMENT                                            │
│                                                                                         │
│   Entry Filters:                                                                        │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ • positions.count < max_positions (5)                                           │  │
│   │ • confidence ≥ min_confidence (0.5)                                             │  │
│   │ • signal_strength ≥ min_strength (MODERATE = 3)                                 │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   Exit Conditions:                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ • STOP_LOSS:      pnl_pct ≤ -50%                                                │  │
│   │ • TAKE_PROFIT:    pnl_pct ≥ +100%                                               │  │
│   │ • DTE_EXIT:       remaining_dte ≤ 7 days                                        │  │
│   │ • SIGNAL_REVERSAL: sentiment reversed against position                          │  │
│   │ • END_OF_BACKTEST: close all remaining positions                                │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   Position Monitoring:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ current_value = Σ(Black-Scholes price for each leg)                            │  │
│   │ pnl = (current_value - entry_price) × 100 × contracts                          │  │
│   │ pnl_pct = pnl / initial_investment                                             │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TRADE EXECUTION                                            │
│                                                                                         │
│   Backtest Mode:                           Live Mode:                                   │
│   ┌─────────────────────────┐              ┌─────────────────────────┐                 │
│   │ • Theoretical pricing   │              │ • Alpaca API            │                 │
│   │ • Simulated fills       │              │ • Real order execution  │                 │
│   │ • 2% slippage estimate  │              │ • Position tracking     │                 │
│   │ • Results to JSON       │              │ • Real-time monitoring  │                 │
│   └─────────────────────────┘              └─────────────────────────┘                 │
│                                                                                         │
│   Output:                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │ Trade Record                                                                    │  │
│   │ ├── entry_date, exit_date                                                       │  │
│   │ ├── entry_price, exit_price                                                     │  │
│   │ ├── underlying_at_entry, underlying_at_exit                                     │  │
│   │ ├── gross_pnl, net_pnl (after 2% costs)                                        │  │
│   │ ├── pnl_pct                                                                     │  │
│   │ ├── exit_reason (stop_loss, take_profit, dte_exit, signal_reversal)            │  │
│   │ ├── strategy, regime, confidence                                                │  │
│   │ └── contracts                                                                   │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════
                                    END OF FLOW
═══════════════════════════════════════════════════════════════════════════════════════════
```

---

## Data Transformation Summary

| Layer | Input | Processing | Output |
|-------|-------|------------|--------|
| **1. Data Sources** | API Requests | REST/WebSocket | Raw JSON |
| **2. Adapters** | Raw JSON | Normalization | Bar, Quote, Flow dataclasses |
| **3. Engines** | OHLCV, Volume | Indicators, Analysis | Particle, Sentiment, Liquidity, Prediction states |
| **4. Regime** | Engine outputs | Weighted combination | MarketRegime enum |
| **5. Strategy** | Regime, IV, Signals | Decision tree | OptionsStrategy enum |
| **6. Position** | Strategy, Price, Vol | Black-Scholes | OptionsPosition with Greeks |
| **7. Execution** | Position, Risk params | Entry/Exit logic | Trade records, P&L |

---

## Key Data Structures

### PriceParticle (Physics Engine Output)
```python
@dataclass
class PriceParticle:
    price: float           # Current price
    velocity: float        # Rate of change (%)
    acceleration: float    # Change in velocity
    mass: float           # 0.5 (small cap) to 10 (mega cap)
    energy: float         # Volume above average (%)
    kinetic_energy: float # 0.5 * mass * velocity²
    momentum: float       # mass * velocity
    force: float          # energy * direction
    uncertainty: float    # Volatility-based range
```

### SentimentState (Sentiment Engine Output)
```python
@dataclass
class SentimentState:
    rsi: float                    # 0-100
    rsi_signal: str               # overbought, oversold, neutral
    macd: float                   # MACD line value
    macd_histogram: float         # MACD - Signal
    macd_cross: str               # bullish_cross, bearish_cross, none
    momentum_signal: str          # bullish, bearish, neutral
    overall_sentiment: float      # -1 to 1
    sentiment_strength: int       # 0-5
    confidence: float             # 0-1
```

### LiquidityState (Liquidity Engine Output)
```python
@dataclass
class LiquidityState:
    ad_trend: str             # accumulation, distribution, neutral
    bb_squeeze: bool          # Bollinger inside Keltner
    bb_width: float           # Band width %
    bb_position: float        # 0 (at lower) to 1 (at upper)
    obv_trend: str            # bullish, bearish, neutral
    volume_ratio: float       # Current / Average
    liquidity_score: float    # -1 to 1
    breakout_probability: float  # 0-1
```

### PENTAState (PENTA Methodology Output)
```python
@dataclass
class PENTAState:
    # Confluence
    confluence_level: str         # PENTA, QUAD, TRIPLE, DOUBLE, SINGLE
    combined_bias: str            # bullish, bearish, neutral
    combined_confidence: float    # 0-1 (with confluence bonus)
    agreeing_methodologies: int   # 0-5
    
    # Individual methodologies
    wyckoff_bias: str
    ict_bias: str
    order_flow_bias: str
    sd_bias: str
    lc_trend: str
```

---

## Signal Weighting

### Combined Direction Score
```
direction_score = (sentiment × 0.4) + (liquidity × 0.3) + (physics_momentum × 0.3)
```

### Confidence Calculation
```
base_confidence = average(all_engine_confidences)
confluence_bonus = {PENTA: 0.30, QUAD: 0.25, TRIPLE: 0.15, DOUBLE: 0.08}
final_confidence = base_confidence × (1 + confluence_bonus)
```

### IV Percentile
```
iv_percentile = percentile_rank(current_vol, historical_vol_252d)
iv_high = iv_percentile > 60
iv_low = iv_percentile < 40
```

---

## Example Flow (Long Straddle Entry)

```
1. DATA: SPY daily bars from Massive.com
   └── OHLCV: Open=450, High=455, Low=448, Close=452, Vol=80M

2. PHYSICS: PriceParticle
   └── mass=10 (mega-cap), velocity=0.5%, energy=20%, momentum=5

3. SENTIMENT: SentimentState
   └── RSI=55 (neutral), MACD=bullish_cross, overall=0.2, confidence=0.6

4. LIQUIDITY: LiquidityState
   └── bb_squeeze=True, breakout_prob=0.7, liquidity_score=0.1

5. PREDICTION: 
   └── Cone: [±2% at 5d, ±4% at 10d, ±6% at 21d]
   └── Support: 445, Resistance: 460

6. REGIME: LOW_VOL (bb_squeeze=True)
   └── combined_score = (0.2×0.4) + (0.1×0.3) + (0.05×0.3) = 0.125

7. STRATEGY: LONG_STRADDLE
   └── Low conviction (|0.125| < 0.3), bb_squeeze=True, iv_low=True

8. POSITION: ATM Straddle at 450 strike
   └── Call: $5.50, Put: $5.00, Total: $10.50
   └── Breakeven: [439.50, 460.50]
   └── Max Loss: $10.50, Max Profit: Unlimited

9. EXECUTION: Enter with 2 contracts
   └── Capital required: $2,100
   └── Stop loss at -50%: -$1,050
   └── Take profit at +100%: +$2,100
```

---

*GNOSIS Trading System - Complete Information Flow Documentation*
