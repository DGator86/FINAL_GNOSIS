# 🧠 GNOSIS Trading System - Complete Architecture

## 📊 System Overview

**GNOSIS** (Generative Network for Optimal Strategy and Intelligent Signals) is an institutional-grade algorithmic trading platform combining:
- Technical Analysis
- Machine Learning (LSTM, Transformers, XGBoost, RL)
- Options Flow Analysis
- Sentiment Analysis
- Liquidity Analysis (PENTA Methodology)
- Risk Management
- Price-as-Particle Physics Model

---

## 📁 Complete Directory Structure

```
/home/root/webapp/
├── 🤖 AGENTS (Agent Layer)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py, base_agent.py          # Base agent classes
│   │   ├── confidence_builder.py            # Signal confidence calculation
│   │   ├── meta_controller.py               # Master agent orchestration
│   │   ├── ml_adaptation_agent.py           # ML model adaptation
│   │   ├── regime_detection_agent.py        # Market regime classification
│   │   ├── risk_management_agent.py         # Risk management rules
│   │   │
│   │   ├── composer/                        # Signal Composition
│   │   │   ├── composer_agent_v1.py         # Basic composition
│   │   │   ├── composer_agent_v2.py         # Enhanced composition
│   │   │   ├── composer_agent_v3.py         # Advanced composition
│   │   │   ├── composer_agent_v4.py         # PENTA integration
│   │   │   └── prediction_cone.py           # Predictive cones
│   │   │
│   │   ├── hedge_agent_v3.py                # Volatility surface analysis
│   │   ├── hedge_agent_v3_enhanced.py       # Enhanced greeks management
│   │   ├── hedge_agent_v4.py                # Latest hedge agent
│   │   │
│   │   ├── liquidity_agent_v1-v5.py         # Liquidity analysis
│   │   │   └── liquidity_agent_v5.py        # PENTA methodology (Latest)
│   │   │
│   │   ├── sentiment_agent_v1-v3.py         # Sentiment analysis
│   │   │   └── sentiment_agent_v3.py        # Multi-source sentiment (Latest)
│   │   │
│   │   ├── memory/                          # Agent Memory
│   │   │   ├── episodic_memory.py           # Trade history memory
│   │   │   └── semantic_memory.py           # Pattern memory
│   │   │
│   │   └── monitoring/
│   │       └── gnosis_monitor.py            # System monitoring
│
├── ⚙️ ENGINES (Analysis Engines)
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── base.py                          # Base engine class
│   │   ├── dynamic_universe.py              # Dynamic stock universe
│   │   ├── engine_factory.py                # Engine instantiation
│   │   ├── scanner.py                       # Market scanner
│   │   │
│   │   ├── inputs/                          # Data Adapters
│   │   │   ├── adapter_factory.py           # Adapter factory
│   │   │   ├── alpaca_market_adapter.py     # Alpaca market data
│   │   │   ├── massive_market_adapter.py    # Massive.com (Polygon) data
│   │   │   ├── massive_options_adapter.py   # Options flow from Massive
│   │   │   ├── market_data_adapter.py       # Base market adapter
│   │   │   ├── news_adapter.py              # News data
│   │   │   ├── options_chain_adapter.py     # Options chains
│   │   │   ├── polygon_options_adapter.py   # Polygon direct
│   │   │   └── unusual_whales_adapter.py    # Unusual Whales flow
│   │   │
│   │   ├── hedge/                           # Hedge/Volatility Engine
│   │   │   ├── hedge_engine_v3.py           # Main hedge engine
│   │   │   ├── regime_models.py             # Regime detection models
│   │   │   ├── volatility_intel_v2.py       # Volatility intelligence
│   │   │   └── volatility_intel_v3.py       # Enhanced vol intel
│   │   │
│   │   ├── liquidity/                       # Liquidity Engine (PENTA)
│   │   │   ├── liquidity_engine_v1-v5.py    # Progression of liquidity
│   │   │   ├── liquidity_engine_v5.py       # PENTA unified engine (Latest)
│   │   │   ├── ict_engine.py                # ICT methodology
│   │   │   ├── order_flow_engine.py         # Order flow analysis
│   │   │   ├── supply_demand_engine.py      # Supply/Demand zones
│   │   │   ├── liquidity_concepts_engine.py # Liquidity pools/voids
│   │   │   └── options_execution_v2.py      # Options execution
│   │   │
│   │   ├── sentiment/                       # Sentiment Engine
│   │   │   ├── sentiment_engine_v1-v3.py    # Sentiment analysis
│   │   │   └── sentiment_engine_v3.py       # Multi-source (Latest)
│   │   │
│   │   ├── ml/                              # ML Engines
│   │   │   ├── lstm_engine.py               # LSTM predictions
│   │   │   ├── forecasting.py               # Time series forecasting
│   │   │   ├── enhancement_engine.py        # ML signal enhancement
│   │   │   ├── anomaly.py                   # Anomaly detection
│   │   │   ├── curriculum.py                # Curriculum learning
│   │   │   ├── similarity.py                # Pattern similarity
│   │   │   ├── validation.py                # Model validation
│   │   │   └── massive_options_integration.py # Massive.com integration
│   │   │
│   │   ├── elasticity/
│   │   │   └── elasticity_engine_v1.py      # Price elasticity
│   │   │
│   │   └── orchestration/
│   │       ├── pipeline_runner.py           # Pipeline execution
│   │       └── strategy_selector.py         # Strategy selection
│
├── 📈 BACKTESTING
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── metrics.py                       # Performance metrics
│   │   │
│   │   ├── gnosis_options_backtest.py       # ⭐ GNOSIS Options Engine (NEW)
│   │   │   ├── Price-as-Particle Physics Model
│   │   │   ├── Sentiment Engine (RSI, MACD, Momentum)
│   │   │   ├── Liquidity Engine (A/D, Bollinger, OBV)
│   │   │   ├── Predictive Cones
│   │   │   ├── Support/Resistance
│   │   │   └── Black-Scholes Options Pricing
│   │   │
│   │   ├── mtf_backtest_engine.py           # ⭐ MTF Backtest (NEW)
│   │   │   ├── Multi-timeframe alignment (W1, D1, H4, H1, M15)
│   │   │   ├── HTF Bias confirmation
│   │   │   ├── LTF Entry timing
│   │   │   └── Entry quality grading
│   │   │
│   │   ├── gnosis_v2_backtest.py            # Full GNOSIS backtest
│   │   ├── gnosis_v2_full_simulation.py     # Complete simulation
│   │   ├── elite_backtest_engine.py         # Elite tier backtest
│   │   ├── liquidity_sentiment_backtest.py  # L+S focused backtest
│   │   ├── composer_backtest.py             # Composer agent backtest
│   │   ├── options_backtest_engine.py       # Standard options backtest
│   │   ├── ml_backtest_engine.py            # ML model backtest
│   │   ├── ml_hyperparameter_backtest.py    # Hyperparameter optimization
│   │   ├── walk_forward_engine.py           # Walk-forward analysis
│   │   ├── strategy_optimizer.py            # Strategy optimization
│   │   ├── historical_options_manager.py    # Historical data management
│   │   ├── synthetic_options_data.py        # Synthetic data generation
│   │   └── backtest_runner_v2.py            # Generic backtest runner
│
├── 💹 TRADE EXECUTION
│   ├── trade/
│   │   ├── __init__.py
│   │   ├── gnosis_trade_agent.py            # Main trade agent
│   │   ├── gnosis_trade_agent_v2.py         # Enhanced trade agent
│   │   ├── elite_trade_agent.py             # Elite tier trading
│   │   ├── ml_trading_engine.py             # ML-driven trading
│   │   │
│   │   ├── order_executor.py                # Order execution
│   │   ├── execution_mapper.py              # Trade to order mapping
│   │   ├── trade_agent_router.py            # Agent routing
│   │   ├── options_trade_agent.py           # Options-specific trading
│   │   │
│   │   ├── portfolio_optimizer.py           # Portfolio optimization
│   │   ├── portfolio_greeks.py              # Portfolio greeks
│   │   ├── greeks_hedger.py                 # Greeks hedging
│   │   │
│   │   ├── paper_trading_engine.py          # Paper trading
│   │   ├── trading_safety.py                # Safety checks
│   │   ├── risk_analysis.py                 # Risk analysis
│   │   ├── position_lifecycle_manager.py    # Position management
│   │   ├── event_risk_manager.py            # Event risk (earnings)
│   │   │
│   │   ├── regime_classifier.py             # Regime classification
│   │   ├── structure_selector.py            # Options structure selection
│   │   └── cone_metrics.py                  # Prediction cone metrics
│
├── 🧠 ML MODELS
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                          # Base model class
│   │   │
│   │   ├── time_series/
│   │   │   ├── lstm_forecaster.py           # LSTM forecasting
│   │   │   ├── gnosis_lstm_forecaster.py    # GNOSIS LSTM
│   │   │   ├── transformer_forecaster.py    # Transformer model
│   │   │   └── attention_mechanism.py       # Attention layers
│   │   │
│   │   ├── ensemble/
│   │   │   └── xgboost_model.py             # XGBoost ensemble
│   │   │
│   │   ├── features/
│   │   │   └── feature_builder.py           # Feature engineering
│   │   │
│   │   ├── rl_agents/                       # Reinforcement Learning
│   │   │
│   │   ├── transformer_predictor.py         # Transformer predictions
│   │   ├── rl_agent.py                      # RL trading agent
│   │   ├── volatility_surface.py            # Vol surface modeling
│   │   ├── lstm_lookahead.py                # Lookahead LSTM
│   │   ├── lookahead_model.py               # Lookahead predictions
│   │   ├── hyperparameter_optimizer.py      # Hyperparameter tuning
│   │   ├── options_contracts.py             # Options modeling
│   │   │
│   │   └── trained/                         # Trained model artifacts
│
├── 🎯 ALPHA GENERATION
│   ├── alpha/
│   │   ├── __init__.py
│   │   ├── alpha_config.py                  # Alpha configuration
│   │   ├── alpha_trader.py                  # Alpha trading logic
│   │   ├── signal_generator.py              # Signal generation
│   │   ├── options_signal.py                # Options signals
│   │   ├── options_trader.py                # Options trading
│   │   ├── technical_analyzer.py            # Technical analysis
│   │   ├── zero_dte.py                      # 0-DTE strategies
│   │   ├── pdt_tracker.py                   # PDT rule tracking
│   │   ├── cli.py                           # CLI interface
│   │   │
│   │   └── ml/
│   │       ├── models.py                    # ML models
│   │       ├── features.py                  # Feature engineering
│   │       ├── trainer.py                   # Model training
│   │       └── backtest.py                  # ML backtesting
│
├── 🔧 CORE GNOSIS
│   ├── gnosis/
│   │   ├── __init__.py
│   │   ├── unified_trading_bot.py           # Main unified bot
│   │   ├── dynamic_universe_manager.py      # Universe management
│   │   ├── experiment_tracking.py           # MLflow tracking
│   │   ├── mlops_deployment.py              # MLOps deployment
│   │   ├── market_utils.py                  # Market utilities
│   │   ├── timeframe_manager.py             # Timeframe handling
│   │   │
│   │   ├── trading/                         # Trading core
│   │   ├── dashboard/                       # Dashboards
│   │   ├── scanner/                         # Market scanners
│   │   ├── memory/                          # Memory systems
│   │   └── utils/                           # Utilities
│
├── ⚙️ CONFIGURATION
│   ├── config/
│   │   ├── __init__.py
│   │   ├── gnosis_config_v2.py              # ⭐ Main config
│   │   ├── options_config_v2.py             # Options config
│   │   ├── config_models.py                 # Config models
│   │   ├── credentials.py                   # API credentials
│   │   ├── loader.py                        # Config loading
│   │   ├── validator.py                     # Config validation
│   │   ├── hyperparameters/                 # ML hyperparameters
│   │   └── research/                        # Research configs
│
├── 📊 BACKTEST RESULTS
│   ├── runs/
│   │   ├── gnosis_options_backtests/        # Options backtest results
│   │   ├── mtf_backtests/                   # MTF backtest results
│   │   ├── elite_backtests/                 # Elite tier results
│   │   ├── liquidity_sentiment/             # L+S results
│   │   ├── ml_hyperparameter_backtests/     # ML tuning results
│   │   └── walk_forward/                    # Walk-forward results
│
├── 🔌 API & SERVICES
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                          # FastAPI main
│   ├── routers/                             # API routes
│   ├── schemas/                             # Pydantic schemas
│   ├── crud/                                # Database CRUD
│   ├── db_models/                           # SQLAlchemy models
│   ├── middleware/                          # API middleware
│   │
│   ├── brokers/
│   │   ├── __init__.py
│   │   └── alpaca_client.py                 # Alpaca integration
│   │
│   └── execution/
│       └── broker_adapters/                 # Broker adapters
│
├── 🖥️ USER INTERFACE
│   ├── dashboard/
│   │   ├── trading_dashboard.py             # Main dashboard
│   │   ├── portfolio_analytics.py           # Analytics
│   │   └── magnificent7_web.py              # Mag7 dashboard
│   │
│   ├── ui/                                  # UI components
│   ├── templates/                           # HTML templates
│   └── gnosis_dashboard.py                  # Root dashboard
│
├── 📜 SCRIPTS & CLI
│   ├── scripts/
│   │   ├── gnosis_service.py                # Main service
│   │   └── run_liquidity_sentiment_backtest.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── lstm_train.py                    # LSTM training CLI
│   │   ├── pipeline_builder.py              # Pipeline CLI
│   │   ├── result_formatter.py              # Result formatting
│   │   └── commands/                        # CLI commands
│
├── 🔄 PIPELINE & ML OPS
│   ├── pipeline/                            # Data pipelines
│   ├── ml/                                  # ML workflows
│   ├── feedback/                            # Feedback loops
│   ├── notifications/                       # Alerts
│   └── storage/                             # Data storage
│
└── 📚 DOCUMENTATION
    └── docs/
        ├── guides/                          # User guides
        ├── implementation/                  # Implementation docs
        └── reviews/                         # Code reviews
```

---

## 🧬 Core Components Deep Dive

### 1. 🎯 Price-as-Particle Physics Model (NEW)

**Location:** `backtesting/gnosis_options_backtest.py`

Models price behavior using physics principles:

```python
@dataclass
class PriceParticle:
    # State
    price: float           # Current price
    velocity: float        # Rate of change (momentum)
    acceleration: float    # Change in velocity
    
    # Physical Properties
    mass: float            # Market cap inertia (1-10 scale)
    energy: float          # Volume-weighted energy
    kinetic_energy: float  # 0.5 * mass * velocity²
    potential_energy: float # Distance from equilibrium (MAs)
    
    # Derived Metrics
    momentum: float        # mass × velocity
    force: float           # Volume pressure
    friction: float        # Spread/liquidity resistance
    
    # Quantum Uncertainty
    position_uncertainty: float  # Price volatility range
    momentum_uncertainty: float  # Velocity uncertainty
```

**Mass Calculation:**
| Market Cap | Mass | Description |
|------------|------|-------------|
| ≥$1T (Mega) | 10.0 | Very hard to move (AAPL, MSFT) |
| ≥$200B (Large) | 5.0 | Hard to move |
| ≥$10B (Mid) | 2.0 | Moderate resistance |
| ≥$2B (Small) | 1.0 | Easier to move |
| <$2B (Micro) | 0.5 | Very easy to move |

**Key Physics Equations:**
- `Momentum = Mass × Velocity`
- `Kinetic Energy = 0.5 × Mass × Velocity²`
- `Force = Energy × Direction - Friction`
- `Acceleration = Force / Mass`

---

### 2. 📊 Sentiment Engine

**Location:** `backtesting/gnosis_options_backtest.py` (backtest) + `engines/sentiment/sentiment_engine_v3.py` (live)

**Indicators:**
```python
@dataclass
class SentimentState:
    # RSI (14-period)
    rsi: float                    # 0-100
    rsi_signal: str               # overbought (>70), oversold (<30), neutral
    rsi_divergence: str           # bullish_div, bearish_div, none
    
    # MACD (12, 26, 9)
    macd: float                   # MACD line
    macd_signal: float            # Signal line
    macd_histogram: float         # MACD - Signal
    macd_cross: str               # bullish_cross, bearish_cross
    macd_trend: str               # bullish, bearish, neutral
    
    # Momentum (5, 10, 20-period)
    momentum_5: float
    momentum_10: float
    momentum_20: float
    momentum_signal: str
    
    # Stochastic (14, 3)
    stoch_k: float               # %K line
    stoch_d: float               # %D line (3-period SMA of %K)
    stoch_signal: str
    
    # Williams %R (14-period)
    williams_r: float            # -100 to 0
    williams_signal: str
    
    # Combined Sentiment
    overall_sentiment: float     # -1 to +1
    sentiment_strength: SignalStrength
    confidence: float            # 0-1
```

**Sentiment Weights:**
| Indicator | Weight |
|-----------|--------|
| RSI | 0.20 |
| MACD | 0.30 |
| Momentum | 0.20 |
| Stochastic | 0.15 |
| Williams %R | 0.15 |

---

### 3. 💧 Liquidity Engine (PENTA Methodology)

**Location:** `engines/liquidity/liquidity_engine_v5.py` + `agents/liquidity_agent_v5.py`

**PENTA Sub-Engines:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    LIQUIDITY ENGINE V5 (PENTA)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │  Wyckoff  │ │    ICT    │ │Order Flow │ │Supply/Dem │       │
│  │   (VSA)   │ │(FVG, OB)  │ │(Footprint)│ │  (Zones)  │       │
│  │   18%     │ │   18%     │ │    18%    │ │    18%    │       │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │
│                         ↓                                       │
│  ┌───────────────────────────────────────────────────────┐     │
│  │              Liquidity Concepts (18%)                  │     │
│  │         (Pools, Voids, Inducements)                   │     │
│  └───────────────────────────────────────────────────────┘     │
│                         ↓                                       │
│  ┌───────────────────────────────────────────────────────┐     │
│  │                 Base Analysis (10%)                    │     │
│  │     (Bid-Ask, Depth, Tradability, Volume Profile)     │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
│  Output: PENTA Confluence (5/5, 4/5, 3/5...)                   │
└─────────────────────────────────────────────────────────────────┘
```

**Backtest Liquidity Indicators:**
```python
@dataclass  
class LiquidityState:
    # Accumulation/Distribution
    ad_line: float               # A/D line value
    ad_trend: str               # accumulating, distributing, neutral
    ad_divergence: str          # bullish_div, bearish_div, none
    
    # Bollinger Bands (20, 2)
    bb_middle: float            # SMA(20)
    bb_upper: float             # Middle + 2*std
    bb_lower: float             # Middle - 2*std
    bb_width: float             # (Upper - Lower) / Middle
    bb_position: float          # -1 to +1 (position within bands)
    bb_squeeze: bool            # True if width < threshold
    
    # On-Balance Volume
    obv: float                  # OBV value
    obv_trend: str              # bullish, bearish, neutral
    
    # Money Flow Index (14-period)
    mfi: float                  # 0-100
    mfi_signal: str             # overbought, oversold, neutral
    
    # VWAP
    vwap: float                 # Volume-weighted average price
    vwap_position: str          # above, below, at
    
    # Combined
    overall_liquidity: float    # -1 to +1
    liquidity_strength: SignalStrength
    confidence: float           # 0-1
```

---

### 4. 📐 Predictive Cones

**Location:** `backtesting/gnosis_options_backtest.py` + `agents/composer/prediction_cone.py`

```python
@dataclass
class PredictiveCone:
    current_price: float
    
    # 1-day cone (1σ, 2σ)
    day_1_upper_1s: float      # +1 std dev
    day_1_lower_1s: float      # -1 std dev
    day_1_upper_2s: float      # +2 std dev
    day_1_lower_2s: float      # -2 std dev
    
    # 5-day cone
    day_5_upper_1s: float
    day_5_lower_1s: float
    # ... etc
    
    # 10-day and 21-day similarly
    
    # Calculation: σ_T = σ_daily × √T
    # 1σ range: ±σ_T
    # 2σ range: ±2×σ_T
```

**Cone Horizons:**
| Horizon | Use Case |
|---------|----------|
| 1-day | 0-DTE options |
| 5-day | Weekly options |
| 10-day | Swing trades |
| 21-day | Monthly options |

---

### 5. 📊 Multi-Timeframe (MTF) Engine

**Location:** `backtesting/mtf_backtest_engine.py`

**Timeframe Weights:**
```python
TIMEFRAME_WEIGHTS = {
    Timeframe.W1:  0.30,   # Weekly - strongest
    Timeframe.D1:  0.25,   # Daily
    Timeframe.H4:  0.20,   # 4-hour
    Timeframe.H1:  0.15,   # 1-hour
    Timeframe.M15: 0.10,   # 15-min - entry timing
}
```

**MTF Signal Structure:**
```python
@dataclass
class MTFSignal:
    # Alignment Metrics
    alignment_score: float       # -1 to +1
    alignment_count: int         # 0-5 timeframes aligned
    weighted_confidence: float   # Weighted by TF importance
    
    # Higher Timeframe Bias
    htf_bias: str               # bullish, bearish, neutral
    htf_confidence: float       # From W1 + D1
    
    # Lower Timeframe Confirmation  
    ltf_confirms: bool          # H4/H1 confirm HTF
    ltf_confidence: float       # Entry timing quality
    
    # Entry Quality
    entry_quality: str          # perfect, strong, moderate, weak, none
    final_direction: str        # bullish, bearish, neutral
    final_confidence: float     # Overall confidence
```

**Entry Quality Grading:**
| Grade | Criteria |
|-------|----------|
| Perfect | 4/4 TF aligned + HTF confirms + LTF confirms |
| Strong | 3/4 TF aligned + HTF confirms + LTF confirms |
| Moderate | 3/4 TF aligned + (HTF or LTF confirms) |
| Weak | 2/4 TF aligned |
| None | <2 TF aligned or conflicting signals |

---

### 6. 📊 Options Strategy Selection

**Strategy Matrix:**
```
               │ Bullish      │ Bearish      │ Neutral
───────────────┼──────────────┼──────────────┼──────────────
High Vol       │ Long Call    │ Long Put     │ Long Straddle
               │ Bull Spread  │ Bear Spread  │ Long Strangle
───────────────┼──────────────┼──────────────┼──────────────
Normal Vol     │ Bull Spread  │ Bear Spread  │ Iron Condor
               │ Long Call    │ Long Put     │ Butterfly
───────────────┼──────────────┼──────────────┼──────────────
Low Vol        │ Bull Spread  │ Bear Spread  │ Calendar
(Squeeze)      │ Long Call    │ Long Put     │ (Short Straddle)
```

**Black-Scholes Implementation:**
```python
def black_scholes(S, K, T, r, sigma, option_type):
    """
    S: Spot price
    K: Strike price
    T: Time to expiry (years)
    r: Risk-free rate
    sigma: Implied volatility
    
    Returns: (price, delta, gamma, theta, vega)
    """
```

---

## 🔄 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                   │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│ Massive.com │   Alpaca    │  Unusual    │  News APIs  │ Social Media    │
│  (Polygon)  │ (Execution) │   Whales    │             │                 │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴────────┬────────┘
       │             │             │             │               │
       └─────────────┴─────────────┴─────────────┴───────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │     INPUT ADAPTERS          │
                    │ (engines/inputs/)           │
                    └──────────────┬──────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  HEDGE ENGINE   │     │LIQUIDITY ENGINE │     │SENTIMENT ENGINE │
│   (V3/V4)       │     │ (V5 - PENTA)    │     │     (V3)        │
│                 │     │                 │     │                 │
│ • Vol Surface   │     │ • Wyckoff VSA   │     │ • News          │
│ • Greeks        │     │ • ICT           │     │ • Options Flow  │
│ • Regime        │     │ • Order Flow    │     │ • Technical     │
│ • LSTM Predict  │     │ • Supply/Demand │     │ • Social        │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │      AGENT LAYER        │
                    ├─────────────────────────┤
                    │ • Hedge Agent V3/V4     │
                    │ • Liquidity Agent V5    │
                    │ • Sentiment Agent V3    │
                    │ • ML Adaptation Agent   │
                    │ • Regime Detection      │
                    │ • Risk Management       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    COMPOSER AGENT V4    │
                    │  (Signal Aggregation)   │
                    │                         │
                    │ Weights:                │
                    │ • Hedge: 40%            │
                    │ • Liquidity: 35%        │
                    │ • Sentiment: 25%        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │     TRADE AGENT V2      │
                    │  (Order Generation)     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   EXECUTION LAYER       │
                    │  (Alpaca/Paper Trade)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    FEEDBACK LOOP        │
                    │ • Trade Results         │
                    │ • Model Retraining      │
                    │ • Parameter Adjustment  │
                    └─────────────────────────┘
```

---

## 📈 Latest Backtest Results

### GNOSIS Options Backtest (2020-2024)
```
Configuration:
• Symbols: SPY, QQQ, AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOGL
• Period: 2020-01-01 to 2024-12-01
• Initial Capital: $100,000

Results:
• Total Trades: 298
• Win Rate: 37.9%
• Profit Factor: 0.86

Strategy Performance:
┌─────────────────┬────────┬──────────┬─────────────┐
│ Strategy        │ Trades │ Win Rate │ P&L         │
├─────────────────┼────────┼──────────┼─────────────┤
│ Straddles       │ 225    │ 41.3%    │ +$102,263   │
│ Spreads         │ 55     │ 21.8%    │ -$157,224   │
└─────────────────┴────────┴──────────┴─────────────┘

Market Regime Performance:
┌─────────────────┬──────────┬─────────────┐
│ Regime          │ Win Rate │ Avg P&L     │
├─────────────────┼──────────┼─────────────┤
│ High Volatility │ 34.3%    │ +$154/trade │
│ Neutral         │ 47.9%    │ +$178/trade │
│ Bear            │ 46.2%    │ -$1,455/tr  │
│ Bull            │ 33.3%    │ -$2,805/tr  │
└─────────────────┴──────────┴─────────────┘

KEY INSIGHT: Straddles in High Vol are highly profitable!
```

### MTF Backtest (2020-2024)
```
Configuration:
• Symbols: SPY, QQQ, AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOGL
• Timeframes: 1W, 1D, 4H, 1H
• Min Alignment: 3/4

Results:
• Initial Capital: $100,000
• Final Capital: $111,008.92
• Total Return: +11.01%
• Max Drawdown: 11.05%
• Total Trades: 335
• Win Rate: 32.8%
• Profit Factor: 1.13

Alignment Analysis:
┌─────────────────┬──────────┬─────────────┐
│ Alignment       │ Win Rate │ P&L         │
├─────────────────┼──────────┼─────────────┤
│ 4/4 TF Aligned  │ 33.6%    │ +$7,541     │
│ 3/4 TF Aligned  │ 30.9%    │ +$3,467     │
└─────────────────┴──────────┴─────────────┘

HTF Bias Performance:
┌─────────────────┬──────────┬─────────────┐
│ HTF Bias        │ Win Rate │ P&L         │
├─────────────────┼──────────┼─────────────┤
│ Bullish         │ 40.7%    │ +$13,867    │
│ Bearish         │ 24.1%    │ -$2,858     │
└─────────────────┴──────────┴─────────────┘

KEY INSIGHT: 4/4 alignment + Bullish HTF = Best performance!
```

---

## 🚀 Key Entry Points

### Live Trading
```bash
# Full GNOSIS trading
python gnosis_live_trading_quickstart.py

# Paper trading
python trade/paper_trading_engine.py
```

### Backtesting
```bash
# GNOSIS Options Backtest
cd backtesting && python gnosis_options_backtest.py

# MTF Backtest
cd backtesting && python mtf_backtest_engine.py

# Liquidity-Sentiment Backtest
python scripts/run_liquidity_sentiment_backtest.py
```

### Dashboards
```bash
# Main dashboard
python gnosis_dashboard.py

# Trading dashboard
python dashboard/trading_dashboard.py
```

---

## 🔑 API Integrations

| Provider | Purpose | Module |
|----------|---------|--------|
| Massive.com (Polygon) | Historical data, Options flow | `engines/inputs/massive_*.py` |
| Alpaca | Execution, Real-time data | `brokers/alpaca_client.py` |
| Unusual Whales | Options flow alerts | `engines/inputs/unusual_whales_adapter.py` |
| News APIs | Sentiment | `engines/inputs/news_adapter.py` |

---

## 📝 Configuration

**Main Config:** `config/gnosis_config_v2.py`

```python
@dataclass
class EngineConfig:
    hedge: Dict       # Regime components, gamma/vanna weights
    liquidity: Dict   # PENTA weights, thresholds
    sentiment: Dict   # Source weights (news, flow, technical)

@dataclass
class AgentConfig:
    hedge_agent: Dict      # Min confidence, energy threshold
    liquidity_agent: Dict  # Confluence threshold
    sentiment_agent: Dict  # Sentiment threshold

@dataclass
class ComposerConfig:
    weights: Dict  # hedge: 0.40, liquidity: 0.35, sentiment: 0.25
    min_confidence: float
    max_positions: int
```

---

**Version:** 2.0.0  
**Last Updated:** 2024-12-24  
**Author:** GNOSIS Trading System
