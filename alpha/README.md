# Gnosis Alpha

**Short-term Directional Trading - Stocks & Options - PDT Friendly**

Gnosis Alpha is a simplified trading signal system optimized for:
- 📈 **0-7 day holding periods** (swing trading)
- 🎯 **Directional signals** (BUY/SELL/HOLD for stocks, CALL/PUT for options)
- 📱 **Robinhood/Webull friendly** (simple strategies only)
- ⚖️ **PDT compliant** (tracks day trades automatically)

## Quick Start

### Stock Trading
```bash
# Scan for stock signals
python -m alpha.cli scan

# Get signal for specific symbol
python -m alpha.cli signal AAPL

# Execute a stock trade
python -m alpha.cli trade AAPL
```

### Options Trading
```bash
# Scan for options opportunities
python -m alpha.cli options scan

# Get options signal (auto-selects call/put based on direction)
python -m alpha.cli options signal AAPL

# Get specific strategy
python -m alpha.cli options signal AAPL -s put

# View options chain
python -m alpha.cli options chain AAPL

# Execute options trade
python -m alpha.cli options trade AAPL call

# View options positions
python -m alpha.cli options positions
```

### ⚡ 0DTE Options (HIGH RISK)
```bash
# Scan for 0DTE opportunities
python -m alpha.cli 0dte scan

# Get 0DTE signal (shows disclaimer first)
python -m alpha.cli 0dte signal SPY

# Specific 0DTE strategy
python -m alpha.cli 0dte signal TSLA -s lotto_call

# Execute 0DTE trade (RISKY!)
python -m alpha.cli 0dte trade SPY -m 100  # Max $100

# Skip disclaimer for repeated use
python -m alpha.cli 0dte scan --no-disclaimer -y
```

## Features

### Stock Signals
Simple directional signals based on technical analysis:

```
🟢 AAPL: BUY 💪
Confidence: 78%
Entry: $192.50
Stop: $182.88
Target: $211.75
Hold: 5 days
```

### Options Signals
Retail-friendly options strategies:

```
🟢 AAPL: Buy Call
Confidence: 72%
Strike: $195.00 CALL
Expires: 2024-01-19 (21d)
Price: $3.50/contract
Stock: $192.50
Max Loss: $350.00
Break-Even: $198.50
```

## Supported Options Strategies

| Strategy | Direction | Description | Max Loss |
|----------|-----------|-------------|----------|
| **Long Call** | Bullish | Buy call option | Premium paid |
| **Long Put** | Bearish | Buy put option | Premium paid |
| **Covered Call** | Neutral/Bullish | Own shares + sell call | Stock price - premium |
| **Cash-Secured Put** | Bullish | Sell put with cash collateral | Strike - premium |

### ⚡ 0DTE Strategies (HIGH RISK)

| Strategy | Direction | Description | Risk Level |
|----------|-----------|-------------|------------|
| **Scalp Call** | Bullish | Near ATM, quick in/out | 🔥🔥 HIGH |
| **Scalp Put** | Bearish | Near ATM, quick in/out | 🔥🔥 HIGH |
| **Momentum Call** | Bullish | Slightly OTM, ride momentum | 🔥🔥 HIGH |
| **Momentum Put** | Bearish | Slightly OTM, ride momentum | 🔥🔥 HIGH |
| **Lotto Call** | Bullish | Deep OTM lottery ticket | 🔥🔥🔥 EXTREME |
| **Lotto Put** | Bearish | Deep OTM lottery ticket | 🔥🔥🔥 EXTREME |

⚠️ **0DTE WARNING**: These options expire TODAY. 100% loss is common!

### Strategy Selection Guide

| Market View | Own Shares? | Recommended Strategy |
|-------------|-------------|---------------------|
| Bullish | No | Long Call |
| Bullish | Yes | Covered Call (income) |
| Bearish | No | Long Put |
| Neutral | Yes | Covered Call |
| Neutral | No | Cash-Secured Put |

## PDT Compliance
Pattern Day Trader rules are enforced automatically:
- Tracks day trades over rolling 5-day window
- Warns before exceeding 3 day trades
- Suggests swing trades for PDT-restricted accounts

## Configuration

Set in `.env` file:

```bash
# Alpha Alpaca API
ALPHA_ALPACA_API_KEY=your_key
ALPHA_ALPACA_SECRET_KEY=your_secret
ALPHA_ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2

# Signal Settings
ALPHA_MIN_CONFIDENCE=0.65       # Minimum signal confidence
ALPHA_MAX_HOLDING_DAYS=7        # Maximum hold period
ALPHA_MAX_POSITIONS=5           # Max concurrent positions
ALPHA_STOP_LOSS_PCT=0.05        # 5% stop loss
ALPHA_TAKE_PROFIT_PCT=0.10      # 10% take profit

# Universe (comma-separated symbols)
ALPHA_UNIVERSE=AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA
```

## CLI Reference

### Stock Commands

| Command | Description |
|---------|-------------|
| `scan` | Scan universe for stock signals |
| `signal SYMBOL` | Get stock signal for symbol |
| `trade SYMBOL` | Execute stock trade |
| `status` | Show account and PDT status |
| `close SYMBOL` | Close stock position |

### Options Commands

| Command | Description |
|---------|-------------|
| `options scan` | Scan for options opportunities |
| `options signal SYMBOL` | Get options signal |
| `options chain SYMBOL` | View options chain |
| `options trade SYMBOL call/put` | Execute options trade |
| `options positions` | View options positions |

### ⚡ 0DTE Commands (HIGH RISK)

| Command | Description |
|---------|-------------|
| `0dte scan` | Scan for 0DTE opportunities |
| `0dte signal SYMBOL` | Get 0DTE signal for symbol |
| `0dte trade SYMBOL` | Execute 0DTE trade (RISKY!) |

### Examples

```bash
# Stock trading
python -m alpha.cli scan -s AAPL,TSLA,NVDA    # Scan specific symbols
python -m alpha.cli scan -d BUY               # Show only BUY signals
python -m alpha.cli signal AAPL --json        # JSON output

# Options trading
python -m alpha.cli options scan -t call      # Scan for calls only
python -m alpha.cli options scan -c 0.7       # Min 70% confidence
python -m alpha.cli options chain AAPL --min-dte 14  # 14+ days to exp
python -m alpha.cli options trade AAPL call -n 2     # Buy 2 contracts

# 0DTE trading (HIGH RISK)
python -m alpha.cli 0dte scan -s SPY,QQQ      # Scan specific symbols
python -m alpha.cli 0dte signal SPY -s scalp_call  # Scalp strategy
python -m alpha.cli 0dte signal TSLA -s lotto_call # Lotto ticket
python -m alpha.cli 0dte trade SPY -m 100     # Max $100 position
```

## Python API

### Stock Trading
```python
from alpha import AlphaSignalGenerator, AlphaTrader, SignalDirection

# Generate stock signal
generator = AlphaSignalGenerator()
signal = generator.generate_signal("AAPL")

print(signal.direction)     # SignalDirection.BUY
print(signal.confidence)    # 0.78
print(signal.entry_price)   # 192.50

# Execute trade
trader = AlphaTrader(paper=True)
order = trader.execute_signal(signal)
```

### Options Trading
```python
from alpha import OptionsSignalGenerator, OptionsTrader, OptionStrategy

# Generate options signal
generator = OptionsSignalGenerator(
    api_key="your_key",
    secret_key="your_secret",
)

# Auto-select strategy based on direction
signal = generator.generate_signal("AAPL")

# Or specify strategy
signal = generator.generate_signal("AAPL", strategy=OptionStrategy.LONG_CALL)

print(signal.strategy)       # OptionStrategy.LONG_CALL
print(signal.contracts[0])   # OptionContract with strike, expiry, etc.
print(signal.max_loss)       # Maximum possible loss
print(signal.break_even)     # Break-even price

# Execute trade
trader = OptionsTrader(paper=True)
order = trader.execute_signal(signal, contracts=1)
```

### Scan Multiple Symbols
```python
# Stock scan
stock_signals = generator.scan_universe(min_confidence=0.65)
for s in stock_signals:
    print(f"{s.symbol}: {s.direction.value}")

# Options scan
opt_generator = OptionsSignalGenerator(api_key, secret_key)
opt_signals = opt_generator.scan_for_options(
    symbols=["AAPL", "TSLA", "NVDA"],
    strategies=[OptionStrategy.LONG_CALL, OptionStrategy.LONG_PUT],
)
for s in opt_signals:
    print(f"{s.symbol}: {s.strategy.value} ({s.confidence:.0%})")
```

### ⚡ 0DTE Options (HIGH RISK)
```python
from alpha import ZeroDTEGenerator, ZeroDTEStrategy, print_0dte_disclaimer

# Always show disclaimer first!
print_0dte_disclaimer()

# Initialize 0DTE generator
generator = ZeroDTEGenerator(
    api_key="your_key",
    secret_key="your_secret",
    max_position_dollars=200,  # Keep it small!
)

# Check if 0DTE is available
if generator.is_0dte_available("SPY"):
    # Generate signal
    signal = generator.generate_signal("SPY")
    if signal:
        print(signal.to_robinhood_format())
        print(f"Risk Level: {signal.risk_level.value}")
        print(f"Max Loss: ${signal.max_loss:.2f}")
        
        # Show warnings
        for warning in signal.warnings:
            print(f"⚠️ {warning}")

# Scan multiple symbols
signals = generator.scan_0dte(
    symbols=["SPY", "QQQ", "TSLA"],
    max_dollars=200,
)
for s in signals:
    print(f"{s.symbol}: {s.strategy.value} - Risk: {s.risk_level.value}")
```

### 🤖 Machine Learning
```python
from alpha.ml import AlphaFeatureEngine, AlphaTrainer, DirectionalClassifier

# Quick training
from alpha.ml.trainer import quick_train
result = quick_train(symbols=["AAPL", "TSLA", "NVDA"], days=180)
result.print_summary()

# Extract features for prediction
engine = AlphaFeatureEngine()
features = engine.extract("AAPL")
print(f"Features: {len(features.features)}")

# Predict with trained model
prediction = result.model.predict(features.to_array())
print(f"Direction: {prediction.direction}")
print(f"Confidence: {prediction.confidence:.1%}")
print(f"Expected Return: {prediction.expected_return:+.2%}")

# Full training pipeline
from alpha.ml import AlphaTrainer, TrainingConfig
config = TrainingConfig(
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    train_start_days_ago=365,
    tune_hyperparameters=True,
)
trainer = AlphaTrainer(config)
result = trainer.train(save_model=True)
```

## Architecture

```
alpha/
├── __init__.py              # Package exports
├── alpha_config.py          # Configuration management
├── signal_generator.py      # Stock signal generation
├── technical_analyzer.py    # Technical analysis (SMA, RSI, MACD)
├── pdt_tracker.py           # PDT compliance tracking
├── alpha_trader.py          # Stock trading execution
├── options_signal.py        # Options signal generation
├── options_trader.py        # Options trading execution
├── zero_dte.py              # ⚡ 0DTE options (HIGH RISK)
├── cli.py                   # Command-line interface
├── ml/                      # 🤖 Machine Learning module
│   ├── __init__.py          # ML exports
│   ├── features.py          # Feature engineering (50 features)
│   ├── models.py            # ML models (GBM, RF, LightGBM, XGBoost)
│   ├── backtest.py          # Walk-forward backtesting
│   └── trainer.py           # Training pipeline
└── README.md                # This file
```

## Machine Learning

### ML Commands
```bash
# Train a model
python -m alpha.cli ml train -s AAPL,TSLA,NVDA -d 180

# Train with hyperparameter tuning
python -m alpha.cli ml train --tune

# Get ML prediction
python -m alpha.cli ml predict AAPL

# Run backtest
python -m alpha.cli ml backtest -s AAPL,MSFT,GOOGL -d 90

# View features for a symbol
python -m alpha.cli ml features TSLA
```

### Features (50 total)
| Category | Features |
|----------|----------|
| **Price/Trend** | price_vs_sma (5,10,20,50), sma crossovers, price changes (1-20d) |
| **Momentum** | RSI (7,14), MACD, Stochastic K/D |
| **Volatility** | ATR, Bollinger Bands, realized volatility |
| **Volume** | Volume ratios, OBV, price-volume trend |
| **Pattern** | Higher highs/lows, trend strength, gaps, range position |
| **Market Relative** | SPY relative strength, beta, correlation |

### Model Types
| Model | Best For |
|-------|----------|
| **Gradient Boosting** | General purpose, good accuracy (default) |
| **LightGBM** | Fast training, large datasets |
| **XGBoost** | Best accuracy, slower |
| **Random Forest** | Robust, less overfitting |
| **Ensemble** | Combines multiple models |

### Walk-Forward Backtesting
The backtester uses walk-forward validation to prevent look-ahead bias:
- Trains on historical data
- Retrains periodically (default: every 30 days)
- Applies realistic costs (slippage, commissions)
- Tracks stop loss, take profit, max hold time

## Options Risk Management

### Max Loss Calculation

| Strategy | Max Loss |
|----------|----------|
| Long Call | Premium paid (e.g., $3.50 × 100 = $350) |
| Long Put | Premium paid |
| Covered Call | Stock price - premium received |
| Cash-Secured Put | Strike - premium (if stock goes to $0) |

### Position Sizing
- Default: 5% of account per trade max risk
- Automatically calculates number of contracts based on max loss
- Considers buying power requirements

### Strike Selection
| Strategy | Strike Selection |
|----------|------------------|
| Long Call (high confidence) | ATM or slightly ITM |
| Long Call (moderate confidence) | Slightly OTM |
| Long Put (high confidence) | ATM or slightly ITM |
| Covered Call | 5% OTM (want to keep shares) |
| Cash-Secured Put | 5% OTM (buy at discount) |

## Comparison: Gnosis vs Gnosis Alpha

| Feature | Full Gnosis | Gnosis Alpha |
|---------|-------------|--------------|
| Instruments | Stocks + Complex Options | Stocks + Simple Options |
| Stock Strategies | Full analysis | BUY/SELL/HOLD |
| Options Strategies | 14+ (spreads, condors) | 4 (call, put, covered, CSP) |
| Multi-leg Options | Yes | No |
| Time Horizon | Multi-timeframe | 0-7 days |
| Complexity | High | Simple |
| PDT Tracking | No | Yes |
| Target User | Active traders | Robinhood/Webull retail |

## Risk Disclaimer

**This is paper trading only by default.** 

Options trading involves substantial risk of loss. You can lose your entire investment. Past performance is not indicative of future results. This software is for educational purposes only. 

**Understand the risks before trading options:**
- Time decay (theta) works against long options
- Options can expire worthless
- Volatility changes affect option prices
- Always know your max loss before entering a trade

### ⚠️ 0DTE EXTREME RISK WARNING

**0DTE (Zero Days to Expiration) options are GAMBLING, not investing:**

- 💸 **100% loss is COMMON** - Most 0DTE options expire worthless
- ⏰ **No time to recover** - Options expire at market close TODAY
- 📉 **Extreme theta decay** - Value evaporates rapidly
- 📊 **High gamma risk** - Extreme sensitivity to price changes
- 💵 **Wide spreads** - Hard to exit at fair price

**0DTE should only be traded with money you can afford to lose ENTIRELY.**

Recommended 0DTE position size: **< 1% of account** or a fixed small dollar amount (e.g., $100-$200).

## Support

See the main [FINAL_GNOSIS README](../README.md) for full documentation.
