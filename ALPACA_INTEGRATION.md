# Alpaca Integration Guide

## 🎯 Overview

Alpaca is now fully integrated into Super Gnosis for **live and paper trading execution**. This guide shows you how to set up Alpaca and start trading with real market data from Unusual Whales.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Getting Started](#getting-started)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Data Flow](#data-flow)
6. [Execution Modes](#execution-modes)
7. [API Reference](#api-reference)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

### Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPER GNOSIS PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌──────────────────┐            │
│  │ Unusual Whales  │─────▶│ DataSourceManager│            │
│  │   (PRIMARY)     │      │   Unified Data   │            │
│  └─────────────────┘      └──────────────────┘            │
│         │                         │                        │
│         │ Options Flow            │ Real-time Quotes       │
│         │ Market Tide             │ Historical Data        │
│         │ Sentiment               │                        │
│         ▼                         ▼                        │
│  ┌─────────────────────────────────────────┐              │
│  │         STRATEGY ENGINE                 │              │
│  │  - Regime Detection                     │              │
│  │  - ML Models (LightGBM/XGBoost)        │              │
│  │  - Risk Management                      │              │
│  └─────────────────────────────────────────┘              │
│                    │                                       │
│                    │ Trading Signals                       │
│                    ▼                                       │
│  ┌─────────────────────────────────────────┐              │
│  │        EXECUTION LAYER                  │              │
│  │  ┌───────────────────────────────────┐  │              │
│  │  │  AlpacaBrokerAdapter              │  │              │
│  │  │  - Market/Limit/Stop Orders       │  │              │
│  │  │  - Position Management            │  │              │
│  │  │  - Account Info                   │  │              │
│  │  │  - Real-time Fills                │  │              │
│  │  └───────────────────────────────────┘  │              │
│  └─────────────────────────────────────────┘              │
│                    │                                       │
│                    ▼                                       │
│  ┌─────────────────────────────────────────┐              │
│  │         ALPACA BROKER                   │              │
│  │  - Paper Trading (FREE)                 │              │
│  │  - Live Trading (Commission-Free)       │              │
│  └─────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

✅ **Commission-Free Trading**: No fees on stocks/ETFs  
✅ **Paper Trading**: Free testing environment with $100K paper capital  
✅ **Real-time Data**: Level 1 market data included  
✅ **Fractional Shares**: Trade partial shares  
✅ **Extended Hours**: Pre-market and after-hours trading  
✅ **Multiple Order Types**: Market, limit, stop, stop-limit  
✅ **Position Tracking**: Real-time P&L and position management  

---

## 🚀 Getting Started

### Step 1: Create Alpaca Account

1. Visit [Alpaca Markets](https://alpaca.markets/)
2. Sign up for a free account
3. Choose **Paper Trading** for testing (recommended)
4. Navigate to **Dashboard** → **API Keys**

### Step 2: Get API Keys

#### Paper Trading Keys (Recommended for Testing)
- URL: https://paper-api.alpaca.markets
- Dashboard: https://app.alpaca.markets/paper/dashboard/overview
- Starting Capital: $100,000 (paper money)

#### Live Trading Keys (Real Money)
- URL: https://api.alpaca.markets
- Dashboard: https://app.alpaca.markets/live/dashboard/overview
- **WARNING**: Real money at risk!

### Step 3: Configure Environment

Copy your API keys to `.env`:

```bash
# Copy template
cp .env.example .env

# Edit .env and add your keys:
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
```

### Step 4: Install Dependencies

```bash
# Install Alpaca SDK
pip install alpaca-py>=0.43.0

# Or install all project dependencies
pip install -r requirements.txt
```

### Step 5: Verify Connection

```bash
# Test Alpaca connection
python examples/alpaca_quick_test.py
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ALPACA_API_KEY` | Your Alpaca API key | `PKELRVPCSCXJOWMMQI...` |
| `ALPACA_SECRET_KEY` | Your Alpaca secret key | `Ezf2dYZD1M85tYBGUb...` |
| `ALPACA_BASE_URL` | API endpoint | `https://paper-api.alpaca.markets` |

### Paper vs Live Trading

**Paper Trading** (Recommended for Development):
```bash
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Live Trading** (Real Money):
```bash
ALPACA_BASE_URL=https://api.alpaca.markets
```

### Risk Management Settings

Add to your `.env`:
```bash
# Trading Configuration
MAX_POSITION_SIZE_PCT=2.0      # Max 2% of capital per trade
MAX_DAILY_LOSS_USD=5000.0      # Max $5k daily loss
MAX_PORTFOLIO_LEVERAGE=1.0     # No leverage by default
DEFAULT_CAPITAL=100000.0       # Default capital for paper trading
ORDER_TIMEOUT_SECONDS=60       # Order timeout
```

---

## 📚 Usage Examples

### Example 1: Basic Account Info

```python
from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
import os

# Initialize adapter (paper trading)
adapter = AlpacaBrokerAdapter(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
    paper=True
)

# Get account info
account = adapter.get_account()
print(f"Account ID: {account.account_id}")
print(f"Buying Power: ${account.buying_power:,.2f}")
print(f"Cash: ${account.cash:,.2f}")
print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
```

### Example 2: Place Market Order

```python
from execution.schemas import OrderRequest, OrderSide, OrderType, AssetClass

# Create order request
order = OrderRequest(
    asset_class=AssetClass.STOCK,
    symbol="SPY",
    side=OrderSide.BUY,
    quantity=10,
    order_type=OrderType.MARKET
)

# Submit order
result = adapter.place_order(order)
print(f"Order ID: {result.order_id}")
print(f"Status: {result.status}")
print(f"Symbol: {result.symbol}")
```

### Example 3: Place Limit Order

```python
from execution.schemas import TimeInForce

order = OrderRequest(
    asset_class=AssetClass.STOCK,
    symbol="SPY",
    side=OrderSide.BUY,
    quantity=5,
    order_type=OrderType.LIMIT,
    limit_price=450.00,
    time_in_force=TimeInForce.DAY
)

result = adapter.place_order(order)
print(f"Limit order placed at ${order.limit_price}")
```

### Example 4: Get Real-time Quote

```python
# Get quote from Alpaca
quote = adapter.get_quote("SPY")
print(f"Symbol: {quote.symbol}")
print(f"Bid: ${quote.bid:.2f}")
print(f"Ask: ${quote.ask:.2f}")
print(f"Mid: ${quote.mid:.2f}")
print(f"Spread: {quote.spread_pct:.2%}")
```

### Example 5: Check Positions

```python
# Get all open positions
positions = adapter.get_positions()

for pos in positions:
    print(f"\n{pos.symbol}:")
    print(f"  Quantity: {pos.quantity}")
    print(f"  Entry: ${pos.avg_entry_price:.2f}")
    print(f"  Current: ${pos.current_price:.2f}")
    print(f"  P&L: ${pos.unrealized_pnl:,.2f}")
```

### Example 6: Cancel Order

```python
# Cancel pending order
success = adapter.cancel_order("order_123")
print(f"Order cancelled: {success}")
```

### Example 7: Full Trading Workflow

```python
from engines.inputs.data_source_manager import DataSourceManager
from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
from execution.schemas import OrderRequest, OrderSide, OrderType, AssetClass
import os

# 1. Initialize data manager (Unusual Whales primary)
data_mgr = DataSourceManager(
    unusual_whales_api_key=os.getenv("UNUSUAL_WHALES_API_KEY"),
    alpaca_api_key=os.getenv("ALPACA_API_KEY"),
    alpaca_api_secret=os.getenv("ALPACA_SECRET_KEY"),
    alpaca_paper=True
)

# 2. Get real-time quote (Unusual Whales → fallback to Alpaca)
symbol = "SPY"
quote = data_mgr.fetch_quote(symbol)
print(f"Current price: ${quote['last']:.2f}")

# 3. Initialize broker
broker = AlpacaBrokerAdapter(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
    paper=True
)

# 4. Check buying power
account = broker.get_account()
print(f"Buying power: ${account.buying_power:,.2f}")

# 5. Place order with limit price
order = OrderRequest(
    asset_class=AssetClass.STOCK,
    symbol=symbol,
    side=OrderSide.BUY,
    quantity=10,
    order_type=OrderType.LIMIT,
    limit_price=quote['last'] * 0.99  # Limit at 1% below current
)

# 6. Submit order
result = broker.place_order(order)
print(f"Order submitted: {result.order_id}")
print(f"Status: {result.status}")

# 7. Monitor order status
import time
time.sleep(2)  # Wait for fill

status = broker.get_order_status(result.order_id)
print(f"Updated status: {status.status}")
if status.filled_quantity > 0:
    print(f"Filled: {status.filled_quantity} @ ${status.avg_fill_price:.2f}")
```

---

## 🔄 Data Flow

### Quote Fallback Chain

When you call `DataSourceManager.fetch_quote()`:

```
1. Unusual Whales (PRIMARY)
   ↓ (if fails)
2. Public.com (BACKUP)
   ↓ (if fails)
3. IEX Cloud (BACKUP)
   ↓ (if fails)
4. Alpaca (FINAL FALLBACK) ✅ Most Reliable
```

### Why Alpaca is Last?

- **Most Reliable**: Official broker feed, always available during market hours
- **Real-time**: Sub-second latency
- **Commission-Free**: No cost for quote data
- **Regulatory Compliant**: SEC-registered broker-dealer

---

## 🎮 Execution Modes

### 1. Paper Trading (Recommended)

**Setup:**
```python
adapter = AlpacaBrokerAdapter(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
    paper=True  # Paper trading
)
```

**Benefits:**
- ✅ Free $100K paper capital
- ✅ Real market prices
- ✅ Test strategies safely
- ✅ No financial risk
- ✅ Reset account anytime

**Use Cases:**
- Strategy development
- Backtesting validation
- Algorithm tuning
- Learning to trade

### 2. Live Trading (Real Money)

**Setup:**
```python
adapter = AlpacaBrokerAdapter(
    api_key=os.getenv("ALPACA_API_KEY"),  # LIVE keys
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
    paper=False  # LIVE TRADING
)
```

**Requirements:**
- ✅ Funded account
- ✅ Complete identity verification
- ✅ Tested strategy
- ✅ Risk management in place

**WARNING:**
⚠️ **Real money at risk!** Only use live trading after thorough paper trading and backtesting.

---

## 📖 API Reference

### AlpacaBrokerAdapter

```python
class AlpacaBrokerAdapter:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True
    )
    
    def get_account() -> AccountInfo
    def get_positions() -> List[Position]
    def get_quote(symbol: str) -> Quote
    def get_quotes_batch(symbols: List[str]) -> List[Quote]
    
    def place_order(order: OrderRequest) -> OrderResult
    def get_order_status(order_id: str) -> OrderResult
    def cancel_order(order_id: str) -> bool
```

### Order Types

```python
OrderType.MARKET       # Execute immediately at market price
OrderType.LIMIT        # Execute at specified price or better
OrderType.STOP         # Stop-loss order
OrderType.STOP_LIMIT   # Stop-loss with limit price
```

### Time in Force

```python
TimeInForce.DAY  # Valid until market close
TimeInForce.GTC  # Good til canceled
TimeInForce.IOC  # Immediate or cancel
TimeInForce.FOK  # Fill or kill
```

---

## 🧪 Testing

### Unit Tests (Mocked)

```bash
# Run unit tests (no API credentials needed)
pytest tests/execution/broker_adapters/test_alpaca_adapter.py -v
```

### Integration Tests (Real Paper Trading)

```bash
# Set up paper trading credentials first
export ALPACA_API_KEY=your_paper_key
export ALPACA_SECRET_KEY=your_paper_secret

# Run integration tests
pytest tests/execution/broker_adapters/test_alpaca_adapter.py -m integration -v
```

### Manual Testing Script

```bash
# Quick test with paper trading account
python examples/alpaca_quick_test.py
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Authentication Failed

**Error**: `BrokerConnectionError: Failed to connect to Alpaca`

**Solution:**
- Verify API keys are correct
- Check if using paper keys with paper URL
- Ensure keys haven't expired

```bash
# Test credentials
curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/account
```

#### 2. Insufficient Buying Power

**Error**: `InsufficientFundsError: Insufficient buying power`

**Solution:**
- Check account cash balance
- Reduce order size
- Wait for pending orders to settle

```python
account = adapter.get_account()
print(f"Available: ${account.buying_power:,.2f}")
```

#### 3. Symbol Not Found

**Error**: `InvalidOrderError: Symbol not found`

**Solution:**
- Verify symbol is valid (e.g., "SPY" not "S&P500")
- Check if symbol is tradeable on Alpaca
- Use Alpaca's asset search: https://alpaca.markets/docs/trading/assets/

#### 4. Market Closed

**Error**: Order rejected during market hours

**Solution:**
- Check market hours (9:30 AM - 4:00 PM ET)
- Use extended hours trading if needed:
  ```python
  order.extended_hours = True  # For AlpacaExecutor
  ```

#### 5. Rate Limiting

**Error**: API rate limit exceeded

**Solution:**
- Alpaca has generous rate limits: 200 req/min
- Add delays between requests
- Use batch operations when possible

```python
import time
time.sleep(0.3)  # 300ms delay = max 200/min
```

---

## 🎓 Best Practices

### 1. Always Start with Paper Trading

```python
# GOOD: Paper trading first
adapter = AlpacaBrokerAdapter(paper=True)

# BAD: Live trading without testing
adapter = AlpacaBrokerAdapter(paper=False)  # ❌ DON'T DO THIS
```

### 2. Implement Position Sizing

```python
from decimal import Decimal

MAX_POSITION_PCT = 0.02  # Max 2% per trade

account = adapter.get_account()
max_position_size = account.portfolio_value * MAX_POSITION_PCT

# Calculate quantity
price = quote['last']
quantity = int(max_position_size / price)
```

### 3. Use Limit Orders for Better Fills

```python
# GOOD: Limit order with smart pricing
limit_price = quote['ask'] * 0.999  # Just inside ask

order = OrderRequest(
    symbol="SPY",
    side=OrderSide.BUY,
    quantity=10,
    order_type=OrderType.LIMIT,
    limit_price=limit_price
)

# BAD: Market order (may get bad fills)
order = OrderRequest(..., order_type=OrderType.MARKET)  # ⚠️ Slippage risk
```

### 4. Monitor Order Status

```python
# Submit order
result = adapter.place_order(order)

# Monitor until filled or timeout
import time
timeout = 60  # seconds
elapsed = 0

while elapsed < timeout:
    status = adapter.get_order_status(result.order_id)
    
    if status.status == OrderStatus.FILLED:
        print("Order filled!")
        break
    elif status.status == OrderStatus.REJECTED:
        print("Order rejected!")
        break
    
    time.sleep(1)
    elapsed += 1
```

### 5. Implement Stop Losses

```python
# After entering position, place stop-loss
entry_price = 450.0
stop_price = entry_price * 0.98  # 2% stop loss

stop_order = OrderRequest(
    symbol="SPY",
    side=OrderSide.SELL,
    quantity=10,
    order_type=OrderType.STOP,
    stop_price=stop_price
)

adapter.place_order(stop_order)
```

---

## 📊 Performance Monitoring

### Track Your Trading

```python
# Get all positions
positions = adapter.get_positions()

# Calculate totals
total_pnl = sum(pos.unrealized_pnl for pos in positions)
total_value = sum(pos.market_value for pos in positions)

print(f"Total P&L: ${total_pnl:,.2f}")
print(f"Total Value: ${total_value:,.2f}")
```

### Log All Trades

```python
import json
from datetime import datetime

# Log order
log_entry = {
    "timestamp": datetime.now().isoformat(),
    "symbol": result.symbol,
    "side": result.side,
    "quantity": result.quantity,
    "price": result.avg_fill_price,
    "order_id": result.order_id
}

with open("trades.log", "a") as f:
    f.write(json.dumps(log_entry) + "\n")
```

---

## 🔗 Additional Resources

- [Alpaca Docs](https://alpaca.markets/docs/)
- [Alpaca Python SDK](https://github.com/alpacahq/alpaca-py)
- [Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)
- [API Status](https://status.alpaca.markets/)
- [Community Forum](https://forum.alpaca.markets/)

---

## 🎉 Next Steps

1. ✅ Set up paper trading account
2. ✅ Test connection with example scripts
3. ✅ Run backtests with historical data
4. ✅ Paper trade for at least 1 month
5. ✅ Review performance metrics
6. ⚠️ Consider live trading (with caution!)

---

**Happy Trading! 🚀**

*Remember: Past performance is not indicative of future results. Always trade responsibly.*
