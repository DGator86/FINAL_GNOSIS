# Multi-Leg Options Trading - Quick Reference

## Available Scripts

### 1. Trade Multi-Leg Options

**File:** `scripts/trade_multi_leg_options.py`

Interactive script for executing complex options strategies:

- Bull Call Spread (bullish, limited risk)
- Bear Put Spread (bearish, limited risk)
- Long Straddle (high volatility play)
- Iron Condor (low volatility play)

**Usage:**

```bash
python scripts/trade_multi_leg_options.py
```

### 2. Get Option Greeks

**File:** `scripts/get_option_greeks.py`

Fetch real-time Greeks for any option contract:

- Delta, Gamma, Theta, Vega, Rho
- Bid/Ask spreads
- Implied Volatility

**Usage:**

```bash
python scripts/get_option_greeks.py SPY241220C00600000
python scripts/get_option_greeks.py SPY241220C00600000 SPY241220P00600000
```

## Direct API Usage

### Place Multi-Leg Order

```python
from brokers.alpaca_client import AlpacaClient

client = AlpacaClient.from_env(mode="paper")

# Bull call spread
legs = [
    {"symbol": "SPY241220C00600000", "side": "buy", "ratio_qty": 1},
    {"symbol": "SPY241220C00610000", "side": "sell", "ratio_qty": 1}
]

result = client.place_multi_leg_option_order(legs, quantity=1)
print(f"Order ID: {result['id']}")
```

### Get Option Snapshot

```python
from brokers.alpaca_client import AlpacaClient

client = AlpacaClient.from_env(mode="paper")

snapshot = client.get_option_snapshot("SPY241220C00600000")
print(f"Delta: {snapshot['SPY241220C00600000']['greeks']['delta']}")
```

## Strategy Examples

### Bull Call Spread

- **Outlook:** Moderately bullish
- **Max Profit:** Strike difference - net premium
- **Max Loss:** Net premium paid
- **Example:** Buy $600 call, Sell $610 call

### Bear Put Spread

- **Outlook:** Moderately bearish
- **Max Profit:** Strike difference - net premium
- **Max Loss:** Net premium paid
- **Example:** Buy $600 put, Sell $590 put

### Long Straddle

- **Outlook:** High volatility expected
- **Max Profit:** Unlimited
- **Max Loss:** Total premium paid
- **Example:** Buy $600 call + Buy $600 put

### Iron Condor

- **Outlook:** Low volatility expected
- **Max Profit:** Net premium received
- **Max Loss:** Spread width - net premium
- **Example:** Sell $610 call, Buy $620 call, Sell $590 put, Buy $580 put

## Option Symbol Format

Format: `UNDERLYING + YYMMDD + C/P + STRIKE`

Examples:

- `SPY241220C00600000` = SPY Dec 20, 2024 $600 Call
- `SPY241220P00590000` = SPY Dec 20, 2024 $590 Put
- `AAPL250117C00150000` = AAPL Jan 17, 2025 $150 Call

## Notes

- All scripts use **paper trading** by default
- Real-time market data during trading hours
- Greeks updated continuously
- Multi-leg orders execute atomically (all-or-nothing)
- Maximum 4 legs per order (Alpaca limit)

## Integration with Gnosis

The multi-leg functionality is integrated into:

- `brokers/alpaca_client.py` - Core API methods
- `engines/liquidity/options_execution_v2.py` - Execution module
- `pipeline/options_pipeline_v2.py` - V2 pipeline

All existing Gnosis functionality remains unchanged.
