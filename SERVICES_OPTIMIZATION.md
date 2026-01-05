# Services Optimization Guide

A comprehensive guide to optimizing costs for Super Gnosis Trading System while maintaining data quality.

---

## Current Services Analysis

| Service | Current Cost | Data Quality | Required? |
|---------|--------------|--------------|-----------|
| Alpaca (Paper) | **FREE** | High | Yes - Core broker |
| Unusual Whales | **$99-299/mo** | Premium | No - Has alternatives |
| Massive.com | **$50-200/mo** | Premium | No - Has alternatives |
| Polygon.io | **FREE-$199/mo** | High | Optional |
| Yahoo Finance | **FREE** | Good | Yes - Fallback |
| PostgreSQL | **FREE** (self-hosted) | N/A | Yes |
| Redis | **FREE** (self-hosted) | N/A | No - Memory fallback |

---

## Recommended Free/Cheap Stack

### Tier 1: Completely Free (Best for Getting Started)

**Total Cost: $0/month**

| Service | Purpose | Quality |
|---------|---------|---------|
| **Alpaca Paper Trading** | Broker + Market Data (IEX feed) | High - Real-time quotes, free paper trading |
| **Yahoo Finance (yfinance)** | Historical OHLCV, Options Chains | Good - Slight delays, reliable |
| **Polygon.io Basic** | Historical Options Data | Good - End-of-day + minute aggregates |
| **Railway Starter** | Hosting (500 hours/month) | Good - $5 credit included |

**Limitations:**
- No real-time options flow alerts
- Limited Greeks data (use Black-Scholes calculations)
- No premium sentiment indicators

---

### Tier 2: Budget-Friendly ($30-50/month)

**Total Cost: ~$40/month**

| Service | Cost | Purpose |
|---------|------|---------|
| **Alpaca Paper Trading** | FREE | Broker + IEX Market Data |
| **Polygon.io Starter** | $29/mo | 15-min delayed options data |
| **Yahoo Finance** | FREE | Historical data fallback |
| **Railway Hobby** | $5/mo | Hosting with more resources |

**Improvements over Free Tier:**
- Better options historical data
- Minute-level aggregates
- More API calls/minute

---

### Tier 3: Optimal Value ($100-150/month)

**Total Cost: ~$130/month**

| Service | Cost | Purpose |
|---------|------|---------|
| **Alpaca (SIP data)** | ~$50/mo | Real-time SIP feed (professional quotes) |
| **Polygon.io Developer** | $79/mo | Full tick data + trades |
| **Yahoo Finance** | FREE | Fallback |
| **Railway Pro** | $20/mo | Production hosting |

**Best balance of cost vs. data quality.**

---

## Service-by-Service Alternatives

### 1. Unusual Whales Alternatives

**Current:** $99-299/month for options flow + Greeks

| Alternative | Cost | Features | Data Quality |
|-------------|------|----------|--------------|
| **Polygon.io** | FREE-$79/mo | Historical options, Greeks (calculated) | High |
| **CBOE DataShop** | $50/mo | Institutional flow data | Premium |
| **Alpaca Options** | FREE (paper) | Basic options data | Good |
| **Calculate Greeks** | FREE | Black-Scholes in-code | Good |

**Recommendation:** Use **Polygon.io Starter ($29/mo)** + calculate Greeks using Black-Scholes formula (already in codebase).

### 2. Massive.com Alternatives

**Current:** $50-200/month for comprehensive market data

| Alternative | Cost | Features | Data Quality |
|-------------|------|----------|--------------|
| **Alpaca + yfinance** | FREE | OHLCV, quotes, fundamentals | Good |
| **Polygon.io** | FREE-$79/mo | Full market data suite | High |
| **Alpha Vantage** | FREE (5/min) | Fundamentals, technicals | Good |
| **Financial Modeling Prep** | FREE-$29/mo | Financials, news, ratings | Good |

**Recommendation:** Use **Alpaca (free)** for real-time + **Yahoo Finance** for historical. No need for Massive.com.

### 3. Options Greeks Sources

**Current:** Fetched from Unusual Whales API

| Alternative | Cost | Method |
|-------------|------|--------|
| **Alpaca Options Snapshot** | FREE | Native Greeks in API response |
| **Black-Scholes Calculation** | FREE | Calculate from price, IV, time |
| **Polygon.io Options** | $29+/mo | Greeks in snapshots |

**Recommendation:** Use **Alpaca's built-in Greeks** (free with paper account) or calculate using Black-Scholes (already implemented in `engines/ml/greeks_calculator.py`).

### 4. Market Data (OHLCV)

| Source | Cost | Latency | Quality |
|--------|------|---------|---------|
| **Alpaca IEX** | FREE | Real-time | Good |
| **Alpaca SIP** | $50/mo | Real-time | Premium |
| **Yahoo Finance** | FREE | 15-min delay | Good |
| **Polygon.io** | FREE | End-of-day | Good |

**Recommendation:** Alpaca IEX (free) is sufficient for most use cases.

---

## Environment Configuration by Tier

### Free Tier `.env`

```bash
# === FREE TIER CONFIGURATION ===

# Alpaca Paper Trading (FREE)
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true
ALPACA_DATA_FEED=iex

# Disable Premium Services
UNUSUAL_WHALES_API_TOKEN=
MASSIVE_API_KEY=
MASSIVE_API_ENABLED=false

# Polygon.io Free Tier (optional)
POLYGON_API_KEY=your_free_polygon_key

# Use built-in Greeks calculation
USE_CALCULATED_GREEKS=true

# Railway will provide these
DATABASE_URL=
REDIS_URL=
```

### Budget Tier `.env`

```bash
# === BUDGET TIER CONFIGURATION ===

# Alpaca Paper Trading (FREE)
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true
ALPACA_DATA_FEED=iex

# Polygon.io Starter ($29/mo)
POLYGON_API_KEY=your_polygon_starter_key
POLYGON_TIER=starter

# Disable Expensive Services
UNUSUAL_WHALES_API_TOKEN=
MASSIVE_API_KEY=
MASSIVE_API_ENABLED=false

# Use calculated Greeks
USE_CALCULATED_GREEKS=true
```

---

## What You Lose vs. Premium

### Free Tier Limitations

| Feature | Premium | Free Alternative |
|---------|---------|------------------|
| Real-time flow alerts | Unusual Whales | Not available (use scheduled scans) |
| GoldenSweep/BlockTrade | Unusual Whales | Volume spike detection |
| Market Tide sentiment | Unusual Whales | Technical indicators |
| Greek Flow | Unusual Whales | Calculate from options chain |
| Benzinga news | Massive.com | Yahoo Finance news (delayed) |

### Data Quality Comparison

| Metric | Premium Stack | Free Stack |
|--------|---------------|------------|
| Quote latency | <100ms | ~1-5 seconds |
| Greeks accuracy | Pre-calculated | 95%+ (Black-Scholes) |
| Options chains | Full chain | Major strikes |
| Historical depth | 20+ years | 5-10 years |
| API rate limits | High | Moderate |

---

## Migration Steps

### Step 1: Switch to Free Data Sources

1. Set `MASSIVE_API_ENABLED=false` in environment
2. Remove or comment out `UNUSUAL_WHALES_API_TOKEN`
3. Ensure `ALPACA_DATA_FEED=iex` (free feed)

### Step 2: Enable Fallback Calculations

The codebase already has fallbacks:
- `utils/price_provider.py` - Uses yfinance when APIs unavailable
- `engines/ml/greeks_calculator.py` - Black-Scholes calculations
- `utils/redis_cache.py` - Falls back to memory cache

### Step 3: Get Free API Keys

1. **Alpaca**: https://alpaca.markets (free paper trading)
2. **Polygon.io**: https://polygon.io (free basic tier)
3. **Alpha Vantage**: https://www.alphavantage.co (free, 5 req/min)

---

## Code Changes for Free Tier

No code changes required! The system already has:

1. **Graceful fallbacks**: All premium services degrade gracefully
2. **yfinance integration**: `utils/price_provider.py`
3. **Built-in Greeks**: `engines/ml/greeks_calculator.py`
4. **Memory cache**: Falls back when Redis unavailable

Simply configure environment variables to disable premium services.

---

## Cost Summary

| Configuration | Monthly Cost | Best For |
|--------------|--------------|----------|
| **Free Tier** | $0 | Learning, paper trading, testing |
| **Budget Tier** | ~$35/mo | Serious paper trading |
| **Optimal Tier** | ~$130/mo | Production-ready |
| **Premium (Current)** | ~$400/mo | Institutional-grade |

**Recommendation for Railway deployment: Start with Free Tier, upgrade as needed.**
