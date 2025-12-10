# 🐋 Unusual Whales API - Available Endpoints

**Base URL**: `https://api.unusualwhales.com`
**Authentication**: `Authorization: Bearer {UUID_TOKEN}`
**Your Token**: `8932cd23-72b3-4f74-9848-13f9103b9df5`

---

## ✅ Verified Working Endpoints (Tested by Unusual Whales Support - Dec 2025)

These endpoints have been confirmed to work with your token by Dan from Unusual Whales API support:

### 1. **Flow Alerts** - Significant Options Flow
```bash
GET /api/option-trades/flow-alerts
```

**Example (Unusually Bullish filter)**:
```bash
curl -G \
  --data-urlencode "limit=5" \
  --data-urlencode "issue_types[]=Common Stock" \
  --data-urlencode "issue_types[]=ADR" \
  --data-urlencode "rule_name[]=RepeatedHits" \
  --data-urlencode "rule_name[]=RepeatedHitsAscendingFill" \
  --data-urlencode "is_call=true" \
  --data-urlencode "is_ask_side=true" \
  --data-urlencode "is_otm=true" \
  --data-urlencode "size_greater_oi=true" \
  --data-urlencode "is_multi_leg=false" \
  --data-urlencode "min_premium=250000" \
  --data-urlencode "max_dte=183" \
  --url "https://api.unusualwhales.com/api/option-trades/flow-alerts" \
  --header "Accept: application/json, text/plain" \
  --header "Authorization: Bearer 8932cd23-72b3-4f74-9848-13f9103b9df5"
```

**Use Case**: Identify significant bullish/bearish options flow in real-time

---

### 2. **Market Tide** - Overall Market Flow Sentiment
```bash
GET /api/market/market-tide
```

**Example**:
```bash
curl -G \
  --data-urlencode "interval_5m=false" \
  --url "https://api.unusualwhales.com/api/market/market-tide" \
  --header "Accept: application/json, text/plain" \
  --header "Authorization: Bearer 8932cd23-72b3-4f74-9848-13f9103b9df5"
```

**Use Case**: Monitor overall market sentiment from options flow (like the Market Tide chart on UW website)

---

### 3. **Dark Pool Trades by Ticker**
```bash
GET /api/darkpool/{ticker}
```

**Example (SPY dark pool trades ≥ $10M)**:
```bash
curl -G \
  --data-urlencode "limit=3" \
  --data-urlencode "min_premium=10000000" \
  --url "https://api.unusualwhales.com/api/darkpool/SPY" \
  --header "Accept: application/json, text/plain" \
  --header "Authorization: Bearer 8932cd23-72b3-4f74-9848-13f9103b9df5"
```

**Use Case**: Track large institutional block trades (dark pool activity)

---

## 📊 Options Chain & Contract Data

### **Option Contracts** (Currently used in adapter)
```bash
GET /api/stock/{ticker}/option-contracts
```
**Parameters**:
- `expiration_date` (optional) - Filter by specific expiration
- `limit` (default: 100, max: 500)

**Returns**: Full options chain with OCC symbols, strikes, bid/ask, volume, OI, greeks

---

### **Option Chains** (Alternative format)
```bash
GET /api/stock/{ticker}/option-chains
```
**Returns**: Full options chain data in alternative format

---

### **ATM Chains** - At-the-Money Options Only
```bash
GET /api/stock/{ticker}/atm-chains
```
**Returns**: Only ATM strikes (most liquid options)

---

## 🔥 Options Flow Data

### **Recent Flow by Ticker**
```bash
GET /api/stock/{ticker}/flow-recent
```
**Returns**: Most recent options flow for a specific ticker

---

### **Flow Alerts by Ticker**
```bash
GET /api/stock/{ticker}/flow-alerts
```
**Returns**: Flow alerts specific to a ticker

---

### **Flow Per Expiry**
```bash
GET /api/stock/{ticker}/flow-per-expiry
```
**Returns**: Options flow aggregated by expiration date

---

### **Flow Per Strike**
```bash
GET /api/stock/{ticker}/flow-per-strike
```
**Returns**: Options flow aggregated by strike price

---

### **Flow Per Strike (Intraday)**
```bash
GET /api/stock/{ticker}/flow-per-strike-intraday
```
**Returns**: Intraday flow data by strike

---

## 🎯 Greek Exposure & Analysis

### **Greek Exposure (Total)**
```bash
GET /api/stock/{ticker}/greek-exposure
```
**Returns**: Total delta, gamma, vanna, charm exposure for ticker

---

### **Greek Exposure by Expiry**
```bash
GET /api/stock/{ticker}/greek-exposure/expiry
```
**Returns**: Greek exposure broken down by expiration

---

### **Greek Exposure by Strike**
```bash
GET /api/stock/{ticker}/greek-exposure/strike
```
**Returns**: Greek exposure by strike price (GEX levels)

---

### **Greek Flow**
```bash
GET /api/stock/{ticker}/greek-flow
```
**Returns**: Flow of greeks over time

---

### **Spot Exposures** (Key Price Levels)
```bash
GET /api/stock/{ticker}/spot-exposures
```
**Returns**: Key price levels based on gamma/delta exposure

---

## 📈 Volatility & IV Data

### **Realized Volatility**
```bash
GET /api/stock/{ticker}/volatility/realized
```
**Parameters**: `timeframe` (e.g., "30d")
**Returns**: Historical/realized volatility data

---

### **Volatility Stats**
```bash
GET /api/stock/{ticker}/volatility/stats
```
**Returns**: IV percentile, rank, and volatility statistics

---

### **Volatility Term Structure**
```bash
GET /api/stock/{ticker}/volatility/term-structure
```
**Returns**: IV across different expirations (term structure curve)

---

### **Interpolated IV**
```bash
GET /api/stock/{ticker}/interpolated-iv
```
**Returns**: Interpolated implied volatility surface

---

### **IV Rank**
```bash
GET /api/stock/{ticker}/iv-rank
```
**Returns**: IV rank and percentile metrics

---

## 🎲 Open Interest & Volume Analysis

### **OI Change**
```bash
GET /api/stock/{ticker}/oi-change
```
**Returns**: Open interest changes (signals positioning changes)

---

### **OI Per Expiry**
```bash
GET /api/stock/{ticker}/oi-per-expiry
```
**Returns**: Open interest by expiration

---

### **OI Per Strike**
```bash
GET /api/stock/{ticker}/oi-per-strike
```
**Returns**: Open interest by strike (shows dealer hedging walls)

---

### **Max Pain**
```bash
GET /api/stock/{ticker}/max-pain
```
**Returns**: Max pain price level (where most options expire worthless)

---

### **Options Volume**
```bash
GET /api/stock/{ticker}/options-volume
```
**Returns**: Total options volume for ticker

---

## 📊 Market-Wide Data

### **Top Net Impact** (Trending Tickers)
```bash
GET /api/market/top-net-impact
```
**Returns**: Top tickers by net premium flow (what's moving the market)

---

### **Sector Tide**
```bash
GET /api/market/{sector}/sector-tide
```
**Returns**: Options flow sentiment for entire sector

---

### **OI Change (Market-Wide)**
```bash
GET /api/market/oi-change
```
**Returns**: Biggest open interest changes across all tickers

---

### **Spike** (Unusual Activity)
```bash
GET /api/market/spike
```
**Returns**: Tickers with spiking unusual options activity

---

### **Total Options Volume**
```bash
GET /api/market/total-options-volume
```
**Returns**: Market-wide options volume metrics

---

## 📰 News & Fundamentals

### **News Headlines**
```bash
GET /api/news/headlines
```
**Returns**: Recent news affecting stocks/options

---

### **Stock Info**
```bash
GET /api/stock/{ticker}/info
```
**Returns**: Company fundamentals, sector, industry, market cap

---

### **Earnings (Ticker)**
```bash
GET /api/earnings/{ticker}
```
**Returns**: Earnings data, EPS, guidance for ticker

---

### **Earnings (Premarket)**
```bash
GET /api/earnings/premarket
```
**Returns**: Companies reporting before market open

---

### **Earnings (After Hours)**
```bash
GET /api/earnings/afterhours
```
**Returns**: Companies reporting after market close

---

## 👔 Institutional & Insider Activity

### **Congress Trades**
```bash
GET /api/congress/recent-trades
```
**Returns**: Recent congressional stock trades (Nancy Pelosi tracker)

---

### **Congress Trader**
```bash
GET /api/congress/congress-trader
```
**Parameters**: `name` (congress member name)
**Returns**: Trades by specific congress member

---

### **Insider Trades (Ticker)**
```bash
GET /api/insider/{ticker}
```
**Returns**: Corporate insider buying/selling for ticker

---

### **Insider Transactions**
```bash
GET /api/insider/transactions
```
**Returns**: Recent insider transactions across all tickers

---

### **Institution Holdings**
```bash
GET /api/institution/{name}/holdings
```
**Returns**: Holdings of specific institution (e.g., "BlackRock")

---

### **Institution Activity**
```bash
GET /api/institution/{name}/activity
```
**Returns**: Recent buying/selling activity by institution

---

## 🌑 Dark Pool & Short Data

### **Recent Dark Pool Trades**
```bash
GET /api/darkpool/recent
```
**Returns**: Most recent dark pool trades across all tickers

---

### **Short Interest**
```bash
GET /api/shorts/{ticker}/interest-float
```
**Returns**: Short interest as % of float

---

### **Fails to Deliver (FTDs)**
```bash
GET /api/shorts/{ticker}/ftds
```
**Returns**: Fails to deliver data (potential short squeezes)

---

## 🔍 Screeners

### **Option Contracts Screener**
```bash
GET /api/screener/option-contracts
```
**Returns**: Filter options by criteria (volume, OI, IV, etc.)

---

### **Stock Screener**
```bash
GET /api/screener/stocks
```
**Returns**: Screen stocks by fundamentals, technicals

---

### **Analyst Ratings**
```bash
GET /api/screener/analysts
```
**Returns**: Latest analyst upgrades/downgrades

---

## 📡 Real-Time WebSocket Feeds

### **Live Options Flow**
```bash
WS /api/socket/option_trades
```
**Returns**: Real-time stream of options trades

---

### **Live Flow Alerts**
```bash
WS /api/socket/flow_alerts
```
**Returns**: Real-time flow alerts as they trigger

---

### **Live GEX Updates**
```bash
WS /api/socket/gex
```
**Returns**: Real-time gamma exposure updates

---

### **Live Price Data**
```bash
WS /api/socket/price
```
**Returns**: Real-time stock price updates

---

## 🎯 Priority Endpoints for Your Trading System

Based on your current implementation and Dan's verified examples, these are the **highest value endpoints**:

### **Tier 1: Critical for Trading**
1. ✅ `/api/stock/{ticker}/option-contracts` - **Already in use**
2. ✅ `/api/option-trades/flow-alerts` - **Verified working**
3. ✅ `/api/market/market-tide` - **Verified working**
4. `/api/stock/{ticker}/greek-exposure` - Greek exposure for hedging
5. `/api/stock/{ticker}/volatility/realized` - IV for options pricing

### **Tier 2: High Value Signals**
6. `/api/market/top-net-impact` - Find trending tickers
7. `/api/stock/{ticker}/flow-recent` - Recent significant flow
8. `/api/stock/{ticker}/spot-exposures` - Key price levels (GEX walls)
9. ✅ `/api/darkpool/{ticker}` - **Verified working** - Institutional flow
10. `/api/stock/{ticker}/oi-change` - Positioning changes

### **Tier 3: Supplemental Data**
11. `/api/congress/recent-trades` - Political trades signal
12. `/api/insider/{ticker}` - Insider buying/selling
13. `/api/earnings/{ticker}` - Earnings catalysts
14. `/api/stock/{ticker}/max-pain` - Max pain analysis
15. `/api/market/spike` - Unusual activity scanner

---

## 📝 Authentication Best Practices

All requests require the Bearer token in the Authorization header:

```bash
Authorization: Bearer 8932cd23-72b3-4f74-9848-13f9103b9df5
```

**Python Example**:
```python
import httpx

headers = {
    "Accept": "application/json",
    "Authorization": "Bearer 8932cd23-72b3-4f74-9848-13f9103b9df5"
}

response = httpx.get(
    "https://api.unusualwhales.com/api/market/market-tide",
    headers=headers
)

data = response.json()
```

---

## 🔗 Resources

- **API Documentation**: https://api.unusualwhales.com/docs
- **OpenAPI Spec**: `/home/user/FINAL_GNOSIS/api-spec.yaml`
- **Adapter Code**: `engines/inputs/unusual_whales_adapter.py`

---

**Last Updated**: December 2025
**Verified By**: Dan @ Unusual Whales API Support
**Token Format**: UUID Bearer Token (NOT JWT)
