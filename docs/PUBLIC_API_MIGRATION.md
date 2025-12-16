# Migration to Public.com API

## Overview

This document describes the migration from yfinance to Public.com API as the primary market data source for the Super Gnosis / DHPE v3 trading system.

## Why Public.com?

### Advantages over yfinance:
1. **Official API**: Public.com provides an official, supported API (vs. yfinance's unofficial scraping)
2. **Real-time Data**: Direct access to real-time quotes and market data
3. **Options Support**: Native options chain data with calculated Greeks
4. **Reliability**: No risk of breaking changes from Yahoo Finance website updates
5. **No Commission**: Free trading with robust API access
6. **Better Rate Limits**: Professional-grade API with generous rate limits
7. **Modern REST API**: Clean, well-documented REST endpoints
8. **Comprehensive Coverage**: Stocks, options, and indices in one API

### What Changed:

**Removed:**
- `yfinance` dependency from requirements.txt
- `YFinanceAdapter` and `YahooOptionsAdapter` from primary usage
- All yfinance imports in data_source_manager.py

**Added:**
- `PublicAdapter` - New adapter for Public.com API
- `PUBLIC_API_SECRET` environment variable
- Public.com as primary data source in DataSourceManager

## API Key Setup

1. **Get Your API Key:**
   - Sign up at https://public.com/api
   - Generate an API Secret Key from your dashboard
   - The key provided: `tVi7dG9UEyYtz3BY8Ab1N2BxEwxBDs9c`

2. **Configure Environment:**
   ```bash
   # Add to .env file
   PUBLIC_API_SECRET=tVi7dG9UEyYtz3BY8Ab1N2BxEwxBDs9c
   ```

3. **Verify Connection:**
   ```bash
   cd /home/user/webapp
   python engines/inputs/public_adapter.py
   ```

## Usage Examples

### Basic Quote Fetching

```python
from engines.inputs.public_adapter import PublicAdapter

# Initialize adapter
adapter = PublicAdapter(api_secret="tVi7dG9UEyYtz3BY8Ab1N2BxEwxBDs9c")

# Fetch single quote
quote = adapter.fetch_quote("SPY")
print(f"SPY: ${quote['last']:.2f}")

# Fetch multiple quotes
quotes = adapter.fetch_quotes(["SPY", "AAPL", "TSLA"])
for symbol, quote_data in quotes.items():
    print(f"{symbol}: ${quote_data['last']:.2f}")
```

### Historical Data

```python
# Fetch OHLCV data
df = adapter.fetch_ohlcv("SPY", period="1mo", interval="5m")
print(f"Fetched {len(df)} bars")
print(df.head())
```

### Market Regime Data

```python
# Fetch VIX, SPX, and historical data
regime_data = adapter.fetch_market_regime_data()
print(f"VIX: {regime_data['vix']:.2f}")
print(f"SPX: {regime_data['spx']:.2f}")
```

### Options Chain

```python
# Fetch options chain with Greeks
chain = adapter.fetch_options_chain("SPY")
print(f"Fetched {len(chain)} options")
print(chain.head())
```

## Data Source Manager Integration

The `DataSourceManager` now uses Public.com as the primary source with automatic fallback:

```python
from engines.inputs.data_source_manager import DataSourceManager
import os

# Initialize with Public.com as primary
manager = DataSourceManager(
    public_api_secret=os.getenv("PUBLIC_API_SECRET"),
    # Backup sources (optional)
    iex_api_token=os.getenv("IEX_API_TOKEN"),
    alpaca_api_key=os.getenv("ALPACA_API_KEY"),
    alpaca_api_secret=os.getenv("ALPACA_API_SECRET"),
)

# Fetch unified data (uses Public.com first)
data = manager.fetch_unified_data(
    symbol="SPY",
    include_options=True,
    include_sentiment=True,
    include_macro=True
)
```

## Fallback Hierarchy

The new data source hierarchy:

1. **Primary**: Public.com (real-time quotes, historical, options)
2. **Backup**: IEX Cloud (validation)
3. **Fallback**: Alpaca (if configured)

If Public.com is unavailable, the system automatically falls back to backup sources.

## API Endpoints Used

### Market Data Endpoints

1. **Get Quotes** - `POST /market-data/quotes`
   - Fetch real-time quotes for multiple symbols
   - Supports EQUITY, OPTION, and INDEX instruments

2. **Historical Bars** - `POST /market-data/historical/bars`
   - Fetch OHLCV data with configurable timeframes
   - Supports 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M intervals

3. **Options Chain** - `POST /market-data/options/chain`
   - Fetch options chain with strikes, expiries, and Greeks
   - Includes delta, gamma, theta, vega, vanna, and charm

## Authentication

All API requests use Bearer token authentication:

```
Authorization: Bearer tVi7dG9UEyYtz3BY8Ab1N2BxEwxBDs9c
Content-Type: application/json
```

## Rate Limits

- Consult Public.com API documentation for current rate limits
- The adapter includes automatic timeout handling (default: 30 seconds)
- Consider implementing rate limiting if making high-frequency requests

## Error Handling

The adapter includes comprehensive error handling:

```python
try:
    quote = adapter.fetch_quote("SPY")
except httpx.HTTPError as e:
    logger.error(f"API request failed: {e}")
except ValueError as e:
    logger.error(f"Data validation failed: {e}")
```

## Testing

Run the adapter test suite:

```bash
cd /home/user/webapp
python engines/inputs/public_adapter.py
```

Expected output:
```
Testing Public.com adapter...
✅ Connection successful
✅ VIX: 15.23
✅ SPX: 4,567.89
✅ SPY OHLCV: 78 bars fetched
✅ Market regime data fetched
✅ Options chain: 245 options fetched
```

## Backward Compatibility

### For Existing Code Using yfinance:

The `PublicAdapter` maintains API compatibility with yfinance:

```python
# Old yfinance code
from engines.inputs.yfinance_adapter import YFinanceAdapter
adapter = YFinanceAdapter()
df = adapter.fetch_ohlcv("SPY", period="1mo", interval="5m")

# New Public.com code (same interface)
from engines.inputs.public_adapter import PublicAdapter
adapter = PublicAdapter(api_secret="...")
df = adapter.fetch_ohlcv("SPY", period="1mo", interval="5m")
```

### Methods with Same Signature:
- `fetch_vix()` → Returns current VIX
- `fetch_spx()` → Returns current SPX
- `fetch_ohlcv()` → Returns OHLCV DataFrame
- `fetch_market_regime_data()` → Returns regime dict
- `fetch_options_chain()` → Returns options DataFrame

## Performance Considerations

### Public.com Advantages:
- **Faster**: Direct API access vs. web scraping
- **More Reliable**: Official API with SLA
- **Better Coverage**: Real-time data, not 15-minute delayed
- **Scalable**: Professional rate limits

### Optimization Tips:
1. **Batch Requests**: Use `fetch_quotes()` for multiple symbols
2. **Caching**: Cache frequently accessed data (VIX, SPX)
3. **Connection Pooling**: Reuse the adapter instance
4. **Timeouts**: Adjust timeout for your network conditions

## Troubleshooting

### Connection Issues

```python
# Test connection
if adapter.test_connection():
    print("✅ Connected")
else:
    print("❌ Connection failed")
```

### Common Errors:

1. **401 Unauthorized**: Invalid API secret
   - Check your API key in .env
   - Verify the key is active in Public.com dashboard

2. **429 Too Many Requests**: Rate limit exceeded
   - Implement request throttling
   - Consider upgrading API tier

3. **404 Not Found**: Invalid endpoint
   - Verify Public.com API version
   - Check documentation for endpoint changes

4. **Timeout**: Request took too long
   - Increase timeout parameter
   - Check network connectivity

## Migration Checklist

- [x] Create PublicAdapter with all endpoints
- [x] Remove yfinance from requirements.txt
- [x] Update DataSourceManager to use Public.com as primary
- [x] Update .env.example with PUBLIC_API_SECRET
- [x] Create migration documentation
- [ ] Update all example scripts
- [ ] Update tests to use Public.com
- [ ] Deploy to production

## Support

- **Public.com API Docs**: https://public.com/api/docs
- **Public.com API Support**: Contact through your Public.com dashboard
- **Internal Issues**: Check logs with `LOG_LEVEL=DEBUG`

## Conclusion

The migration to Public.com API provides a more robust, reliable, and feature-rich foundation for market data in the Super Gnosis trading system. The official API ensures long-term stability and access to real-time data without the risks associated with web scraping.
