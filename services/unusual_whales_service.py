"""
Unusual Whales API Service - Comprehensive Options Flow & Greeks Data

Provides access to:
- Options Greeks (delta, gamma, theta, vega, rho, charm, vanna)
- Flow Alerts (unusual options activity)
- Greek Flow (directional delta/vega flow)
- Market Tide (market-wide sentiment)
- Max Pain (options pain points)
- Expiry Breakdown (OI by expiration)
- Realized Volatility

API Docs: https://api.unusualwhales.com/docs
"""

import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum

import aiohttp
from loguru import logger


class FlowAlertType(Enum):
    """Types of flow alerts."""
    REPEATED_HITS = "RepeatedHits"
    GOLDEN_SWEEP = "GoldenSweep"
    BLOCK_TRADE = "BlockTrade"
    UNUSUAL_VOLUME = "UnusualVolume"


@dataclass
class OptionGreeks:
    """Greeks for a single option contract."""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    charm: float = 0.0
    vanna: float = 0.0
    implied_volatility: float = 0.0


@dataclass
class FlowAlert:
    """Unusual options flow alert."""
    id: str
    ticker: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    alert_rule: str
    total_premium: float
    volume: int
    open_interest: int
    trade_count: int
    iv: float
    underlying_price: float
    has_sweep: bool
    has_floor: bool
    created_at: datetime
    bid: float = 0.0
    ask: float = 0.0
    

@dataclass 
class GreekFlow:
    """Directional greek flow for a ticker."""
    ticker: str
    timestamp: datetime
    transactions: int
    volume: int
    dir_delta_flow: float  # Directional delta
    dir_vega_flow: float   # Directional vega
    total_delta_flow: float
    total_vega_flow: float
    otm_dir_delta_flow: float = 0.0
    otm_dir_vega_flow: float = 0.0


@dataclass
class MarketTide:
    """Market-wide options sentiment."""
    date: str
    call_premium: float
    put_premium: float
    call_volume: int
    put_volume: int
    net_premium: float
    put_call_ratio: float


@dataclass
class MaxPain:
    """Max pain calculation for a ticker."""
    ticker: str
    date: str
    max_pain_strike: float
    current_price: float
    distance_pct: float


class UnusualWhalesService:
    """
    Comprehensive service for Unusual Whales API.
    
    Usage:
        async with UnusualWhalesService() as uw:
            greeks = await uw.get_greeks('SPY')
            alerts = await uw.get_flow_alerts()
            greek_flow = await uw.get_greek_flow('SPY')
    """
    
    BASE_URL = "https://api.unusualwhales.com"
    
    def __init__(self, token: Optional[str] = None):
        """Initialize with API token from env or parameter."""
        self.token = token or os.getenv('UNUSUAL_WHALES_API_TOKEN')
        if not self.token:
            raise ValueError("UNUSUAL_WHALES_API_TOKEN not set")
        
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'application/json'
        }
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 100
        self._rate_limit_reset = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with error handling."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with self.session.get(url, params=params, timeout=30) as resp:
                # Track rate limits
                self._rate_limit_remaining = int(resp.headers.get('X-RateLimit-Remaining', 100))
                
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning("Rate limited by Unusual Whales API")
                    raise Exception("Rate limited")
                elif resp.status == 401:
                    logger.error("Invalid Unusual Whales API token")
                    raise Exception("Invalid API token")
                else:
                    text = await resp.text()
                    logger.warning(f"UW API error {resp.status}: {text[:100]}")
                    return {}
        except asyncio.TimeoutError:
            logger.warning(f"Timeout requesting {endpoint}")
            return {}
        except Exception as e:
            logger.error(f"Error requesting {endpoint}: {e}")
            raise

    # ==================== GREEKS ====================
    
    async def get_greeks(self, ticker: str) -> List[OptionGreeks]:
        """
        Get options Greeks for all contracts of a ticker.
        
        Endpoint: /api/stock/{ticker}/greeks
        
        Returns Greeks including delta, gamma, theta, vega, rho, charm, vanna
        for both calls and puts at each strike/expiry.
        """
        data = await self._request(f"/api/stock/{ticker}/greeks")
        greeks_data = data.get('data', [])
        
        result = []
        for row in greeks_data:
            # Each row has both call and put Greeks
            strike = float(row.get('strike', 0))
            expiry = row.get('expiry', '')
            
            # Call Greeks
            if row.get('call_option_symbol'):
                result.append(OptionGreeks(
                    symbol=row['call_option_symbol'],
                    strike=strike,
                    expiry=expiry,
                    option_type='call',
                    delta=float(row.get('call_delta', 0) or 0),
                    gamma=float(row.get('call_gamma', 0) or 0),
                    theta=float(row.get('call_theta', 0) or 0),
                    vega=float(row.get('call_vega', 0) or 0),
                    rho=float(row.get('call_rho', 0) or 0),
                    charm=float(row.get('call_charm', 0) or 0),
                    vanna=float(row.get('call_vanna', 0) or 0),
                    implied_volatility=float(row.get('call_volatility', 0) or 0),
                ))
            
            # Put Greeks
            if row.get('put_option_symbol'):
                result.append(OptionGreeks(
                    symbol=row['put_option_symbol'],
                    strike=strike,
                    expiry=expiry,
                    option_type='put',
                    delta=float(row.get('put_delta', 0) or 0),
                    gamma=float(row.get('put_gamma', 0) or 0),
                    theta=float(row.get('put_theta', 0) or 0),
                    vega=float(row.get('put_vega', 0) or 0),
                    rho=float(row.get('put_rho', 0) or 0),
                    charm=float(row.get('put_charm', 0) or 0),
                    vanna=float(row.get('put_vanna', 0) or 0),
                    implied_volatility=float(row.get('put_volatility', 0) or 0),
                ))
        
        logger.info(f"Retrieved {len(result)} option Greeks for {ticker}")
        return result

    # ==================== FLOW ALERTS ====================
    
    async def get_flow_alerts(
        self, 
        ticker: Optional[str] = None,
        limit: int = 50
    ) -> List[FlowAlert]:
        """
        Get unusual options flow alerts.
        
        Endpoint: /api/option-trades/flow-alerts
        
        Returns alerts for unusual activity like repeated hits, sweeps, blocks.
        """
        params = {'limit': limit}
        
        if ticker:
            data = await self._request(f"/api/stock/{ticker}/flow-alerts", params)
        else:
            data = await self._request("/api/option-trades/flow-alerts", params)
        
        alerts_data = data.get('data', [])
        
        result = []
        for alert in alerts_data:
            try:
                result.append(FlowAlert(
                    id=alert.get('id', ''),
                    ticker=alert.get('ticker', ''),
                    strike=float(alert.get('strike', 0) or 0),
                    expiry=alert.get('expiry', ''),
                    option_type=alert.get('type', 'unknown'),
                    alert_rule=alert.get('alert_rule', ''),
                    total_premium=float(alert.get('total_premium', 0) or 0),
                    volume=int(alert.get('volume', 0) or 0),
                    open_interest=int(alert.get('open_interest', 0) or 0),
                    trade_count=int(alert.get('trade_count', 0) or 0),
                    iv=float(alert.get('iv_start', alert.get('iv', 0)) or 0),
                    underlying_price=float(alert.get('underlying_price', 0) or 0),
                    has_sweep=alert.get('has_sweep', False),
                    has_floor=alert.get('has_floor', False),
                    created_at=datetime.fromisoformat(alert['created_at'].replace('Z', '+00:00')) if alert.get('created_at') else datetime.now(),
                    bid=float(alert.get('bid', 0) or 0),
                    ask=float(alert.get('ask', 0) or 0),
                ))
            except Exception as e:
                logger.debug(f"Error parsing flow alert: {e}")
                continue
        
        logger.info(f"Retrieved {len(result)} flow alerts")
        return result

    # ==================== GREEK FLOW ====================
    
    async def get_greek_flow(
        self, 
        ticker: str,
        expiry: Optional[str] = None
    ) -> List[GreekFlow]:
        """
        Get directional greek flow for a ticker.
        
        Endpoint: /api/stock/{ticker}/greek-flow
        
        Returns minute-by-minute delta and vega flow data.
        """
        if expiry:
            data = await self._request(f"/api/stock/{ticker}/greek-flow/{expiry}")
        else:
            data = await self._request(f"/api/stock/{ticker}/greek-flow")
        
        flow_data = data.get('data', [])
        
        result = []
        for row in flow_data:
            try:
                result.append(GreekFlow(
                    ticker=row.get('ticker', ticker),
                    timestamp=datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')) if row.get('timestamp') else datetime.now(),
                    transactions=int(row.get('transactions', 0) or 0),
                    volume=int(row.get('volume', 0) or 0),
                    dir_delta_flow=float(row.get('dir_delta_flow', 0) or 0),
                    dir_vega_flow=float(row.get('dir_vega_flow', 0) or 0),
                    total_delta_flow=float(row.get('total_delta_flow', 0) or 0),
                    total_vega_flow=float(row.get('total_vega_flow', 0) or 0),
                    otm_dir_delta_flow=float(row.get('otm_dir_delta_flow', 0) or 0),
                    otm_dir_vega_flow=float(row.get('otm_dir_vega_flow', 0) or 0),
                ))
            except Exception as e:
                logger.debug(f"Error parsing greek flow: {e}")
                continue
        
        logger.info(f"Retrieved {len(result)} greek flow records for {ticker}")
        return result

    # ==================== MARKET TIDE ====================
    
    async def get_market_tide(self) -> Optional[MarketTide]:
        """
        Get market-wide options sentiment (Market Tide).
        
        Endpoint: /api/market/market-tide
        
        Returns aggregate call/put premium and volume.
        """
        data = await self._request("/api/market/market-tide")
        tide_data = data.get('data', {})
        
        if not tide_data:
            return None
        
        # Handle both dict and list responses
        if isinstance(tide_data, list) and tide_data:
            tide_data = tide_data[0]
        
        call_prem = float(tide_data.get('call_premium', 0) or 0)
        put_prem = float(tide_data.get('put_premium', 0) or 0)
        call_vol = int(tide_data.get('call_volume', 0) or 0)
        put_vol = int(tide_data.get('put_volume', 0) or 0)
        
        return MarketTide(
            date=data.get('date', str(date.today())),
            call_premium=call_prem,
            put_premium=put_prem,
            call_volume=call_vol,
            put_volume=put_vol,
            net_premium=call_prem - put_prem,
            put_call_ratio=put_vol / call_vol if call_vol > 0 else 0,
        )

    # ==================== MAX PAIN ====================
    
    async def get_max_pain(self, ticker: str) -> Optional[MaxPain]:
        """
        Get max pain strike for a ticker.
        
        Endpoint: /api/stock/{ticker}/max-pain
        
        Returns the strike price where option holders would lose the most.
        """
        data = await self._request(f"/api/stock/{ticker}/max-pain")
        pain_data = data.get('data', {})
        
        if not pain_data:
            return None
        
        if isinstance(pain_data, list) and pain_data:
            pain_data = pain_data[0]
        
        max_pain_strike = float(pain_data.get('price', pain_data.get('max_pain', 0)) or 0)
        current_price = float(pain_data.get('spot_price', pain_data.get('underlying_price', 0)) or 0)
        
        distance_pct = 0.0
        if current_price > 0 and max_pain_strike > 0:
            distance_pct = (max_pain_strike - current_price) / current_price * 100
        
        return MaxPain(
            ticker=ticker,
            date=data.get('date', str(date.today())),
            max_pain_strike=max_pain_strike,
            current_price=current_price,
            distance_pct=distance_pct,
        )

    # ==================== EXPIRY BREAKDOWN ====================
    
    async def get_expiry_breakdown(self, ticker: str) -> Dict[str, Dict]:
        """
        Get open interest breakdown by expiration.
        
        Endpoint: /api/stock/{ticker}/expiry-breakdown
        
        Returns OI and volume for each expiration date.
        """
        data = await self._request(f"/api/stock/{ticker}/expiry-breakdown")
        return data.get('data', {})

    # ==================== VOLATILITY ====================
    
    async def get_realized_volatility(self, ticker: str) -> Dict[str, float]:
        """
        Get realized volatility data.
        
        Endpoint: /api/stock/{ticker}/volatility/realized
        """
        data = await self._request(f"/api/stock/{ticker}/volatility/realized")
        vol_data = data.get('data', [])
        
        if vol_data and isinstance(vol_data, list):
            return vol_data[0] if vol_data else {}
        return vol_data if isinstance(vol_data, dict) else {}

    # ==================== OPTION CONTRACTS ====================
    
    async def get_option_contracts(
        self, 
        ticker: str,
        expiration: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict]:
        """
        Get option contracts with quotes.
        
        Endpoint: /api/stock/{ticker}/option-contracts
        """
        params = {'limit': limit}
        if expiration:
            params['expiration_date'] = expiration
        
        data = await self._request(f"/api/stock/{ticker}/option-contracts", params)
        return data.get('data', [])

    # ==================== CONVENIENCE METHODS ====================
    
    async def get_trading_signals(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive trading signals combining multiple data sources.
        
        Returns:
            - Greeks summary
            - Flow sentiment  
            - Max pain distance
            - Greek flow direction
        """
        # Fetch data in parallel
        greeks_task = self.get_greeks(ticker)
        alerts_task = self.get_flow_alerts(ticker, limit=20)
        greek_flow_task = self.get_greek_flow(ticker)
        max_pain_task = self.get_max_pain(ticker)
        
        greeks, alerts, greek_flow, max_pain = await asyncio.gather(
            greeks_task, alerts_task, greek_flow_task, max_pain_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(greeks, Exception):
            greeks = []
        if isinstance(alerts, Exception):
            alerts = []
        if isinstance(greek_flow, Exception):
            greek_flow = []
        if isinstance(max_pain, Exception):
            max_pain = None
        
        # Calculate aggregate Greeks
        total_call_delta = sum(g.delta for g in greeks if g.option_type == 'call')
        total_put_delta = sum(g.delta for g in greeks if g.option_type == 'put')
        total_gamma = sum(g.gamma for g in greeks)
        
        # Calculate flow sentiment
        call_alerts = [a for a in alerts if a.option_type == 'call']
        put_alerts = [a for a in alerts if a.option_type == 'put']
        call_premium = sum(a.total_premium for a in call_alerts)
        put_premium = sum(a.total_premium for a in put_alerts)
        
        # Recent greek flow direction
        recent_flow = greek_flow[-10:] if greek_flow else []
        avg_delta_flow = sum(f.dir_delta_flow for f in recent_flow) / len(recent_flow) if recent_flow else 0
        avg_vega_flow = sum(f.dir_vega_flow for f in recent_flow) / len(recent_flow) if recent_flow else 0
        
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'greeks': {
                'total_call_delta': total_call_delta,
                'total_put_delta': total_put_delta,
                'net_delta': total_call_delta + total_put_delta,
                'total_gamma': total_gamma,
                'num_contracts': len(greeks),
            },
            'flow': {
                'call_premium': call_premium,
                'put_premium': put_premium,
                'net_premium': call_premium - put_premium,
                'call_alerts': len(call_alerts),
                'put_alerts': len(put_alerts),
                'bullish_bias': call_premium > put_premium * 1.2,
                'bearish_bias': put_premium > call_premium * 1.2,
            },
            'greek_flow': {
                'avg_delta_flow': avg_delta_flow,
                'avg_vega_flow': avg_vega_flow,
                'delta_bullish': avg_delta_flow > 0,
                'records': len(greek_flow),
            },
            'max_pain': {
                'strike': max_pain.max_pain_strike if max_pain else 0,
                'current_price': max_pain.current_price if max_pain else 0,
                'distance_pct': max_pain.distance_pct if max_pain else 0,
            } if max_pain else None,
        }


# Convenience function for quick access
async def get_unusual_whales_data(ticker: str) -> Dict[str, Any]:
    """Quick access to Unusual Whales trading signals."""
    async with UnusualWhalesService() as uw:
        return await uw.get_trading_signals(ticker)


# Test function
async def test_service():
    """Test the Unusual Whales service."""
    print("=" * 60)
    print("  UNUSUAL WHALES SERVICE TEST")
    print("=" * 60)
    
    async with UnusualWhalesService() as uw:
        # Test Greeks
        print("\n[1] Testing Greeks...")
        greeks = await uw.get_greeks('SPY')
        print(f"  Retrieved {len(greeks)} option Greeks")
        if greeks:
            g = greeks[0]
            print(f"  Sample: {g.symbol} delta={g.delta:.4f} gamma={g.gamma:.6f}")
        
        # Test Flow Alerts
        print("\n[2] Testing Flow Alerts...")
        alerts = await uw.get_flow_alerts(limit=10)
        print(f"  Retrieved {len(alerts)} flow alerts")
        if alerts:
            a = alerts[0]
            print(f"  Sample: {a.ticker} {a.strike} {a.option_type} ${a.total_premium:,.0f}")
        
        # Test Greek Flow
        print("\n[3] Testing Greek Flow...")
        flow = await uw.get_greek_flow('SPY')
        print(f"  Retrieved {len(flow)} greek flow records")
        if flow:
            f = flow[-1]
            print(f"  Latest: delta_flow={f.dir_delta_flow:,.0f} vega_flow={f.dir_vega_flow:,.0f}")
        
        # Test Market Tide
        print("\n[4] Testing Market Tide...")
        tide = await uw.get_market_tide()
        if tide:
            print(f"  Call Premium: ${tide.call_premium:,.0f}")
            print(f"  Put Premium: ${tide.put_premium:,.0f}")
            print(f"  Net Premium: ${tide.net_premium:,.0f}")
            print(f"  P/C Ratio: {tide.put_call_ratio:.2f}")
        
        # Test Max Pain
        print("\n[5] Testing Max Pain...")
        pain = await uw.get_max_pain('SPY')
        if pain:
            print(f"  Max Pain: ${pain.max_pain_strike}")
            print(f"  Current: ${pain.current_price}")
            print(f"  Distance: {pain.distance_pct:.2f}%")
        
        # Test Trading Signals
        print("\n[6] Testing Trading Signals...")
        signals = await uw.get_trading_signals('SPY')
        print(f"  Net Delta: {signals['greeks']['net_delta']:,.0f}")
        print(f"  Net Flow Premium: ${signals['flow']['net_premium']:,.0f}")
        print(f"  Bullish Bias: {signals['flow']['bullish_bias']}")
        print(f"  Bearish Bias: {signals['flow']['bearish_bias']}")
    
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / '.env')
    asyncio.run(test_service())
