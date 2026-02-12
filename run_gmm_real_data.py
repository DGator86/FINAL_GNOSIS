import os
import sys
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from loguru import logger
from alpaca.data.timeframe import TimeFrame

from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from engines.inputs.unusual_whales_adapter import UnusualWhalesOptionsAdapter
from engines.inputs.options_chain_adapter import OptionContract

from market_model import (
    GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m,
    FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot, GaussianComponent,
    gmm_step, make_forecast, profit_score, ModelConfig,
    build_micro_state, build_liquidity_field
)

# ---------------------------------------------------------
# Adapter Setup
# ---------------------------------------------------------
def init_adapters():
    try:
        alpaca = AlpacaMarketDataAdapter()
    except Exception as e:
        logger.error(f"Failed to init Alpaca adapter: {e}")
        alpaca = None

    try:
        # Check if UW token exists, otherwise might fail
        if os.getenv("UNUSUAL_WHALES_API_TOKEN"):
            uw = UnusualWhalesOptionsAdapter()
        else:
            logger.warning("UNUSUAL_WHALES_API_TOKEN not set, skipping UW adapter")
            uw = None
    except Exception as e:
        logger.error(f"Failed to init Unusual Whales adapter: {e}")
        uw = None
        
    return alpaca, uw

# ---------------------------------------------------------
# Data Fetching & Transformation
# ---------------------------------------------------------
def fetch_recent_history(alpaca: AlpacaMarketDataAdapter, symbol: str, lookback_minutes: int = 60):
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(minutes=lookback_minutes + 10) # buffer
    
    logger.info(f"Fetching bars for {symbol} from {start_dt} to {end_dt}")
    bars = alpaca.get_bars(symbol, start_dt, end_dt, timeframe="1Min")
    
    # Sort by timestamp just in case
    bars.sort(key=lambda x: x.timestamp)
    return bars

def get_current_quote(alpaca: AlpacaMarketDataAdapter, symbol: str) -> QuoteL1:
    q = alpaca.get_quote(symbol)
    return QuoteL1(
        ts=q.timestamp,
        bid=q.bid,
        ask=q.ask,
        bid_size=q.bid_size,
        ask_size=q.ask_size
    )

def fetch_greeks_field(uw: UnusualWhalesOptionsAdapter, symbol: str, ts: datetime) -> GreeksField1m:
    if not uw:
        return GreeksField1m(ts=ts)
    
    try:
        # Get chain (this might be heavy to do every minute, but okay for demo)
        # In prod, you'd cache or use a stream
        contracts = uw.get_chain(symbol, ts)
        
        if not contracts:
            return GreeksField1m(ts=ts)
            
        # Calculate Net GEX
        # GEX = Gamma * OI * 100 * Spot (Approx)
        # We don't have spot inside `get_chain` result easily unless we check `last` or pass it.
        # We'll use a simplified proxy: Sum(Gamma * OI)
        # Calls > 0, Puts < 0
        
        net_gex = 0.0
        net_charm = 0.0
        
        # Identify magnets (High Gamma Levels)
        gamma_levels = {}
        
        for c in contracts:
            sign = 1.0 if c.option_type.lower() == "call" else -1.0
            gex_contrib = c.gamma * c.open_interest * 100
            net_gex += gex_contrib * sign
            
            charm_contrib = c.delta * c.open_interest # Charm proxy often relates to delta decay or vanna
            # Actually Charm is dDelta/dTime. Using provided 'charm' field if available?
            # The adapter doesn't seem to expose 'charm' explicitly in OptionContract schema we saw earlier?
            # Wait, I saw `charm` in `UnusualWhalesOptionsAdapter._fetch_greeks` but `OptionContract` has it?
            # Let's check `OptionContract` definition in `options_chain_adapter.py`.
            # It has `delta`, `gamma`, `theta`, `vega`, `rho`. No `charm`.
            # So we can't compute true charm. We'll use theta as a proxy for time-decay pressure.
            
            net_charm += 0.0 # Placeholder
            
            # Strike magnets
            k = c.strike
            if k not in gamma_levels:
                gamma_levels[k] = 0.0
            gamma_levels[k] += abs(gex_contrib)
            
        # Top strike levels
        sorted_strikes = sorted(gamma_levels.items(), key=lambda x: x[1], reverse=True)
        top_strikes = [k for k, v in sorted_strikes[:5]]
        top_strengths = [v / (sorted_strikes[0][1] + 1e-9) for k, v in sorted_strikes[:5]]
        
        return GreeksField1m(
            ts=ts,
            gex=net_gex,
            charm_ex=net_charm,
            theta_near=0.0,
            strike_levels=top_strikes,
            strike_strength=top_strengths
        )
        
    except Exception as e:
        logger.error(f"Error fetching Greeks: {e}")
        return GreeksField1m(ts=ts)

def fetch_flow_agg(uw: UnusualWhalesOptionsAdapter, symbol: str, ts: datetime) -> FlowAgg1m:
    if not uw:
        return FlowAgg1m(ts=ts, buy_vol=0, sell_vol=0)
    
    try:
        summary = uw.get_flow_summary(symbol)
        # summary has call_volume, put_volume... not exactly buy/sell volume of the underlying.
        # FlowAgg1m usually expects UNDERLYING flow.
        # But if we only have UW options flow, we can use that as a proxy for "sentiment flow".
        # Or, we assume Alpaca bars volume is total, and we don't have split.
        
        # For this demo, let's use 50/50 split of bar volume + UW sentiment influence
        # This is a hack because we lack tick-level data for buy/sell volume classification.
        return FlowAgg1m(
            ts=ts,
            buy_vol=summary.get("call_volume", 0),  # Using Option flow as proxy for intent
            sell_vol=summary.get("put_volume", 0)
        )
    except Exception as e:
        logger.error(f"Error fetching Flow: {e}")
        return FlowAgg1m(ts=ts, buy_vol=0, sell_vol=0)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def run_real_data_loop():
    symbol = "SPY" # SPX usually not tradeable on basic plans, SPY is good proxy
    
    print(f"Initializing adapters for {symbol}...")
    alpaca, uw = init_adapters()
    
    if not alpaca:
        print("❌ Alpaca adapter missing. Cannot proceed.")
        return

    # 1. Fetch History
    print("Fetching historical bars...")
    bars = fetch_recent_history(alpaca, symbol, lookback_minutes=60)
    if not bars:
        print("❌ No historical data found.")
        return
        
    print(f"✅ Loaded {len(bars)} bars.")
    
    # 2. Init GMM State
    cfg = ModelConfig()
    start_price = bars[0].close
    initial_comps = [
        GaussianComponent(w=0.5, mu=start_price, var=0.05),
        GaussianComponent(w=0.3, mu=start_price*1.0002, var=0.05),
        GaussianComponent(w=0.2, mu=start_price*0.9998, var=0.05)
    ]
    state = GMMState(ts=bars[0].timestamp, components=initial_comps)
    
    print("\nRunning GMM on Real Data Trace...")
    print(f"{'Time':<20} | {'Price':<10} | {'Forecast':<10} | {'Sigma':<8} | {'P_Up':<6} | {'K':<3} | {'GEX (Proxy)':<12}")
    print("-" * 100)
    
    # 3. Iterate
    for i, bar_obj in enumerate(bars):
        if i == 0: continue
        
        ts = bar_obj.timestamp
        
        # --- Build Inputs ---
        # Bar
        bar = Bar1m(
            ts=ts,
            open=bar_obj.open, high=bar_obj.high, low=bar_obj.low, close=bar_obj.close,
            volume=bar_obj.volume, vwap=None # Alpaca bars might not have VWAP in simple OHLCV, defaulting
        )
        
        # Quote (Mocking from Bar since we don't have historical quotes easily available in this loop)
        # In live loop we'd call `get_current_quote`. Here we approximate.
        spread = 0.03 # 3 cents
        quote = QuoteL1(
            ts=ts,
            bid=bar.close - spread/2,
            ask=bar.close + spread/2,
            bid_size=100, ask_size=100
        )
        
        # Flow (Mocking or fetching real if recent)
        # Since this is historical loop, we can't easily get historical flow tick-by-tick from current adapters
        # We will use dummy flow or the UW summary if it's the LAST bar (real-time)
        flow = FlowAgg1m(ts=ts, buy_vol=bar.volume/2, sell_vol=bar.volume/2)
        
        # Greeks (Real fetch if possible, else mock)
        # Fetching chain 60 times is slow. We'll do it only for the last bar or every 10th bar.
        greeks = GreeksField1m(ts=ts)
        if i % 10 == 0 and uw:
             # Warning: `fetch_greeks_field` hits API. 
             # For backfill we might skip or assume static.
             pass 
        
        # Wyckoff / L2
        wyck = WyckoffState1m(ts=ts)
        l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
        
        # --- Engine Step ---
        liq = build_liquidity_field(l2, bar, quote)
        micro = build_micro_state(quote, bar, flow, liq, greeks, wyck)
        
        obs_price = bar.close # VWAP missing
        state = gmm_step(state, micro, liq, greeks, wyck, obs_price, cfg)
        
        # Forecast
        fc = make_forecast(ts, 1, state.components, micro.anchor_price, delta_tail=0.5)
        
        print(f"{ts.strftime('%H:%M:%S'):<20} | {obs_price:<10.2f} | {fc.mean:<10.2f} | {math.sqrt(fc.var):<8.2f} | {fc.p_up:<6.2f} | {len(state.components):<3} | {greeks.gex:<12.2f}")

    # 4. Live "Tick" (Optional: Fetch current quote and run one live step)
    if alpaca:
        print("\nFetching Live Quote...")
        try:
            q = get_current_quote(alpaca, symbol)
            print(f"Live Quote: {q.bid} / {q.ask}")
            
            # You could run one more step here
            
        except Exception as e:
            print(f"Could not get live quote: {e}")

if __name__ == "__main__":
    run_real_data_loop()
