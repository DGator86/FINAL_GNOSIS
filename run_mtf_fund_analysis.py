import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath("."))

from loguru import logger
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from engines.inputs.unusual_whales_adapter import UnusualWhalesOptionsAdapter
from market_model import (
    GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m,
    FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot, GaussianComponent,
    gmm_step, make_forecast, profit_score, ModelConfig,
    build_micro_state, build_liquidity_field
)

# ---------------------------------------------------------
# Configuration & Structures
# ---------------------------------------------------------

@dataclass
class TimeframeConfig:
    name: str
    minutes: int
    lookback_bars: int
    diffusion_scale: float  # Multiplier for q0

TF_CONFIGS = {
    "1m": TimeframeConfig("1m", 1, 120, 1.0),
    "15m": TimeframeConfig("15m", 15, 100, math.sqrt(15)),
    "1h": TimeframeConfig("1h", 60, 50, math.sqrt(60)),
}

FUND_UNIVERSE = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "GLD"]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_alpaca_tf(tf_name: str) -> str:
    if tf_name == "1m": return "1Min"
    if tf_name == "15m": return "15Min"
    if tf_name == "1h": return "1Hour"
    return "1Day"

def fetch_history(alpaca, symbol: str, tf_cfg: TimeframeConfig) -> List[Any]:
    end_dt = datetime.now(timezone.utc)
    # Add buffer for weekends/holidays/gaps
    start_dt = end_dt - timedelta(minutes=tf_cfg.minutes * tf_cfg.lookback_bars * 2)
    
    bars = alpaca.get_bars(
        symbol, 
        start_dt, 
        end_dt, 
        timeframe=get_alpaca_tf(tf_cfg.name)
    )
    
    # Return only the requested number of recent bars
    return bars[-tf_cfg.lookback_bars:] if bars else []

def analyze_ticker_mtf(alpaca, uw, symbol: str) -> Dict[str, Any]:
    """Run GMM analysis across multiple timeframes for a single ticker."""
    
    mtf_results = {}
    
    for tf_name, tf_cfg in TF_CONFIGS.items():
        bars = fetch_history(alpaca, symbol, tf_cfg)
        if not bars or len(bars) < 10:
            mtf_results[tf_name] = None
            continue
            
        # Adjust Model Config for Timeframe
        cfg = ModelConfig()
        cfg.dyn.q0 *= tf_cfg.diffusion_scale
        
        # Initialize State (Simple Warmup)
        start_price = bars[0].close
        state = GMMState(
            ts=bars[0].timestamp,
            components=[GaussianComponent(w=1.0, mu=start_price, var=0.1 * tf_cfg.diffusion_scale)]
        )
        
        # Run Filter Loop
        final_forecast = None
        final_micro = None
        
        for bar_obj in bars[1:]:
            ts = bar_obj.timestamp
            
            # --- Build Inputs (Approximate for speed/demo) ---
            bar = Bar1m(
                ts=ts,
                open=bar_obj.open, high=bar_obj.high, low=bar_obj.low, close=bar_obj.close,
                volume=bar_obj.volume, vwap=bar_obj.close # Alpaca OHLCV might not have vwap, fallback to close
            )
            
            # Synthetic Quote/Flow for historical bars
            spread = bar.close * 0.0005 # 5bps spread assumption
            quote = QuoteL1(
                ts=ts, bid=bar.close - spread/2, ask=bar.close + spread/2,
                bid_size=100, ask_size=100
            )
            
            # In a real system, we'd aggregate flow per timeframe. 
            # Here we use volume as proxy.
            flow = FlowAgg1m(ts=ts, buy_vol=bar.volume/2, sell_vol=bar.volume/2)
            
            greeks = GreeksField1m(ts=ts) # Missing historical greeks in this lightweight loop
            wyck = WyckoffState1m(ts=ts)
            l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
            
            liq = build_liquidity_field(l2, bar, quote)
            micro = build_micro_state(quote, bar, flow, liq, greeks, wyck)
            
            obs_price = bar.vwap if bar.vwap else bar.close
            state = gmm_step(state, micro, liq, greeks, wyck, obs_price, cfg)
            
            final_micro = micro
            
        # Generate Forecast
        if state and final_micro:
            final_forecast = make_forecast(
                state.ts, 
                tf_cfg.minutes, 
                state.components, 
                final_micro.anchor_price, 
                delta_tail=0.5 * tf_cfg.diffusion_scale
            )
            
            mtf_results[tf_name] = {
                "price": bars[-1].close,
                "forecast_mean": final_forecast.mean,
                "sigma": math.sqrt(final_forecast.var),
                "p_up": final_forecast.p_up,
                "entropy": state.entropy,
                "trend_strength": (final_forecast.mean - bars[-1].close) / max(math.sqrt(final_forecast.var), 1e-9)
            }
            
    return mtf_results

# ---------------------------------------------------------
# Fund Level Aggregation
# ---------------------------------------------------------

def run_fund_analysis():
    print(f"🚀 Starting MTF Fund Level Analysis for: {FUND_UNIVERSE}")
    print("=" * 80)
    
    try:
        alpaca = AlpacaMarketDataAdapter()
    except:
        print("❌ Alpaca not connected. Check API keys.")
        return
        
    uw = None # Skipping UW for this broad loop to save API calls/time
    
    fund_data = []
    
    for symbol in FUND_UNIVERSE:
        print(f"Analyzing {symbol}...")
        res = analyze_ticker_mtf(alpaca, uw, symbol)
        
        # Flatten for DataFrame
        row = {"Symbol": symbol}
        has_data = False
        for tf, data in res.items():
            if data:
                has_data = True
                row[f"{tf}_Price"] = data["price"]
                row[f"{tf}_Trend"] = data["trend_strength"]
                row[f"{tf}_Vol"] = data["sigma"]
                row[f"{tf}_Pup"] = data["p_up"]
                row[f"{tf}_Entropy"] = data["entropy"]
        
        if has_data:
            fund_data.append(row)
            
    if not fund_data:
        print("No data collected.")
        return

    df = pd.DataFrame(fund_data)
    
    # ---------------------------------------------------------
    # Report Generation
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 FUND LEVEL QUANT DASHBOARD (MTF GMM)")
    print("=" * 80)
    
    # 1. Alignment Score
    # Simple metric: Weighted sum of trends across timeframes
    # 1h carries more weight for structural direction, 1m for momentum
    if "1h_Trend" in df.columns and "15m_Trend" in df.columns and "1m_Trend" in df.columns:
        df["Composite_Score"] = (df["1h_Trend"] * 0.5 + df["15m_Trend"] * 0.3 + df["1m_Trend"] * 0.2)
    else:
        df["Composite_Score"] = 0.0
        
    # Rank by Opportunity
    df_sorted = df.sort_values("Composite_Score", ascending=False)
    
    print("\n🏆 Top Opportunities (MTF Alignment):")
    print(df_sorted[["Symbol", "Composite_Score", "1h_Trend", "15m_Trend", "1m_Trend"]].to_string(index=False, float_format="%.2f"))
    
    # 2. Risk / Regime Analysis
    print("\n⚠️  Risk & Regime Overview:")
    if "1h_Entropy" in df.columns:
        avg_entropy = df["1h_Entropy"].mean()
        regime = "CHAOTIC/HIGH VOL" if avg_entropy > 1.0 else "ORDERLY/TRENDING"
        print(f"Fund-wide 1H Entropy: {avg_entropy:.2f} => Regime: {regime}")
        
    # 3. Liquidity/Volatility Map
    print("\n💧 Volatility Profile (1H Sigma):")
    if "1h_Vol" in df.columns:
        print(df[["Symbol", "1h_Vol", "1h_Price"]].to_string(index=False, float_format="%.2f"))

    # 4. Correlation of Trends (Fund Cohesion)
    # Are all assets moving together?
    if "1h_Trend" in df.columns:
        trend_cohesion = abs(df["1h_Trend"].mean())
        print(f"\n🔗 Fund Directional Cohesion: {trend_cohesion:.2f} (0=Divergent, >1=Strong Unidirectional)")

if __name__ == "__main__":
    run_fund_analysis()
