from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Optional, Any

from loguru import logger

# Internal GMM modules
from .gmm_filter import gmm_step
from .features import build_micro_state, build_liquidity_field
from .forecast import make_forecast
from .gmm_config import ModelConfig
from .internal_schemas import (
    GMMState, GaussianComponent, MicroState1m, LiquidityField1m, 
    GreeksField1m, WyckoffState1m, FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot
)

# External Adapters
from engines.inputs.options_chain_adapter import OptionsChainAdapter
from engines.inputs.market_data_adapter import MarketDataAdapter

class PhysicsEngine:
    """
    Physics Engine v1.0 (GMM / Gaussian Mechanics)
    
    Models price as a particle in a potential field defined by:
    - Liquidity (Wells)
    - Option Strikes (Magnets)
    - Supply/Demand Intercept (Restoring Force)
    
    Produces a probabilistic forecast and regime classification (Entropy).
    """
    
    def __init__(self, 
                 market_adapter: MarketDataAdapter,
                 options_adapter: Optional[OptionsChainAdapter],
                 config: Dict[str, Any]):
        
        self.market_adapter = market_adapter
        self.options_adapter = options_adapter
        self.config = ModelConfig(**config.get("gmm_config", {}))
        
        # State persistence: {symbol: GMMState}
        self.states: Dict[str, GMMState] = {}
        
        logger.info("PhysicsEngine initialized with GMM mechanics")

    def run(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Run the physics filter for a single step.
        """
        # 1. Fetch Data Inputs
        # We need Bar, Quote, Flow, Greeks.
        # In a real pipeline, some of these might be passed in or cached.
        # Here we fetch fresh or use naive defaults if adapters missing.
        
        # Bar (Latest)
        # Note: MarketAdapter usually returns history. We need the latest finalized bar.
        # Ideally, we'd pass the current 'bar' from the pipeline runner.
        # Assuming the pipeline calls this *after* a bar close.
        # We will fetch 1 bar lookup.
        
        # Hack for fetching single bar efficiently or using cached data context
        # For now, we assume we fetch "latest available"
        
        # Quote
        try:
            q_data = self.market_adapter.get_quote(symbol)
            quote = QuoteL1(
                ts=q_data.timestamp,
                bid=q_data.bid,
                ask=q_data.ask,
                bid_size=q_data.bid_size,
                ask_size=q_data.ask_size
            )
        except Exception:
            # Fallback quote
            quote = QuoteL1(ts=timestamp, bid=0, ask=0, bid_size=0, ask_size=0)

        # Bar (OHLCV) - Essential
        # We need the bar ending at 'timestamp'
        # MarketAdapter usually gets a range.
        try:
            bars = self.market_adapter.get_bars(symbol, timestamp, timestamp, "1Min")
            if bars:
                b = bars[-1]
                bar = Bar1m(
                    ts=b.timestamp,
                    open=b.open, high=b.high, low=b.low, close=b.close,
                    volume=b.volume, vwap=getattr(b, 'vwap', b.close)
                )
            else:
                # If no bar (e.g. illiquid or data lag), we can't step physics accurately.
                # Return empty/previous.
                logger.warning(f"PhysicsEngine: No bar data for {symbol} at {timestamp}")
                return self._empty_snapshot(timestamp)
        except Exception:
            return self._empty_snapshot(timestamp)

        # Flow Aggregation
        # This usually comes from a specialized FlowAdapter or aggregating ticks.
        # We'll use a placeholder or check if OptionsAdapter has flow summary.
        flow = FlowAgg1m(ts=timestamp, buy_vol=bar.volume/2, sell_vol=bar.volume/2)
        if self.options_adapter and hasattr(self.options_adapter, 'get_flow_summary'):
            try:
                f_sum = self.options_adapter.get_flow_summary(symbol)
                # Map option flow to "FlowAgg" proxy
                flow = FlowAgg1m(
                    ts=timestamp, 
                    buy_vol=f_sum.get('call_volume', 0) + bar.volume/2, # Naive mix
                    sell_vol=f_sum.get('put_volume', 0) + bar.volume/2
                )
            except Exception:
                pass

        # Greeks / L2 / Wyckoff
        greeks = GreeksField1m(ts=timestamp) # Populate if logic available
        l2 = L2DepthSnapshot(ts=timestamp, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
        wyck = WyckoffState1m(ts=timestamp)

        # 2. Build Physics Inputs
        liq = build_liquidity_field(l2, bar, quote)
        micro = build_micro_state(quote, bar, flow, liq, greeks, wyck)

        # 3. State Management (Recursive)
        prev_state = self.states.get(symbol)
        if not prev_state:
            # Cold Start: Initialize Gaussian packet at current price
            # Initial variance based on price (0.1% volatility approx)
            start_var = (bar.close * 0.001) ** 2
            prev_state = GMMState(
                ts=timestamp, 
                components=[GaussianComponent(w=1.0, mu=bar.close, var=start_var)]
            )

        # 4. Run Physics Step
        # Observation y is VWAP (best proxy for value traded) or Close
        obs_price = bar.vwap if bar.vwap else bar.close
        
        new_state = gmm_step(prev_state, micro, liq, greeks, wyck, obs_price, self.config)
        self.states[symbol] = new_state

        # 5. Forecast & Output
        # 1-minute horizon
        forecast = make_forecast(timestamp, 1, new_state.components, micro.anchor_price, delta_tail=micro.spread)
        
        # Calculate derived metrics for the snapshot
        stiffness = micro.beta
        restoring_force = (micro.anchor_price - forecast.mean) # Rough proxy for pull
        
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "price": obs_price,
            "forecast_mean": forecast.mean,
            "sigma": math.sqrt(forecast.var),
            "p_up": forecast.p_up,
            "entropy": new_state.entropy,
            "stiffness": stiffness,
            "regime": "stable" if new_state.entropy < 1.0 else "chaotic",
            "component_count": len(new_state.components)
        }

    def _empty_snapshot(self, ts):
        return {
            "timestamp": ts,
            "error": "No Data"
        }
