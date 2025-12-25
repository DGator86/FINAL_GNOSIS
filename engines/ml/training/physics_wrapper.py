import numpy as np
from engines.physics.gmm_filter import gmm_step
from engines.physics.gmm_config import ModelConfig
from engines.physics.internal_schemas import (
    GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m,
    FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot, GaussianComponent
)
from engines.physics.features import build_micro_state, build_liquidity_field
from engines.physics.forecast import make_forecast
from datetime import datetime

class FastPhysicsWrapper:
    def __init__(self):
        self.config = ModelConfig()
        self.state = None
        
    def reset(self):
        self.state = None
        
    def update(self, bar_dict):
        """
        bar_dict: {'ts', 'open', 'high', 'low', 'close', 'volume'}
        """
        ts = bar_dict.get('ts', datetime.now())
        close = float(bar_dict['close'])
        
        # Build Inputs (Simplified for speed)
        quote = QuoteL1(ts=ts, bid=close, ask=close, bid_size=100, ask_size=100)
        flow = FlowAgg1m(ts=ts, buy_vol=bar_dict['volume']/2, sell_vol=bar_dict['volume']/2)
        internal_bar = Bar1m(
            ts=ts, 
            open=bar_dict['open'], high=bar_dict['high'], low=bar_dict['low'], close=close, 
            volume=bar_dict['volume'], vwap=close
        )
        # Empty L2/Greeks/Wyckoff for base physics training
        l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
        greeks = GreeksField1m(ts=ts)
        wyck = WyckoffState1m(ts=ts)
        
        liq = build_liquidity_field(l2, internal_bar, quote)
        micro = build_micro_state(quote, internal_bar, flow, liq, greeks, wyck)
        
        if not self.state:
            self.state = GMMState(ts=ts, components=[GaussianComponent(w=1.0, mu=close, var=close*0.001)])
            
        self.state = gmm_step(self.state, micro, liq, greeks, wyck, close, self.config)
        
        # Forecast
        fc = make_forecast(ts, 1, self.state.components, close, delta_tail=close*0.002)
        
        return {
            'entropy': self.state.entropy,
            'stiffness': micro.beta,
            'p_up': fc.p_up,
            'sigma': np.sqrt(fc.var)
        }
