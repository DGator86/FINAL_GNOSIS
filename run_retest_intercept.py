import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath("."))

from loguru import logger
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from market_model import (
    GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m,
    FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot, GaussianComponent,
    gmm_step, make_forecast, build_micro_state, build_liquidity_field, ModelConfig
)

# Q2 2025
SIM_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
SIM_END = datetime(2025, 6, 30, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1_000_000.0
SYMBOLS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "GLD"]

@dataclass
class Trade:
    ts: datetime
    symbol: str
    side: str
    qty: float
    price: float
    pnl: float = 0.0
    cost: float = 0.0

class Fund:
    def __init__(self, capital):
        self.cash = capital
        self.positions = {}
        self.entry_prices = {}
        self.last_prices = {}
        self.ledger = []
        self.equity_curve = []
        
    @property
    def equity(self):
        return self.cash + sum(self.positions.get(s,0)*self.last_prices.get(s,0) for s in self.positions)

    def rebalance(self, ts, target_weights):
        current_eq = self.equity
        self.equity_curve.append({"ts": ts, "equity": current_eq})
        
        target_qtys = {}
        for sym, w in target_weights.items():
            if sym not in self.last_prices: continue
            target_qtys[sym] = int((current_eq * w) / self.last_prices[sym])
            
        for sym in set(list(self.positions.keys()) + list(target_qtys.keys())):
            if sym not in self.last_prices: continue
            price = self.last_prices[sym]
            curr = self.positions.get(sym, 0)
            targ = target_qtys.get(sym, 0)
            diff = targ - curr
            
            if diff == 0: continue
            if abs(diff * price) < current_eq * 0.02: continue # 2% buffer
            
            cost = abs(diff * price) * 0.0005
            self.cash -= (diff * price + cost)
            
            # PnL approx
            pnl = 0.0
            if (curr > 0 and diff < 0) or (curr < 0 and diff > 0):
                entry = self.entry_prices.get(sym, price)
                closed = min(abs(diff), abs(curr))
                direction = 1 if curr > 0 else -1
                pnl = (price - entry) * closed * direction - cost
                
            self.ledger.append(Trade(ts, sym, "BUY" if diff > 0 else "SELL", diff, price, pnl=pnl, cost=cost))
            self.positions[sym] = targ
            
            if (curr == 0) or (curr * diff > 0):
                old = curr * self.entry_prices.get(sym, 0)
                new = diff * price
                if targ != 0: self.entry_prices[sym] = (old+new)/targ
                
            if targ == 0 and sym in self.positions:
                del self.positions[sym]
                if sym in self.entry_prices: del self.entry_prices[sym]

def run():
    print("🚀 Running Retest with Intercept Physics (Native Trend Following)")
    alpaca = AlpacaMarketDataAdapter()
    data = {}
    for sym in SYMBOLS:
        bars = alpaca.get_bars(sym, SIM_START, SIM_END, timeframe="1Hour")
        if bars: data[sym] = {b.timestamp: b for b in bars}
    
    all_ts = sorted(list(set(ts for sym in data for ts in data[sym].keys())))
    fund = Fund(INITIAL_CAPITAL)
    cfg = ModelConfig()
    cfg.dyn.q0 *= 7.0 
    gmm_states = {sym: None for sym in SYMBOLS}
    
    print(f"Loaded {len(all_ts)} steps. Running...")
    
    for i, ts in enumerate(all_ts):
        prices = {}
        weights = {}
        
        for sym in SYMBOLS:
            bar = data.get(sym, {}).get(ts)
            if not bar: continue
            close = float(bar.close)
            prices[sym] = close
            
            # GMM Inputs
            quote = QuoteL1(ts=ts, bid=close, ask=close, bid_size=100, ask_size=100)
            flow = FlowAgg1m(ts=ts, buy_vol=bar.volume/2, sell_vol=bar.volume/2)
            internal_bar = Bar1m(ts=ts, open=bar.open, high=bar.high, low=bar.low, close=close, volume=bar.volume, vwap=close)
            l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
            
            liq = build_liquidity_field(l2, internal_bar, quote)
            micro = build_micro_state(quote, internal_bar, flow, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts))
            
            state = gmm_states[sym]
            if not state:
                state = GMMState(ts=ts, components=[GaussianComponent(w=1.0, mu=close, var=close*0.001)])
            
            state = gmm_step(state, micro, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts), close, cfg)
            gmm_states[sym] = state
            
            # Signal: P_up (Native)
            # IMPORTANT: We are NOT using "ALWAYS_FADE" here.
            # We are testing if the Physics Fix makes the native signal accurate.
            fc = make_forecast(ts, 60, state.components, close, delta_tail=close*0.002)
            
            signal = (fc.p_up - 0.5) * 2.0 # -1 to 1
            
            # Only trade if confident
            if abs(signal) > 0.2:
                # Vol Sizing
                ann_vol = math.sqrt(fc.var) / close * 37
                w = signal * (0.10 / max(ann_vol, 0.10))
                weights[sym] = max(min(w, 0.25), -0.25)
                
        fund.last_prices = prices
        if i % 4 == 0:
            fund.rebalance(ts, weights)
        else:
            fund.equity_curve.append({"ts": ts, "equity": fund.equity})
            
        if i % 100 == 0:
            print(f"   {ts.date()} | Eq: ${fund.equity:,.0f}")
            
    # Result
    ret = (fund.equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    print(f"💰 Final: ${fund.equity:,.2f} ({ret:.2%})")
    if fund.ledger:
        df = pd.DataFrame([vars(t) for t in fund.ledger])
        win = len(df[df.pnl > 0]) / len(df)
        print(f"🎯 Win Rate: {win:.2%}")
        
    eqs = [x['equity'] for x in fund.equity_curve]
    peak = eqs[0]
    dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        dd = max(dd, (peak - e)/peak)
    print(f"📉 DD: {dd:.2%}")

if __name__ == "__main__":
    run()
