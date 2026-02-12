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
from market_model import (
    GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m,
    FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot, GaussianComponent,
    gmm_step, make_forecast, build_micro_state, build_liquidity_field, ModelConfig
)

# Q4 2025 Real Data Period
START_DATE = datetime(2025, 10, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 20, tzinfo=timezone.utc)
SYMBOL = "SPY"
INITIAL_CAPITAL = 1_000_000.0

@dataclass
class Trade:
    ts: datetime
    symbol: str
    side: str
    qty: float
    price: float
    pnl: float = 0.0
    cost: float = 0.0

class PhysicsFund:
    def __init__(self, capital):
        self.cash = capital
        self.positions = {} 
        self.entry_prices = {}
        self.last_prices = {}
        self.ledger = []
        self.equity_curve = []
        
    @property
    def equity(self):
        mkt_val = sum(self.positions.get(s, 0) * self.last_prices.get(s, 0) for s in self.positions)
        return self.cash + mkt_val

    def rebalance(self, ts, target_weights):
        current_eq = self.equity
        self.equity_curve.append({"ts": ts, "equity": current_eq})
        
        target_qtys = {}
        for sym, w in target_weights.items():
            if sym not in self.last_prices: continue
            price = self.last_prices[sym]
            target_qtys[sym] = int((current_eq * w) / price)
            
        for sym in list(set(list(self.positions.keys()) + list(target_qtys.keys()))):
            if sym not in self.last_prices: continue
            price = self.last_prices[sym]
            curr = self.positions.get(sym, 0)
            targ = target_qtys.get(sym, 0)
            
            diff = targ - curr
            if diff == 0: continue
            
            # Buffer check
            if abs(diff * price) < current_eq * 0.02: continue 
            
            cost = abs(diff * price) * 0.0005
            self.cash -= (diff * price + cost)
            
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
                if targ != 0: self.entry_prices[sym] = (old + new) / targ
            
            if targ == 0 and sym in self.positions:
                del self.positions[sym]
                if sym in self.entry_prices: del self.entry_prices[sym]

def run():
    print(f"⚛️  Starting Physics-Only Backtest for {SYMBOL} ({START_DATE.date()} - {END_DATE.date()})")
    
    alpaca = AlpacaMarketDataAdapter()
    bars = alpaca.get_bars(SYMBOL, START_DATE, END_DATE, timeframe="1Hour")
    
    if not bars:
        print("❌ No historical bars found.")
        return
        
    print(f"✅ Loaded {len(bars)} hourly steps.")
    
    fund = PhysicsFund(INITIAL_CAPITAL)
    cfg = ModelConfig()
    cfg.dyn.q0 *= 7.0 
    
    state = None
    
    print("\n🚀 Running Physics Loop...")
    
    for i, bar in enumerate(bars):
        ts = bar.timestamp
        close = float(bar.close)
        fund.last_prices[SYMBOL] = close
        
        # --- Physics Inputs ---
        quote = QuoteL1(ts=ts, bid=close, ask=close, bid_size=100, ask_size=100)
        flow = FlowAgg1m(ts=ts, buy_vol=bar.volume/2, sell_vol=bar.volume/2)
        internal_bar = Bar1m(ts=ts, open=bar.open, high=bar.high, low=bar.low, close=close, volume=bar.volume, vwap=close)
        l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
        
        liq = build_liquidity_field(l2, internal_bar, quote)
        micro = build_micro_state(quote, internal_bar, flow, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts))
        
        # --- Physics Step ---
        if not state:
            state = GMMState(ts=ts, components=[GaussianComponent(w=1.0, mu=close, var=close*0.001)])
        
        state = gmm_step(state, micro, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts), close, cfg)
        
        # --- Signal ---
        fc = make_forecast(ts, 60, state.components, close, delta_tail=close*0.002)
        
        # Native Trend Logic (With Fixed Physics)
        # (P_up - 0.5) * 2
        raw_signal = (fc.p_up - 0.5) * 2.0
        
        target_weights = {}
        
        # Confidence Gate
        if abs(raw_signal) > 0.2:
            # Vol Sizing
            ann_vol = math.sqrt(fc.var) / close * 37
            w = raw_signal * (0.15 / max(ann_vol, 0.10))
            target_weights[SYMBOL] = max(min(w, 0.50), -0.50) # Allow up to 50% concentration since single symbol
            
        # Rebalance every 4 hours
        if i % 4 == 0:
            fund.rebalance(ts, target_weights)
        else:
            fund.equity_curve.append({"ts": ts, "equity": fund.equity})
            
        if i % 50 == 0:
            print(f"   {ts.strftime('%Y-%m-%d %H:%M')} | Eq: ${fund.equity:,.0f} | Signal: {raw_signal:+.2f}")

    # Results
    final_eq = fund.equity
    ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    print("-" * 60)
    print(f"💰 Final Equity: ${final_eq:,.2f}")
    print(f"📈 Total Return: {ret:.2%}")
    
    if fund.ledger:
        df = pd.DataFrame([vars(t) for t in fund.ledger])
        win_rate = len(df[df.pnl > 0]) / len(df)
        print(f"🎯 Win Rate: {win_rate:.2%}")
        print(f"🔄 Trades: {len(df)}")
        print(f"💸 Fees: ${sum(t.cost for t in fund.ledger):,.2f}")
        
    # Drawdown
    eqs = [x['equity'] for x in fund.equity_curve]
    peak = eqs[0]
    dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        dd = max(dd, (peak - e)/peak)
    print(f"📉 Max Drawdown: {dd:.2%}")

if __name__ == "__main__":
    run()
