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

# ---------------------------------------------------------
# OPTIMIZED CONFIGURATION (Derived from Grid Search)
# ---------------------------------------------------------
SIM_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
SIM_END = datetime(2025, 6, 30, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1_000_000.0
SYMBOLS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "GLD", "JPM", "XOM"]

# Best Settings Found:
# {'ml_window': 4, 'vol_target': 0.1, 'base_strategy': 'ALWAYS_FADE', 'rebalance_freq': 4, 'stop_loss': 0.05}
# Note: "ALWAYS_FADE" means we treat GMM P_up as a contrarian indicator.
# If P_up > 0.5, we Short. If P_up < 0.5, we Long.

STRATEGY_CONFIG = {
    "vol_target": 0.10,        # 10% Vol Target (More conservative than 15%)
    "rebalance_freq": 4,       # 4 Hours
    "stop_loss": 0.05,         # 5% Stop Loss
    "mode": "ALWAYS_FADE"      # Mean Reversion
}

@dataclass
class Trade:
    ts: datetime
    symbol: str
    side: str
    qty: float
    price: float
    pnl: float = 0.0
    cost: float = 0.0

class OptimalFund:
    def __init__(self, capital):
        self.cash = capital
        self.positions = {} 
        self.entry_prices = {}
        self.last_prices = {}
        self.ledger = []
        self.equity_curve = []
        self.peak_equity = capital
        
    @property
    def equity(self):
        mkt_val = sum(self.positions.get(s, 0) * self.last_prices.get(s, 0) for s in self.positions)
        return self.cash + mkt_val

    def check_stops(self):
        """Execute stop losses intra-step"""
        for sym in list(self.positions.keys()):
            entry = self.entry_prices.get(sym)
            curr = self.last_prices.get(sym)
            if not entry or not curr: continue
            
            qty = self.positions[sym]
            pnl_pct = (curr - entry)/entry if qty > 0 else (entry - curr)/entry
            
            if pnl_pct < -STRATEGY_CONFIG["stop_loss"]:
                # CLOSE
                cost = abs(qty * curr) * 0.0005
                self.cash += (qty * curr - cost)
                
                # Log
                pnl = (curr - entry) * qty - cost
                self.ledger.append(Trade(datetime.now(), sym, "STOP_LOSS", -qty, curr, pnl=pnl, cost=cost))
                
                del self.positions[sym]

    def rebalance(self, ts, target_weights):
        current_eq = self.equity
        self.equity_curve.append({"ts": ts, "equity": current_eq})
        self.peak_equity = max(self.peak_equity, current_eq)
        
        # 1. Net Targets
        target_qtys = {}
        for sym, w in target_weights.items():
            if sym not in self.last_prices: continue
            price = self.last_prices[sym]
            target_qtys[sym] = int((current_eq * w) / price)
            
        # 2. Execution
        for sym in list(set(list(self.positions.keys()) + list(target_qtys.keys()))):
            if sym not in self.last_prices: continue
            
            price = self.last_prices[sym]
            curr_qty = self.positions.get(sym, 0)
            target_qty = target_qtys.get(sym, 0)
            
            diff = target_qty - curr_qty
            if diff == 0: continue
            
            # Buffer check (5% of position value change)
            if abs(diff * price) < current_eq * 0.02: continue 
            
            cost = abs(diff * price) * 0.0005
            self.cash -= (diff * price + cost)
            
            # PnL calc for ledger
            trade_pnl = 0.0
            if (curr_qty > 0 and diff < 0) or (curr_qty < 0 and diff > 0):
                entry = self.entry_prices.get(sym, price)
                closed_qty = min(abs(diff), abs(curr_qty))
                direction = 1 if curr_qty > 0 else -1
                trade_pnl = (price - entry) * closed_qty * direction - cost
            
            self.ledger.append(Trade(ts, sym, "BUY" if diff > 0 else "SELL", diff, price, pnl=trade_pnl, cost=cost))
            
            self.positions[sym] = target_qty
            
            # Update avg entry
            if (curr_qty == 0) or (curr_qty * diff > 0):
                old_val = curr_qty * self.entry_prices.get(sym, 0)
                new_val = diff * price
                if target_qty != 0:
                    self.entry_prices[sym] = (old_val + new_val) / target_qty
            
            if target_qty == 0 and sym in self.positions:
                del self.positions[sym]
                if sym in self.entry_prices: del self.entry_prices[sym]

def run():
    print(f"🚀 Launching FINAL OPTIMIZED STRATEGY ({STRATEGY_CONFIG['mode']})")
    
    # 1. Data
    alpaca = AlpacaMarketDataAdapter()
    data = {}
    print("📥 Fetching 1H Bars...")
    for sym in SYMBOLS:
        bars = alpaca.get_bars(sym, SIM_START, SIM_END, timeframe="1Hour")
        if bars:
            data[sym] = {b.timestamp: b for b in bars}
            
    all_ts = sorted(list(set(ts for sym in data for ts in data[sym].keys())))
    print(f"✅ Loaded {len(all_ts)} hourly steps.")
    
    # 2. Init
    fund = OptimalFund(INITIAL_CAPITAL)
    cfg = ModelConfig()
    cfg.dyn.q0 *= 7.0 
    gmm_states = {sym: None for sym in SYMBOLS}
    
    # 3. Loop
    for i, ts in enumerate(all_ts):
        prices = {}
        target_weights = {}
        
        for sym in SYMBOLS:
            bar = data.get(sym, {}).get(ts)
            if not bar: continue
            
            prices[sym] = bar.close
            
            # --- GMM ---
            # Inputs
            close_price = float(bar.close)
            vwap_val = getattr(bar, 'vwap', close_price) or close_price
            quote = QuoteL1(ts=ts, bid=close_price*0.9998, ask=close_price*1.0002, bid_size=500, ask_size=500)
            flow = FlowAgg1m(ts=ts, buy_vol=bar.volume/2, sell_vol=bar.volume/2)
            internal_bar = Bar1m(ts=ts, open=bar.open, high=bar.high, low=bar.low, close=close_price, volume=bar.volume, vwap=vwap_val)
            l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
            
            liq = build_liquidity_field(l2, internal_bar, quote)
            micro = build_micro_state(quote, internal_bar, flow, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts))
            
            state = gmm_states[sym]
            if not state:
                state = GMMState(ts=ts, components=[GaussianComponent(w=1.0, mu=close_price, var=close_price*0.001)])
            
            state = gmm_step(state, micro, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts), close_price, cfg)
            gmm_states[sym] = state
            
            # --- Signal Generation ---
            fc = make_forecast(ts, 60, state.components, close_price, delta_tail=close_price*0.002)
            
            # Raw Signal: (P_up - 0.5) * 2 -> Range [-1, 1]
            raw_signal = (fc.p_up - 0.5) * 2.0
            
            # STRATEGY: ALWAYS FADE (Mean Reversion)
            final_signal = -raw_signal 
            
            # Volatility Sizing (Risk Parity)
            # 1H Vol -> Annualized
            ann_vol = math.sqrt(fc.var) / close_price * 37
            if ann_vol < 0.10: ann_vol = 0.10
            
            size_scaler = STRATEGY_CONFIG["vol_target"] / ann_vol
            final_weight = final_signal * size_scaler
            
            # Cap
            final_weight = max(min(final_weight, 0.25), -0.25)
            
            target_weights[sym] = final_weight
            
        # Execute
        fund.last_prices = prices
        fund.check_stops()
        
        if i % STRATEGY_CONFIG["rebalance_freq"] == 0:
            fund.rebalance(ts, target_weights)
        else:
            fund.equity_curve.append({"ts": ts, "equity": fund.equity})
            
        if i % 100 == 0:
            print(f"   {ts.date()} | Eq: ${fund.equity:,.0f}")

    # 4. Results
    final_ret = (fund.equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    print("-" * 60)
    print(f"💰 Final Equity: ${fund.equity:,.2f}")
    print(f"📈 Return: {final_ret:.2%}")
    
    if fund.ledger:
        df = pd.DataFrame([vars(t) for t in fund.ledger])
        win_rate = len(df[df.pnl > 0]) / len(df)
        print(f"🎯 Win Rate: {win_rate:.2%}")
        print(f"💸 Total Fees: ${sum(t.cost for t in fund.ledger):,.2f}")
        print(f"🔄 Total Trades: {len(fund.ledger)}")
        
    eqs = [x['equity'] for x in fund.equity_curve]
    peak = eqs[0]
    max_dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)
    print(f"📉 Max Drawdown: {max_dd:.2%}")

if __name__ == "__main__":
    run()
