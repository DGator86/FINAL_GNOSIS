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
    gmm_step, make_forecast, profit_score, ModelConfig,
    build_micro_state, build_liquidity_field
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SIM_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
SIM_END = datetime(2025, 6, 30, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1_000_000.0
SYMBOLS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "GLD", "JPM", "XOM"]

# Industry Standard Constraints
MAX_POSITIONS = 5
MAX_LEVERAGE = 1.2
TARGET_VOL = 0.12  # 12% Annualized Vol
REBALANCE_FREQ_MIN = 60  # 1 Hour
FRICTION_BPS = 0.0005    # 5 bps

@dataclass
class Trade:
    ts: datetime
    symbol: str
    side: str
    qty: float
    price: float
    pnl: float = 0.0
    cost: float = 0.0

class QuantitativeFund:
    def __init__(self, capital):
        self.cash = capital
        self.positions = {} # {sym: qty}
        self.entry_prices = {} # {sym: price}
        self.ledger = []
        self.equity_curve = []
        self.peak_equity = capital
        
    @property
    def equity(self):
        mkt_val = sum(self.positions.get(s, 0) * self.last_prices.get(s, 0) for s in self.positions)
        return self.cash + mkt_val

    def set_prices(self, prices):
        self.last_prices = prices

    def rebalance(self, ts, signals):
        """
        Execute trades based on signals with risk management.
        signals: dict {sym: weight} (target weight)
        """
        current_eq = self.equity
        self.equity_curve.append({"ts": ts, "equity": current_eq})
        self.peak_equity = max(self.peak_equity, current_eq)
        
        # 1. Calculate Target Holdings
        target_holdings = {}
        used_leverage = 0.0
        
        # Rank signals by absolute weight (conviction)
        sorted_sigs = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Select top N ideas
        selected = sorted_sigs[:MAX_POSITIONS]
        
        # Normalize if leverage exceeded
        total_w = sum(abs(w) for s, w in selected)
        scaler = min(1.0, MAX_LEVERAGE / total_w) if total_w > 0 else 0.0
        
        for sym, w in selected:
            target_holdings[sym] = w * scaler
            
        # 2. Execute Diff
        # First Sell (raise cash)
        for sym in list(self.positions.keys()):
            # Only execute sell if we have a current price for the symbol
            if sym not in self.last_prices:
                continue
                
            if sym not in target_holdings:
                self._execute_trade(ts, sym, 0) # Close
            elif abs(target_holdings[sym] * current_eq / self.last_prices[sym] - self.positions[sym]) > 0:
                pass # Will handle in Buy step (netting)
                
        # Then Buy/Adjust
        for sym, target_w in target_holdings.items():
            if sym not in self.last_prices: continue
            
            target_val = target_w * current_eq
            target_qty = int(target_val / self.last_prices[sym])
            self._execute_trade(ts, sym, target_qty)
            
    def _execute_trade(self, ts, sym, target_qty):
        curr_qty = self.positions.get(sym, 0)
        diff = target_qty - curr_qty
        
        if diff == 0: return
        
        price = self.last_prices[sym]
        cost = abs(diff * price) * FRICTION_BPS
        
        # Record PnL if closing/reducing
        if (curr_qty > 0 and diff < 0) or (curr_qty < 0 and diff > 0):
            entry = self.entry_prices.get(sym, price)
            # FIFO PnL approx
            closed_qty = min(abs(diff), abs(curr_qty))
            direction = 1 if curr_qty > 0 else -1
            trade_pnl = (price - entry) * closed_qty * direction - cost
            self.ledger.append(Trade(ts, sym, "SELL" if diff < 0 else "BUY", diff, price, pnl=trade_pnl, cost=cost))
        else:
            # Increasing pos
            self.ledger.append(Trade(ts, sym, "BUY" if diff > 0 else "SELL", diff, price, pnl=-cost, cost=cost))
            
        # Update State
        self.cash -= (diff * price + cost)
        self.positions[sym] = target_qty
        
        # Update avg entry if increasing size
        if (curr_qty == 0) or (curr_qty * diff > 0): 
            # Weighted average
            old_val = curr_qty * self.entry_prices.get(sym, 0)
            new_val = diff * price
            self.entry_prices[sym] = (old_val + new_val) / target_qty if target_qty != 0 else 0
            
        if target_qty == 0:
            del self.positions[sym]
            if sym in self.entry_prices: del self.entry_prices[sym]

# ---------------------------------------------------------
# Data Loader
# ---------------------------------------------------------
def load_real_data():
    alpaca = AlpacaMarketDataAdapter()
    data = {}
    print(f"📥 Fetching Real Data (1Hour) from {SIM_START.date()} to {SIM_END.date()}...")
    
    for sym in SYMBOLS:
        # Use 1Hour bars for industry-standard backtest cadence (avoids microstructure noise)
        bars = alpaca.get_bars(sym, SIM_START, SIM_END, timeframe="1Hour")
        if not bars:
            print(f"⚠️  No data for {sym}, skipping.")
            continue
        # Index by timestamp
        data[sym] = {b.timestamp: b for b in bars}
        
    return data

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def run():
    # 1. Data
    market_data = load_real_data()
    if not market_data:
        print("❌ No data available.")
        return
        
    # Build timeline
    all_ts = sorted(list(set(ts for sym in market_data for ts in market_data[sym].keys())))
    print(f"✅ Loaded {len(all_ts)} hourly steps.")
    
    # 2. Engines
    fund = QuantitativeFund(INITIAL_CAPITAL)
    gmm_states = {sym: None for sym in SYMBOLS}
    cfg = ModelConfig()
    cfg.dyn.q0 *= 7.0 # Scale diffusion for 1H bars (sqrt(60))
    
    # 3. Loop
    print("\n🚀 Running Strategy Loop...")
    for i, ts in enumerate(all_ts):
        prices = {}
        signals = {}
        
        for sym in SYMBOLS:
            bar = market_data.get(sym, {}).get(ts)
            if not bar: continue
            
            prices[sym] = bar.close
            
            # --- Engine Update ---
            # Construct simplified state for backtest speed
            # Standardize bar access
            close_price = float(bar.close)
            high_price = float(bar.high)
            low_price = float(bar.low)
            open_price = float(bar.open)
            vol_val = float(bar.volume)
            
            # Use close as vwap proxy if unavailable
            vwap_val = getattr(bar, 'vwap', close_price)
            if vwap_val is None: vwap_val = close_price

            spread = close_price * 0.0002
            quote = QuoteL1(ts=ts, bid=close_price-spread, ask=close_price+spread, bid_size=500, ask_size=500)
            flow = FlowAgg1m(ts=ts, buy_vol=vol_val/2, sell_vol=vol_val/2)
            
            # Reconstruct Bar1m for internal use
            internal_bar = Bar1m(
                ts=ts,
                open=open_price, high=high_price, low=low_price, close=close_price,
                volume=vol_val, vwap=vwap_val
            )
            
            # Use VWAP as liquidity magnet
            l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
            liq = build_liquidity_field(l2, internal_bar, quote)
            
            # GMM Step
            micro = build_micro_state(quote, internal_bar, flow, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts))
            state = gmm_states[sym]
            if not state:
                state = GMMState(ts=ts, components=[GaussianComponent(w=1.0, mu=close_price, var=close_price*0.001)])
            
            state = gmm_step(state, micro, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts), close_price, cfg)
            gmm_states[sym] = state
            
            # --- Alpha Logic (Industry Best Practice) ---
            # 1. Forecast
            fc = make_forecast(ts, 60, state.components, close_price, delta_tail=close_price*0.002)
            
            # 2. Regime Filter (Entropy)
            # Low Entropy = Ordered market (Safe to trade)
            if state.entropy > 1.2: 
                continue # Skip chaos
                
            # 3. Signal
            # Probabilistic Edge > 55%
            score = 0.0
            if fc.p_up > 0.55:
                score = (fc.p_up - 0.5) # Positive Weight
            elif fc.p_up < 0.45:
                score = (fc.p_up - 0.5) # Negative Weight
                
            # 4. Volatility Scaling (Risk Parity ish)
            # Higher vol -> Lower weight
            vol_pct = math.sqrt(fc.var) / close_price
            if vol_pct > 0:
                weight = score / (vol_pct * 100) # Normalize
                signals[sym] = max(min(weight, 0.20), -0.20) # Cap at 20%
                
        # Execute
        fund.set_prices(prices)
        fund.rebalance(ts, signals)
        
        if i % 100 == 0:
            print(f"   {ts.date()} | Eq: ${fund.equity:,.0f} | Pos: {len(fund.positions)}")
            
    # 4. Analysis
    final_ret = (fund.equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    print("-" * 60)
    print(f"💰 Final Equity: ${fund.equity:,.2f}")
    print(f"📈 Return: {final_ret:.2%}")
    
    if fund.ledger:
        df = pd.DataFrame([vars(t) for t in fund.ledger])
        win_rate = len(df[df.pnl > 0]) / len(df)
        print(f"🎯 Win Rate: {win_rate:.2%}")
        print(f"💸 Total Fees: ${sum(t.cost for t in fund.ledger):,.2f}")
    
    # Drawdown
    dd = 0.0
    if len(fund.equity_curve) > 0:
        eqs = [x['equity'] for x in fund.equity_curve]
        peak = eqs[0]
        max_dd = 0.0
        for e in eqs:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        print(f"📉 Max Drawdown: {max_dd:.2%}")

if __name__ == "__main__":
    run()
