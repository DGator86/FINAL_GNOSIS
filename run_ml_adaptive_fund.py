import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Deque
from dataclasses import dataclass
from collections import deque

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
# Configuration
# ---------------------------------------------------------
SIM_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
SIM_END = datetime(2025, 6, 30, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1_000_000.0
SYMBOLS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "GLD", "JPM", "XOM"]

# Adaptive Parameters
LOOKBACK_CYCLE = 12  # 12 * 1H = 12 Hours rolling window for ML regime detection
REBALANCE_FREQ = 4   # Every 4 Hours
ALLOCATION_BUFFER = 0.05 # Don't trade unless target changes by 5%

@dataclass
class Trade:
    ts: datetime
    symbol: str
    side: str
    qty: float
    price: float
    pnl: float = 0.0
    cost: float = 0.0

class MLCycleManager:
    """
    Online Learner that tracks the efficacy of the GMM signal 
    and adapts the strategy regime (Trend vs Mean Reversion).
    """
    def __init__(self, window=LOOKBACK_CYCLE):
        self.window = window
        self.predictions = deque(maxlen=window) # (pred_dir, actual_return)
        self.regime_score = 0.0 # -1 (Mean Rev) to +1 (Trend)
        
    def update(self, pred_dir: float, actual_ret: float):
        # pred_dir is +1 (Bullish), -1 (Bearish) from previous step
        # actual_ret is return of this step
        self.predictions.append((pred_dir, actual_ret))
        
        if len(self.predictions) < 3:
            self.regime_score = 0.0
            return

        # Correlation between prediction and outcome
        # If High Correlation -> Trend Following works
        # If Negative Correlation -> Mean Reversion dominates (Fade signals)
        
        preds = np.array([x[0] for x in self.predictions])
        rets = np.array([x[1] for x in self.predictions])
        
        # Simple directional accuracy: sum(sign(pred) == sign(ret))
        # But better: alignment score
        alignment = np.mean(preds * np.sign(rets))
        self.regime_score = alignment

    def get_adaptive_weight(self, raw_signal: float) -> float:
        """
        Adjust raw signal based on regime.
        raw_signal: -1.0 to 1.0 (from GMM)
        Returns: Adjusted weight -1.0 to 1.0
        """
        # If regime is Trend (score > 0), use signal as is
        # If regime is Mean Rev (score < 0), flip signal
        
        adaptive_signal = raw_signal * self.regime_score
        
        # Convexity: Only trade if regime is strong (> 0.2 or < -0.2)
        if abs(self.regime_score) < 0.2:
            return 0.0
            
        return adaptive_signal

class QuantitativeFund:
    def __init__(self, capital):
        self.cash = capital
        self.positions = {} 
        self.entry_prices = {}
        self.ledger = []
        self.equity_curve = []
        self.last_prices = {}
        
    @property
    def equity(self):
        mkt_val = sum(self.positions.get(s, 0) * self.last_prices.get(s, 0) for s in self.positions)
        return self.cash + mkt_val

    def rebalance(self, ts, target_weights):
        current_eq = self.equity
        self.equity_curve.append({"ts": ts, "equity": current_eq})
        
        # 1. Net Targets
        target_qtys = {}
        for sym, w in target_weights.items():
            if sym not in self.last_prices: continue
            price = self.last_prices[sym]
            target_qtys[sym] = int((current_eq * w) / price)
            
        # 2. Hysteresis Execution
        # First Sell
        for sym in list(self.positions.keys()):
            current_qty = self.positions[sym]
            target_qty = target_qtys.get(sym, 0)
            
            if sym not in self.last_prices: continue
            price = self.last_prices[sym]
            
            # Check buffer: % change in position value vs equity
            # diff_value = abs(target_qty - current_qty) * price
            # if diff_value < current_eq * ALLOCATION_BUFFER: continue -> logic handled in execute
            
            # We execute sells first to free cash
            if abs(target_qty) < abs(current_qty): # Reducing exposure
                self._execute_with_buffer(ts, sym, target_qty, current_eq)

        # Then Buy
        for sym, target_qty in target_qtys.items():
            current_qty = self.positions.get(sym, 0)
            if abs(target_qty) >= abs(current_qty): # Increasing exposure or flipping
                self._execute_with_buffer(ts, sym, target_qty, current_eq)

    def _execute_with_buffer(self, ts, sym, target_qty, total_equity):
        curr_qty = self.positions.get(sym, 0)
        diff = target_qty - curr_qty
        if diff == 0: return
        
        price = self.last_prices[sym]
        diff_val = abs(diff * price)
        
        # Buffer Check: Only trade if change is meaningful (> 5% of NAV) to stop churn
        if diff_val < total_equity * ALLOCATION_BUFFER:
            return

        cost = diff_val * 0.0005 # 5 bps
        
        # PnL calc
        if (curr_qty > 0 and diff < 0) or (curr_qty < 0 and diff > 0):
            entry = self.entry_prices.get(sym, price)
            closed_qty = min(abs(diff), abs(curr_qty))
            direction = 1 if curr_qty > 0 else -1
            trade_pnl = (price - entry) * closed_qty * direction - cost
            self.ledger.append(Trade(ts, sym, "SELL" if diff < 0 else "BUY", diff, price, pnl=trade_pnl, cost=cost))
        else:
            self.ledger.append(Trade(ts, sym, "BUY" if diff > 0 else "SELL", diff, price, pnl=-cost, cost=cost))
            
        self.cash -= (diff * price + cost)
        self.positions[sym] = target_qty
        
        # Entry price update
        if (curr_qty == 0) or (curr_qty * diff > 0):
            old_val = curr_qty * self.entry_prices.get(sym, 0)
            new_val = diff * price
            self.entry_prices[sym] = (old_val + new_val) / target_qty if target_qty != 0 else 0
            
        if target_qty == 0:
            del self.positions[sym]
            if sym in self.entry_prices: del self.entry_prices[sym]

# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------
def run():
    print(f"🧠 Starting ML-Adaptive Fund Simulation ({SIM_START.date()} - {SIM_END.date()})")
    
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
    fund = QuantitativeFund(INITIAL_CAPITAL)
    cfg = ModelConfig()
    cfg.dyn.q0 *= 7.0 
    
    # ML State
    gmm_states = {sym: None for sym in SYMBOLS}
    ml_learners = {sym: MLCycleManager() for sym in SYMBOLS}
    last_predictions = {sym: 0.0 for sym in SYMBOLS} # Store prev prediction sign
    
    # 3. Loop
    print("\n🚀 Running Adaptive Loop...")
    
    for i, ts in enumerate(all_ts):
        prices = {}
        target_weights = {}
        
        for sym in SYMBOLS:
            bar = data.get(sym, {}).get(ts)
            if not bar: continue
            
            prices[sym] = bar.close
            
            # Update ML Learner with result of previous prediction
            prev_pred = last_predictions[sym]
            if prev_pred != 0 and i > 0:
                # Find previous price (approx) to calc return
                # Simply: if we are long (pred > 0) and price went up, good.
                # Since we don't store prev price easily here, we use the fact that
                # gmm_state was updated with previous bar last step.
                # Actually, simpler: just use current bar open vs close? 
                # No, we predict t to t+1. 
                # Let's use bar return: (close - open) approx for intra-bar, or (close - prev_close).
                # We'll use (close - open) as proxy for "did the hour move as expected?"
                # Or better: (close - prev_close) if we tracked it.
                # Let's assume bar.open is close enough to prev_close for 1H continuous.
                ret = (bar.close - bar.open) / bar.open
                ml_learners[sym].update(prev_pred, ret)
            
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
            
            # ML Adaptation: Scale by Regime Score
            # If correlated, weight = signal * score (positive)
            # If anti-correlated, weight = signal * score (negative) -> Flips signal
            adaptive_weight = ml_learners[sym].get_adaptive_weight(raw_signal)
            
            # Volatility Sizing (Kelly-ish)
            # Target Vol / Realized Vol
            # Sigma from forecast is 1H sigma. Annualize -> * sqrt(1400) ~ 37
            ann_vol = math.sqrt(fc.var) / close_price * 37
            if ann_vol < 0.10: ann_vol = 0.10
            
            size_scaler = 0.15 / ann_vol # Target 15% vol
            final_weight = adaptive_weight * size_scaler
            
            # Cap
            final_weight = max(min(final_weight, 0.25), -0.25)
            
            target_weights[sym] = final_weight
            last_predictions[sym] = 1.0 if raw_signal > 0 else -1.0
            
        # Execute (Only every 4 hours to save fees)
        fund.last_prices = prices # Update prices every hour for equity calc
        
        if i % REBALANCE_FREQ == 0:
            fund.rebalance(ts, target_weights)
        else:
            # Still record equity curve
            fund.equity_curve.append({"ts": ts, "equity": fund.equity})
            
        if i % 100 == 0:
            print(f"   {ts.date()} | Eq: ${fund.equity:,.0f} | Regime (SPY): {ml_learners['SPY'].regime_score:.2f}")

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
        
    # Stats
    eqs = [x['equity'] for x in fund.equity_curve]
    peak = eqs[0]
    max_dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)
    print(f"📉 Max Drawdown: {max_dd:.2%}")

if __name__ == "__main__":
    run()
