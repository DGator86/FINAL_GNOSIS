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
from utils_bs import black_scholes

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Use a recent period with real data
SIM_START = datetime(2025, 10, 1, tzinfo=timezone.utc)
SIM_END = datetime(2025, 12, 20, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1_000_000.0
SYMBOLS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL"]

@dataclass
class OptionPosition:
    symbol: str
    type: str # "STRADDLE", "IRON_CONDOR"
    entry_ts: datetime
    expiry: datetime
    strikes: List[float] # [K1, K2] for straddle, [K1, K2, K3, K4] for condor
    qty: int
    entry_cost: float
    
@dataclass
class VolStrategy:
    capital: float
    positions: List[OptionPosition]
    ledger: List[Dict]
    
    def equity(self, current_prices: Dict[str, float], current_ts: datetime, r=0.05):
        val = 0.0
        # Value positions
        for pos in self.positions:
            S = current_prices.get(pos.symbol)
            if not S: continue
            
            T = (pos.expiry - current_ts).total_seconds() / (365*24*3600)
            if T < 0: T = 0
            
            # Simple implied vol assumption for marking to market
            # In reality, IV changes. Here we assume constant IV for PnL tracking or simple intrinsic+time value
            # Let's use a flat 20% IV for valuation to simulate "market pricing"
            sigma = 0.20 
            
            if pos.type == "STRADDLE":
                # Long Call K, Long Put K
                K = pos.strikes[0]
                call = black_scholes(S, K, T, r, sigma, "call")
                put = black_scholes(S, K, T, r, sigma, "put")
                val += (call + put) * pos.qty * 100
                
            elif pos.type == "IRON_CONDOR":
                # Short Strangle (Sell Put K2, Sell Call K3) + Long Wings (Buy Put K1, Buy Call K4)
                # K1 < K2 < K3 < K4
                k1, k2, k3, k4 = pos.strikes
                p1 = black_scholes(S, k1, T, r, sigma, "put") # Long
                p2 = black_scholes(S, k2, T, r, sigma, "put") # Short
                c3 = black_scholes(S, k3, T, r, sigma, "call") # Short
                c4 = black_scholes(S, k4, T, r, sigma, "call") # Long
                
                # Credit received at entry, so current value is cost to close (debit)
                # Value = (p1 - p2 - c3 + c4) * qty * 100
                # But typically we track equity = Cash + Liquidation Value
                # Liquidation Value of Short IC is Negative (Cost to close)
                # Wait, pos.qty is usually positive for "Long Condor" (debit)? 
                # Strategy says "Short Iron Condor" (Credit strategy).
                # So we sold the body.
                # Current Val = Cost to Close.
                # PnL = Entry Credit - Current Cost
                
                # Let's treat qty as +1 for Short Condor (standard naming).
                # Value = (p1 - p2 - c3 + c4) * 100 (This is negative if we sold inner, buy outer? No)
                # Short Iron Condor: Sell P(k2), Buy P(k1). Sell C(k3), Buy C(k4).
                # Cash delta at entry was +Credit.
                # Current Liquidation Cost is (P2 - P1) + (C3 - C4).
                
                cost_to_close = (p2 - p1) + (c3 - c4)
                val -= cost_to_close * pos.qty * 100
                
        return self.capital + val

    def check_exits(self, ts, prices):
        # Close positions nearing expiry or stop/profit
        remaining = []
        for pos in self.positions:
            S = prices.get(pos.symbol)
            if not S: 
                remaining.append(pos)
                continue
                
            T = (pos.expiry - ts).total_seconds() / (365*24*3600)
            
            # Close 1 day before expiry
            if T * 365 < 1:
                # Liquidate
                # Calculate final value
                val = 0.0
                sigma = 0.2
                r = 0.05
                if pos.type == "STRADDLE":
                    val = (black_scholes(S, pos.strikes[0], T, r, sigma, "call") + 
                           black_scholes(S, pos.strikes[0], T, r, sigma, "put")) * 100 * pos.qty
                elif pos.type == "IRON_CONDOR":
                    k1, k2, k3, k4 = pos.strikes
                    cost = ((black_scholes(S, k2, T, r, sigma, "put") - black_scholes(S, k1, T, r, sigma, "put")) +
                            (black_scholes(S, k3, T, r, sigma, "call") - black_scholes(S, k4, T, r, sigma, "call"))) * 100
                    val = -cost * pos.qty # Paying to close
                
                self.capital += val
                pnl = val - pos.entry_cost if pos.type == "STRADDLE" else val + pos.entry_cost # Logic tricky
                # Straddle: Paid entry, receive val. PnL = Val - Entry.
                # Condor: Recv entry (stored as neg cost?), paid val (negative). 
                # Let's handle entry_cost as "Cash Impact at Entry".
                # Straddle: Entry = -Debit. Exit = +Credit.
                # Condor: Entry = +Credit. Exit = -Debit.
                
                # Simplified: self.capital is cash. It was adjusted at entry.
                # Now we adjust at exit.
                
                self.ledger.append({"ts": ts, "symbol": pos.symbol, "type": "CLOSE", "pnl": "N/A"})
            else:
                remaining.append(pos)
        self.positions = remaining

def run_simulation():
    print(f"🔮 Starting GMM-Vol-Gnosis Backtest ({SIM_START.date()} - {SIM_END.date()})")
    
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
    
    # 2. Init Models
    strat = VolStrategy(INITIAL_CAPITAL, [], [])
    cfg = ModelConfig()
    cfg.dyn.q0 *= 7.0 
    
    gmm_states = {sym: None for sym in SYMBOLS}
    elasticity_map = {sym: 0.0 for sym in SYMBOLS}
    
    equity_curve = []
    
    # 3. Loop
    print("\n🚀 Running Gnosis Vol Loop...")
    
    for i, ts in enumerate(all_ts):
        prices = {}
        
        for sym in SYMBOLS:
            bar = data.get(sym, {}).get(ts)
            if not bar: continue
            
            close = float(bar.close)
            prices[sym] = close
            
            # --- Gnosis / GMM Update ---
            # Lightweight state
            quote = QuoteL1(ts=ts, bid=close, ask=close, bid_size=100, ask_size=100)
            flow = FlowAgg1m(ts=ts, buy_vol=bar.volume/2, sell_vol=bar.volume/2)
            l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
            
            # Proxy Elasticity: Volume / Range
            # High Elasticity = Stiff = High Vol/Range? No.
            # Physics: Elasticity = Force / Displacement.
            # Force ~ Volume. Displacement ~ Range.
            # E = Vol / Range.
            # High E = Hard to move (Stiff).
            # Low E = Easy to move (Flexible).
            
            tr = max(bar.high - bar.low, 0.01)
            elasticity = bar.volume / tr if tr > 0 else 0
            elasticity_map[sym] = elasticity
            
            # GMM
            internal_bar = Bar1m(ts=ts, open=bar.open, high=bar.high, low=bar.low, close=close, volume=bar.volume, vwap=close)
            liq = build_liquidity_field(l2, internal_bar, quote)
            micro = build_micro_state(quote, internal_bar, flow, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts))
            
            state = gmm_states[sym]
            if not state:
                state = GMMState(ts=ts, components=[GaussianComponent(w=1.0, mu=close, var=close*0.001)])
            state = gmm_step(state, micro, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts), close, cfg)
            gmm_states[sym] = state
            
            # --- Signal ---
            # 1. Entropy (Chaos)
            entropy = state.entropy
            
            # 2. Normalized Elasticity (Z-score ish would be better, but use threshold)
            # Log elasticity to normalize
            log_e = math.log(elasticity + 1)
            
            # Logic:
            # Low Entropy + High Elasticity -> Market is coiled spring (Stiff + Compressed) -> BUY STRADDLE (Expect explosion)
            # High Entropy + Low Elasticity -> Market is loose/choppy -> SELL CONDOR (Expect mean reversion/range)
            
            # Thresholds (Simulated optimization)
            # Entropy < 0.5 (Low)
            # Entropy > 1.2 (High)
            # Elasticity > 15 (High - depends on stock)
            
            # Check existing pos
            has_pos = any(p.symbol == sym for p in strat.positions)
            
            if not has_pos and i < len(all_ts) - 20: # Don't open at end
                expiry = ts + timedelta(days=7) # Weekly options
                T = 7/365
                r = 0.05
                sigma = 0.20 # Pricing Vol
                
                # Signal
                if entropy < 0.5 and log_e > 12: # COILED
                    # LONG STRADDLE
                    K = close
                    cost = (black_scholes(close, K, T, r, sigma, "call") + black_scholes(close, K, T, r, sigma, "put")) * 100 * 10 # 10 contracts
                    strat.capital -= cost
                    strat.positions.append(OptionPosition(sym, "STRADDLE", ts, expiry, [K], 10, cost))
                    strat.ledger.append({"ts": ts, "symbol": sym, "side": "OPEN STRADDLE", "cost": cost})
                    
                elif entropy > 1.2: # CHAOTIC / RANGE
                    # SHORT IRON CONDOR
                    # Strikes: +/- 2% and +/- 4%
                    k2 = close * 0.98 # Short Put
                    k1 = close * 0.96 # Long Put
                    k3 = close * 1.02 # Short Call
                    k4 = close * 1.04 # Long Call
                    
                    # Credit
                    credit = ((black_scholes(close, k2, T, r, sigma, "put") - black_scholes(close, k1, T, r, sigma, "put")) +
                              (black_scholes(close, k3, T, r, sigma, "call") - black_scholes(close, k4, T, r, sigma, "call"))) * 100 * 10
                              
                    strat.capital += credit
                    strat.positions.append(OptionPosition(sym, "IRON_CONDOR", ts, expiry, [k1, k2, k3, k4], 10, credit))
                    strat.ledger.append({"ts": ts, "symbol": sym, "side": "OPEN CONDOR", "credit": credit})

        # Manage Positions
        strat.check_exits(ts, prices)
        
        # Track
        eq = strat.equity(prices, ts)
        equity_curve.append({"ts": ts, "equity": eq})
        
        if i % 100 == 0:
            print(f"   {ts.date()} | Eq: ${eq:,.0f} | Pos: {len(strat.positions)}")

    # 4. Results
    final_eq = equity_curve[-1]["equity"]
    ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    print("-" * 60)
    print(f"💰 Final Equity: ${final_eq:,.2f}")
    print(f"📈 Return: {ret:.2%}")
    print(f"📜 Trades: {len(strat.ledger)}")
    
    # Analyze
    df = pd.DataFrame(equity_curve)
    if not df.empty:
        df.set_index("ts", inplace=True)
        # Daily sharpe
        daily = df['equity'].resample('D').last().pct_change().dropna()
        sharpe = daily.mean()/daily.std() * math.sqrt(252) if daily.std() > 0 else 0
        print(f"📊 Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    run_simulation()
