import os
import sys
import math
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath("."))

from loguru import logger
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from market_model import (
    GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m,
    FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot, GaussianComponent,
    gmm_step, make_forecast, profit_score, ModelConfig,
    build_micro_state, build_liquidity_field, update_universe
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
START_DATE = datetime(2025, 4, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 6, 30, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1_000_000.0
SYMBOLS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "GLD", "JPM", "XOM"]

@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    current_price: float
    entry_time: datetime

@dataclass
class TradeRecord:
    ts: datetime
    symbol: str
    side: str
    qty: float
    price: float
    cost: float
    pnl: float = 0.0
    reason: str = ""

# ---------------------------------------------------------
# Synthetic Data Generator (Heston-like for missing dates)
# ---------------------------------------------------------
def generate_synthetic_series(symbol: str, start_dt, end_dt, start_price=100.0, vol=0.2):
    """Generate 1-minute OHLCV bars using stochastic volatility."""
    minutes = int((end_dt - start_dt).total_seconds() / 60)
    # Don't generate millions of bars for a quick sim, step by 15m to be practical
    step_min = 15 
    steps = int(minutes / step_min)
    
    dt = step_min / (252 * 390) 
    
    data = []
    price = start_price
    variance = vol ** 2
    kappa = 2.0
    theta = variance
    xi = 0.3
    
    curr_time = start_dt
    
    for _ in range(steps):
        # Heston Vol
        dw_var = random.gauss(0, 1) * math.sqrt(dt)
        variance = max(0.01**2, variance + kappa*(theta - variance)*dt + xi*math.sqrt(variance)*dw_var)
        sigma = math.sqrt(variance)
        
        # Price drift/diff
        dw_price = random.gauss(0, 1) * math.sqrt(dt)
        # Random trend per symbol to make it interesting
        trend = 0.05 if "NVDA" in symbol else (0.02 if "SPY" in symbol else 0.0)
        
        ret = (trend - 0.5*variance)*dt + sigma*dw_price
        price *= math.exp(ret)
        
        # Bar construction
        high = price * (1 + 0.5*sigma*math.sqrt(dt))
        low = price * (1 - 0.5*sigma*math.sqrt(dt))
        open_p = (high + low)/2 # approx
        
        # Market hours only (9:30 - 16:00 ET)
        # Simplified: just filter by hour in UTC roughly (13:30 to 20:00)
        if 13 <= curr_time.hour < 20:
            data.append(Bar1m(
                ts=curr_time,
                open=open_p, high=high, low=low, close=price,
                volume=int(10000/price), vwap=price
            ))
            
        curr_time += timedelta(minutes=step_min)
        
    return data

# ---------------------------------------------------------
# Portfolio Manager (Best Practices)
# ---------------------------------------------------------
class PortfolioManager:
    def __init__(self, capital):
        self.initial_capital = capital
        self.cash = capital
        self.positions: Dict[str, Position] = {}
        self.ledger: List[TradeRecord] = []
        self.equity_curve = []
        
    @property
    def total_equity(self):
        pos_val = sum(p.qty * p.current_price for p in self.positions.values())
        return self.cash + pos_val
        
    def update_prices(self, current_prices: Dict[str, float]):
        for sym, price in current_prices.items():
            if sym in self.positions:
                self.positions[sym].current_price = price
                
    def execute_signals(self, ts: datetime, signals: List[Dict], current_prices: Dict[str, float]):
        # signals: [{symbol, score, volatility, side, forecast_mean, ...}]
        
        # 1. Volatility Targeting (Industry Standard)
        # Target portfolio vol: 15% annualized
        TARGET_VOL = 0.15
        
        for sig in signals:
            sym = sig['symbol']
            price = current_prices.get(sym)
            if not price: continue
            
            # Kelly-like sizing or Vol-Target
            # Weight = (TargetVol / InstrumentVol) * ForecastConfidence
            
            # Annualize sigma (sig['vol'] is 15m vol approx)
            # 15m vol -> annual = sigma * sqrt(4 * 6.5 * 252) approx 80
            ann_vol = sig['vol'] * 80 
            if ann_vol < 0.05: ann_vol = 0.05 # floor
            
            base_weight = TARGET_VOL / ann_vol
            
            # Confidence scaler (0.5 to 1.0 -> 0 to 1)
            conf_scale = max(0, (sig['p_up'] - 0.5) * 2) if sig['side'] == 'BUY' else max(0, (0.5 - sig['p_up']) * 2)
            
            target_weight = base_weight * conf_scale * 0.2 # Cap single pos at 20% leverage-adjusted
            
            # Cap max weight
            target_weight = min(target_weight, 0.10) # 10% max allocation per ticker
            
            target_value = self.total_equity * target_weight
            
            current_pos = self.positions.get(sym)
            current_qty = current_pos.qty if current_pos else 0
            
            desired_qty = int(target_value / price) if sig['side'] == 'BUY' else -int(target_value / price)
            
            # Trade Delta
            delta_qty = desired_qty - current_qty
            
            # Friction Check (Don't trade small noise)
            trade_val = abs(delta_qty * price)
            if trade_val < self.total_equity * 0.01: 
                continue
                
            # EXECUTE
            cost = trade_val * 0.0005 # 5bps slippage+comm
            self.cash -= (delta_qty * price + cost)
            
            # Update Position
            new_qty = current_qty + delta_qty
            if new_qty == 0:
                if sym in self.positions:
                    # Realized PnL
                    entry = self.positions[sym].entry_price
                    # simple fifo approx
                    pnl = (price - entry) * self.positions[sym].qty * (1 if self.positions[sym].qty > 0 else -1)
                    del self.positions[sym]
            else:
                if sym not in self.positions:
                    self.positions[sym] = Position(sym, new_qty, price, price, ts)
                else:
                    # Avg Cost Update
                    prev = self.positions[sym]
                    if (prev.qty > 0 and delta_qty > 0) or (prev.qty < 0 and delta_qty < 0):
                        # Increasing size, blend price
                        total_val = prev.qty * prev.entry_price + delta_qty * price
                        prev.entry_price = total_val / new_qty
                    prev.qty = new_qty
                    prev.current_price = price
            
            self.ledger.append(TradeRecord(
                ts=ts, symbol=sym, side="BUY" if delta_qty > 0 else "SELL",
                qty=delta_qty, price=price, cost=cost, reason=f"GMM:{sig['p_up']:.2f}"
            ))
            
        # Rebalance / Stop Loss check (simplified)
        self.equity_curve.append({"ts": ts, "equity": self.total_equity})

# ---------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------
def run_simulation():
    print(f"🔬 Simulating Q2 2025 Fund Strategy ({START_DATE.date()} to {END_DATE.date()})")
    print("-" * 80)
    
    # 1. Load Data
    alpaca = None
    try:
        alpaca = AlpacaMarketDataAdapter()
        print("✅ Alpaca Adapter Loaded (checking for real future data...)")
    except:
        print("⚠️ Alpaca not available, forcing synthetic mode.")

    market_data_map = {}
    
    for sym in SYMBOLS:
        bars = []
        if alpaca:
            # Try fetching "future" data (if this is a time-shifted sandbox)
            bars = alpaca.get_bars(sym, START_DATE, END_DATE, timeframe="15Min")
        
        if not bars:
            print(f"   Generating synthetic Q2 2025 data for {sym}...")
            # Seed prices based on late 2024 ballpark
            seed_price = 600.0 if sym == "SPY" else (150.0 if sym == "NVDA" else 200.0)
            bars = generate_synthetic_series(sym, START_DATE, END_DATE, start_price=seed_price)
            
        market_data_map[sym] = bars
        
    # OHLCV from Alpaca adapter uses 'timestamp', our internal Bar1m uses 'ts'
    # We need to handle both if mix of synthetic and real
    all_timestamps = set()
    for bars in market_data_map.values():
        for b in bars:
            if hasattr(b, 'ts'):
                all_timestamps.add(b.ts)
            elif hasattr(b, 'timestamp'):
                all_timestamps.add(b.timestamp)
    
    all_timestamps = sorted(list(all_timestamps))
    print(f"🗓️  Timeline: {len(all_timestamps)} simulation steps (15m cadence)")
    
    # 2. Initialize Models
    portfolio = PortfolioManager(INITIAL_CAPITAL)
    gmm_states = {sym: None for sym in SYMBOLS}
    cfg = ModelConfig()
    cfg.dyn.q0 *= math.sqrt(15) # scaling for 15m
    
    # 3. Main Loop
    start_time = time.time()
    
    # Optimization: pre-index bars by timestamp
    bars_by_ts = {}
    for sym, bars in market_data_map.items():
        bars_by_ts[sym] = {}
        for b in bars:
            key = b.ts if hasattr(b, 'ts') else b.timestamp
            bars_by_ts[sym][key] = b

    for i, ts in enumerate(all_timestamps):
        current_prices = {}
        signals = []
        
        for sym in SYMBOLS:
            bar = bars_by_ts[sym].get(ts)
            if not bar: continue
            
            current_prices[sym] = bar.close
            
            # --- GMM Step ---
            # Lightweight inputs
            # Standardize bar access
            close_price = float(bar.close)
            high_price = float(bar.high)
            low_price = float(bar.low)
            open_price = float(bar.open)
            vol_val = float(bar.volume)
            
            quote = QuoteL1(ts=ts, bid=close_price-0.01, ask=close_price+0.01, bid_size=100, ask_size=100)
            flow = FlowAgg1m(ts=ts, buy_vol=vol_val/2, sell_vol=vol_val/2)
            greeks = GreeksField1m(ts=ts)
            wyck = WyckoffState1m(ts=ts)
            l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
            
            # Reconstruct Bar1m for internal use if it's an OHLCV object
            internal_bar = Bar1m(
                ts=ts,
                open=open_price, high=high_price, low=low_price, close=close_price,
                volume=vol_val, vwap=close_price
            )
            
            liq = build_liquidity_field(l2, internal_bar, quote)
            micro = build_micro_state(quote, internal_bar, flow, liq, greeks, wyck)
            
            state = gmm_states[sym]
            if state is None:
                # Cold start
                state = GMMState(ts=ts, components=[GaussianComponent(w=1.0, mu=close_price, var=0.1)])
            
            state = gmm_step(state, micro, liq, greeks, wyck, close_price, cfg)
            gmm_states[sym] = state
            
            # Forecast & Signal
            # 1-step forecast (15m ahead)
            fc = make_forecast(ts, 1, state.components, close_price, delta_tail=close_price*0.001)
            
            # Logic: Trend Following with Mean Rev Filter
            # If P_up > 0.6 => Bullish
            # If P_up < 0.4 => Bearish
            
            side = "NEUTRAL"
            if fc.p_up > 0.60: side = "BUY"
            elif fc.p_up < 0.40: side = "SELL"
            
            if side != "NEUTRAL":
                signals.append({
                    "symbol": sym,
                    "side": side,
                    "p_up": fc.p_up,
                    "vol": math.sqrt(fc.var) / close_price, # pct vol
                    "forecast_mean": fc.mean
                })
                
        # Portfolio Update
        portfolio.update_prices(current_prices)
        portfolio.execute_signals(ts, signals, current_prices)
        
        # Progress
        if i % 500 == 0:
            print(f"   Step {i}/{len(all_timestamps)} | Eq: ${portfolio.total_equity:,.0f} | Pos: {len(portfolio.positions)}")

    # 4. Results
    duration = time.time() - start_time
    print("-" * 80)
    print(f"🏁 Simulation Complete in {duration:.2f}s")
    
    final_eq = portfolio.total_equity
    ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    print(f"💰 Final Equity: ${final_eq:,.2f}")
    print(f"📈 Total Return: {ret:.2%}")
    print(f"📜 Total Trades: {len(portfolio.ledger)}")
    
    if len(portfolio.ledger) > 0:
        win_rate = len([t for t in portfolio.ledger if t.qty < 0 and t.price < t.cost]) # Rough approx, need closed trade logic for real wr
        # Actually PnL is tracked on closing trades in positions logic? No, simplified logic didn't store closed pnl list.
        # Let's just output the positions.
        print("\nOpen Positions:")
        for sym, pos in portfolio.positions.items():
            print(f"   {sym}: {pos.qty:.0f} @ ${pos.current_price:.2f} (Entry: ${pos.entry_price:.2f})")

    # Generate Performance Chart Data
    eq_df = pd.DataFrame(portfolio.equity_curve)
    eq_df.set_index("ts", inplace=True)
    
    # Calculate Sharpe
    if len(eq_df) > 1:
        daily = eq_df['equity'].resample('D').last().pct_change().dropna()
        sharpe = daily.mean() / daily.std() * math.sqrt(252) if daily.std() > 0 else 0
        print(f"📊 Sharpe Ratio: {sharpe:.2f}")
        
    print("-" * 80)

if __name__ == "__main__":
    run_simulation()
