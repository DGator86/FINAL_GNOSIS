#!/usr/bin/env python3
"""
ADD STOP-LOSS ORDERS - Fixed for Alpaca API constraints

Handles:
- Fractional shares (DAY orders only)
- Options (DAY orders only)
- Whole share positions (GTC allowed)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import math
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

# Load environment
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

load_env()

ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").replace("/v2", "")

headers = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


def get_positions():
    """Get all positions from Alpaca."""
    response = httpx.get(f"{ALPACA_BASE_URL}/v2/positions", headers=headers, timeout=30.0)
    if response.status_code == 200:
        return response.json()
    return []


def place_stop_order(symbol: str, qty: float, stop_price: float, side: str = "sell", is_option: bool = False):
    """Place a stop order with proper time_in_force."""
    # Determine if fractional
    is_fractional = qty != int(qty)
    
    # Options and fractional shares must use DAY orders
    tif = "day" if (is_option or is_fractional) else "gtc"
    
    # Round qty for whole shares
    if not is_fractional:
        qty = int(qty)
    
    order_data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "stop",
        "time_in_force": tif,
        "stop_price": str(round(stop_price, 2)),
    }
    
    response = httpx.post(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=headers,
        json=order_data,
        timeout=30.0
    )
    
    return response.status_code in [200, 201], response.text


def place_trailing_stop(symbol: str, qty: float, trail_percent: float, side: str = "sell"):
    """Place a trailing stop order."""
    is_fractional = qty != int(qty)
    tif = "day" if is_fractional else "gtc"
    
    if not is_fractional:
        qty = int(qty)
    
    order_data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "trailing_stop",
        "time_in_force": tif,
        "trail_percent": str(trail_percent),
    }
    
    response = httpx.post(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=headers,
        json=order_data,
        timeout=30.0
    )
    
    return response.status_code in [200, 201], response.text


def main():
    print("\n" + "="*70)
    print("🛡️ ADDING STOP-LOSS ORDERS TO ALL POSITIONS")
    print("="*70)
    print(f"Time: {datetime.now().isoformat()}")
    print("="*70 + "\n")
    
    positions = get_positions()
    if not positions:
        print("No positions found.")
        return
    
    print(f"📊 Found {len(positions)} positions\n")
    
    success_count = 0
    fail_count = 0
    
    for p in positions:
        symbol = p["symbol"]
        qty = float(p["qty"])
        current_price = float(p.get("current_price", 0))
        avg_entry = float(p.get("avg_entry_price", 0))
        unrealized_plpc = float(p.get("unrealized_plpc", 0)) * 100
        side = p.get("side", "long")
        
        # Determine if option
        is_option = len(symbol) > 10
        
        # Calculate stop price
        if is_option:
            # Options: Stop at 40% below current (protect remaining value)
            stop_price = current_price * 0.60
            stop_type = "OPTION STOP"
        else:
            # Stocks: Trailing stop at 8%
            # Or fixed stop at 12% below current
            if unrealized_plpc > 5:
                # In profit - use tighter stop to protect gains
                stop_price = current_price * 0.92  # 8% trailing
                stop_type = "PROFIT PROTECT"
            elif unrealized_plpc > 0:
                # Small profit - moderate stop
                stop_price = current_price * 0.90  # 10% stop
                stop_type = "GAIN PROTECT"
            else:
                # In loss - wider stop to avoid whipsaw
                stop_price = current_price * 0.88  # 12% stop
                stop_type = "LOSS LIMIT"
        
        order_side = "sell" if side == "long" else "buy"
        
        print(f"📍 {symbol}")
        print(f"   Qty: {qty} | Current: ${current_price:.2f} | P&L: {unrealized_plpc:+.2f}%")
        print(f"   Type: {stop_type} | Stop Price: ${stop_price:.2f}")
        
        success, result = place_stop_order(symbol, qty, stop_price, order_side, is_option)
        
        if success:
            print(f"   ✅ Stop order placed!")
            success_count += 1
        else:
            # Try trailing stop for stocks
            if not is_option and "fractional" not in result.lower():
                trail_pct = 8.0 if unrealized_plpc > 5 else 10.0 if unrealized_plpc > 0 else 12.0
                success2, result2 = place_trailing_stop(symbol, qty, trail_pct, order_side)
                if success2:
                    print(f"   ✅ Trailing stop ({trail_pct}%) placed instead!")
                    success_count += 1
                else:
                    print(f"   ⚠️ Failed: {result2[:80]}")
                    fail_count += 1
            else:
                print(f"   ⚠️ Failed: {result[:80]}")
                fail_count += 1
        
        print()
        time.sleep(0.3)
    
    print("="*70)
    print(f"📊 SUMMARY: {success_count} stops placed, {fail_count} failed")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
