#!/usr/bin/env python3
"""
EMERGENCY POSITION CLEANUP - Close losing positions NOW!

This script:
1. Identifies positions beyond stop-loss threshold
2. Closes them immediately via Alpaca API
3. Places stop-loss orders for remaining positions

Author: GNOSIS Emergency Risk Management
"""

import os
import sys
from pathlib import Path
from datetime import datetime
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

# Risk thresholds
STOP_LOSS_PCT = -30.0  # Close anything down more than 30%
CRITICAL_LOSS_PCT = -50.0  # Immediately close anything down 50%+

headers = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


def get_positions():
    """Get all positions from Alpaca."""
    response = httpx.get(f"{ALPACA_BASE_URL}/v2/positions", headers=headers, timeout=30.0)
    if response.status_code == 200:
        return response.json()
    print(f"❌ Failed to get positions: {response.status_code}")
    return []


def close_position(symbol: str, qty: float = None):
    """Close a position immediately."""
    url = f"{ALPACA_BASE_URL}/v2/positions/{symbol}"
    if qty:
        url += f"?qty={qty}"
    
    response = httpx.delete(url, headers=headers, timeout=30.0)
    if response.status_code in [200, 204]:
        return True, response.json() if response.text else {}
    return False, response.text


def place_stop_loss_order(symbol: str, qty: float, stop_price: float, side: str = "sell"):
    """Place a stop-loss order."""
    order_data = {
        "symbol": symbol,
        "qty": str(abs(qty)),
        "side": side,
        "type": "stop",
        "time_in_force": "gtc",
        "stop_price": str(round(stop_price, 2)),
    }
    
    response = httpx.post(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=headers,
        json=order_data,
        timeout=30.0
    )
    
    if response.status_code in [200, 201]:
        return True, response.json()
    return False, response.text


def main():
    print("\n" + "="*70)
    print("🚨 EMERGENCY POSITION CLEANUP 🚨")
    print("="*70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Stop-Loss Threshold: {STOP_LOSS_PCT}%")
    print(f"Critical Loss Threshold: {CRITICAL_LOSS_PCT}%")
    print("="*70 + "\n")
    
    positions = get_positions()
    if not positions:
        print("No positions found or API error.")
        return
    
    print(f"📊 Found {len(positions)} positions\n")
    
    # Categorize positions
    critical = []  # > 50% loss - CLOSE NOW
    stop_loss = []  # > 30% loss - CLOSE
    at_risk = []    # > 15% loss - ADD STOP
    healthy = []    # < 15% loss - ADD STOP
    
    for p in positions:
        symbol = p["symbol"]
        qty = float(p["qty"])
        unrealized_plpc = float(p.get("unrealized_plpc", 0)) * 100
        unrealized_pl = float(p.get("unrealized_pl", 0))
        current_price = float(p.get("current_price", 0))
        avg_entry = float(p.get("avg_entry_price", 0))
        
        pos_data = {
            "symbol": symbol,
            "qty": qty,
            "entry": avg_entry,
            "current": current_price,
            "pl": unrealized_pl,
            "pl_pct": unrealized_plpc,
            "side": p.get("side", "long"),
        }
        
        if unrealized_plpc <= CRITICAL_LOSS_PCT:
            critical.append(pos_data)
        elif unrealized_plpc <= STOP_LOSS_PCT:
            stop_loss.append(pos_data)
        elif unrealized_plpc <= -15:
            at_risk.append(pos_data)
        else:
            healthy.append(pos_data)
    
    # === PHASE 1: CLOSE CRITICAL POSITIONS ===
    print("="*70)
    print("🚨 PHASE 1: CLOSING CRITICAL POSITIONS (> 50% loss)")
    print("="*70)
    
    if critical:
        for p in critical:
            print(f"\n❌ CLOSING: {p['symbol']}")
            print(f"   Loss: ${p['pl']:,.2f} ({p['pl_pct']:.2f}%)")
            print(f"   Entry: ${p['entry']:.2f} → Current: ${p['current']:.2f}")
            
            success, result = close_position(p['symbol'])
            if success:
                print(f"   ✅ CLOSED successfully!")
            else:
                print(f"   ⚠️ Close failed: {result}")
            time.sleep(0.5)  # Rate limiting
    else:
        print("✅ No critical positions to close.")
    
    # === PHASE 2: CLOSE STOP-LOSS POSITIONS ===
    print("\n" + "="*70)
    print("⚠️ PHASE 2: CLOSING STOP-LOSS POSITIONS (> 30% loss)")
    print("="*70)
    
    if stop_loss:
        for p in stop_loss:
            print(f"\n❌ CLOSING: {p['symbol']}")
            print(f"   Loss: ${p['pl']:,.2f} ({p['pl_pct']:.2f}%)")
            
            success, result = close_position(p['symbol'])
            if success:
                print(f"   ✅ CLOSED successfully!")
            else:
                print(f"   ⚠️ Close failed: {result}")
            time.sleep(0.5)
    else:
        print("✅ No stop-loss positions to close.")
    
    # === PHASE 3: ADD STOP-LOSS ORDERS ===
    print("\n" + "="*70)
    print("🛡️ PHASE 3: ADDING STOP-LOSS ORDERS TO REMAINING POSITIONS")
    print("="*70)
    
    remaining = at_risk + healthy
    
    for p in remaining:
        symbol = p['symbol']
        qty = p['qty']
        current = p['current']
        entry = p['entry']
        
        # Skip options (complex symbols) for now - they need different handling
        is_option = len(symbol) > 10
        
        if is_option:
            # For options, set stop at 50% of current value
            stop_price = current * 0.5
            print(f"\n📍 {symbol} (OPTION)")
            print(f"   Current: ${current:.2f} | Setting stop at: ${stop_price:.2f} (50% of value)")
        else:
            # For stocks, set stop at 10% below current
            stop_price = current * 0.90
            print(f"\n📍 {symbol} (STOCK)")
            print(f"   Current: ${current:.2f} | Setting stop at: ${stop_price:.2f} (-10%)")
        
        # Determine side based on position
        side = "sell" if p['side'] == "long" else "buy"
        
        success, result = place_stop_loss_order(symbol, qty, stop_price, side)
        if success:
            print(f"   ✅ Stop-loss order placed!")
        else:
            print(f"   ⚠️ Failed: {result[:100] if isinstance(result, str) else result}")
        
        time.sleep(0.3)
    
    # === SUMMARY ===
    print("\n" + "="*70)
    print("📊 CLEANUP SUMMARY")
    print("="*70)
    print(f"Critical positions closed: {len(critical)}")
    print(f"Stop-loss positions closed: {len(stop_loss)}")
    print(f"Stop-loss orders added: {len(remaining)}")
    
    total_loss_closed = sum(p['pl'] for p in critical + stop_loss)
    print(f"\nTotal realized loss from cleanup: ${total_loss_closed:,.2f}")
    print("="*70)
    print("✅ EMERGENCY CLEANUP COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Confirmation
    print("\n⚠️  WARNING: This will CLOSE losing positions and place STOP-LOSS orders!")
    print("    This action cannot be undone.\n")
    
    confirm = input("Type 'CLEANUP' to proceed: ")
    if confirm.strip().upper() == "CLEANUP":
        main()
    else:
        print("Aborted.")
