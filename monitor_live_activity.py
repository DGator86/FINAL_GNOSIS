#!/usr/bin/env python3
"""
Live Activity Monitor - See what tickers the engines are analyzing
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def monitor_logs():
    """Monitor log files for ticker activity"""

    print("="*80)
    print("🔍 LIVE ACTIVITY MONITOR")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Monitoring for ticker analysis, greek exposure, and dark pool data...")
    print("Press Ctrl+C to stop")
    print("="*80)
    print()

    # Find latest log file
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print(f"❌ Log directory '{log_dir}' not found. Start the trading system first!")
        return

    log_files = [f for f in os.listdir(log_dir) if f.startswith("dynamic_trading")]
    if not log_files:
        print(f"❌ No log files found in '{log_dir}'. Start the trading system first!")
        return

    latest_log = os.path.join(log_dir, sorted(log_files)[-1])
    print(f"📄 Monitoring: {latest_log}")
    print()

    # Track last position
    last_pos = 0

    # Keywords to filter for
    keywords = [
        "Testing:",           # When testing a ticker
        "Retrieved",          # When data is retrieved
        "greek exposure",     # Greek exposure data
        "dark pool",          # Dark pool data
        "GEX",                # Gamma exposure
        "VEX",                # Vanna exposure
        "Charm",              # Charm exposure
        "TRADE",              # Trade execution
        "ORDER",              # Order placement
        "Universe",           # Universe updates
        "symbols",            # Symbol lists
    ]

    try:
        while True:
            with open(latest_log, 'r') as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                last_pos = f.tell()

                for line in new_lines:
                    # Filter for relevant lines
                    if any(keyword in line for keyword in keywords):
                        # Color code different types
                        if "Testing:" in line or "Retrieved" in line:
                            print(f"🎯 {line.strip()}")
                        elif "GEX" in line or "greek exposure" in line:
                            print(f"📊 {line.strip()}")
                        elif "dark pool" in line:
                            print(f"🌑 {line.strip()}")
                        elif "TRADE" in line or "ORDER" in line:
                            print(f"💰 {line.strip()}")
                        elif "Universe" in line or "symbols" in line:
                            print(f"🌐 {line.strip()}")
                        else:
                            print(f"   {line.strip()}")

            time.sleep(1)  # Check every second

    except KeyboardInterrupt:
        print()
        print("="*80)
        print("Monitor stopped")
        print("="*80)

def show_current_universe():
    """Show current universe from ledger"""

    ledger_path = "data/ledger.jsonl"
    if not os.path.exists(ledger_path):
        print("❌ No ledger found. System hasn't run yet.")
        return

    # Read last few lines to see recent symbols
    with open(ledger_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("❌ Ledger is empty")
        return

    import json

    # Get unique symbols from last 100 entries
    recent_symbols = set()
    for line in lines[-100:]:
        try:
            entry = json.loads(line)
            if 'symbol' in entry:
                recent_symbols.add(entry['symbol'])
        except:
            continue

    print()
    print("📊 Recent Active Tickers:")
    print("-" * 40)
    for symbol in sorted(recent_symbols):
        print(f"  • {symbol}")
    print()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "universe":
        show_current_universe()
    else:
        monitor_logs()
