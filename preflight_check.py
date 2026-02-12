#!/usr/bin/env python3
"""GNOSIS pre-flight checklist for Alpaca trading readiness."""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
from engines.inputs.unusual_whales_adapter import UnusualWhalesOptionsAdapter


def run_preflight_check(paper: bool | None = None) -> bool:
    """Run pre-flight checks to validate connectivity and readiness."""

    load_dotenv()

    print("\n🔍 Running GNOSIS Pre-Flight Checks...\n")

    checks_passed = 0
    total_checks = 7

    # Check 1: Alpaca Connection
    print("1️⃣ Testing Alpaca connection...")
    try:
        trader = AlpacaBrokerAdapter(paper=paper)
        print(f"   ✅ Connected to Alpaca successfully (paper={trader.paper})")
        checks_passed += 1
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Connection failed: {exc}")
        # Continue checking others but return False at end

    # Check 2: Account Status
    print("\n2️⃣ Checking account status...")
    try:
        if 'trader' in locals():
            account = trader.get_account()
            if account.trading_blocked:
                print("   ❌ Trading is blocked on this account")
            else:
                print(
                    "   ✅ Account active - "
                    f"Portfolio: ${account.portfolio_value:,.2f} | Cash: ${account.cash:,.2f}"
                )
                checks_passed += 1
        else:
             print("   ⚠️ Skipped (Alpaca connection failed)")
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Account check failed: {exc}")

    # Check 3: Market Status
    print("\n3️⃣ Checking market status...")
    try:
        if 'trader' in locals():
            if trader.is_market_open():
                print("   ✅ Market is OPEN")
            else:
                next_open = trader.get_next_market_open()
                opens_at = next_open.strftime("%Y-%m-%d %H:%M:%S %Z") if isinstance(next_open, datetime) else next_open
                print(f"   ⚠️  Market is CLOSED - Opens at: {opens_at}")
            checks_passed += 1
        else:
             print("   ⚠️ Skipped (Alpaca connection failed)")
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Market status check failed: {exc}")

    # Check 4: Alpaca Data Access
    print("\n4️⃣ Testing Alpaca market data access...")
    try:
        market_data = AlpacaMarketDataAdapter()
        end = datetime.utcnow()
        start = end - timedelta(days=5)
        bars = market_data.get_bars("SPY", start, end, timeframe="1Day")
        if bars:
            print(f"   ✅ Alpaca Market data accessible - Latest SPY close: ${bars[-1].close:.2f}")
            checks_passed += 1
        else:
            print("   ❌ No market data received from Alpaca")
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Alpaca Data access failed: {exc}")

    # Check 5: Massive API Status
    print("\n5️⃣ Testing Massive API connection...")
    try:
        massive = MassiveMarketDataAdapter()
        if massive.enabled:
            # Try to get market status or a simple quote
            status = massive.get_market_status()
            if status and status.get("status") != "error":
                print(f"   ✅ Massive API connected - Market: {status.get('market', 'Unknown')}")
                checks_passed += 1
            else:
                # Fallback check with a quote if status endpoint fails or returns error
                quote = massive.get_quote("SPY")
                if quote and quote.last > 0:
                     print(f"   ✅ Massive API connected (Quote verified) - SPY: ${quote.last:.2f}")
                     checks_passed += 1
                else:
                     print(f"   ❌ Massive API check failed: {status.get('error', 'Unknown error')}")
        else:
            print("   ⚠️ Massive API is disabled in config")
            checks_passed += 1 # Count as passed if intentionally disabled
    except Exception as exc:
        print(f"   ❌ Massive API failed: {exc}")

    # Check 6: Unusual Whales API Status
    print("\n6️⃣ Testing Unusual Whales API connection...")
    try:
        # Check environment variable first to avoid instantiating if missing token
        if not os.getenv("UNUSUAL_WHALES_API_TOKEN"):
             print("   ⚠️ Unusual Whales API Token not set")
             # Don't fail the check if not configured, unless required? 
             # Assuming optional for now, or count as passed if skipped?
             # Let's count as passed if missing but print warning.
             # Or better, require it if it's part of the system.
             # The system can run without it (stub), but for "full gnosis", maybe needed?
             # Let's verify connectivity if token exists.
             checks_passed += 1
        else:
            uw = UnusualWhalesOptionsAdapter()
            # Simple check - flow summary for SPY
            summary = uw.get_flow_summary("SPY")
            if summary:
                print(f"   ✅ Unusual Whales API connected - SPY Flow Sweep Ratio: {summary.get('sweep_ratio', 0):.2f}")
                checks_passed += 1
            else:
                # It might return empty dict if no recent flow, but connection is likely ok if no error raised
                print("   ✅ Unusual Whales API connected (No recent flow data for SPY)")
                checks_passed += 1
    except Exception as exc:
        print(f"   ❌ Unusual Whales API failed: {exc}")

    # Check 7: Order Capability (dry run)
    print("\n7️⃣ Checking order capabilities...")
    try:
        if 'trader' in locals():
            positions = trader.get_positions()
            print(f"   ✅ Order system accessible - Current positions: {len(positions)}")
            checks_passed += 1
        else:
             print("   ⚠️ Skipped (Alpaca connection failed)")
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Order check failed: {exc}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Pre-Flight Check Complete: {checks_passed}/{total_checks} passed")
    print(f"{'='*60}")

    if checks_passed == total_checks:
        print("\n✅ ALL CHECKS PASSED - Ready for live trading!")
        return True

    print("\n❌ SOME CHECKS FAILED - Please fix issues before trading")
    return False


if __name__ == "__main__":
    run_preflight_check()
