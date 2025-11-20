#!/usr/bin/env python3
"""GNOSIS pre-flight checklist for Alpaca trading readiness."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter


def run_preflight_check(paper: bool | None = None) -> bool:
    """Run pre-flight checks to validate Alpaca connectivity and readiness."""

    load_dotenv()

    print("\n🔍 Running GNOSIS Pre-Flight Checks...\n")

    checks_passed = 0
    total_checks = 5

    # Check 1: Alpaca Connection
    print("1️⃣ Testing Alpaca connection...")
    try:
        trader = AlpacaBrokerAdapter(paper=paper)
        print(f"   ✅ Connected to Alpaca successfully (paper={trader.paper})")
        checks_passed += 1
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Connection failed: {exc}")
        return False

    # Check 2: Account Status
    print("\n2️⃣ Checking account status...")
    try:
        account = trader.get_account()
        if account.trading_blocked:
            print("   ❌ Trading is blocked on this account")
            return False
        print(
            "   ✅ Account active - "
            f"Portfolio: ${account.portfolio_value:,.2f} | Cash: ${account.cash:,.2f}"
        )
        checks_passed += 1
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Account check failed: {exc}")
        return False

    # Check 3: Market Status
    print("\n3️⃣ Checking market status...")
    try:
        if trader.is_market_open():
            print("   ✅ Market is OPEN")
        else:
            next_open = trader.get_next_market_open()
            opens_at = next_open.strftime("%Y-%m-%d %H:%M:%S %Z") if isinstance(next_open, datetime) else next_open
            print(f"   ⚠️  Market is CLOSED - Opens at: {opens_at}")
        checks_passed += 1
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Market status check failed: {exc}")
        return False

    # Check 4: Data Access
    print("\n4️⃣ Testing market data access...")
    try:
        market_data = AlpacaMarketDataAdapter()
        end = datetime.utcnow()
        start = end - timedelta(days=5)
        bars = market_data.get_bars("SPY", start, end, timeframe="1Day")
        if bars:
            print(f"   ✅ Market data accessible - Latest SPY close: ${bars[-1].close:.2f}")
            checks_passed += 1
        else:
            print("   ❌ No market data received")
            return False
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Data access failed: {exc}")
        return False

    # Check 5: Order Capability (dry run)
    print("\n5️⃣ Checking order capabilities...")
    try:
        positions = trader.get_positions()
        print(f"   ✅ Order system accessible - Current positions: {len(positions)}")
        checks_passed += 1
    except Exception as exc:  # pragma: no cover - connectivity check
        print(f"   ❌ Order check failed: {exc}")
        return False

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
