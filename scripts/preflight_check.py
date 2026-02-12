#!/usr/bin/env python3
"""Pre-flight check script for paper trading readiness.

Run this BEFORE starting paper trading to verify:
1. Environment configuration
2. API connectivity (Alpaca)
3. Account status and permissions
4. Risk parameters
5. System dependencies

Usage:
    python scripts/preflight_check.py
    
Exit codes:
    0 - All checks passed, ready for trading
    1 - Critical failure, do not trade
    2 - Warnings present, review before trading
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


class PreflightChecker:
    """Comprehensive pre-flight check for paper trading."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
    def run_all_checks(self) -> int:
        """Run all pre-flight checks.
        
        Returns:
            0 = success, 1 = critical failure, 2 = warnings
        """
        print("\n" + "="*70)
        print("🔍 SUPER GNOSIS PRE-FLIGHT CHECK")
        print("="*70)
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print("="*70 + "\n")
        
        # Run checks
        self._check_environment()
        self._check_alpaca_connection()
        self._check_account_status()
        self._check_risk_parameters()
        self._check_market_status()
        self._check_data_adapters()
        self._check_system_dependencies()
        self._check_data_directories()
        
        # Report results
        return self._report_results()
    
    def _check_environment(self):
        """Check environment variables."""
        print("📋 Checking environment configuration...")
        
        # Required for paper trading
        required_vars = [
            ("ALPACA_API_KEY", "Alpaca API key"),
            ("ALPACA_SECRET_KEY", "Alpaca secret key"),
        ]
        
        # Also check legacy alias
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        
        for var, description in required_vars:
            value = os.getenv(var)
            if var == "ALPACA_SECRET_KEY":
                value = alpaca_secret
            
            if not value or value.startswith("your_"):
                self.errors.append(f"Missing {description} ({var})")
            else:
                self.info.append(f"✓ {description} configured")
        
        # Check paper mode
        paper_mode = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        if paper_mode:
            self.info.append("✓ Paper trading mode enabled (safe)")
        else:
            self.warnings.append("⚠️ LIVE TRADING MODE - Are you sure?")
        
        # Optional but recommended
        optional_vars = [
            ("UNUSUAL_WHALES_API_TOKEN", "Unusual Whales API (options data)"),
            ("MAX_POSITION_SIZE_PCT", "Max position size %"),
            ("MAX_DAILY_LOSS_USD", "Max daily loss $"),
        ]
        
        for var, description in optional_vars:
            value = os.getenv(var)
            if value and not value.startswith("your_"):
                self.info.append(f"✓ {description} configured")
            else:
                self.warnings.append(f"Optional: {description} not set")
        
        print()
    
    def _check_alpaca_connection(self):
        """Check Alpaca API connectivity."""
        print("🔌 Checking Alpaca API connection...")
        
        try:
            from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
            from execution.broker_adapters.settings import get_alpaca_paper_setting
            
            paper = get_alpaca_paper_setting()
            adapter = AlpacaBrokerAdapter(paper=paper)
            
            # Test account endpoint
            account = adapter.get_account()
            self.info.append(f"✓ Connected to Alpaca ({'Paper' if paper else 'LIVE'})")
            self.info.append(f"  Account ID: {account.account_id[:8]}...")
            
        except Exception as e:
            self.errors.append(f"Alpaca connection failed: {e}")
        
        print()
    
    def _check_account_status(self):
        """Check account status and buying power."""
        print("💰 Checking account status...")
        
        try:
            from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
            from execution.broker_adapters.settings import get_alpaca_paper_setting
            
            adapter = AlpacaBrokerAdapter(paper=get_alpaca_paper_setting())
            account = adapter.get_account()
            
            # Check trading blocked
            if account.trading_blocked:
                self.errors.append("Trading is BLOCKED on this account")
            else:
                self.info.append("✓ Trading is enabled")
            
            # Check PDT status
            if account.pattern_day_trader:
                self.warnings.append("Account is flagged as Pattern Day Trader")
            
            # Check buying power
            if account.buying_power < 1000:
                self.warnings.append(f"Low buying power: ${account.buying_power:,.2f}")
            else:
                self.info.append(f"✓ Buying power: ${account.buying_power:,.2f}")
            
            # Portfolio value
            self.info.append(f"✓ Portfolio value: ${account.portfolio_value:,.2f}")
            self.info.append(f"✓ Cash: ${account.cash:,.2f}")
            
            # Options level
            if account.options_trading_level:
                self.info.append(f"✓ Options level: {account.options_trading_level}")
            else:
                self.warnings.append("Options trading level not reported")
            
        except Exception as e:
            self.errors.append(f"Account check failed: {e}")
        
        print()
    
    def _check_risk_parameters(self):
        """Check risk management configuration."""
        print("⚠️ Checking risk parameters...")
        
        max_position_pct = float(os.getenv("MAX_POSITION_SIZE_PCT", "2.0"))
        max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_USD", "5000.0"))
        max_leverage = float(os.getenv("MAX_PORTFOLIO_LEVERAGE", "1.0"))
        
        self.info.append(f"✓ Max position size: {max_position_pct}% of portfolio")
        self.info.append(f"✓ Max daily loss: ${max_daily_loss:,.2f}")
        self.info.append(f"✓ Max leverage: {max_leverage}x")
        
        # Sanity checks
        if max_position_pct > 10:
            self.warnings.append(f"High position size ({max_position_pct}%) - risky!")
        
        if max_daily_loss > 10000:
            self.warnings.append(f"High daily loss limit (${max_daily_loss:,.0f}) - review!")
        
        if max_leverage > 2:
            self.warnings.append(f"High leverage ({max_leverage}x) - dangerous!")
        
        print()
    
    def _check_market_status(self):
        """Check market hours and status."""
        print("🕐 Checking market status...")
        
        try:
            from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
            from execution.broker_adapters.settings import get_alpaca_paper_setting
            
            adapter = AlpacaBrokerAdapter(paper=get_alpaca_paper_setting())
            clock = adapter.get_market_clock()
            
            if clock:
                if clock["is_open"]:
                    self.info.append("✓ Market is OPEN")
                else:
                    self.info.append(f"Market is CLOSED")
                    if clock.get("next_open"):
                        self.info.append(f"  Next open: {clock['next_open']}")
            else:
                self.warnings.append("Could not retrieve market clock")
            
        except Exception as e:
            self.warnings.append(f"Market status check failed: {e}")
        
        print()
    
    def _check_data_adapters(self):
        """Check data adapter availability."""
        print("📊 Checking data adapters...")
        
        try:
            from engines.inputs.adapter_factory import (
                create_market_data_adapter,
                create_options_adapter,
            )
            
            # Market data
            market_adapter = create_market_data_adapter(prefer_real=True)
            adapter_name = type(market_adapter).__name__
            if "Static" in adapter_name or "Stub" in adapter_name:
                self.warnings.append(f"Using stub market data adapter: {adapter_name}")
            else:
                self.info.append(f"✓ Market data: {adapter_name}")
            
            # Options data
            options_adapter = create_options_adapter(prefer_real=True)
            adapter_name = type(options_adapter).__name__
            if "Static" in adapter_name or "Stub" in adapter_name:
                self.warnings.append(f"Using stub options adapter: {adapter_name}")
            else:
                self.info.append(f"✓ Options data: {adapter_name}")
            
        except Exception as e:
            self.errors.append(f"Adapter check failed: {e}")
        
        print()
    
    def _check_system_dependencies(self):
        """Check required Python packages."""
        print("📦 Checking system dependencies...")
        
        required_packages = [
            ("alpaca", "alpaca-py"),
            ("pydantic", "pydantic"),
            ("loguru", "loguru"),
            ("typer", "typer"),
        ]
        
        for module, package in required_packages:
            try:
                __import__(module)
                self.info.append(f"✓ {package} installed")
            except ImportError:
                self.errors.append(f"Missing package: {package}")
        
        # Optional packages
        optional_packages = [
            ("torch", "PyTorch (ML)"),
            ("streamlit", "Streamlit (Dashboard)"),
        ]
        
        for module, description in optional_packages:
            try:
                __import__(module)
                self.info.append(f"✓ {description} available")
            except ImportError:
                self.warnings.append(f"Optional: {description} not installed")
        
        print()
    
    def _check_data_directories(self):
        """Check required data directories exist."""
        print("📁 Checking data directories...")
        
        required_dirs = [
            Path("data"),
            Path("logs"),
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                self.info.append(f"✓ {dir_path}/ exists")
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.info.append(f"✓ Created {dir_path}/")
        
        # Check ledger
        ledger_path = Path("data/ledger.jsonl")
        if ledger_path.exists():
            self.info.append(f"✓ Ledger exists ({ledger_path.stat().st_size} bytes)")
        else:
            self.info.append("Ledger will be created on first trade")
        
        print()
    
    def _report_results(self) -> int:
        """Report check results and return exit code."""
        print("="*70)
        print("📊 PRE-FLIGHT CHECK RESULTS")
        print("="*70)
        
        # Info
        if self.info:
            print("\n✅ PASSED:")
            for item in self.info:
                print(f"   {item}")
        
        # Warnings
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for item in self.warnings:
                print(f"   {item}")
        
        # Errors
        if self.errors:
            print("\n❌ ERRORS:")
            for item in self.errors:
                print(f"   {item}")
        
        # Summary
        print("\n" + "="*70)
        
        if self.errors:
            print("❌ PRE-FLIGHT CHECK FAILED")
            print("   Fix the errors above before trading.")
            print("="*70 + "\n")
            return 1
        elif self.warnings:
            print("⚠️  PRE-FLIGHT CHECK PASSED WITH WARNINGS")
            print("   Review warnings before proceeding.")
            print("="*70 + "\n")
            return 2
        else:
            print("✅ PRE-FLIGHT CHECK PASSED")
            print("   System is ready for paper trading!")
            print("="*70 + "\n")
            return 0


def main():
    """Run pre-flight check."""
    checker = PreflightChecker()
    exit_code = checker.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
