#!/usr/bin/env python3
"""
Gnosis Alpha - Quick Start Script

Run this to scan for signals and see the Alpha system in action.
"""

import sys
from pathlib import Path

# Ensure alpha module is importable
sys.path.insert(0, str(Path(__file__).parent))

from alpha import AlphaSignalGenerator, AlphaConfig, SignalDirection


def main():
    print("\n" + "="*60)
    print("  🔮 GNOSIS ALPHA - Short-Term Trading Signals")
    print("="*60 + "\n")
    
    # Load configuration
    config = AlphaConfig.from_env()
    
    # Validate config
    issues = config.validate()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print()
    
    # Show config summary
    print("📋 Configuration:")
    print(f"   Universe: {len(config.universe)} symbols")
    print(f"   Min Confidence: {config.min_confidence:.0%}")
    print(f"   Max Holding: {config.max_holding_days} days")
    print(f"   Stop Loss: {config.stop_loss_pct:.0%}")
    print(f"   Take Profit: {config.take_profit_pct:.0%}")
    print(f"   PDT Enabled: {config.pdt_enabled}")
    print()
    
    # Initialize signal generator
    print("🔄 Initializing signal generator...")
    generator = AlphaSignalGenerator(config=config)
    
    # Scan for signals
    print(f"📡 Scanning {len(config.universe)} symbols...\n")
    signals = generator.scan_universe()
    
    # Display results
    generator.print_signals(signals)
    
    # Summary
    buy_count = len([s for s in signals if s.direction == SignalDirection.BUY])
    sell_count = len([s for s in signals if s.direction == SignalDirection.SELL])
    
    print(f"\n📊 Summary:")
    print(f"   Total Signals: {len(signals)}")
    print(f"   BUY Signals: {buy_count}")
    print(f"   SELL Signals: {sell_count}")
    
    # Save signals
    if signals:
        filepath = generator.save_signals(signals)
        print(f"\n💾 Signals saved to: {filepath}")
    
    print("\n" + "="*60)
    print("  Run `python -m alpha.cli status` for account info")
    print("  Run `python -m alpha.cli trade SYMBOL` to execute")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
