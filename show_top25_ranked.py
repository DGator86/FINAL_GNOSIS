#!/usr/bin/env python3
"""
Display the top 25 ranked options underlyings with all metrics.
"""

import sys

from config.loader import load_config
from engines.dynamic_universe import DynamicUniverseRanker


def main():
    """Show ranked top 25 with full metrics."""
    print("\n" + "="*100)
    print("🎯 SUPER GNOSIS DHPE v3 - DYNAMIC TOP 25 OPTIONS UNDERLYINGS")
    print("="*100)
    print()
    
    # Load config
    config = load_config()
    
    # Create ranker
    print("📊 Analyzing market-wide options activity...")
    ranker = DynamicUniverseRanker(config.scanner.model_dump())
    
    # Get ranked metrics
    top_metrics = ranker.get_ranked_metrics(n=25)
    
    if not top_metrics:
        print("❌ No symbols met the minimum criteria")
        sys.exit(1)
    
    print(f"✅ Found {len(top_metrics)} high-activity options underlyings\n")
    
    # Display header
    print(f"{'Rank':<6} {'Symbol':<8} {'Score':<10} {'OptVol':<10} {'OI':<10} {'Gamma':<10} {'Liquid':<10} {'Flow':<10}")
    print("-" * 100)
    
    # Display each symbol
    for m in top_metrics:
        print(
            f"{m.rank:<6} "
            f"{m.symbol:<8} "
            f"{m.composite_score:<10.2f} "
            f"{m.options_volume:<10.1f} "
            f"{m.open_interest:<10.1f} "
            f"{m.gamma_exposure:<10.1f} "
            f"{m.liquidity_score:<10.1f} "
            f"{m.unusual_flow_score:<10.1f}"
        )
    
    print("-" * 100)
    print()
    print("📈 RANKING CRITERIA:")
    print(f"   • Options Volume:    {config.scanner.ranking_criteria.options_volume_weight*100:.0f}%")
    print(f"   • Open Interest:     {config.scanner.ranking_criteria.open_interest_weight*100:.0f}%")
    print(f"   • Gamma Exposure:    {config.scanner.ranking_criteria.gamma_exposure_weight*100:.0f}%")
    print(f"   • Liquidity Score:   {config.scanner.ranking_criteria.liquidity_score_weight*100:.0f}%")
    print(f"   • Unusual Flow:      {config.scanner.ranking_criteria.unusual_flow_weight*100:.0f}%")
    print()
    print("💡 These 25 symbols are now your permanent trading universe.")
    print("   The system automatically adapts as market conditions change.")
    print("   Just run: python main.py live-loop")
    print()
    print("="*100)
    print()


if __name__ == "__main__":
    main()
