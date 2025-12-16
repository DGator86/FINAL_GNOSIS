#!/usr/bin/env python3
"""
Display the top 25 ranked options underlyings with elasticity analysis.

This shows the complete picture: ranking metrics + elasticity physics.
"""

import sys
from datetime import datetime

from config.loader import load_config
from engines.dynamic_universe import DynamicUniverseRanker
from engines.hedge.hedge_engine_v3 import HedgeEngineV3
from adapters.adapter_factory import (
    create_market_data_adapter,
    create_options_adapter,
)


def main():
    """Show ranked top 25 with elasticity values."""
    print("\n" + "="*120)
    print("🎯 SUPER GNOSIS DHPE v3 - DYNAMIC TOP 25 WITH ELASTICITY ANALYSIS")
    print("="*120)
    print()
    
    # Load config
    config = load_config()
    
    # Create ranker
    print("📊 Phase 1: Ranking universe by options activity...")
    ranker = DynamicUniverseRanker(config.scanner.model_dump())
    top_metrics = ranker.get_ranked_metrics(n=25)
    
    if not top_metrics:
        print("❌ No symbols met the minimum criteria")
        sys.exit(1)
    
    print(f"✅ Found {len(top_metrics)} high-activity options underlyings")
    print()
    
    # Initialize hedge engine for elasticity analysis
    print("⚡ Phase 2: Calculating elasticity for each symbol...")
    try:
        options_adapter = create_options_adapter(prefer_real=True)
        market_adapter = create_market_data_adapter(prefer_real=True)
        hedge_engine = HedgeEngineV3(
            options_adapter=options_adapter,
            market_adapter=market_adapter,
            config=config.engines.hedge.model_dump()
        )
        print("✅ Elasticity engine initialized")
    except Exception as e:
        print(f"⚠️  Using fallback mode for elasticity (API data unavailable): {e}")
        hedge_engine = None
    
    print()
    
    # Calculate elasticity for each symbol
    elasticity_data = {}
    if hedge_engine:
        for m in top_metrics:
            try:
                snapshot = hedge_engine.run(m.symbol, datetime.now())
                if snapshot and snapshot.elasticity is not None:
                    elasticity_data[m.symbol] = {
                        'elasticity': snapshot.elasticity,
                        'movement_energy': snapshot.movement_energy or 0.0,
                        'dealer_gamma_sign': snapshot.dealer_gamma_sign or 0,
                    }
                else:
                    elasticity_data[m.symbol] = {
                        'elasticity': None,
                        'movement_energy': 0.0,
                        'dealer_gamma_sign': 0,
                    }
            except Exception as e:
                elasticity_data[m.symbol] = {
                    'elasticity': None,
                    'movement_energy': 0.0,
                    'dealer_gamma_sign': 0,
                }
    
    # Display comprehensive report
    print("="*120)
    print("📊 COMPLETE RANKED LIST WITH ELASTICITY")
    print("="*120)
    print()
    print(f"{'Rank':<6} {'Symbol':<8} {'Score':<8} {'Elasticity':<12} {'MovEnergy':<12} {'DealerΓ':<10} {'OptVol':<8} {'Gamma':<8} {'Flow':<8}")
    print("-" * 120)
    
    for m in top_metrics:
        elast_info = elasticity_data.get(m.symbol, {})
        elasticity_str = f"{elast_info.get('elasticity', 0.0):.3f}" if elast_info.get('elasticity') is not None else "N/A"
        movement_str = f"{elast_info.get('movement_energy', 0.0):.2f}"
        dealer_gamma = elast_info.get('dealer_gamma_sign', 0)
        dealer_str = "SHORT" if dealer_gamma < 0 else "LONG" if dealer_gamma > 0 else "-"
        
        print(
            f"{m.rank:<6} "
            f"{m.symbol:<8} "
            f"{m.composite_score:<8.2f} "
            f"{elasticity_str:<12} "
            f"{movement_str:<12} "
            f"{dealer_str:<10} "
            f"{m.options_volume:<8.1f} "
            f"{m.gamma_exposure:<8.1f} "
            f"{m.unusual_flow_score:<8.1f}"
        )
    
    print("-" * 120)
    print()
    
    # Summary
    print("📈 SYSTEM CONFIGURATION:")
    print(f"   • Ranking Mode:      Dynamic Top-N (auto-adapts to market)")
    print(f"   • Default Universe:  {config.scanner.default_top_n} symbols")
    print(f"   • Min Volume:        {config.scanner.min_daily_options_volume:,.0f} options contracts/day")
    print()
    print("⚖️  RANKING WEIGHTS:")
    print(f"   • Options Volume:    {config.scanner.ranking_criteria.options_volume_weight*100:.0f}%")
    print(f"   • Open Interest:     {config.scanner.ranking_criteria.open_interest_weight*100:.0f}%")
    print(f"   • Gamma Exposure:    {config.scanner.ranking_criteria.gamma_exposure_weight*100:.0f}%")
    print(f"   • Liquidity Score:   {config.scanner.ranking_criteria.liquidity_score_weight*100:.0f}%")
    print(f"   • Unusual Flow:      {config.scanner.ranking_criteria.unusual_flow_weight*100:.0f}%")
    print()
    print("🔬 ELASTICITY PHYSICS:")
    print("   • Elasticity:       Dealer gamma hedging resistance (higher = more pinning pressure)")
    print("   • Movement Energy:  Kinetic energy of price movement (momentum measure)")
    print("   • Dealer Gamma:     Dealer positioning (SHORT = supportive, LONG = resistive)")
    print()
    print("💡 USAGE:")
    print("   The system is now permanently locked to these dynamic top 25 symbols.")
    print("   No manual updates needed - adapts automatically as market conditions change.")
    print()
    print("   To start trading:")
    print("      python main.py live-loop          # Trade all 25 symbols")
    print("      python main.py scan-opportunities # Find current best setups")
    print()
    print("="*120)
    print()


if __name__ == "__main__":
    main()
