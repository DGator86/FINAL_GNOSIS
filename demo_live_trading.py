#!/usr/bin/env python3
"""
Live Trading Demo for Super Gnosis DHPE v3
Shows the system in action with real Alpaca data
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from main import build_pipeline, load_config

def demo_pipeline():
    """Run a single pipeline iteration and show results beautifully."""
    
    print("\n" + "="*80)
    print("🚀 SUPER GNOSIS DHPE v3 - LIVE TRADING DEMO")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Symbol: SPY (S&P 500 ETF)")
    print("Mode: Live Analysis with Real Alpaca Data")
    print("="*80 + "\n")
    
    # Load config and build pipeline
    print("📋 Loading configuration...")
    config = load_config()
    
    print("🔧 Building pipeline with real adapters...")
    runner = build_pipeline("SPY", config)
    
    print("🏃 Running analysis...\n")
    
    # Run pipeline
    result = runner.run_once(datetime.now())
    
    # Display results beautifully
    print("\n" + "="*80)
    print("📊 ANALYSIS RESULTS")
    print("="*80)
    
    if result.hedge_snapshot:
        h = result.hedge_snapshot
        print("\n🎯 HEDGE ENGINE v3.0 (Elasticity Theory)")
        print("-" * 80)
        print(f"   Market Elasticity:    {h.elasticity:.2f} (resistance to price movement)")
        print(f"   Movement Energy:      {h.movement_energy:.2f} (energy required to move price)")
        print(f"   Energy Asymmetry:     {h.energy_asymmetry:+.3f} ({'↑ BULLISH' if h.energy_asymmetry > 0 else '↓ BEARISH' if h.energy_asymmetry < 0 else '↔ NEUTRAL'})")
        print(f"   Pressure Net:         {h.pressure_net:+,.2f}")
        print(f"   Dealer Gamma Sign:    {h.dealer_gamma_sign:+.3f} ({'LONG γ' if h.dealer_gamma_sign > 0 else 'SHORT γ' if h.dealer_gamma_sign < 0 else 'NEUTRAL'})")
        print(f"   Regime:               {h.regime.upper()}")
        print(f"   Confidence:           {h.confidence:.1%}")
    
    if result.liquidity_snapshot:
        l = result.liquidity_snapshot
        print("\n💧 LIQUIDITY ENGINE")
        print("-" * 80)
        print(f"   Liquidity Score:      {l.liquidity_score:.3f} ({'Excellent' if l.liquidity_score > 0.8 else 'Good' if l.liquidity_score > 0.5 else 'Poor'})")
        print(f"   Bid-Ask Spread:       {l.bid_ask_spread:.4f}% ({'Tight' if l.bid_ask_spread < 0.1 else 'Wide'})")
        print(f"   Impact Cost:          {l.impact_cost:.4f}%")
    
    if result.sentiment_snapshot:
        s = result.sentiment_snapshot
        print("\n📰 SENTIMENT ENGINE")
        print("-" * 80)
        print(f"   Overall Sentiment:    {s.sentiment_score:+.3f} ({'BULLISH' if s.sentiment_score > 0.2 else 'BEARISH' if s.sentiment_score < -0.2 else 'NEUTRAL'})")
        print(f"   News Sentiment:       {s.news_sentiment:+.3f}")
        print(f"   Flow Sentiment:       {s.flow_sentiment:+.3f}")
        print(f"   Technical Sentiment:  {s.technical_sentiment:+.3f}")
        print(f"   Confidence:           {s.confidence:.1%}")
    
    if result.elasticity_snapshot:
        e = result.elasticity_snapshot
        print("\n⚡ ELASTICITY ENGINE")
        print("-" * 80)
        print(f"   Volatility:           {e.volatility:.2%} ({e.volatility_regime.upper()})")
        print(f"   Trend Strength:       {e.trend_strength:.3f}")
    
    if result.suggestions:
        print("\n🤖 AGENT SUGGESTIONS")
        print("-" * 80)
        for sug in result.suggestions:
            print(f"   {sug.agent_name}:")
            print(f"      Direction: {sug.direction.value.upper()}")
            print(f"      Confidence: {sug.confidence:.1%}")
            print(f"      Reasoning: {sug.reasoning}")
    
    if result.consensus:
        c = result.consensus
        print("\n🎯 CONSENSUS")
        print("-" * 80)
        print(f"   Direction:            {c['direction'].upper()}")
        print(f"   Confidence:           {c['confidence']:.1%}")
        print(f"   Consensus Value:      {c['consensus_value']:+.3f}")
        print(f"   Agents Contributing:  {c['num_agents']}")
    
    if result.trade_ideas:
        print("\n💡 TRADE IDEAS")
        print("-" * 80)
        for idea in result.trade_ideas:
            print(f"   Strategy: {idea.strategy_type.value.upper()}")
            print(f"   Direction: {idea.direction.value.upper()}")
            print(f"   Confidence: {idea.confidence:.1%}")
            print(f"   Size: ${idea.size:,.2f}")
            print(f"   Reasoning: {idea.reasoning}")
    else:
        print("\n💡 TRADE IDEAS")
        print("-" * 80)
        print("   No trade ideas generated (consensus neutral or low confidence)")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results stored in: {config.tracking.ledger_path}")
    print("="*80 + "\n")
    
    # Show what happens next
    print("📈 NEXT STEPS:")
    print("   • Run 'python main.py live-loop --symbol SPY' for continuous trading")
    print("   • Run 'python main.py scan-opportunities' to find the best symbols")
    print("   • Check your Alpaca paper account for positions")
    print("   • View ledger.jsonl for historical results")
    print("\n🎉 System is LIVE and ready to trade!\n")


if __name__ == "__main__":
    try:
        demo_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
