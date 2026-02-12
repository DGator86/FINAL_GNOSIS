"""Test script for Phase 3: Multi-Timeframe Primary Agents.

Tests the new multi-timeframe agents to ensure they properly:
1. Accept multi-timeframe engine outputs
2. Generate TimeframeSignals
3. Detect confluence/divergence
4. Integrate with ConfidenceBuilder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime


from agents.confidence_builder import ConfidenceBuilder
from agents.hedge_agent_v4 import HedgeAgentV4
from agents.liquidity_agent_v2 import LiquidityAgentV2
from agents.sentiment_agent_v2 import SentimentAgentV2


# Mock HedgeSnapshot
class MockHedgeSnapshot:
    def __init__(self, confidence, energy_asymmetry, movement_energy):
        self.confidence = confidence
        self.energy_asymmetry = energy_asymmetry
        self.movement_energy = movement_energy


# Mock LiquiditySnapshot
class MockLiquiditySnapshot:
    def __init__(self, confidence, bid_ask_imbalance):
        self.confidence = confidence
        self.bid_ask_imbalance = bid_ask_imbalance
        self.total_depth = 500


# Mock SentimentSnapshot
class MockSentimentSnapshot:
    def __init__(self, confidence, sentiment_score):
        self.confidence = confidence
        self.sentiment_score = sentiment_score


def test_phase3_agents():
    """Test all Phase 3 multi-timeframe agents."""
    
    print("=" * 80)
    print("PHASE 3 MULTI-TIMEFRAME AGENTS TEST")
    print("=" * 80)
    
    # Initialize agents
    hedge_agent = HedgeAgentV4({"min_confidence": 0.5})
    liquidity_agent = LiquidityAgentV2({"min_confidence": 0.5})
    sentiment_agent = SentimentAgentV2({"min_confidence": 0.5})
    confidence_builder = ConfidenceBuilder()
    
    # Create mock multi-timeframe data (bullish across all timeframes)
    hedge_snapshots = {
        '1Min': MockHedgeSnapshot(0.7, 0.4, 50),    # Bullish
        '5Min': MockHedgeSnapshot(0.75, 0.45, 60),  # Bullish
        '15Min': MockHedgeSnapshot(0.8, 0.5, 70),   # Bullish
        '1Hour': MockHedgeSnapshot(0.85, 0.55, 80), # Bullish
        '1Day': MockHedgeSnapshot(0.9, 0.6, 90),    # Bullish
    }
    
    liquidity_snapshots = {
        '1Min': MockLiquiditySnapshot(0.7, 0.2),   # Bullish (bid pressure)
        '5Min': MockLiquiditySnapshot(0.75, 0.25),
        '15Min': MockLiquiditySnapshot(0.8, 0.3),
        '1Hour': MockLiquiditySnapshot(0.85, 0.35),
        '1Day': MockLiquiditySnapshot(0.9, 0.4),
    }
    
    sentiment_snapshots = {
        '1Min': MockSentimentSnapshot(0.7, 0.3),   # Bullish sentiment
        '5Min': MockSentimentSnapshot(0.75, 0.35),
        '15Min': MockSentimentSnapshot(0.8, 0.4),
        '1Hour': MockSentimentSnapshot(0.85, 0.45),
        '1Day': MockSentimentSnapshot(0.9, 0.5),
    }
    
    # Test 1: HedgeAgent
    print("\n1. Testing HedgeAgentV4...")
    print("-" * 80)
    hedge_signals = hedge_agent.suggest_multiframe(
        hedge_snapshots,
        symbol="SPY",
        timestamp=datetime.now()
    )
    print(f"   Generated {len(hedge_signals)} timeframe signals")
    for signal in hedge_signals:
        print(f"   {signal.timeframe}: direction={signal.direction:+.2f}, "
              f"strength={signal.strength:.2f}, confidence={signal.confidence:.2f}")
    
    # Test confluence detection
    confluence = hedge_agent.detect_confluence(hedge_snapshots)
    print(f"\n   Confluence Detection:")
    print(f"   - Has confluence: {confluence['has_confluence']}")
    print(f"   - Direction: {confluence['direction']}")
    print(f"   - Agreement: {confluence['agreement_ratio']:.1%}")
    
    # Test 2: LiquidityAgent
    print("\n2. Testing LiquidityAgentV2...")
    print("-" * 80)
    liquidity_signals = liquidity_agent.suggest_multiframe(
        liquidity_snapshots,
        symbol="SPY",
        timestamp=datetime.now()
    )
    print(f"   Generated {len(liquidity_signals)} timeframe signals")
    for signal in liquidity_signals:
        print(f"   {signal.timeframe}: direction={signal.direction:+.2f}, "
              f"strength={signal.strength:.2f}, confidence={signal.confidence:.2f}")
    
    # Test support/resistance
    sr_analysis = liquidity_agent.detect_support_resistance(liquidity_snapshots, 450.0)
    print(f"\n   Support/Resistance Analysis:")
    print(f"   - Strong support: {sr_analysis['has_strong_support']}")
    print(f"   - Strong resistance: {sr_analysis['has_strong_resistance']}")
    
    # Test 3: SentimentAgent
    print("\n3. Testing SentimentAgentV2...")
    print("-" * 80)
    sentiment_signals = sentiment_agent.suggest_multiframe(
        sentiment_snapshots,
        symbol="SPY",
        timestamp=datetime.now()
    )
    print(f"   Generated {len(sentiment_signals)} timeframe signals")
    for signal in sentiment_signals:
        print(f"   {signal.timeframe}: direction={signal.direction:+.2f}, "
              f"strength={signal.strength:.2f}, confidence={signal.confidence:.2f}")
    
    # Test divergence detection
    divergence = sentiment_agent.detect_divergences(sentiment_snapshots)
    print(f"\n   Divergence Analysis:")
    print(f"   - Has divergence: {divergence['has_divergence']}")
    print(f"   - Type: {divergence.get('type', 'None')}")
    
    # Test 4: ConfidenceBuilder Integration
    print("\n4. Testing ConfidenceBuilder Integration...")
    print("-" * 80)
    
    # Combine all signals
    all_signals = hedge_signals + liquidity_signals + sentiment_signals
    print(f"   Total signals from all agents: {len(all_signals)}")
    
    # Calculate confidence
    confidence_score = confidence_builder.calculate_confidence(all_signals)
    print(f"\n   Confidence Score:")
    print(f"   - Overall confidence: {confidence_score.overall_confidence:.2%}")
    print(f"   - Weighted direction: {confidence_score.weighted_direction:+.2f}")
    print(f"   - Alignment score: {confidence_score.alignment_score:.1%}")
    print(f"   - Dominant timeframe: {confidence_score.dominant_timeframe}")
    print(f"   - Reasoning: {confidence_score.reasoning}")
    
    # Check if meets threshold
    meets_threshold = confidence_builder.meets_threshold(confidence_score)
    print(f"\n   Meets threshold for action: {meets_threshold}")
    
    # Get recommended timeframe
    rec_tf = confidence_builder.get_recommended_timeframe(confidence_score)
    print(f"   Recommended trade timeframe: {rec_tf}")
    
    print("\n" + "=" * 80)
    print("PHASE 3 TEST COMPLETE ✅")
    print("=" * 80)
    print("\nNext Steps:")
    print("  - Integrate agents with PipelineRunner")
    print("  - Run live data test with TimeframeManager")
    print("  - Proceed to Phase 4 (Composer enhancement)")
    print("=" * 80)


if __name__ == "__main__":
    test_phase3_agents()
