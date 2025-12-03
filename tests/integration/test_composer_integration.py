"""Integration tests for Composer Agent."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from agents.composer.composer_agent_v1 import ComposerAgentV1
from schemas.core_schemas import (
    HedgeSnapshot,
    LiquiditySnapshot,
    SentimentSnapshot,
    AgentSignal,
)


class TestComposerIntegration:
    """Integration tests for ComposerAgentV1 consensus mechanism."""

    @pytest.fixture
    def composer(self):
        """Create composer with standard weights."""
        weights = {
            "hedge": 0.4,
            "liquidity": 0.2,
            "sentiment": 0.4,
        }
        config = {"enabled": True, "min_consensus_score": 0.5}
        return ComposerAgentV1(weights, config)

    @pytest.fixture
    def timestamp(self):
        """Common timestamp for tests."""
        return datetime.now(timezone.utc)

    def test_strong_consensus_bullish(self, composer, timestamp):
        """Test strong bullish consensus from all agents."""
        # All agents agree on bullish signal
        hedge_signal = AgentSignal(
            timestamp=timestamp,
            symbol="SPY",
            signal="bullish",
            confidence=0.9,
            reasoning="High call pressure, positive energy asymmetry",
        )

        liquidity_signal = AgentSignal(
            timestamp=timestamp,
            symbol="SPY",
            signal="bullish",
            confidence=0.8,
            reasoning="High liquidity, tight spreads",
        )

        sentiment_signal = AgentSignal(
            timestamp=timestamp,
            symbol="SPY",
            signal="bullish",
            confidence=0.85,
            reasoning="Positive news and flow sentiment",
        )

        signals = {
            "hedge": hedge_signal,
            "liquidity": liquidity_signal,
            "sentiment": sentiment_signal,
        }

        decision = composer.compose(signals)

        # Verify strong bullish consensus
        assert decision.signal == "bullish"
        assert decision.confidence > 0.8
        assert decision.strength > 0.7

        # Verify weighted consensus score
        # 0.4 * 0.9 + 0.2 * 0.8 + 0.4 * 0.85 = 0.36 + 0.16 + 0.34 = 0.86
        expected_consensus = 0.86
        assert abs(decision.consensus_score - expected_consensus) < 0.01

    def test_conflicting_signals(self, composer, timestamp):
        """Test resolution when agents disagree."""
        # Hedge is bullish, sentiment is bearish, liquidity is neutral
        signals = {
            "hedge": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bullish",
                confidence=0.8,
                reasoning="Positive dealer positioning",
            ),
            "liquidity": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="neutral",
                confidence=0.6,
                reasoning="Moderate liquidity",
            ),
            "sentiment": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bearish",
                confidence=0.7,
                reasoning="Negative news sentiment",
            ),
        }

        decision = composer.compose(signals)

        # With conflicting signals, decision should be neutral or low confidence
        assert decision.confidence < 0.7
        # Consensus score should reflect disagreement
        assert decision.consensus_score < 0.6

    def test_weighted_influence(self, composer, timestamp):
        """Test that hedge agent has higher influence (40% vs 20% liquidity)."""
        # Strong hedge signal, weak liquidity signal
        signals = {
            "hedge": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bullish",
                confidence=0.9,
                reasoning="Strong hedge signal",
            ),
            "liquidity": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bearish",
                confidence=0.8,
                reasoning="Poor liquidity",
            ),
            "sentiment": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="neutral",
                confidence=0.5,
                reasoning="Mixed sentiment",
            ),
        }

        decision = composer.compose(signals)

        # Hedge (40%) should dominate over liquidity (20%)
        # Expected: 0.4 * 0.9 + 0.2 * (-0.8) + 0.4 * 0 = 0.36 - 0.16 + 0 = 0.20
        # Should lean bullish due to hedge weight
        assert decision.signal in ["bullish", "neutral"]

    def test_all_neutral_signals(self, composer, timestamp):
        """Test when all agents are neutral."""
        signals = {
            "hedge": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="neutral",
                confidence=0.5,
                reasoning="No clear hedge signal",
            ),
            "liquidity": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="neutral",
                confidence=0.5,
                reasoning="Average liquidity",
            ),
            "sentiment": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="neutral",
                confidence=0.5,
                reasoning="Mixed sentiment",
            ),
        }

        decision = composer.compose(signals)

        # Should result in neutral decision
        assert decision.signal == "neutral"
        assert decision.confidence <= 0.6

    def test_missing_agent_signal(self, composer, timestamp):
        """Test handling when an agent signal is missing."""
        # Only hedge and sentiment, no liquidity
        signals = {
            "hedge": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bullish",
                confidence=0.8,
                reasoning="Strong hedge signal",
            ),
            "sentiment": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bullish",
                confidence=0.75,
                reasoning="Positive sentiment",
            ),
        }

        decision = composer.compose(signals)

        # Should still produce valid decision
        assert isinstance(decision, AgentSignal)
        assert decision.signal in ["bullish", "neutral", "bearish"]
        assert 0.0 <= decision.confidence <= 1.0

    def test_very_low_confidence_signals(self, composer, timestamp):
        """Test behavior with very low confidence signals."""
        signals = {
            "hedge": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bullish",
                confidence=0.2,
                reasoning="Weak signal",
            ),
            "liquidity": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bullish",
                confidence=0.15,
                reasoning="Low liquidity",
            ),
            "sentiment": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bullish",
                confidence=0.25,
                reasoning="Unclear sentiment",
            ),
        }

        decision = composer.compose(signals)

        # Even if all agree, low confidence should propagate
        assert decision.confidence < 0.4

    def test_bearish_consensus(self, composer, timestamp):
        """Test strong bearish consensus."""
        signals = {
            "hedge": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bearish",
                confidence=0.85,
                reasoning="Heavy put pressure",
            ),
            "liquidity": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bearish",
                confidence=0.75,
                reasoning="Widening spreads",
            ),
            "sentiment": AgentSignal(
                timestamp=timestamp,
                symbol="SPY",
                signal="bearish",
                confidence=0.9,
                reasoning="Negative news flow",
            ),
        }

        decision = composer.compose(signals)

        # Should produce strong bearish signal
        assert decision.signal == "bearish"
        assert decision.confidence > 0.8
