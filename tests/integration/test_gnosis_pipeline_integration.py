"""
GNOSIS Pipeline Integration Tests

Tests end-to-end data flow through the entire GNOSIS architecture:
1. Engine Layer → Primary Agent Layer
2. Primary Agent Layer → Composer Agent Layer
3. Composer Agent Layer → Trade Agent Layer
4. Trade Agent Layer → Monitoring Agent Layer

Run with: pytest tests/integration/test_gnosis_pipeline_integration.py -v
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "symbol": "AAPL",
        "price": 230.0,
        "volume": 50_000_000,
        "high": 232.0,
        "low": 228.0,
        "open": 229.0,
        "change_pct": 0.5,
    }


@pytest.fixture
def sample_hedge_engine_output():
    """Sample output from HedgeEngineV3."""
    return {
        "direction": "bullish",
        "confidence": 0.75,
        "energy_asymmetry": 0.4,
        "dealer_gamma": 0.3,
        "regime": "trending",
        "gamma_pressure": 0.5,
        "vanna_pressure": 0.2,
    }


@pytest.fixture
def sample_sentiment_engine_output():
    """Sample output from SentimentEngine."""
    return {
        "direction": "bullish",
        "confidence": 0.65,
        "sentiment_score": 0.5,
        "news_sentiment": 0.6,
        "social_sentiment": 0.4,
    }


@pytest.fixture
def sample_liquidity_engine_output():
    """Sample output from LiquidityEngineV5."""
    return {
        "direction": "bullish",
        "confidence": 0.60,
        "liquidity_score": 0.7,
        "spread_quality": "good",
        "market_depth": "adequate",
        # PENTA components
        "wyckoff": {"phase": "markup", "confidence": 0.6},
        "ict": {"bias": "bullish", "confidence": 0.7},
        "order_flow": {"imbalance": 0.3, "confidence": 0.65},
        "supply_demand": {"zone": "demand", "confidence": 0.6},
        "liquidity_concepts": {"trend": "bullish", "confidence": 0.7},
    }


# ============================================================================
# LAYER 1: ENGINE → PRIMARY AGENT TESTS
# ============================================================================

class TestEngineToAgentIntegration:
    """Test data flow from Engines to Primary Agents."""
    
    def test_hedge_agent_initialization(self):
        """Test HedgeAgentV3 can be initialized."""
        from agents.hedge_agent_v3 import HedgeAgentV3
        
        # Create agent
        agent = HedgeAgentV3(config={})
        
        # Verify agent has suggest method
        assert hasattr(agent, 'suggest')
    
    def test_sentiment_agent_initialization(self):
        """Test SentimentAgentV1 can be initialized."""
        from agents.sentiment_agent_v1 import SentimentAgentV1
        
        agent = SentimentAgentV1(config={})
        assert hasattr(agent, 'suggest')
    
    def test_liquidity_agent_v5_initialization(self):
        """Test LiquidityAgentV5 can be initialized."""
        from agents.liquidity_agent_v5 import LiquidityAgentV5
        
        agent = LiquidityAgentV5(config={})
        
        # V5 should have the suggest method and confluence analysis
        assert hasattr(agent, 'suggest')
        assert hasattr(agent, 'get_confluence_analysis')


# ============================================================================
# LAYER 2: PRIMARY AGENT → COMPOSER TESTS
# ============================================================================

class TestAgentToComposerIntegration:
    """Test data flow from Primary Agents to Composer."""
    
    def test_all_agents_feed_composer(self):
        """Test all three primary agents feed into ComposerAgentV4."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        # Create composer
        composer = ComposerAgentV4()
        
        # Create agent signals
        hedge_signal = {"direction": "bullish", "confidence": 0.75}
        sentiment_signal = {"direction": "bullish", "confidence": 0.65}
        liquidity_signal = {"direction": "bullish", "confidence": 0.60}
        
        # Compose signals
        output = composer.compose(
            hedge_signal=hedge_signal,
            sentiment_signal=sentiment_signal,
            liquidity_signal=liquidity_signal,
        )
        
        # Verify output
        assert output.direction == "LONG"
        assert output.confidence > 0
        assert output.consensus_score >= 0
    
    def test_weighted_composition(self):
        """Test weighted composition with custom weights."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        # Custom weights: hedge heavy
        custom_weights = {"hedge": 0.6, "sentiment": 0.2, "liquidity": 0.2}
        composer = ComposerAgentV4(weights=custom_weights)
        
        # Strong hedge bullish, weak others
        output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.9},
            sentiment_signal={"direction": "neutral", "confidence": 0.5},
            liquidity_signal={"direction": "neutral", "confidence": 0.5},
        )
        
        # Should still be LONG due to heavy hedge weight
        assert output.direction == "LONG"
    
    def test_penta_confluence_bonus_applied(self):
        """Test PENTA confluence bonus is applied correctly."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        composer = ComposerAgentV4()
        
        # Base signals
        hedge_signal = {"direction": "bullish", "confidence": 0.7}
        sentiment_signal = {"direction": "bullish", "confidence": 0.65}
        liquidity_signal = {"direction": "bullish", "confidence": 0.6}
        
        # Without PENTA
        output_no_penta = composer.compose(
            hedge_signal=hedge_signal,
            sentiment_signal=sentiment_signal,
            liquidity_signal=liquidity_signal,
        )
        
        # With PENTA confluence
        output_penta = composer.compose(
            hedge_signal=hedge_signal,
            sentiment_signal=sentiment_signal,
            liquidity_signal=liquidity_signal,
            penta_confluence="PENTA",
        )
        
        # PENTA should have higher confidence
        assert output_penta.penta_confidence_bonus == 0.30
        assert output_penta.penta_confluence == "PENTA"


# ============================================================================
# LAYER 3: COMPOSER → TRADE AGENT TESTS
# ============================================================================

class TestComposerToTradeAgentIntegration:
    """Test data flow from Composer to Trade Agents."""
    
    def test_composer_to_alpha_trade_agent(self):
        """Test ComposerAgentV4 → AlphaTradeAgentV2 data flow."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4, ComposerOutput
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        
        # Create agents with required config
        composer = ComposerAgentV4()
        trade_agent = AlphaTradeAgentV2(config={})
        
        # Generate composer output
        composer_output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.75},
            sentiment_signal={"direction": "bullish", "confidence": 0.7},
            liquidity_signal={"direction": "bullish", "confidence": 0.65},
        )
        
        # Process through trade agent
        signal = trade_agent.process_composer_output(
            composer_output,
            symbol="AAPL",
            current_price=230.0
        )
        
        # Verify signal
        assert signal.symbol == "AAPL"
        assert signal.direction in ["BUY", "SELL", "HOLD"]
        if signal.direction == "BUY":
            assert signal.entry_price == 230.0
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price
    
    def test_composer_to_full_gnosis_agent(self):
        """Test ComposerAgentV4 → FullGnosisTradeAgentV2 data flow."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2
        
        composer = ComposerAgentV4()
        trade_agent = FullGnosisTradeAgentV2(
            config={"min_confidence": 0.5, "portfolio_value": 100000}
        )
        
        composer_output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.8},
            sentiment_signal={"direction": "bullish", "confidence": 0.75},
            liquidity_signal={"direction": "bullish", "confidence": 0.7},
        )
        
        action = trade_agent.process_composer_output(
            composer_output,
            symbol="NVDA",
            current_price=145.0
        )
        
        assert action is not None
        assert action.symbol == "NVDA"
    
    def test_hold_signal_generated_for_low_confidence(self):
        """Test HOLD signal generated when confidence is too low."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        
        composer = ComposerAgentV4()
        trade_agent = AlphaTradeAgentV2(config={"min_confidence": 0.7, "strong_confidence_threshold": 0.8})
        
        # Low confidence signals
        composer_output = composer.compose(
            hedge_signal={"direction": "neutral", "confidence": 0.3},
            sentiment_signal={"direction": "neutral", "confidence": 0.3},
            liquidity_signal={"direction": "neutral", "confidence": 0.3},
        )
        
        signal = trade_agent.process_composer_output(
            composer_output,
            symbol="SPY",
            current_price=600.0
        )
        
        assert signal.direction == "HOLD"


# ============================================================================
# LAYER 4: TRADE AGENT → MONITOR TESTS
# ============================================================================

class TestTradeAgentToMonitorIntegration:
    """Test data flow from Trade Agents to Monitors."""
    
    def test_alpha_agent_to_alpha_monitor(self):
        """Test AlphaTradeAgentV2 → AlphaMonitor data flow."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        from agents.monitoring import AlphaMonitor
        
        # Create components
        composer = ComposerAgentV4()
        trade_agent = AlphaTradeAgentV2(config={})
        monitor = AlphaMonitor({})
        
        # Generate signal
        composer_output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.8},
            sentiment_signal={"direction": "bullish", "confidence": 0.75},
            liquidity_signal={"direction": "bullish", "confidence": 0.7},
        )
        
        signal = trade_agent.process_composer_output(
            composer_output,
            symbol="GOOGL",
            current_price=175.0
        )
        
        # Track signal in monitor
        monitor.update(signal={
            "symbol": signal.symbol,
            "signal_type": signal.direction,
            "confidence": signal.confidence,
        })
        
        # Verify tracking
        assert monitor.metrics.signals_generated == 1
    
    def test_gnosis_agent_to_gnosis_monitor(self):
        """Test FullGnosisTradeAgentV2 → GnosisMonitor data flow."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2
        from agents.monitoring import GnosisMonitor
        
        composer = ComposerAgentV4()
        trade_agent = FullGnosisTradeAgentV2(
            config={"min_confidence": 0.5, "portfolio_value": 100000}
        )
        monitor = GnosisMonitor({"initial_equity": 100000})
        
        # Generate action
        composer_output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.8},
            sentiment_signal={"direction": "bullish", "confidence": 0.75},
            liquidity_signal={"direction": "bullish", "confidence": 0.7},
        )
        
        action = trade_agent.process_composer_output(
            composer_output,
            symbol="AMZN",
            current_price=220.0
        )
        
        # Update monitor with position
        positions = {
            "AMZN": {
                "direction": action.direction,
                "quantity": action.quantity,
                "entry_price": action.price,
            }
        }
        
        monitor.update(positions=positions, current_prices={"AMZN": 225.0})
        
        # Monitor should track unrealized P&L
        assert monitor.metrics.unrealized_pnl != 0


# ============================================================================
# END-TO-END PIPELINE TESTS
# ============================================================================

class TestFullPipelineIntegration:
    """Test complete end-to-end pipeline."""
    
    def test_complete_bullish_pipeline(self):
        """Test complete pipeline with bullish signals."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        from agents.monitoring import AlphaMonitor
        
        # Setup
        composer = ComposerAgentV4()
        trade_agent = AlphaTradeAgentV2(config={})
        monitor = AlphaMonitor({})
        
        # Layer 1 → Layer 2: Agents to Composer
        composer_output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.85},
            sentiment_signal={"direction": "bullish", "confidence": 0.80},
            liquidity_signal={"direction": "bullish", "confidence": 0.75},
            penta_confluence="QUAD",
        )
        
        # Layer 2 → Layer 3: Composer to Trade Agent
        signal = trade_agent.process_composer_output(
            composer_output,
            symbol="MSFT",
            current_price=430.0
        )
        
        # Layer 3 → Layer 4: Trade Agent to Monitor
        monitor.update(signal={
            "symbol": signal.symbol,
            "signal_type": signal.direction,
            "confidence": signal.confidence,
        })
        
        # Verify complete pipeline
        assert composer_output.direction == "LONG"
        assert signal.direction == "BUY"
        assert monitor.metrics.signals_generated == 1
        
        # Verify QUAD confluence bonus
        assert composer_output.penta_confluence == "QUAD"
    
    def test_complete_bearish_pipeline(self):
        """Test complete pipeline with bearish signals."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        from agents.monitoring import AlphaMonitor
        
        composer = ComposerAgentV4()
        trade_agent = AlphaTradeAgentV2(config={})
        monitor = AlphaMonitor({})
        
        # Bearish signals
        composer_output = composer.compose(
            hedge_signal={"direction": "bearish", "confidence": 0.80},
            sentiment_signal={"direction": "bearish", "confidence": 0.75},
            liquidity_signal={"direction": "bearish", "confidence": 0.70},
        )
        
        signal = trade_agent.process_composer_output(
            composer_output,
            symbol="TSLA",
            current_price=350.0
        )
        
        monitor.update(signal={
            "symbol": signal.symbol,
            "signal_type": signal.direction,
            "confidence": signal.confidence,
        })
        
        assert composer_output.direction == "SHORT"
        assert signal.direction == "SELL"
        # For SELL, stop should be above price, target below
        if signal.stop_loss and signal.take_profit:
            assert signal.stop_loss > signal.entry_price
            assert signal.take_profit < signal.entry_price
    
    def test_penta_methodology_pipeline(self):
        """Test pipeline with full PENTA methodology."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        
        composer = ComposerAgentV4()
        trade_agent = AlphaTradeAgentV2(config={})
        
        # Full PENTA confluence
        composer_output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.9},
            sentiment_signal={"direction": "bullish", "confidence": 0.85},
            liquidity_signal={"direction": "bullish", "confidence": 0.80},
            penta_confluence="PENTA",  # All 5 methodologies align
        )
        
        signal = trade_agent.process_composer_output(
            composer_output,
            symbol="SPY",
            current_price=600.0
        )
        
        # PENTA should give maximum confidence bonus
        assert composer_output.penta_confidence_bonus == 0.30
        assert signal.penta_confluence == "PENTA"
        
        # PENTA target should be extended (higher than base target)
        base_target_pct = 0.05  # 5% base
        base_target = 600.0 * (1 + base_target_pct)  # 630
        
        # PENTA signal should have higher target than base
        assert signal.take_profit > base_target
    
    def test_express_mode_0dte(self):
        """Test 0DTE express mode pipeline."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4, ComposerMode
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        
        composer = ComposerAgentV4()
        trade_agent = AlphaTradeAgentV2(config={})
        
        # Express 0DTE mode
        composer_output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.8},
            sentiment_signal={"direction": "bullish", "confidence": 0.75},
            liquidity_signal={"direction": "bullish", "confidence": 0.70},
            mode=ComposerMode.EXPRESS_0DTE,
        )
        
        signal = trade_agent.process_composer_output(
            composer_output,
            symbol="SPY",
            current_price=600.0
        )
        
        # 0DTE mode should be set
        assert composer_output.mode == "0dte"


# ============================================================================
# ALPHA SIGNAL GENERATOR INTEGRATION
# ============================================================================

class TestAlphaSignalGeneratorIntegration:
    """Test AlphaSignalGenerator integration with architecture."""
    
    def test_alpha_generator_import(self):
        """Test AlphaSignalGenerator can be imported."""
        from alpha.signal_generator import AlphaSignalGenerator
        assert AlphaSignalGenerator is not None
    
    def test_alpha_generator_uses_composer(self):
        """Test AlphaSignalGenerator uses Composer for consensus."""
        from alpha.signal_generator import AlphaSignalGenerator
        
        generator = AlphaSignalGenerator()
        
        # Should have composer (named 'composer' not 'composer_agent')
        assert hasattr(generator, 'composer')
        # Should have engines and agents
        assert hasattr(generator, 'engines')
        assert hasattr(generator, 'agents')


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
