"""
Comprehensive Unit Tests for GNOSIS Architecture V2 Components.

Tests all layers of the architecture:
1. Engine Layer (LiquidityEngineV5, PENTA sub-engines)
2. Primary Agent Layer (LiquidityAgentV5)
3. Composer Agent Layer (ComposerAgentV4)
4. Trade Agent Layer (FullGnosisTradeAgentV2, AlphaTradeAgentV2)
5. Monitoring Agent Layer (GnosisMonitor, AlphaMonitor)

Author: GNOSIS Trading System
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_hedge_signal():
    """Mock hedge agent signal."""
    return {
        "direction": "bullish",
        "confidence": 0.75,
        "energy_asymmetry": 0.4,
        "dealer_gamma": 0.3,
    }


@pytest.fixture
def mock_sentiment_signal():
    """Mock sentiment agent signal."""
    return {
        "direction": "bullish",
        "confidence": 0.65,
        "sentiment_score": 0.5,
    }


@pytest.fixture
def mock_liquidity_signal():
    """Mock liquidity agent signal."""
    return {
        "direction": "neutral",
        "confidence": 0.5,
        "liquidity_score": 0.7,
    }


@pytest.fixture
def composer_config():
    """Composer configuration."""
    return {
        "min_confidence": 0.5,
        "bullish_threshold": 0.3,
        "bearish_threshold": -0.3,
    }


@pytest.fixture
def trade_agent_config():
    """Trade agent configuration."""
    return {
        "min_confidence": 0.5,
        "strong_confidence_threshold": 0.8,
        "moderate_confidence_threshold": 0.65,
        "default_stop_pct": 0.03,
        "default_target_pct": 0.05,
        "default_holding_days": 3,
        "signal_validity_hours": 24,
    }


# ============================================================================
# COMPOSER AGENT V4 TESTS
# ============================================================================

class TestComposerAgentV4:
    """Test suite for ComposerAgentV4."""
    
    def test_import(self):
        """Test that ComposerAgentV4 can be imported."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        assert ComposerAgentV4 is not None
    
    def test_initialization_default_weights(self):
        """Test initialization with default weights."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        composer = ComposerAgentV4()
        
        assert composer.agent_weights["hedge"] == pytest.approx(0.4, rel=0.01)
        assert composer.agent_weights["sentiment"] == pytest.approx(0.4, rel=0.01)
        assert composer.agent_weights["liquidity"] == pytest.approx(0.2, rel=0.01)
    
    def test_initialization_custom_weights(self):
        """Test initialization with custom weights."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        custom_weights = {"hedge": 0.5, "sentiment": 0.3, "liquidity": 0.2}
        composer = ComposerAgentV4(weights=custom_weights)
        
        assert composer.agent_weights["hedge"] == pytest.approx(0.5, rel=0.01)
        assert composer.agent_weights["sentiment"] == pytest.approx(0.3, rel=0.01)
        assert composer.agent_weights["liquidity"] == pytest.approx(0.2, rel=0.01)
    
    def test_compose_bullish_consensus(self, mock_hedge_signal, mock_sentiment_signal, mock_liquidity_signal):
        """Test composition with bullish consensus."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        composer = ComposerAgentV4()
        output = composer.compose(
            hedge_signal=mock_hedge_signal,
            sentiment_signal=mock_sentiment_signal,
            liquidity_signal=mock_liquidity_signal,
        )
        
        assert output.direction == "LONG"
        assert 0 < output.confidence <= 1.0
        assert output.consensus_score >= 0
    
    def test_compose_bearish_consensus(self):
        """Test composition with bearish consensus."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        composer = ComposerAgentV4()
        output = composer.compose(
            hedge_signal={"direction": "bearish", "confidence": 0.8},
            sentiment_signal={"direction": "bearish", "confidence": 0.7},
            liquidity_signal={"direction": "bearish", "confidence": 0.6},
        )
        
        assert output.direction == "SHORT"
        assert output.confidence > 0
    
    def test_compose_neutral_when_mixed(self):
        """Test composition returns neutral when signals are mixed."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        composer = ComposerAgentV4()
        output = composer.compose(
            hedge_signal={"direction": "bullish", "confidence": 0.5},
            sentiment_signal={"direction": "bearish", "confidence": 0.5},
            liquidity_signal={"direction": "neutral", "confidence": 0.5},
        )
        
        assert output.direction == "NEUTRAL"
    
    def test_compose_with_penta_confluence(self, mock_hedge_signal, mock_sentiment_signal, mock_liquidity_signal):
        """Test composition with PENTA confluence bonus."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        composer = ComposerAgentV4()
        
        # Without PENTA
        output_no_penta = composer.compose(
            hedge_signal=mock_hedge_signal,
            sentiment_signal=mock_sentiment_signal,
            liquidity_signal=mock_liquidity_signal,
        )
        
        # With PENTA confluence
        output_with_penta = composer.compose(
            hedge_signal=mock_hedge_signal,
            sentiment_signal=mock_sentiment_signal,
            liquidity_signal=mock_liquidity_signal,
            penta_confluence="PENTA",
        )
        
        assert output_with_penta.penta_confluence == "PENTA"
        assert output_with_penta.penta_confidence_bonus == 0.30
    
    def test_compose_express_mode_0dte(self, mock_hedge_signal, mock_sentiment_signal, mock_liquidity_signal):
        """Test composition with 0DTE express mode."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4, ComposerMode
        
        composer = ComposerAgentV4()
        output = composer.compose(
            hedge_signal=mock_hedge_signal,
            sentiment_signal=mock_sentiment_signal,
            liquidity_signal=mock_liquidity_signal,
            mode=ComposerMode.EXPRESS_0DTE,
        )
        
        assert output.mode == "0dte"
    
    def test_compose_no_signals(self):
        """Test composition with no signals returns neutral."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        composer = ComposerAgentV4()
        output = composer.compose()
        
        assert output.direction == "NEUTRAL"
        assert output.confidence == 0.0
    
    def test_output_to_dict(self, mock_hedge_signal, mock_sentiment_signal, mock_liquidity_signal):
        """Test ComposerOutput to_dict method."""
        from agents.composer.composer_agent_v4 import ComposerAgentV4
        
        composer = ComposerAgentV4()
        output = composer.compose(
            hedge_signal=mock_hedge_signal,
            sentiment_signal=mock_sentiment_signal,
            liquidity_signal=mock_liquidity_signal,
        )
        
        output_dict = output.to_dict()
        
        assert "direction" in output_dict
        assert "confidence" in output_dict
        assert "consensus_score" in output_dict
        assert "reasoning" in output_dict


# ============================================================================
# ALPHA TRADE AGENT V2 TESTS
# ============================================================================

class TestAlphaTradeAgentV2:
    """Test suite for AlphaTradeAgentV2."""
    
    def test_import(self):
        """Test that AlphaTradeAgentV2 can be imported."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        assert AlphaTradeAgentV2 is not None
    
    def test_initialization(self, trade_agent_config):
        """Test initialization."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        assert agent.min_confidence == 0.5
        assert agent.strong_confidence_threshold == 0.8
        assert agent.default_stop_pct == 0.03
    
    def test_process_bullish_signal(self, trade_agent_config):
        """Test processing a bullish composer output."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2, SignalType
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        composer_output = ComposerOutput(
            direction="LONG",
            confidence=0.75,
            consensus_score=0.8,
            reasoning="Test bullish signal",
        )
        
        signal = agent.process_composer_output(
            composer_output, 
            symbol="AAPL", 
            current_price=230.0
        )
        
        assert signal.symbol == "AAPL"
        assert signal.direction == "BUY"
        assert signal.signal_type == SignalType.BUY
        assert signal.entry_price == 230.0
        assert signal.stop_loss is not None
        assert signal.take_profit is not None
    
    def test_process_strong_bullish_signal(self, trade_agent_config):
        """Test processing a strong bullish signal."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2, SignalType, SignalStrength
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        composer_output = ComposerOutput(
            direction="LONG",
            confidence=0.85,
            consensus_score=0.9,
            reasoning="Strong bullish signal",
        )
        
        signal = agent.process_composer_output(
            composer_output,
            symbol="NVDA",
            current_price=145.0
        )
        
        assert signal.signal_type == SignalType.STRONG_BUY
        assert signal.strength == SignalStrength.STRONG
    
    def test_process_bearish_signal(self, trade_agent_config):
        """Test processing a bearish composer output."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2, SignalType
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        composer_output = ComposerOutput(
            direction="SHORT",
            confidence=0.70,
            consensus_score=0.75,
            reasoning="Test bearish signal",
        )
        
        signal = agent.process_composer_output(
            composer_output,
            symbol="TSLA",
            current_price=350.0
        )
        
        assert signal.direction == "SELL"
        assert signal.signal_type == SignalType.SELL
        # For SELL, stop should be above price
        assert signal.stop_loss > signal.entry_price
        assert signal.take_profit < signal.entry_price
    
    def test_process_hold_signal(self, trade_agent_config):
        """Test processing a neutral/hold signal."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2, SignalType
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        composer_output = ComposerOutput(
            direction="NEUTRAL",
            confidence=0.4,
            consensus_score=0.5,
            reasoning="Neutral signal",
        )
        
        signal = agent.process_composer_output(
            composer_output,
            symbol="SPY",
            current_price=600.0
        )
        
        assert signal.direction == "HOLD"
        assert signal.signal_type == SignalType.HOLD
        assert signal.stop_loss is None
        assert signal.take_profit is None
    
    def test_penta_target_bonus(self, trade_agent_config):
        """Test PENTA confluence affects take profit target."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        # Without PENTA
        output_no_penta = ComposerOutput(
            direction="LONG",
            confidence=0.7,
            consensus_score=0.8,
        )
        signal_no_penta = agent.process_composer_output(
            output_no_penta, "TEST", 100.0
        )
        
        # With PENTA
        output_with_penta = ComposerOutput(
            direction="LONG",
            confidence=0.7,
            consensus_score=0.8,
            penta_confluence="PENTA",
            penta_confidence_bonus=0.30,
        )
        signal_with_penta = agent.process_composer_output(
            output_with_penta, "TEST", 100.0
        )
        
        assert signal_with_penta.penta_confluence == "PENTA"
        # PENTA target should be larger
        assert signal_with_penta.take_profit > signal_no_penta.take_profit
    
    def test_options_play_suggestion(self, trade_agent_config):
        """Test options play suggestion."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        composer_output = ComposerOutput(
            direction="LONG",
            confidence=0.7,
            consensus_score=0.8,
        )
        
        signal = agent.process_composer_output(
            composer_output,
            symbol="AAPL",
            current_price=230.0
        )
        
        assert signal.options_play is not None
        assert signal.options_play["option_type"] == "CALL"
        assert "suggested_strike" in signal.options_play
    
    def test_signal_to_dict(self, trade_agent_config):
        """Test signal serialization."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        composer_output = ComposerOutput(
            direction="LONG",
            confidence=0.7,
            consensus_score=0.8,
        )
        
        signal = agent.process_composer_output(
            composer_output,
            symbol="MSFT",
            current_price=430.0
        )
        
        signal_dict = signal.to_dict()
        
        assert "symbol" in signal_dict
        assert "direction" in signal_dict
        assert "confidence" in signal_dict
        assert "entry_price" in signal_dict
    
    def test_robinhood_format(self, trade_agent_config):
        """Test Robinhood-friendly formatting."""
        from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        agent = AlphaTradeAgentV2(config=trade_agent_config)
        
        composer_output = ComposerOutput(
            direction="LONG",
            confidence=0.75,
            consensus_score=0.8,
            penta_confluence="QUAD",
        )
        
        signal = agent.process_composer_output(
            composer_output,
            symbol="GOOGL",
            current_price=175.0
        )
        
        formatted = signal.to_robinhood_format()
        
        assert "GOOGL" in formatted
        assert "BUY" in formatted
        assert "QUAD" in formatted


# ============================================================================
# FULL GNOSIS TRADE AGENT V2 TESTS
# ============================================================================

class TestFullGnosisTradeAgentV2:
    """Test suite for FullGnosisTradeAgentV2."""
    
    def test_import(self):
        """Test that FullGnosisTradeAgentV2 can be imported."""
        from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2
        assert FullGnosisTradeAgentV2 is not None
    
    def test_initialization(self):
        """Test initialization."""
        from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2
        
        config = {
            "min_confidence": 0.6,
            "max_position_size": 0.1,
            "default_stop_pct": 0.02,
        }
        
        agent = FullGnosisTradeAgentV2(config=config)
        
        assert agent.min_confidence == 0.6
        assert agent.max_position_size == 0.1
    
    def test_process_long_entry(self):
        """Test processing a long entry signal."""
        from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2, TradeActionType
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        config = {
            "min_confidence": 0.6,
            "portfolio_value": 100000,
        }
        
        agent = FullGnosisTradeAgentV2(config=config)
        
        composer_output = ComposerOutput(
            direction="LONG",
            confidence=0.75,
            consensus_score=0.8,
        )
        
        action = agent.process_composer_output(
            composer_output,
            symbol="AAPL",
            current_price=230.0
        )
        
        assert action.action_type == TradeActionType.ENTER_LONG
        assert action.direction == "LONG"
        assert action.quantity > 0
        assert action.stop_loss < action.price
        assert action.take_profit > action.price
    
    def test_process_short_entry(self):
        """Test processing a short entry signal."""
        from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2, TradeActionType
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        config = {"min_confidence": 0.6, "portfolio_value": 100000}
        agent = FullGnosisTradeAgentV2(config=config)
        
        composer_output = ComposerOutput(
            direction="SHORT",
            confidence=0.7,
            consensus_score=0.75,
        )
        
        action = agent.process_composer_output(
            composer_output,
            symbol="TSLA",
            current_price=350.0
        )
        
        assert action.action_type == TradeActionType.ENTER_SHORT
        assert action.direction == "SHORT"
    
    def test_no_action_below_confidence(self):
        """Test no action when confidence is below threshold."""
        from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2, TradeActionType
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        config = {"min_confidence": 0.7}
        agent = FullGnosisTradeAgentV2(config=config)
        
        composer_output = ComposerOutput(
            direction="LONG",
            confidence=0.5,  # Below threshold
            consensus_score=0.6,
        )
        
        action = agent.process_composer_output(
            composer_output,
            symbol="SPY",
            current_price=600.0
        )
        
        assert action.action_type == TradeActionType.NO_ACTION
    
    def test_penta_position_bonus(self):
        """Test PENTA confluence affects position sizing."""
        from trade.gnosis_trade_agent_v2 import FullGnosisTradeAgentV2
        from agents.composer.composer_agent_v4 import ComposerOutput
        
        config = {"min_confidence": 0.6, "portfolio_value": 100000}
        agent = FullGnosisTradeAgentV2(config=config)
        
        # Without PENTA
        output_no_penta = ComposerOutput(
            direction="LONG",
            confidence=0.7,
            consensus_score=0.8,
        )
        action_no_penta = agent.process_composer_output(
            output_no_penta, "TEST", 100.0
        )
        
        # With PENTA
        output_with_penta = ComposerOutput(
            direction="LONG",
            confidence=0.7,
            consensus_score=0.8,
            penta_confluence="PENTA",
        )
        action_with_penta = agent.process_composer_output(
            output_with_penta, "TEST", 100.0
        )
        
        # PENTA should allow larger position
        assert action_with_penta.penta_bonus > 0


# ============================================================================
# MONITORING AGENT TESTS
# ============================================================================

class TestGnosisMonitor:
    """Test suite for GnosisMonitor."""
    
    def test_import(self):
        """Test that GnosisMonitor can be imported."""
        from agents.monitoring import GnosisMonitor
        assert GnosisMonitor is not None
    
    def test_initialization(self):
        """Test initialization."""
        from agents.monitoring import GnosisMonitor
        
        monitor = GnosisMonitor({})
        
        assert monitor.metrics.total_trades == 0
        assert monitor.metrics.win_rate == 0.0
    
    def test_update_with_positions(self):
        """Test updating with positions."""
        from agents.monitoring import GnosisMonitor
        
        monitor = GnosisMonitor({"initial_equity": 100000})
        
        positions = {
            "AAPL": {
                "direction": "LONG",
                "quantity": 10,
                "entry_price": 230.0,
            }
        }
        
        current_prices = {"AAPL": 235.0}
        
        monitor.update(positions, current_prices)
        
        # Should have unrealized P&L
        assert monitor.metrics.unrealized_pnl > 0
    
    def test_get_metrics(self):
        """Test getting metrics."""
        from agents.monitoring import GnosisMonitor
        
        monitor = GnosisMonitor({})
        metrics = monitor.get_metrics()
        
        assert metrics is not None
        assert hasattr(metrics, 'total_trades')
        assert hasattr(metrics, 'win_rate')


class TestAlphaMonitor:
    """Test suite for AlphaMonitor."""
    
    def test_import(self):
        """Test that AlphaMonitor can be imported."""
        from agents.monitoring import AlphaMonitor
        assert AlphaMonitor is not None
    
    def test_initialization(self):
        """Test initialization."""
        from agents.monitoring import AlphaMonitor
        
        monitor = AlphaMonitor({})
        
        assert monitor.metrics.signals_generated == 0
        assert monitor.metrics.signal_accuracy == 0.0
    
    def test_track_signal(self):
        """Test tracking a signal."""
        from agents.monitoring import AlphaMonitor
        
        monitor = AlphaMonitor({})
        
        signal = {
            "symbol": "AAPL",
            "signal_type": "BUY",
            "confidence": 0.75,
        }
        
        monitor.update(signal=signal)
        
        assert monitor.metrics.signals_generated == 1
        assert "AAPL" in monitor.pending_signals
    
    def test_record_outcome(self):
        """Test recording signal outcome."""
        from agents.monitoring import AlphaMonitor
        
        monitor = AlphaMonitor({})
        
        # First track a signal
        signal = {
            "symbol": "AAPL",
            "signal_type": "BUY",
            "confidence": 0.75,
        }
        monitor.update(signal=signal)
        
        # Then record outcome
        outcome = {
            "symbol": "AAPL",
            "correct": True,
            "pnl": 150.0,
        }
        monitor.update(outcome=outcome)
        
        assert monitor.metrics.signals_correct == 1
        assert monitor.metrics.signal_accuracy > 0


# ============================================================================
# LIQUIDITY ENGINE V5 TESTS
# ============================================================================

class TestLiquidityEngineV5:
    """Test suite for LiquidityEngineV5."""
    
    def test_import(self):
        """Test that LiquidityEngineV5 can be imported."""
        from engines.liquidity import LiquidityEngineV5
        assert LiquidityEngineV5 is not None
    
    def test_initialization(self):
        """Test initialization."""
        from engines.liquidity import LiquidityEngineV5
        
        engine = LiquidityEngineV5()
        
        assert engine.VERSION == "5.0.0"
        assert hasattr(engine, 'get_penta_engines')
    
    def test_penta_engines_available(self):
        """Test PENTA sub-engines are available."""
        from engines.liquidity import LiquidityEngineV5
        
        engine = LiquidityEngineV5()
        
        # Use the get_penta_engines method to access sub-engines
        penta_engines = engine.get_penta_engines()
        expected_engines = ['wyckoff', 'ict', 'order_flow', 'supply_demand', 'liquidity_concepts']
        
        for engine_name in expected_engines:
            assert engine_name in penta_engines


# ============================================================================
# LIQUIDITY AGENT V5 TESTS
# ============================================================================

class TestLiquidityAgentV5:
    """Test suite for LiquidityAgentV5."""
    
    def test_import(self):
        """Test that LiquidityAgentV5 can be imported."""
        from agents.liquidity_agent_v5 import LiquidityAgentV5
        assert LiquidityAgentV5 is not None
    
    def test_initialization_with_unified_engine(self):
        """Test initialization with unified LiquidityEngineV5."""
        from agents.liquidity_agent_v5 import LiquidityAgentV5
        from engines.liquidity import LiquidityEngineV5
        
        engine = LiquidityEngineV5()
        agent = LiquidityAgentV5(
            config={"min_confidence": 0.5},
            liquidity_engine_v5=engine,
        )
        
        assert agent is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
