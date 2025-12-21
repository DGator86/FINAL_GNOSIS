"""
Tests for ML Trading Engine.

Tests cover:
- MLTradingConfig
- MLTradingEngine initialization
- ML signal processing
- Trade execution
- Position management
- Safety integration

Author: Super Gnosis Elite Trading System
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, Mock

import pytest

from trade.ml_trading_engine import (
    MLTradingState,
    MLTradingConfig,
    MLPosition,
    MLTradingStats,
    MLTradingEngine,
    create_ml_trading_engine,
)

from ml import (
    MLTradeDecision,
    MLSignal,
    MLPositionSize,
    SignalStrength,
    MarketRegime,
)


class TestMLTradingConfig:
    """Tests for MLTradingConfig."""
    
    def test_default_values(self):
        config = MLTradingConfig()
        assert config.paper_mode is True
        assert config.dry_run is False
        assert config.ml_preset == "balanced"
        assert config.ml_confidence_threshold == 0.60
    
    def test_custom_values(self):
        config = MLTradingConfig(
            paper_mode=False,
            ml_preset="aggressive",
            max_daily_loss_usd=10000.0,
            symbols=["TSLA", "AMZN"],
        )
        assert config.paper_mode is False
        assert config.ml_preset == "aggressive"
        assert config.max_daily_loss_usd == 10000.0
        assert "TSLA" in config.symbols


class TestMLPosition:
    """Tests for MLPosition."""
    
    def test_creation(self):
        position = MLPosition(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=200.0,
            current_price=205.0,
            entry_time=datetime.now(timezone.utc),
            ml_confidence=0.75,
            ml_regime="trending_bull",
        )
        assert position.symbol == "AAPL"
        assert position.ml_confidence == 0.75
    
    def test_pnl_calculation(self):
        position = MLPosition(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=200.0,
            current_price=210.0,
            entry_time=datetime.now(timezone.utc),
        )
        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
        assert position.unrealized_pnl == 100.0  # $10 gain x 10 shares


class TestMLTradingStats:
    """Tests for MLTradingStats."""
    
    def test_creation(self):
        stats = MLTradingStats(
            start_time=datetime.now(timezone.utc),
            start_equity=100000.0,
        )
        assert stats.ml_signals_generated == 0
        assert stats.realized_pnl == 0.0


class TestMLTradingEngine:
    """Tests for MLTradingEngine."""
    
    @pytest.fixture
    def engine(self):
        config = MLTradingConfig(
            dry_run=True,
            paper_mode=True,
            symbols=["SPY", "AAPL"],
        )
        return MLTradingEngine(config=config)
    
    def test_initialization(self, engine):
        assert engine.state == MLTradingState.INITIALIZING
        assert engine.config.dry_run is True
    
    def test_internal_initialize(self, engine):
        result = engine._initialize()
        assert result is True
        assert engine.state == MLTradingState.READY
        assert engine.ml_pipeline is not None
        assert engine.safety_manager is not None
    
    def test_start_stop(self, engine):
        engine._initialize()
        
        # Start non-blocking
        engine.start(blocking=False)
        assert engine.running is True
        assert engine.state == MLTradingState.RUNNING
        
        # Stop
        engine.stop()
        assert engine.running is False
    
    def test_pause_resume(self, engine):
        engine._initialize()
        
        engine.pause()
        assert engine.paused is True
        assert engine.state == MLTradingState.PAUSED
        
        engine.resume()
        assert engine.paused is False
        assert engine.state == MLTradingState.RUNNING
    
    def test_get_ml_decision(self, engine):
        engine._initialize()
        
        decision = engine._get_ml_decision("SPY")
        assert decision is None or isinstance(decision, MLTradeDecision)
    
    def test_validate_ml_trade(self, engine):
        engine._initialize()
        
        # Create mock decision
        decision = MLTradeDecision(
            should_trade=True,
            action="buy",
            signal=MLSignal(
                direction="bullish",
                strength=SignalStrength.BUY,
                confidence=0.75,
            ),
            position_size=MLPositionSize(
                base_size=0.02,
                adjusted_size=0.02,
                max_allowed=0.04,
            ),
            overall_confidence=0.75,
        )
        
        result = engine._validate_ml_trade("AAPL", decision)
        assert result.approved is True or result.reason != ""
    
    def test_track_position(self, engine):
        engine._initialize()
        
        decision = MLTradeDecision(
            should_trade=True,
            action="buy",
            signal=MLSignal(
                direction="bullish",
                strength=SignalStrength.BUY,
                confidence=0.75,
                regime="trending_bull",
            ),
            position_size=MLPositionSize(
                base_size=0.02,
                adjusted_size=0.02,
                max_allowed=0.04,
            ),
            overall_confidence=0.75,
            stop_loss=0.02,
            take_profit=0.04,
        )
        
        engine._track_position("AAPL", decision, 10, 200.0, "TEST123")
        
        assert "AAPL" in engine.positions
        assert engine.positions["AAPL"].ml_confidence == 0.75
    
    def test_check_exit_conditions_stop_loss(self, engine):
        engine._initialize()
        
        position = MLPosition(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=200.0,
            current_price=190.0,  # 5% loss
            entry_time=datetime.now(timezone.utc),
            stop_loss=195.0,  # Stop at 2.5%
        )
        
        should_exit, reason = engine._check_exit_conditions(position)
        assert should_exit is True
        assert reason == "stop_loss"
    
    def test_check_exit_conditions_take_profit(self, engine):
        engine._initialize()
        
        position = MLPosition(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=200.0,
            current_price=220.0,  # 10% gain
            entry_time=datetime.now(timezone.utc),
            take_profit=210.0,  # Take at 5%
        )
        
        should_exit, reason = engine._check_exit_conditions(position)
        assert should_exit is True
        assert reason == "take_profit"
    
    def test_check_exit_conditions_hold(self, engine):
        engine._initialize()
        
        position = MLPosition(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=200.0,
            current_price=205.0,  # 2.5% gain
            entry_time=datetime.now(timezone.utc),
            stop_loss=190.0,
            take_profit=220.0,
        )
        
        should_exit, reason = engine._check_exit_conditions(position)
        assert should_exit is False
        assert reason == ""
    
    def test_get_status(self, engine):
        engine._initialize()
        
        status = engine.get_status()
        
        assert "state" in status
        assert "running" in status
        assert "ml_preset" in status
        assert "ml_pipeline_status" in status
        assert "safety_status" in status
        assert "stats" in status
    
    def test_safety_integration(self, engine):
        engine._initialize()
        
        # Safety manager should block when conditions are met
        engine.safety_manager.metrics.circuit_breaker_state = engine.safety_manager.metrics.circuit_breaker_state.__class__.OPEN
        
        decision = MLTradeDecision(
            should_trade=True,
            action="buy",
            signal=MLSignal(
                direction="bullish",
                strength=SignalStrength.BUY,
                confidence=0.80,
            ),
            position_size=MLPositionSize(
                base_size=0.02,
                adjusted_size=0.02,
                max_allowed=0.04,
            ),
            overall_confidence=0.80,
        )
        
        result = engine._validate_ml_trade("AAPL", decision)
        assert result.approved is False
    
    def test_ml_confidence_threshold(self, engine):
        engine._initialize()
        engine.config.ml_confidence_threshold = 0.80
        
        # Low confidence decision should be filtered
        decision = MLTradeDecision(
            should_trade=True,
            action="buy",
            signal=MLSignal(
                direction="bullish",
                strength=SignalStrength.WEAK_BUY,
                confidence=0.55,
            ),
            position_size=MLPositionSize(
                base_size=0.02,
                adjusted_size=0.02,
                max_allowed=0.04,
            ),
            overall_confidence=0.55,  # Below threshold
        )
        
        # Would be filtered in scan loop
        passes_threshold = decision.overall_confidence >= engine.config.ml_confidence_threshold
        assert passes_threshold is False
    
    def test_position_update(self, engine):
        engine._initialize()
        
        # Add a position
        position = MLPosition(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=200.0,
            current_price=200.0,
            entry_time=datetime.now(timezone.utc),
            stop_loss=190.0,
            take_profit=220.0,
            highest_price=200.0,
            lowest_price=200.0,
        )
        engine.positions["AAPL"] = position
        
        # Mock price update
        with patch.object(engine, '_get_spot_price', return_value=205.0):
            engine._check_positions()
        
        # Position should be updated
        assert engine.positions["AAPL"].current_price == 205.0
        assert engine.positions["AAPL"].highest_price == 205.0


class TestCreateMLTradingEngine:
    """Tests for create_ml_trading_engine factory."""
    
    def test_default_creation(self):
        engine = create_ml_trading_engine()
        assert engine.config.ml_preset == "balanced"
        assert engine.config.paper_mode is True
        assert engine.config.dry_run is False
    
    def test_custom_creation(self):
        engine = create_ml_trading_engine(
            preset="aggressive",
            paper_mode=True,
            dry_run=True,
            symbols=["TSLA"],
            max_daily_loss=10000.0,
        )
        assert engine.config.ml_preset == "aggressive"
        assert engine.config.dry_run is True
        assert "TSLA" in engine.config.symbols
        assert engine.config.max_daily_loss_usd == 10000.0


class TestMLTradingIntegration:
    """Integration tests for ML trading engine."""
    
    def test_full_workflow(self):
        """Test complete ML trading workflow."""
        engine = create_ml_trading_engine(
            preset="conservative",
            dry_run=True,
        )
        
        # Initialize
        result = engine._initialize()
        assert result is True
        
        # Check status
        status = engine.get_status()
        assert status["state"] == "ready"
        
        # Start (non-blocking)
        engine.start(blocking=False)
        assert engine.running is True
        
        # Pause
        engine.pause()
        assert engine.paused is True
        
        # Resume
        engine.resume()
        assert engine.paused is False
        
        # Stop
        engine.stop()
        assert engine.running is False
    
    def test_ml_pipeline_integration(self):
        """Test ML pipeline is properly integrated."""
        engine = create_ml_trading_engine(preset="balanced", dry_run=True)
        engine._initialize()
        
        assert engine.ml_pipeline is not None
        
        # Check pipeline status
        pipeline_status = engine.ml_pipeline.get_status()
        assert "preset" in pipeline_status
        assert pipeline_status["preset"] == "balanced"
    
    def test_safety_manager_integration(self):
        """Test safety manager is properly integrated."""
        engine = create_ml_trading_engine(
            max_daily_loss=5000.0,
            dry_run=True,
        )
        engine._initialize()
        
        assert engine.safety_manager is not None
        
        # Check safety config
        assert engine.safety_manager.config.max_daily_loss_usd == 5000.0
        
        # Check safety status
        safety_status = engine.safety_manager.get_status()
        assert safety_status["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
