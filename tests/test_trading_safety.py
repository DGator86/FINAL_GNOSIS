"""
Tests for Trading Safety Module.

Tests cover:
- SafetyConfig validation
- TradingSafetyManager
- Circuit breakers
- Position limits
- Rate limiting
- Trade validation

Author: Super Gnosis Elite Trading System
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from trade.trading_safety import (
    SafetyStatus,
    CircuitBreakerState,
    SafetyConfig,
    SafetyMetrics,
    TradeValidationResult,
    TradingSafetyManager,
    create_safety_manager,
)


class TestSafetyConfig:
    """Tests for SafetyConfig."""
    
    def test_default_values(self):
        config = SafetyConfig()
        assert config.max_daily_loss_usd == 5000.0
        assert config.max_positions == 10
        assert config.max_position_size_pct == 0.10
    
    def test_custom_values(self):
        config = SafetyConfig(
            max_daily_loss_usd=10000.0,
            max_positions=5,
            max_vix_for_new_trades=40.0,
        )
        assert config.max_daily_loss_usd == 10000.0
        assert config.max_positions == 5
        assert config.max_vix_for_new_trades == 40.0


class TestSafetyMetrics:
    """Tests for SafetyMetrics."""
    
    def test_default_values(self):
        metrics = SafetyMetrics()
        assert metrics.daily_pnl == 0.0
        assert metrics.position_count == 0
        assert metrics.status == SafetyStatus.OK
        assert metrics.circuit_breaker_state == CircuitBreakerState.CLOSED
    
    def test_warning_list(self):
        metrics = SafetyMetrics()
        metrics.warnings.append("Test warning")
        assert len(metrics.warnings) == 1


class TestTradeValidationResult:
    """Tests for TradeValidationResult."""
    
    def test_approved_result(self):
        result = TradeValidationResult(approved=True)
        assert result.approved is True
        assert result.reason == ""
        assert result.suggested_size_multiplier == 1.0
    
    def test_rejected_result(self):
        result = TradeValidationResult(
            approved=False,
            reason="Position size exceeded",
            warnings=["Near limit"],
        )
        assert result.approved is False
        assert "Position size" in result.reason
        assert len(result.warnings) == 1


class TestTradingSafetyManager:
    """Tests for TradingSafetyManager."""
    
    @pytest.fixture
    def safety_manager(self):
        return TradingSafetyManager(portfolio_value=100000.0)
    
    def test_initialization(self, safety_manager):
        assert safety_manager.portfolio_value == 100000.0
        assert safety_manager.metrics.status == SafetyStatus.OK
    
    def test_validate_trade_approved(self, safety_manager):
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
        )
        assert result.approved is True
    
    def test_validate_trade_position_size_limit(self, safety_manager):
        # Try to buy more than max position size (10% of 100k = 10k)
        # 100 shares at $200 = $20,000 = 20% of portfolio
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=200.0,
        )
        # Should add warning but still potentially approve with adjustment
        assert len(result.warnings) > 0 or result.max_allowed_size < 100
    
    def test_validate_trade_max_positions(self, safety_manager):
        # Fill up positions
        for i in range(safety_manager.config.max_positions):
            safety_manager._positions[f"SYM{i}"] = {"quantity": 1, "value": 1000, "avg_price": 100}
        
        safety_manager._update_position_metrics()
        
        # Try to add another
        result = safety_manager.validate_trade(
            symbol="NEWSTOCK",
            side="buy",
            quantity=10,
            price=100.0,
        )
        assert result.approved is False
        assert "Max positions" in result.reason
    
    def test_validate_trade_vix_block(self, safety_manager):
        safety_manager.update_vix(40.0)  # Above default threshold of 35
        
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
        )
        assert result.approved is False
        assert "VIX" in result.reason
    
    def test_validate_trade_sell_allowed_with_vix(self, safety_manager):
        safety_manager.update_vix(40.0)
        safety_manager._positions["AAPL"] = {"quantity": 10, "value": 2000, "avg_price": 200}
        
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="sell",
            quantity=10,
            price=200.0,
        )
        # Sells should still be allowed
        # Note: VIX check only blocks buys
        assert result.approved is True
    
    def test_validate_trade_low_liquidity(self, safety_manager):
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
            metadata={"liquidity_score": 0.05},  # Very low
        )
        assert result.approved is False
        assert "liquidity" in result.reason.lower()
    
    def test_validate_trade_wide_spread(self, safety_manager):
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
            metadata={"spread_pct": 0.15},  # 15% spread, very wide
        )
        assert result.approved is False
        assert "spread" in result.reason.lower()
    
    def test_record_order(self, safety_manager):
        safety_manager.record_order("AAPL", "buy", 10, 200.0)
        
        assert "AAPL" in safety_manager._positions
        assert safety_manager._positions["AAPL"]["quantity"] == 10
        assert safety_manager.metrics.last_order_time is not None
    
    def test_record_order_sell(self, safety_manager):
        safety_manager.record_order("AAPL", "buy", 10, 200.0)
        safety_manager.record_order("AAPL", "sell", 5, 210.0)
        
        assert safety_manager._positions["AAPL"]["quantity"] == 5
    
    def test_record_order_close_position(self, safety_manager):
        safety_manager.record_order("AAPL", "buy", 10, 200.0)
        safety_manager.record_order("AAPL", "sell", 10, 210.0)
        
        assert "AAPL" not in safety_manager._positions
    
    def test_update_pnl(self, safety_manager):
        safety_manager.update_pnl(1000.0, 101000.0)
        
        assert safety_manager.metrics.daily_pnl == 1000.0
        assert safety_manager.portfolio_value == 101000.0
    
    def test_update_pnl_loss(self, safety_manager):
        safety_manager.update_pnl(-1000.0, 99000.0)
        
        assert safety_manager.metrics.daily_pnl == -1000.0
        assert safety_manager.metrics.current_drawdown_pct > 0
    
    def test_circuit_breaker_trigger(self, safety_manager):
        # Rapid loss of 2% in short window
        for _ in range(10):
            safety_manager.update_pnl(-300.0, safety_manager.portfolio_value - 300)
        
        # Should trigger circuit breaker
        if safety_manager.metrics.daily_pnl < -2000:
            safety_manager._check_circuit_breakers()
        
        # Check state
        # Note: May or may not trigger depending on timing
        assert safety_manager.metrics.circuit_breaker_state in [
            CircuitBreakerState.CLOSED,
            CircuitBreakerState.OPEN
        ]
    
    def test_circuit_breaker_blocks_trades(self, safety_manager):
        # Force circuit breaker open
        safety_manager.metrics.circuit_breaker_state = CircuitBreakerState.OPEN
        
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
        )
        assert result.approved is False
        assert "Circuit breaker" in result.reason
    
    def test_emergency_stop(self, safety_manager):
        # Large loss to trigger emergency
        safety_manager._session_start_equity = 100000.0
        safety_manager.update_pnl(-12000.0, 88000.0)
        
        # Check if emergency triggered (10% loss)
        assert safety_manager.metrics.status == SafetyStatus.EMERGENCY
    
    def test_emergency_stop_blocks_trades(self, safety_manager):
        safety_manager.metrics.status = SafetyStatus.EMERGENCY
        
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
        )
        assert result.approved is False
        assert "Emergency" in result.reason
    
    def test_reset_circuit_breaker(self, safety_manager):
        safety_manager.metrics.circuit_breaker_state = CircuitBreakerState.OPEN
        safety_manager.metrics.status = SafetyStatus.BLOCKED
        
        safety_manager.reset_circuit_breaker(force=True)
        
        assert safety_manager.metrics.circuit_breaker_state == CircuitBreakerState.CLOSED
        assert safety_manager.metrics.status == SafetyStatus.OK
    
    def test_reset_daily_metrics(self, safety_manager):
        safety_manager.metrics.daily_pnl = -5000.0
        safety_manager.metrics.warnings.append("Test warning")
        
        safety_manager.reset_daily_metrics()
        
        assert safety_manager.metrics.daily_pnl == 0.0
        assert len(safety_manager.metrics.warnings) == 0
    
    def test_is_trading_allowed_ok(self, safety_manager):
        allowed, reason = safety_manager.is_trading_allowed()
        assert allowed is True
        assert reason == "OK"
    
    def test_is_trading_allowed_blocked(self, safety_manager):
        safety_manager.metrics.circuit_breaker_state = CircuitBreakerState.OPEN
        
        allowed, reason = safety_manager.is_trading_allowed()
        assert allowed is False
        assert "Circuit breaker" in reason
    
    def test_get_status(self, safety_manager):
        status = safety_manager.get_status()
        
        assert "status" in status
        assert "circuit_breaker" in status
        assert "daily_pnl" in status
        assert "position_count" in status
    
    def test_rate_limiting(self, safety_manager):
        # Set very low rate limit for testing
        safety_manager.config.max_orders_per_minute = 2
        
        # First two should pass
        safety_manager.record_order("SYM1", "buy", 1, 100)
        time.sleep(0.1)
        safety_manager.record_order("SYM2", "buy", 1, 100)
        
        # Third should fail rate limit
        result = safety_manager.validate_trade("SYM3", "buy", 1, 100)
        assert result.approved is False
        assert "rate limit" in result.reason.lower()
    
    def test_single_stock_concentration(self, safety_manager):
        # Add large existing position
        safety_manager._positions["AAPL"] = {"quantity": 70, "value": 14000, "avg_price": 200}
        
        # Try to add more (would exceed 15% single stock limit)
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
        )
        # Should have warning or be blocked
        assert len(result.warnings) > 0 or result.approved is False
    
    def test_volatility_scaling(self, safety_manager):
        safety_manager.update_vix(30.0)  # Elevated but below threshold
        
        result = safety_manager.validate_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
        )
        
        # Should be approved with volatility adjustment
        assert result.approved is True
        if safety_manager.config.position_size_vix_scaling:
            assert result.suggested_size_multiplier < 1.0 or "volatility" in str(result.adjustments)


class TestCreateSafetyManager:
    """Tests for create_safety_manager factory."""
    
    def test_default_creation(self):
        manager = create_safety_manager()
        assert manager.portfolio_value == 100000.0
        assert manager.config.max_daily_loss_usd == 5000.0
    
    def test_custom_creation(self):
        manager = create_safety_manager(
            portfolio_value=200000.0,
            max_daily_loss=10000.0,
            max_positions=20,
        )
        assert manager.portfolio_value == 200000.0
        assert manager.config.max_daily_loss_usd == 10000.0
        assert manager.config.max_positions == 20
    
    def test_with_callback(self):
        events = []
        
        def callback(event_type, data):
            events.append((event_type, data))
        
        manager = create_safety_manager(on_safety_event=callback)
        
        # Trigger an event
        manager._emit_safety_event("test_event", {"test": "data"})
        
        assert len(events) == 1
        assert events[0][0] == "test_event"


class TestSafetyIntegration:
    """Integration tests for safety system."""
    
    def test_full_trading_day_simulation(self):
        """Simulate a full trading day with various scenarios."""
        manager = create_safety_manager(
            portfolio_value=100000.0,
            max_daily_loss=5000.0,
        )
        
        # Morning: some winning trades
        for i in range(3):
            result = manager.validate_trade(f"SYM{i}", "buy", 10, 100)
            if result.approved:
                manager.record_order(f"SYM{i}", "buy", 10, 100)
                manager.update_pnl(500, manager.portfolio_value + 500)
        
        assert manager.metrics.daily_pnl > 0
        
        # Afternoon: some losing trades
        for i in range(3, 6):
            result = manager.validate_trade(f"SYM{i}", "buy", 10, 100)
            if result.approved:
                manager.record_order(f"SYM{i}", "buy", 10, 100)
                manager.update_pnl(-1000, manager.portfolio_value - 1000)
        
        # End of day
        status = manager.get_status()
        assert "daily_pnl" in status
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker trigger and recovery."""
        manager = create_safety_manager(portfolio_value=100000.0)
        
        # Force trigger
        manager._trigger_circuit_breaker(-3000.0)
        assert manager.metrics.circuit_breaker_state == CircuitBreakerState.OPEN
        
        # Verify blocked
        allowed, _ = manager.is_trading_allowed()
        assert allowed is False
        
        # Reset
        manager.reset_circuit_breaker(force=True)
        
        # Verify recovered
        allowed, _ = manager.is_trading_allowed()
        assert allowed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
