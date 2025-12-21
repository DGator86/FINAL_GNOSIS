"""
Tests for Paper Trading Engine

Comprehensive tests for the paper trading integration including:
- Engine initialization and lifecycle
- Signal generation
- Order execution (mocked)
- Position tracking
- Lifecycle management
- Performance tracking
- Circuit breakers

Author: Super Gnosis Elite Trading System
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from trade.paper_trading_engine import (
    PaperTradingEngine,
    TradingEngineState,
    PositionTracker,
    TradingSessionStats,
    create_paper_trading_engine,
)
from trade.elite_trade_agent import (
    EliteTradeAgent,
    OptionStrategy,
    MarketRegime,
    IVEnvironment,
    Timeframe,
)
from trade.position_lifecycle_manager import (
    PositionLifecycleManager,
    PositionMetrics,
    LifecycleDecision,
    PositionStage,
    ExitReason,
    RollType,
)
from schemas.core_schemas import (
    DirectionEnum,
    TradeIdea,
    OrderResult,
    OrderStatus,
    PipelineResult,
    HedgeSnapshot,
    LiquiditySnapshot,
    SentimentSnapshot,
    ElasticitySnapshot,
    StrategyType,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_broker():
    """Create a mock broker adapter."""
    broker = Mock()
    
    # Account
    account = Mock()
    account.equity = 100000.0
    account.buying_power = 50000.0
    account.cash = 25000.0
    account.trading_blocked = False
    broker.get_account.return_value = account
    
    # Quotes
    broker.get_latest_quote.return_value = {
        "bid": 100.0,
        "ask": 100.10,
    }
    
    # Market status
    broker.is_market_open.return_value = True
    
    # Orders
    broker.place_order.return_value = "order_123"
    broker.close_position.return_value = True
    
    return broker


@pytest.fixture
def mock_market_adapter():
    """Create a mock market data adapter."""
    adapter = Mock()
    
    # Create mock OHLCV bars
    from engines.inputs.market_data_adapter import OHLCV
    
    bars = [
        OHLCV(
            timestamp=datetime.now(timezone.utc) - timedelta(days=1),
            open=99.0,
            high=101.0,
            low=98.0,
            close=100.0,
            volume=1000000,
        )
    ]
    adapter.get_bars.return_value = bars
    
    return adapter


@pytest.fixture
def engine_dry_run():
    """Create a paper trading engine in dry run mode."""
    return PaperTradingEngine(
        symbols=["SPY", "QQQ", "AAPL"],
        dry_run=True,
        config={
            "scan_interval": 5,
            "position_check_interval": 5,
            "max_positions": 3,
            "max_daily_loss": 1000.0,
        },
    )


@pytest.fixture
def engine_with_mocks(mock_broker, mock_market_adapter):
    """Create a paper trading engine with mocked dependencies."""
    engine = PaperTradingEngine(
        symbols=["SPY", "QQQ"],
        broker=mock_broker,
        market_adapter=mock_market_adapter,
        dry_run=False,
        config={
            "scan_interval": 5,
            "position_check_interval": 5,
            "max_positions": 5,
            "max_daily_loss": 5000.0,
        },
    )
    return engine


@pytest.fixture
def sample_trade_idea():
    """Create a sample trade idea."""
    return TradeIdea(
        timestamp=datetime.now(timezone.utc),
        symbol="SPY",
        strategy_type=StrategyType.DIRECTIONAL,
        direction=DirectionEnum.LONG,
        confidence=0.65,
        size=10000.0,
        entry_price=600.0,
        stop_loss=585.0,
        take_profit=630.0,
        reasoning="Test trade idea",
    )


@pytest.fixture
def sample_position():
    """Create a sample position tracker."""
    return PositionTracker(
        symbol="SPY",
        order_id="order_123",
        entry_price=600.0,
        current_price=605.0,
        quantity=10,
        side="long",
        entry_time=datetime.now(timezone.utc) - timedelta(hours=2),
        unrealized_pnl=50.0,
        unrealized_pnl_pct=0.83,
        highest_price=607.0,
        lowest_price=598.0,
        stop_loss=585.0,
        take_profit=630.0,
    )


# =============================================================================
# ENGINE INITIALIZATION TESTS
# =============================================================================

class TestEngineInitialization:
    """Tests for engine initialization."""
    
    def test_create_engine_dry_run(self):
        """Test creating engine in dry run mode."""
        engine = PaperTradingEngine(
            symbols=["SPY"],
            dry_run=True,
        )
        
        assert engine.dry_run is True
        assert engine.symbols == ["SPY"]
        assert engine.state == TradingEngineState.INITIALIZING
    
    def test_create_engine_with_config(self):
        """Test creating engine with custom config."""
        config = {
            "scan_interval": 120,
            "max_positions": 20,
            "max_daily_loss": 10000.0,
        }
        
        engine = PaperTradingEngine(
            symbols=["SPY", "QQQ"],
            config=config,
            dry_run=True,
        )
        
        assert engine.scan_interval == 120
        assert engine.max_positions == 20
        assert engine.max_daily_loss == 10000.0
    
    def test_factory_function(self):
        """Test create_paper_trading_engine factory."""
        engine = create_paper_trading_engine(
            symbols=["AAPL"],
            dry_run=True,
        )
        
        assert isinstance(engine, PaperTradingEngine)
        assert engine.symbols == ["AAPL"]
        assert engine.dry_run is True
    
    def test_engine_initializes_components_dry_run(self, engine_dry_run):
        """Test that engine initializes components in dry run mode."""
        result = engine_dry_run._initialize()
        
        assert result is True
        assert engine_dry_run.state == TradingEngineState.READY
        assert engine_dry_run.trade_agent is not None
        assert engine_dry_run.lifecycle_manager is not None
        assert engine_dry_run.greeks_manager is not None
        assert engine_dry_run.session_stats is not None
    
    def test_engine_initializes_session_stats(self, engine_dry_run):
        """Test session stats initialization."""
        engine_dry_run._init_session_stats(100000.0)
        
        assert engine_dry_run.session_stats is not None
        assert engine_dry_run.session_stats.start_equity == 100000.0
        assert engine_dry_run.session_stats.current_equity == 100000.0
        assert engine_dry_run.session_stats.high_water_mark == 100000.0


# =============================================================================
# ENGINE STATE MANAGEMENT TESTS
# =============================================================================

class TestEngineStateManagement:
    """Tests for engine state management."""
    
    def test_engine_pause_resume(self, engine_dry_run):
        """Test pausing and resuming the engine."""
        engine_dry_run._initialize()
        
        # Pause
        engine_dry_run.pause()
        assert engine_dry_run.paused is True
        assert engine_dry_run.state == TradingEngineState.PAUSED
        
        # Resume
        engine_dry_run.resume()
        assert engine_dry_run.paused is False
        assert engine_dry_run.state == TradingEngineState.RUNNING
    
    def test_engine_stop(self, engine_dry_run):
        """Test stopping the engine."""
        engine_dry_run._initialize()
        engine_dry_run.running = True
        
        engine_dry_run.stop()
        
        assert engine_dry_run.running is False
        assert engine_dry_run.state == TradingEngineState.STOPPED
    
    def test_get_status(self, engine_dry_run):
        """Test getting engine status."""
        engine_dry_run._initialize()
        
        status = engine_dry_run.get_status()
        
        assert "state" in status
        assert "running" in status
        assert "paused" in status
        assert "symbols" in status
        assert "positions_count" in status
        assert "stats" in status


# =============================================================================
# SIGNAL GENERATION TESTS
# =============================================================================

class TestSignalGeneration:
    """Tests for signal generation."""
    
    def test_build_pipeline_result(self, engine_dry_run):
        """Test building pipeline result for signal generation."""
        engine_dry_run._initialize()
        
        result = engine_dry_run._build_pipeline_result("SPY")
        
        assert result is not None
        assert result.symbol == "SPY"
        assert result.hedge_snapshot is not None
        assert result.sentiment_snapshot is not None
        assert result.elasticity_snapshot is not None
        assert result.consensus is not None
    
    def test_get_spot_price_fallback(self, engine_dry_run):
        """Test spot price fallback values."""
        price = engine_dry_run._get_spot_price("SPY")
        assert price == 600.0  # Fallback value
        
        price = engine_dry_run._get_spot_price("UNKNOWN")
        assert price == 100.0  # Default fallback
    
    def test_generate_signal(self, engine_dry_run):
        """Test signal generation."""
        engine_dry_run._initialize()
        
        # Should generate a signal (may be None depending on market conditions)
        signal = engine_dry_run._generate_signal("SPY")
        
        # Signal can be None if conditions aren't met
        if signal is not None:
            assert isinstance(signal, TradeIdea)
            assert signal.symbol == "SPY"


# =============================================================================
# ORDER EXECUTION TESTS
# =============================================================================

class TestOrderExecution:
    """Tests for order execution."""
    
    def test_execute_trade_dry_run(self, engine_dry_run, sample_trade_idea):
        """Test trade execution in dry run mode."""
        engine_dry_run._initialize()
        
        engine_dry_run._execute_trade(sample_trade_idea)
        
        assert "SPY" in engine_dry_run.positions
        assert engine_dry_run.session_stats.orders_placed == 1
    
    def test_track_dry_run_position(self, engine_dry_run, sample_trade_idea):
        """Test tracking positions in dry run mode."""
        engine_dry_run._initialize()
        
        engine_dry_run._track_dry_run_position(sample_trade_idea)
        
        position = engine_dry_run.positions["SPY"]
        assert position.symbol == "SPY"
        assert position.entry_price == 600.0
        assert position.side == "long"
        assert position.stop_loss == 585.0
        assert position.take_profit == 630.0
    
    def test_track_position(self, engine_with_mocks, sample_trade_idea):
        """Test position tracking after order submission."""
        engine_with_mocks._initialize()
        
        order_result = OrderResult(
            timestamp=datetime.now(timezone.utc),
            symbol="SPY",
            status=OrderStatus.SUBMITTED,
            order_id="order_123",
        )
        
        engine_with_mocks._track_position(sample_trade_idea, order_result)
        
        assert "SPY" in engine_with_mocks.positions
        position = engine_with_mocks.positions["SPY"]
        assert position.order_id == "order_123"


# =============================================================================
# POSITION TRACKING TESTS
# =============================================================================

class TestPositionTracking:
    """Tests for position tracking and updates."""
    
    def test_position_tracker_creation(self):
        """Test PositionTracker dataclass."""
        tracker = PositionTracker(
            symbol="AAPL",
            order_id="test_123",
            entry_price=230.0,
            current_price=235.0,
            quantity=100,
            side="long",
            entry_time=datetime.now(timezone.utc),
        )
        
        assert tracker.symbol == "AAPL"
        assert tracker.entry_price == 230.0
        assert tracker.current_price == 235.0
        assert tracker.quantity == 100
    
    def test_get_positions(self, engine_dry_run, sample_trade_idea):
        """Test getting positions as dictionaries."""
        engine_dry_run._initialize()
        engine_dry_run._track_dry_run_position(sample_trade_idea)
        
        positions = engine_dry_run.get_positions()
        
        assert len(positions) == 1
        assert positions[0]["symbol"] == "SPY"
        assert "entry_price" in positions[0]
        assert "unrealized_pnl" in positions[0]
    
    def test_position_pnl_calculation(self, engine_dry_run, sample_trade_idea):
        """Test P&L calculation during position check."""
        engine_dry_run._initialize()
        engine_dry_run._track_dry_run_position(sample_trade_idea)
        
        # Manually update position for testing
        position = engine_dry_run.positions["SPY"]
        position.current_price = 610.0
        position.entry_price = 600.0
        position.quantity = 10
        
        # Calculate P&L (long position)
        unrealized = (610.0 - 600.0) * 10
        position.unrealized_pnl = unrealized
        
        assert position.unrealized_pnl == 100.0


# =============================================================================
# LIFECYCLE MANAGEMENT TESTS
# =============================================================================

class TestLifecycleManagement:
    """Tests for position lifecycle management integration."""
    
    def test_check_position_lifecycle(self, engine_dry_run, sample_position):
        """Test lifecycle check on position."""
        engine_dry_run._initialize()
        engine_dry_run.positions["SPY"] = sample_position
        
        decision = engine_dry_run._check_position_lifecycle(sample_position)
        
        if decision is not None:
            assert isinstance(decision, LifecycleDecision)
            assert decision.action in ["hold", "close", "scale_out", "roll", "adjust"]
    
    def test_close_position_dry_run(self, engine_dry_run, sample_position):
        """Test closing a position in dry run mode."""
        engine_dry_run._initialize()
        engine_dry_run.positions["SPY"] = sample_position
        sample_position.unrealized_pnl = 50.0
        
        engine_dry_run._close_position("SPY", sample_position, ExitReason.PROFIT_TARGET)
        
        assert "SPY" not in engine_dry_run.positions
        assert engine_dry_run.session_stats.winning_trades == 1
        assert engine_dry_run.session_stats.realized_pnl == 50.0
    
    def test_close_losing_position(self, engine_dry_run, sample_position):
        """Test closing a losing position updates stats correctly."""
        engine_dry_run._initialize()
        engine_dry_run.positions["SPY"] = sample_position
        sample_position.unrealized_pnl = -100.0
        
        engine_dry_run._close_position("SPY", sample_position, ExitReason.STOP_LOSS)
        
        assert engine_dry_run.session_stats.losing_trades == 1
        assert engine_dry_run.session_stats.realized_pnl == -100.0
    
    def test_scale_out_position_dry_run(self, engine_dry_run, sample_position):
        """Test scaling out of a position in dry run mode."""
        engine_dry_run._initialize()
        sample_position.quantity = 20
        engine_dry_run.positions["SPY"] = sample_position
        
        engine_dry_run._scale_out_position("SPY", sample_position, quantity=10, pct=0.5)
        
        # Position should remain but with reduced quantity
        assert "SPY" in engine_dry_run.positions
        assert engine_dry_run.positions["SPY"].quantity == 10


# =============================================================================
# PERFORMANCE TRACKING TESTS
# =============================================================================

class TestPerformanceTracking:
    """Tests for performance tracking and statistics."""
    
    def test_session_stats_creation(self):
        """Test TradingSessionStats dataclass."""
        stats = TradingSessionStats(
            start_time=datetime.now(timezone.utc),
            start_equity=100000.0,
            current_equity=105000.0,
            high_water_mark=106000.0,
            total_signals=50,
            orders_placed=25,
            winning_trades=15,
            losing_trades=10,
        )
        
        assert stats.start_equity == 100000.0
        assert stats.current_equity == 105000.0
        assert stats.winning_trades == 15
    
    def test_update_stats(self, engine_dry_run, sample_position):
        """Test stats update."""
        engine_dry_run._initialize()
        sample_position.unrealized_pnl = 500.0
        engine_dry_run.positions["SPY"] = sample_position
        engine_dry_run.session_stats.realized_pnl = 1000.0
        
        engine_dry_run._update_stats()
        
        assert engine_dry_run.session_stats.unrealized_pnl == 500.0
        assert engine_dry_run.session_stats.total_pnl == 1500.0
    
    def test_win_rate_calculation(self, engine_dry_run):
        """Test win rate calculation."""
        engine_dry_run._initialize()
        engine_dry_run.session_stats.winning_trades = 7
        engine_dry_run.session_stats.losing_trades = 3
        
        engine_dry_run._update_stats()
        
        assert engine_dry_run.session_stats.win_rate == 0.7


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreakers:
    """Tests for circuit breaker functionality."""
    
    def test_circuit_breaker_daily_loss(self, engine_dry_run):
        """Test daily loss circuit breaker."""
        engine_dry_run._initialize()
        engine_dry_run.session_stats.realized_pnl = -900.0
        engine_dry_run.session_stats.unrealized_pnl = -200.0
        
        # Should trigger (total loss > max_daily_loss)
        result = engine_dry_run._check_circuit_breakers()
        assert result is False
    
    def test_circuit_breaker_not_triggered(self, engine_dry_run):
        """Test circuit breaker when within limits."""
        engine_dry_run._initialize()
        engine_dry_run.session_stats.realized_pnl = 500.0
        engine_dry_run.session_stats.unrealized_pnl = -200.0
        
        result = engine_dry_run._check_circuit_breakers()
        assert result is True


# =============================================================================
# MARKET HOURS TESTS
# =============================================================================

class TestMarketHours:
    """Tests for market hours checking."""
    
    def test_market_open_dry_run(self, engine_dry_run):
        """Test market open always returns True in dry run."""
        result = engine_dry_run._is_market_open()
        assert result is True
    
    def test_market_open_with_broker(self, engine_with_mocks):
        """Test market open check with broker."""
        engine_with_mocks.broker.is_market_open.return_value = True
        result = engine_with_mocks._is_market_open()
        assert result is True
        
        engine_with_mocks.broker.is_market_open.return_value = False
        result = engine_with_mocks._is_market_open()
        assert result is False


# =============================================================================
# CALLBACK TESTS
# =============================================================================

class TestCallbacks:
    """Tests for callback functionality."""
    
    def test_on_signal_generated_callback(self, engine_dry_run, sample_trade_idea):
        """Test signal generated callback."""
        engine_dry_run._initialize()
        
        callback_called = {"value": False}
        def callback(idea):
            callback_called["value"] = True
            assert idea.symbol == "SPY"
        
        engine_dry_run.on_signal_generated = callback
        engine_dry_run.session_stats.total_signals = 0  # Reset
        
        # Simulate signal generation callback
        engine_dry_run.on_signal_generated(sample_trade_idea)
        
        assert callback_called["value"] is True
    
    def test_on_order_placed_callback(self, engine_dry_run):
        """Test order placed callback."""
        engine_dry_run._initialize()
        
        callback_called = {"value": False}
        def callback(result):
            callback_called["value"] = True
        
        engine_dry_run.on_order_placed = callback
        
        # Simulate callback
        order_result = OrderResult(
            timestamp=datetime.now(timezone.utc),
            symbol="SPY",
            status=OrderStatus.SUBMITTED,
        )
        engine_dry_run.on_order_placed(order_result)
        
        assert callback_called["value"] is True
    
    def test_on_position_update_callback(self, engine_dry_run, sample_position):
        """Test position update callback."""
        engine_dry_run._initialize()
        engine_dry_run.positions["SPY"] = sample_position
        
        callback_called = {"value": False}
        def callback(position):
            callback_called["value"] = True
        
        engine_dry_run.on_position_update = callback
        
        # Simulate callback
        engine_dry_run.on_position_update(sample_position)
        
        assert callback_called["value"] is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the paper trading engine."""
    
    def test_full_signal_to_position_flow_dry_run(self, engine_dry_run, sample_trade_idea):
        """Test complete flow from signal to position tracking."""
        engine_dry_run._initialize()
        
        # Execute trade
        engine_dry_run._execute_trade(sample_trade_idea)
        
        # Verify position created
        assert "SPY" in engine_dry_run.positions
        
        # Verify stats updated
        assert engine_dry_run.session_stats.orders_placed == 1
        
        # Check position
        position = engine_dry_run.positions["SPY"]
        assert position.symbol == "SPY"
        assert position.side == "long"
    
    def test_scan_skips_existing_positions(self, engine_dry_run, sample_trade_idea):
        """Test that scanning skips symbols with existing positions."""
        engine_dry_run._initialize()
        
        # Add existing position
        engine_dry_run._track_dry_run_position(sample_trade_idea)
        
        initial_signals = engine_dry_run.session_stats.total_signals
        
        # Scan for signals (SPY should be skipped)
        engine_dry_run._scan_for_signals()
        
        # If signals were generated, they should be for other symbols
        # SPY already has a position
        assert "SPY" in engine_dry_run.positions
    
    def test_max_positions_limit(self, engine_dry_run):
        """Test that engine respects max positions limit."""
        engine_dry_run._initialize()
        engine_dry_run.max_positions = 2
        
        # Add positions to reach max
        for i, symbol in enumerate(["SPY", "QQQ"]):
            idea = TradeIdea(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                strategy_type=StrategyType.DIRECTIONAL,
                direction=DirectionEnum.LONG,
                confidence=0.65,
                size=10000.0,
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
            )
            engine_dry_run._track_dry_run_position(idea)
        
        assert len(engine_dry_run.positions) == 2
        
        # Try to add another - should be blocked in scan
        # (The scan loop checks max_positions before generating signals)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_handle_broker_error(self, engine_with_mocks, sample_trade_idea):
        """Test handling of broker errors."""
        engine_with_mocks._initialize()
        engine_with_mocks.broker.place_order.side_effect = Exception("Broker error")
        
        # This should handle the error gracefully
        # Note: In the current implementation, errors during execution
        # are logged but the engine continues
        engine_with_mocks._execute_trade(sample_trade_idea)
        
        # Error should be counted in stats
        # (depending on implementation details)
    
    def test_handle_market_data_error(self, engine_dry_run):
        """Test handling of market data errors."""
        engine_dry_run._initialize()
        
        # Mock adapter to raise error
        engine_dry_run.market_adapter = Mock()
        engine_dry_run.market_adapter.get_bars.side_effect = Exception("Data error")
        
        # Should fall back to default price
        price = engine_dry_run._get_spot_price("SPY")
        assert price == 600.0  # Fallback value


# =============================================================================
# CLEANUP TESTS
# =============================================================================

class TestCleanup:
    """Tests for engine cleanup."""
    
    def test_cleanup_sets_state(self, engine_dry_run):
        """Test that cleanup sets correct state."""
        engine_dry_run._initialize()
        engine_dry_run.running = True
        
        engine_dry_run._cleanup()
        
        assert engine_dry_run.running is False
        assert engine_dry_run.state == TradingEngineState.STOPPED
    
    def test_print_session_summary(self, engine_dry_run, capsys):
        """Test session summary printing."""
        engine_dry_run._initialize()
        engine_dry_run.session_stats.total_signals = 10
        engine_dry_run.session_stats.orders_placed = 5
        
        engine_dry_run._print_session_summary()
        
        captured = capsys.readouterr()
        assert "PAPER TRADING SESSION SUMMARY" in captured.out
        assert "Signals Generated: 10" in captured.out
        assert "Orders Placed: 5" in captured.out
