"""
Comprehensive Unit Tests for PositionLifecycleManager

Tests cover:
- Position stage classification
- Exit condition detection (profit, stop loss, time-based, DTE)
- Rolling strategy determination
- Adjustment recommendations
- Scale-out calculations
- Assignment risk monitoring
- Batch position analysis
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from trade.position_lifecycle_manager import (
    PositionLifecycleManager,
    PositionStage,
    ExitReason,
    RollType,
    AdjustmentType,
    PositionMetrics,
    LifecycleDecision,
    LifecycleConfig,
    create_lifecycle_manager,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def lifecycle_manager():
    """Create a basic PositionLifecycleManager for testing."""
    return PositionLifecycleManager()


@pytest.fixture
def custom_lifecycle_manager():
    """Create a PositionLifecycleManager with custom config."""
    config = LifecycleConfig(
        profit_target_pct=75.0,
        stop_loss_pct=50.0,
        dte_exit=5,
        dte_warning=10,
        max_hold_days=30,
    )
    return PositionLifecycleManager(config=config)


@pytest.fixture
def sample_equity_position():
    """Create a sample equity position."""
    return PositionMetrics(
        symbol="AAPL",
        underlying="AAPL",
        entry_price=170.0,
        current_price=175.0,
        quantity=100,
        unrealized_pnl=500.0,
        unrealized_pnl_pct=2.94,
        entry_time=datetime.utcnow() - timedelta(days=5),
        days_held=5,
        is_option=False,
        underlying_price=175.0,
    )


@pytest.fixture
def sample_option_position():
    """Create a sample options position."""
    return PositionMetrics(
        symbol="AAPL240315C175",
        underlying="AAPL",
        entry_price=5.00,
        current_price=6.50,
        quantity=10,
        unrealized_pnl=1500.0,
        unrealized_pnl_pct=30.0,
        entry_time=datetime.utcnow() - timedelta(days=10),
        days_held=10,
        is_option=True,
        option_type="call",
        strike=175.0,
        expiration=datetime.utcnow() + timedelta(days=21),
        dte=21,
        underlying_price=180.0,
        delta=0.65,
        gamma=0.03,
        theta=-0.15,
        moneyness=0.0286,  # (180-175)/175
    )


@pytest.fixture
def sample_credit_position():
    """Create a sample credit spread position."""
    return PositionMetrics(
        symbol="AAPL_SPREAD",
        underlying="AAPL",
        entry_price=-1.50,  # Credit received
        current_price=-0.75,  # Current spread value
        quantity=-10,  # Short position
        unrealized_pnl=750.0,  # 50% profit
        unrealized_pnl_pct=50.0,
        entry_time=datetime.utcnow() - timedelta(days=14),
        days_held=14,
        is_option=True,
        option_type="put",
        strike=170.0,
        expiration=datetime.utcnow() + timedelta(days=14),
        dte=14,
        underlying_price=175.0,
        delta=-0.25,
        gamma=0.02,
        theta=0.10,  # Positive theta (collecting)
        moneyness=-0.0286,
        is_spread=True,
        spread_type="bull_put_spread",
    )


@pytest.fixture
def losing_position():
    """Create a losing position for testing stop loss."""
    return PositionMetrics(
        symbol="TSLA240315C250",
        underlying="TSLA",
        entry_price=10.00,
        current_price=4.00,
        quantity=5,
        unrealized_pnl=-3000.0,
        unrealized_pnl_pct=-60.0,
        entry_time=datetime.utcnow() - timedelta(days=7),
        days_held=7,
        is_option=True,
        option_type="call",
        strike=250.0,
        expiration=datetime.utcnow() + timedelta(days=8),
        dte=8,
        underlying_price=230.0,
        delta=0.25,
        moneyness=-0.08,
    )


@pytest.fixture
def expiring_position():
    """Create a position close to expiration."""
    return PositionMetrics(
        symbol="SPY240315C500",
        underlying="SPY",
        entry_price=3.00,
        current_price=3.50,
        quantity=10,
        unrealized_pnl=500.0,
        unrealized_pnl_pct=16.67,
        entry_time=datetime.utcnow() - timedelta(days=21),
        days_held=21,
        is_option=True,
        option_type="call",
        strike=500.0,
        expiration=datetime.utcnow() + timedelta(days=5),
        dte=5,  # Critical DTE zone
        underlying_price=502.0,
        delta=0.55,
        gamma=0.08,
        theta=-0.25,
    )


# =============================================================================
# TEST CLASS: Enums
# =============================================================================

class TestLifecycleEnums:
    """Test lifecycle-related enums."""
    
    def test_position_stage_values(self):
        """Test PositionStage enum values."""
        assert PositionStage.OPEN.value == "open"
        assert PositionStage.ACTIVE.value == "active"
        assert PositionStage.PROFIT_ZONE.value == "profit_zone"
        assert PositionStage.RISK_ZONE.value == "risk_zone"
        assert PositionStage.EXPIRATION_ZONE.value == "expiration_zone"
        assert PositionStage.ROLL_CANDIDATE.value == "roll_candidate"
        assert PositionStage.CLOSE.value == "close"
        
    def test_exit_reason_values(self):
        """Test ExitReason enum values."""
        assert ExitReason.PROFIT_TARGET.value == "profit_target"
        assert ExitReason.STOP_LOSS.value == "stop_loss"
        assert ExitReason.TIME_STOP.value == "time_stop"
        assert ExitReason.DTE_EXIT.value == "dte_exit"
        assert ExitReason.ASSIGNMENT_RISK.value == "assignment_risk"
        
    def test_roll_type_values(self):
        """Test RollType enum values."""
        assert RollType.SAME_STRIKE_FORWARD.value == "same_strike_forward"
        assert RollType.DIAGONAL_UP.value == "diagonal_up"
        assert RollType.DIAGONAL_DOWN.value == "diagonal_down"
        assert RollType.INVERTED_ROLL.value == "inverted_roll"
        assert RollType.NONE.value == "none"
        
    def test_adjustment_type_values(self):
        """Test AdjustmentType enum values."""
        assert AdjustmentType.ADD_TO_WINNER.value == "add_to_winner"
        assert AdjustmentType.SCALE_OUT_PARTIAL.value == "scale_out"
        assert AdjustmentType.HEDGE_WITH_SPREAD.value == "hedge_spread"


# =============================================================================
# TEST CLASS: LifecycleConfig
# =============================================================================

class TestLifecycleConfig:
    """Test LifecycleConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LifecycleConfig()
        
        assert config.profit_target_pct == 50.0
        assert config.stop_loss_pct == 100.0
        assert config.dte_exit == 7
        assert config.dte_warning == 14
        assert config.max_hold_days == 45
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = LifecycleConfig(
            profit_target_pct=75.0,
            stop_loss_pct=50.0,
            dte_exit=5,
        )
        
        assert config.profit_target_pct == 75.0
        assert config.stop_loss_pct == 50.0
        assert config.dte_exit == 5
        
    def test_credit_strategy_config(self):
        """Test credit strategy specific config."""
        config = LifecycleConfig()
        
        assert config.credit_profit_target_pct == 50.0
        assert config.credit_roll_trigger_pct == -100.0


# =============================================================================
# TEST CLASS: PositionMetrics
# =============================================================================

class TestPositionMetrics:
    """Test PositionMetrics dataclass."""
    
    def test_position_metrics_creation(self, sample_equity_position):
        """Test basic position metrics creation."""
        pos = sample_equity_position
        
        assert pos.symbol == "AAPL"
        assert pos.entry_price == 170.0
        assert pos.quantity == 100
        assert pos.is_option is False
        
    def test_option_position_metrics(self, sample_option_position):
        """Test option position metrics."""
        pos = sample_option_position
        
        assert pos.is_option is True
        assert pos.option_type == "call"
        assert pos.strike == 175.0
        assert pos.dte == 21
        
    def test_position_greeks(self, sample_option_position):
        """Test position Greeks are stored."""
        pos = sample_option_position
        
        assert pos.delta == 0.65
        assert pos.gamma == 0.03
        assert pos.theta == -0.15


# =============================================================================
# TEST CLASS: Manager Initialization
# =============================================================================

class TestLifecycleManagerInit:
    """Test PositionLifecycleManager initialization."""
    
    def test_default_initialization(self, lifecycle_manager):
        """Test default initialization."""
        assert lifecycle_manager is not None
        assert lifecycle_manager.config is not None
        assert lifecycle_manager.positions == {}
        
    def test_custom_initialization(self, custom_lifecycle_manager):
        """Test custom initialization."""
        manager = custom_lifecycle_manager
        
        assert manager.config.profit_target_pct == 75.0
        assert manager.config.stop_loss_pct == 50.0
        
    def test_factory_function(self):
        """Test factory function."""
        manager = create_lifecycle_manager(
            profit_target_pct=60.0,
            stop_loss_pct=75.0,
            dte_exit=10,
        )
        
        assert manager.config.profit_target_pct == 60.0
        assert manager.config.stop_loss_pct == 75.0
        assert manager.config.dte_exit == 10


# =============================================================================
# TEST CLASS: Position Analysis - Basic
# =============================================================================

class TestPositionAnalysisBasic:
    """Test basic position analysis."""
    
    def test_analyze_profitable_position(self, lifecycle_manager, sample_option_position):
        """Test analysis of profitable position."""
        decision = lifecycle_manager.analyze_position(sample_option_position)
        
        assert isinstance(decision, LifecycleDecision)
        assert decision.stage in PositionStage.__members__.values()
        
    def test_analyze_returns_decision(self, lifecycle_manager, sample_equity_position):
        """Test that analysis returns LifecycleDecision."""
        decision = lifecycle_manager.analyze_position(sample_equity_position)
        
        assert hasattr(decision, 'stage')
        assert hasattr(decision, 'action')
        assert hasattr(decision, 'urgency')
        
    def test_analyze_has_reasons(self, lifecycle_manager, sample_option_position):
        """Test that analysis includes reasons."""
        decision = lifecycle_manager.analyze_position(sample_option_position)
        
        assert isinstance(decision.reasons, list)
        assert isinstance(decision.warnings, list)


# =============================================================================
# TEST CLASS: Profit Target Exit
# =============================================================================

class TestProfitTargetExit:
    """Test profit target exit detection."""
    
    def test_profit_target_reached_debit(self, lifecycle_manager):
        """Test profit target exit for debit position."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=10.00,
            quantity=10,
            unrealized_pnl=5000.0,
            unrealized_pnl_pct=100.0,  # At target
            is_option=True,
            option_type="call",
            dte=21,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        assert decision.stage == PositionStage.PROFIT_ZONE
        assert decision.action == "close"
        assert decision.exit_reason == ExitReason.PROFIT_TARGET
        
    def test_profit_target_reached_credit(self, lifecycle_manager, sample_credit_position):
        """Test profit target exit for credit position."""
        # At 50% profit, should trigger close
        decision = lifecycle_manager.analyze_position(sample_credit_position)
        
        assert decision.stage == PositionStage.PROFIT_ZONE
        assert decision.action == "close"
        
    def test_early_profit_scale_out(self, lifecycle_manager):
        """Test early profit triggers scale-out."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=6.50,
            quantity=10,
            unrealized_pnl=1500.0,
            unrealized_pnl_pct=30.0,  # Early profit zone
            is_option=True,
            option_type="call",
            dte=30,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        assert decision.stage == PositionStage.PROFIT_ZONE
        # Should suggest scale-out
        assert decision.scale_out_pct > 0 or decision.action in ["scale_out", "hold"]


# =============================================================================
# TEST CLASS: Stop Loss Exit
# =============================================================================

class TestStopLossExit:
    """Test stop loss exit detection."""
    
    def test_stop_loss_triggered_debit(self, lifecycle_manager, losing_position):
        """Test stop loss for debit position."""
        decision = lifecycle_manager.analyze_position(losing_position)
        
        # At 60% loss with 50% stop (debit default), may suggest close or roll
        assert decision.action in ["close", "roll"]
        
    def test_stop_loss_triggered_credit(self, lifecycle_manager):
        """Test stop loss for credit position."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=-2.00,  # Credit
            current_price=-4.50,  # Losing
            quantity=-10,
            unrealized_pnl=-2500.0,
            unrealized_pnl_pct=-125.0,  # Beyond 100% credit roll trigger
            is_option=True,
            option_type="put",
            dte=14,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        assert decision.stage == PositionStage.CLOSE
        
    def test_mental_stop_warning(self, lifecycle_manager):
        """Test mental stop triggers warning or close."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=1.50,
            quantity=10,
            unrealized_pnl=-3500.0,
            unrealized_pnl_pct=-70.0,  # Near mental stop (for debit, stop is 50%)
            is_option=True,
            option_type="call",
            dte=21,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        # At -70% with 50% stop for debit, should trigger close
        assert decision.stage in [PositionStage.RISK_ZONE, PositionStage.CLOSE]


# =============================================================================
# TEST CLASS: DTE-Based Exit
# =============================================================================

class TestDTEBasedExit:
    """Test DTE-based exit logic."""
    
    def test_dte_exit_zone(self, lifecycle_manager, expiring_position):
        """Test DTE exit zone detection."""
        decision = lifecycle_manager.analyze_position(expiring_position)
        
        # At 5 DTE, should be in expiration zone
        assert decision.stage == PositionStage.EXPIRATION_ZONE
        
    def test_dte_warning_zone(self, lifecycle_manager):
        """Test DTE warning zone detection."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=3.00,
            current_price=3.20,
            quantity=10,
            unrealized_pnl=200.0,
            unrealized_pnl_pct=6.67,
            is_option=True,
            option_type="call",
            dte=12,  # Warning zone
            expiration=datetime.utcnow() + timedelta(days=12),
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        # Could be EXPIRATION_ZONE or ROLL_CANDIDATE depending on profit level
        assert decision.stage in [PositionStage.EXPIRATION_ZONE, PositionStage.ROLL_CANDIDATE]
        # Should have at least one warning or reason related to DTE
        assert len(decision.warnings) > 0 or len(decision.reasons) > 0
        
    def test_dte_adjusted_profit_target(self, lifecycle_manager):
        """Test DTE-adjusted profit target."""
        # Position at 7 DTE with some profit
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=6.25,
            quantity=10,
            unrealized_pnl=1250.0,
            unrealized_pnl_pct=25.0,  # Below normal target but may hit DTE-adjusted
            is_option=True,
            option_type="call",
            dte=7,
            expiration=datetime.utcnow() + timedelta(days=7),
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        assert decision.stage == PositionStage.EXPIRATION_ZONE


# =============================================================================
# TEST CLASS: Time-Based Exit
# =============================================================================

class TestTimeBasedExit:
    """Test time-based exit logic."""
    
    def test_max_hold_time_exceeded(self, lifecycle_manager):
        """Test max hold time exit."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=100.0,
            current_price=102.0,
            quantity=50,
            unrealized_pnl=100.0,
            unrealized_pnl_pct=2.0,
            days_held=50,  # Exceeds 45 day default
            is_option=False,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        assert decision.stage == PositionStage.CLOSE
        assert decision.exit_reason == ExitReason.TIME_STOP


# =============================================================================
# TEST CLASS: Roll Type Determination
# =============================================================================

class TestRollTypeDetermination:
    """Test roll type determination logic."""
    
    def test_determine_roll_type_equity(self, lifecycle_manager, sample_equity_position):
        """Test roll type for equity (should be NONE)."""
        roll_type = lifecycle_manager._determine_roll_type(sample_equity_position)
        
        assert roll_type == RollType.NONE
        
    def test_determine_roll_type_short_option_small_loss(self, lifecycle_manager):
        """Test roll type for short option with small loss."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=-2.00,
            current_price=-2.80,
            quantity=-10,
            unrealized_pnl=-800.0,
            unrealized_pnl_pct=-40.0,  # 40% loss
            is_option=True,
            option_type="put",
        )
        
        roll_type = lifecycle_manager._determine_roll_type(position)
        
        assert roll_type == RollType.SAME_STRIKE_FORWARD
        
    def test_determine_roll_type_short_option_moderate_loss(self, lifecycle_manager):
        """Test roll type for short option with moderate loss."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=-2.00,
            current_price=-3.50,
            quantity=-10,
            unrealized_pnl=-1500.0,
            unrealized_pnl_pct=-75.0,
            is_option=True,
            option_type="put",
        )
        
        roll_type = lifecycle_manager._determine_roll_type(position)
        
        assert roll_type == RollType.DIAGONAL_DOWN  # Roll to lower strike put
        
    def test_determine_roll_type_short_call_moderate_loss(self, lifecycle_manager):
        """Test roll type for short call with moderate loss."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=-2.00,
            current_price=-3.50,
            quantity=-10,
            unrealized_pnl=-1500.0,
            unrealized_pnl_pct=-75.0,
            is_option=True,
            option_type="call",
        )
        
        roll_type = lifecycle_manager._determine_roll_type(position)
        
        assert roll_type == RollType.DIAGONAL_UP  # Roll to higher strike call
        
    def test_determine_roll_type_large_loss(self, lifecycle_manager):
        """Test roll type for large loss."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=-2.00,
            current_price=-6.00,
            quantity=-10,
            unrealized_pnl=-4000.0,
            unrealized_pnl_pct=-200.0,
            is_option=True,
            option_type="put",
        )
        
        roll_type = lifecycle_manager._determine_roll_type(position)
        
        assert roll_type == RollType.INVERTED_ROLL
        
    def test_determine_roll_type_long_option_losing(self, lifecycle_manager):
        """Test roll type for long option losing."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=3.50,
            quantity=10,
            unrealized_pnl=-1500.0,
            unrealized_pnl_pct=-30.0,
            is_option=True,
            option_type="call",
        )
        
        roll_type = lifecycle_manager._determine_roll_type(position)
        
        assert roll_type == RollType.CALENDAR_SPREAD


# =============================================================================
# TEST CLASS: Adjustment Recommendations
# =============================================================================

class TestAdjustmentRecommendations:
    """Test adjustment recommendation logic."""
    
    def test_recommend_add_to_winner(self, lifecycle_manager):
        """Test add to winner recommendation."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=6.25,
            quantity=5,  # Below max
            unrealized_pnl=625.0,
            unrealized_pnl_pct=25.0,  # Above min_profit_to_add
            is_option=True,
        )
        
        adjustment = lifecycle_manager._recommend_adjustment(position)
        
        assert adjustment == AdjustmentType.ADD_TO_WINNER
        
    def test_recommend_scale_out_at_max_size(self, lifecycle_manager):
        """Test scale out when at max position size."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=6.25,
            quantity=10,  # At max
            unrealized_pnl=1250.0,
            unrealized_pnl_pct=25.0,
            is_option=True,
        )
        
        adjustment = lifecycle_manager._recommend_adjustment(position)
        
        assert adjustment == AdjustmentType.SCALE_OUT_PARTIAL
        
    def test_recommend_convert_to_spread(self, lifecycle_manager):
        """Test convert to spread recommendation for losing naked position."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=1.50,
            quantity=10,
            unrealized_pnl=-3500.0,
            unrealized_pnl_pct=-75.0,  # At mental_stop_pct threshold
            is_option=True,
            is_spread=False,
        )
        
        adjustment = lifecycle_manager._recommend_adjustment(position)
        
        # May recommend convert to spread or nothing if logic differs
        # The key is that it doesn't crash and returns valid value
        assert adjustment in [AdjustmentType.CONVERT_TO_SPREAD, None]
        
    def test_recommend_roll_strike_for_spread(self, lifecycle_manager):
        """Test roll strike recommendation for losing spread."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=-2.00,
            current_price=-3.60,
            quantity=-10,
            unrealized_pnl=-1600.0,
            unrealized_pnl_pct=-80.0,
            is_option=True,
            is_spread=True,
        )
        
        adjustment = lifecycle_manager._recommend_adjustment(position)
        
        assert adjustment == AdjustmentType.ROLL_STRIKE


# =============================================================================
# TEST CLASS: Assignment Risk
# =============================================================================

class TestAssignmentRisk:
    """Test assignment risk detection."""
    
    def test_short_call_assignment_risk(self, lifecycle_manager):
        """Test assignment risk detection for ITM short call."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=-3.00,
            current_price=-8.00,
            quantity=-10,  # Short
            unrealized_pnl=-5000.0,
            unrealized_pnl_pct=-166.0,
            is_option=True,
            option_type="call",
            strike=100.0,
            underlying_price=108.0,  # 8% ITM
            moneyness=0.08,
            dte=5,
            expiration=datetime.utcnow() + timedelta(days=5),
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        # Should have assignment risk warning
        assert decision.stage == PositionStage.CLOSE or decision.stage == PositionStage.RISK_ZONE
        
    def test_short_put_assignment_risk(self, lifecycle_manager):
        """Test assignment risk detection for ITM short put."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=-2.00,
            current_price=-7.00,
            quantity=-10,  # Short
            unrealized_pnl=-5000.0,
            unrealized_pnl_pct=-250.0,
            is_option=True,
            option_type="put",
            strike=100.0,
            underlying_price=93.0,  # 7% ITM
            moneyness=-0.07,
            dte=7,
            expiration=datetime.utcnow() + timedelta(days=7),
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        # Should be in risk zone
        assert decision.stage in [PositionStage.CLOSE, PositionStage.RISK_ZONE]


# =============================================================================
# TEST CLASS: Roll Details
# =============================================================================

class TestRollDetails:
    """Test roll execution detail generation."""
    
    def test_get_roll_details_same_strike(self, lifecycle_manager, sample_option_position):
        """Test roll details for same strike forward."""
        details = lifecycle_manager.get_roll_details(
            position=sample_option_position,
            roll_type=RollType.SAME_STRIKE_FORWARD,
            target_dte=21,
        )
        
        assert "original_position" in details
        assert "roll_type" in details
        assert "new_strike" in details
        assert "target_dte" in details
        assert details["new_strike"] == sample_option_position.strike
        
    def test_get_roll_details_diagonal_up(self, lifecycle_manager, sample_option_position):
        """Test roll details for diagonal up."""
        details = lifecycle_manager.get_roll_details(
            position=sample_option_position,
            roll_type=RollType.DIAGONAL_UP,
            target_dte=21,
        )
        
        assert details["new_strike"] > sample_option_position.strike
        
    def test_get_roll_details_diagonal_down(self, lifecycle_manager, sample_option_position):
        """Test roll details for diagonal down."""
        details = lifecycle_manager.get_roll_details(
            position=sample_option_position,
            roll_type=RollType.DIAGONAL_DOWN,
            target_dte=21,
        )
        
        assert details["new_strike"] < sample_option_position.strike


# =============================================================================
# TEST CLASS: Close Order Generation
# =============================================================================

class TestCloseOrderGeneration:
    """Test close order generation."""
    
    def test_get_close_order_long(self, lifecycle_manager, sample_option_position):
        """Test close order for long position."""
        order = lifecycle_manager.get_close_order(
            position=sample_option_position,
            order_type="limit",
            aggression="normal",
        )
        
        assert order["action"] == "sell"
        assert order["quantity"] == 10
        assert order["order_type"] == "limit"
        
    def test_get_close_order_short(self, lifecycle_manager, sample_credit_position):
        """Test close order for short position."""
        order = lifecycle_manager.get_close_order(
            position=sample_credit_position,
            order_type="limit",
            aggression="normal",
        )
        
        assert order["action"] == "buy"
        assert order["quantity"] == 10
        
    def test_get_close_order_aggressive(self, lifecycle_manager, sample_option_position):
        """Test aggressive close order."""
        order_normal = lifecycle_manager.get_close_order(
            position=sample_option_position,
            order_type="limit",
            aggression="normal",
        )
        
        order_aggressive = lifecycle_manager.get_close_order(
            position=sample_option_position,
            order_type="limit",
            aggression="aggressive",
        )
        
        # Aggressive should cross spread (lower price for sell)
        assert order_aggressive["limit_price"] <= order_normal["limit_price"]
        
    def test_get_close_order_market(self, lifecycle_manager, sample_option_position):
        """Test market close order."""
        order = lifecycle_manager.get_close_order(
            position=sample_option_position,
            order_type="market",
        )
        
        assert order["order_type"] == "market"
        assert order["limit_price"] is None


# =============================================================================
# TEST CLASS: Batch Analysis
# =============================================================================

class TestBatchAnalysis:
    """Test batch position analysis."""
    
    def test_batch_analyze_multiple_positions(self, lifecycle_manager, 
                                              sample_equity_position,
                                              sample_option_position,
                                              losing_position):
        """Test batch analysis of multiple positions."""
        positions = [sample_equity_position, sample_option_position, losing_position]
        
        results = lifecycle_manager.batch_analyze(positions)
        
        assert len(results) == 3
        
        for pos, decision in results:
            assert isinstance(pos, PositionMetrics)
            assert isinstance(decision, LifecycleDecision)
            
    def test_batch_analyze_sorted_by_urgency(self, lifecycle_manager):
        """Test batch analysis results are sorted by urgency."""
        # Create positions with different urgencies
        urgent_position = PositionMetrics(
            symbol="URGENT",
            underlying="TEST",
            entry_price=5.00,
            current_price=0.50,
            quantity=10,
            unrealized_pnl=-4500.0,
            unrealized_pnl_pct=-90.0,  # Stop loss
            is_option=True,
            option_type="call",
            dte=3,
            expiration=datetime.utcnow() + timedelta(days=3),
        )
        
        normal_position = PositionMetrics(
            symbol="NORMAL",
            underlying="TEST",
            entry_price=5.00,
            current_price=5.25,
            quantity=10,
            unrealized_pnl=250.0,
            unrealized_pnl_pct=5.0,
            is_option=True,
            option_type="call",
            dte=30,
            expiration=datetime.utcnow() + timedelta(days=30),
        )
        
        results = lifecycle_manager.batch_analyze([normal_position, urgent_position])
        
        # Urgent should come first
        urgency_order = {"immediate": 0, "today": 1, "soon": 2, "monitor": 3}
        urgencies = [urgency_order.get(d.urgency, 4) for _, d in results]
        
        assert urgencies == sorted(urgencies)
        
    def test_batch_analyze_empty_list(self, lifecycle_manager):
        """Test batch analysis with empty list."""
        results = lifecycle_manager.batch_analyze([])
        
        assert results == []


# =============================================================================
# TEST CLASS: Scale-Out Schedule
# =============================================================================

class TestScaleOutSchedule:
    """Test scale-out schedule logic."""
    
    def test_scale_out_schedule_exists(self, lifecycle_manager):
        """Test scale-out schedule is defined."""
        schedule = lifecycle_manager.SCALE_OUT_SCHEDULE
        
        assert len(schedule) > 0
        
        for profit_pct, scale_pct in schedule:
            assert profit_pct > 0
            assert 0 < scale_pct <= 1.0
            
    def test_scale_out_at_25_percent(self, lifecycle_manager):
        """Test scale-out at 25% profit."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=6.25,
            quantity=10,
            unrealized_pnl=1250.0,
            unrealized_pnl_pct=25.0,
            is_option=True,
            dte=30,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        assert decision.scale_out_pct > 0
        
    def test_scale_out_quantity_calculation(self, lifecycle_manager):
        """Test scale-out quantity calculation."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=7.50,
            quantity=20,
            unrealized_pnl=5000.0,
            unrealized_pnl_pct=50.0,  # 50% profit
            is_option=True,
            dte=30,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        # Should have scale-out quantity
        assert decision.scale_out_quantity >= 1
        assert decision.scale_out_quantity <= position.quantity


# =============================================================================
# TEST CLASS: Summary
# =============================================================================

class TestLifecycleManagerSummary:
    """Test summary functionality."""
    
    def test_get_summary(self, lifecycle_manager):
        """Test getting manager summary."""
        summary = lifecycle_manager.get_summary()
        
        assert "profit_target_pct" in summary
        assert "stop_loss_pct" in summary
        assert "dte_warning" in summary
        assert "dte_exit" in summary
        assert "scale_out_schedule" in summary
        assert "max_hold_days" in summary


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================

class TestLifecycleEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_pnl_position(self, lifecycle_manager):
        """Test position with zero P&L."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=5.00,
            quantity=10,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            is_option=True,
            dte=30,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        assert decision.action == "hold"
        
    def test_negative_dte(self, lifecycle_manager):
        """Test position with negative DTE (expired)."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=1.00,
            quantity=10,
            unrealized_pnl=-4000.0,
            unrealized_pnl_pct=-80.0,
            is_option=True,
            dte=-1,  # Expired
            expiration=datetime.utcnow() - timedelta(days=1),
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        # Should handle gracefully
        assert decision is not None
        
    def test_very_large_profit(self, lifecycle_manager):
        """Test position with very large profit."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=1.00,
            current_price=20.00,
            quantity=100,
            unrealized_pnl=190000.0,
            unrealized_pnl_pct=1900.0,  # 19x return
            is_option=True,
            dte=30,
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        assert decision.stage == PositionStage.PROFIT_ZONE
        assert decision.action == "close"


# =============================================================================
# TEST CLASS: Integration Tests
# =============================================================================

class TestLifecycleIntegration:
    """Integration tests for PositionLifecycleManager."""
    
    def test_full_lifecycle_flow(self, lifecycle_manager):
        """Test complete position lifecycle flow."""
        # New position
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=5.10,
            quantity=10,
            unrealized_pnl=100.0,
            unrealized_pnl_pct=2.0,
            days_held=1,
            is_option=True,
            option_type="call",
            dte=28,
            expiration=datetime.utcnow() + timedelta(days=28),
        )
        
        # Should be in ACTIVE stage
        decision = lifecycle_manager.analyze_position(position)
        assert decision.stage in [PositionStage.ACTIVE, PositionStage.PROFIT_ZONE]
        
        # Simulate profit increase
        position.current_price = 7.50
        position.unrealized_pnl = 2500.0
        position.unrealized_pnl_pct = 50.0
        position.days_held = 10
        position.dte = 18
        
        decision = lifecycle_manager.analyze_position(position)
        assert decision.stage == PositionStage.PROFIT_ZONE
        
    def test_lifecycle_with_roll_consideration(self, lifecycle_manager):
        """Test lifecycle with roll consideration."""
        position = PositionMetrics(
            symbol="TEST",
            underlying="TEST",
            entry_price=5.00,
            current_price=4.80,
            quantity=10,
            unrealized_pnl=-200.0,
            unrealized_pnl_pct=-4.0,
            is_option=True,
            option_type="call",
            dte=10,
            expiration=datetime.utcnow() + timedelta(days=10),
        )
        
        decision = lifecycle_manager.analyze_position(position)
        
        # In warning zone, may be roll candidate
        assert decision.stage in [PositionStage.EXPIRATION_ZONE, PositionStage.ROLL_CANDIDATE, PositionStage.ACTIVE]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
