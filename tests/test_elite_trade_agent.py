"""
Comprehensive Unit Tests for EliteTradeAgent

Tests cover:
- Initialization and configuration
- Multi-timeframe signal aggregation
- Strategy selection based on IV environment
- Kelly Criterion position sizing
- Dynamic profit thresholds
- Market context building
- Trade proposal generation
- Validation logic
- Integration with risk managers
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Import the module under test
from trade.elite_trade_agent import (
    EliteTradeAgent,
    MarketRegime,
    IVEnvironment,
    Timeframe,
    OptionStrategy,
    RiskParameters,
    TimeframeConfig,
    LiquidityRequirements,
    DynamicProfitThresholds,
    MarketContext,
    TradeProposal,
)
from schemas.core_schemas import (
    DirectionEnum,
    PipelineResult,
    HedgeSnapshot,
    SentimentSnapshot,
    ElasticitySnapshot,
    LiquiditySnapshot,
    StrategyType,
    TradeIdea,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def elite_agent():
    """Create a basic EliteTradeAgent for testing."""
    return EliteTradeAgent(
        options_adapter=None,
        market_adapter=None,
        broker=None,
        config={"min_confidence": 0.25},
    )


@pytest.fixture
def elite_agent_with_config():
    """Create an EliteTradeAgent with custom config."""
    config = {
        "kelly_fraction": 0.30,
        "min_reward_risk": 1.25,
        "min_confidence": 0.20,
        "min_open_interest": 50,
        "min_volume": 5,
        "max_spread_pct": 0.15,
    }
    return EliteTradeAgent(config=config)


@pytest.fixture
def sample_hedge_snapshot():
    """Create a sample HedgeSnapshot."""
    return HedgeSnapshot(
        timestamp=datetime.utcnow(),
        symbol="AAPL",
        movement_energy=65.0,
        energy_asymmetry=0.4,
        resistance_levels=[180.0, 185.0],
        support_levels=[170.0, 165.0],
        data={"test": "data"},
    )


@pytest.fixture
def sample_sentiment_snapshot():
    """Create a sample SentimentSnapshot."""
    return SentimentSnapshot(
        timestamp=datetime.utcnow(),
        symbol="AAPL",
        news_sentiment=0.3,
        social_sentiment=0.2,
        flow_sentiment=0.5,
        technical_sentiment=0.6,
        composite_score=0.4,
        mtf_score=0.5,
    )


@pytest.fixture
def sample_elasticity_snapshot():
    """Create a sample ElasticitySnapshot."""
    return ElasticitySnapshot(
        timestamp=datetime.utcnow(),
        symbol="AAPL",
        volatility=0.25,
        trend_strength=0.6,
        elasticity_coefficient=1.2,
        momentum=0.3,
    )


@pytest.fixture
def sample_pipeline_result(sample_hedge_snapshot, sample_sentiment_snapshot, sample_elasticity_snapshot):
    """Create a sample PipelineResult for testing."""
    return PipelineResult(
        timestamp=datetime.utcnow(),
        symbol="AAPL",
        hedge_snapshot=sample_hedge_snapshot,
        sentiment_snapshot=sample_sentiment_snapshot,
        elasticity_snapshot=sample_elasticity_snapshot,
        liquidity_snapshot=None,
        consensus={
            "direction": "long",
            "confidence": 0.65,
            "composite_score": 0.55,
        },
    )


@pytest.fixture
def sample_market_context():
    """Create a sample MarketContext."""
    return MarketContext(
        symbol="AAPL",
        spot_price=175.0,
        iv_rank=45.0,
        iv_percentile=50.0,
        historical_vol=0.22,
        implied_vol=0.25,
        vol_skew=0.05,
        put_call_ratio=0.95,
        regime=MarketRegime.RANGE_BOUND,
        iv_environment=IVEnvironment.MEDIUM,
        atr=3.5,
        atr_pct=0.02,
        trend_strength=0.5,
        momentum=0.3,
        support_level=170.0,
        resistance_level=180.0,
    )


# =============================================================================
# TEST CLASS: Initialization and Configuration
# =============================================================================

class TestEliteTradeAgentInit:
    """Test initialization and configuration."""
    
    def test_default_initialization(self, elite_agent):
        """Test default initialization."""
        assert elite_agent is not None
        assert elite_agent.risk_params is not None
        assert elite_agent.liquidity_reqs is not None
        assert elite_agent.portfolio_value > 0
        
    def test_risk_parameters_defaults(self, elite_agent):
        """Test default risk parameters."""
        rp = elite_agent.risk_params
        assert rp.max_position_pct == 0.04  # 4%
        assert rp.max_portfolio_heat == 0.20  # 20%
        assert rp.kelly_fraction == 0.25  # 25%
        assert rp.min_reward_risk == 1.5  # 1.5:1
        
    def test_custom_config(self, elite_agent_with_config):
        """Test custom configuration."""
        agent = elite_agent_with_config
        assert agent.risk_params.kelly_fraction == 0.30
        assert agent.risk_params.min_reward_risk == 1.25
        
    def test_liquidity_requirements(self, elite_agent_with_config):
        """Test liquidity requirements from config."""
        lr = elite_agent_with_config.liquidity_reqs
        assert lr.min_open_interest == 50
        assert lr.min_volume == 5
        assert lr.max_spread_pct == 0.15
        
    def test_timeframe_configs(self, elite_agent):
        """Test timeframe configurations exist."""
        configs = elite_agent.TIMEFRAME_CONFIGS
        assert Timeframe.SCALP in configs
        assert Timeframe.INTRADAY in configs
        assert Timeframe.SWING in configs
        assert Timeframe.POSITION in configs
        
    def test_strategy_matrix(self, elite_agent):
        """Test strategy matrix exists."""
        matrix = elite_agent.STRATEGY_MATRIX
        assert ("bullish", IVEnvironment.HIGH) in matrix
        assert ("bearish", IVEnvironment.LOW) in matrix
        assert ("neutral", IVEnvironment.MEDIUM) in matrix


# =============================================================================
# TEST CLASS: Enums and Data Classes
# =============================================================================

class TestEnumsAndDataClasses:
    """Test enums and data classes."""
    
    def test_market_regime_enum(self):
        """Test MarketRegime enum values."""
        assert MarketRegime.TRENDING_BULL.value == "trending_bull"
        assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"
        assert MarketRegime.BREAKOUT.value == "breakout"
        
    def test_iv_environment_enum(self):
        """Test IVEnvironment enum values."""
        assert IVEnvironment.HIGH.value == "high"
        assert IVEnvironment.MEDIUM.value == "medium"
        assert IVEnvironment.LOW.value == "low"
        
    def test_timeframe_enum(self):
        """Test Timeframe enum values."""
        assert Timeframe.SCALP.value == "scalp"
        assert Timeframe.SWING.value == "swing"
        
    def test_option_strategy_enum(self):
        """Test OptionStrategy enum values."""
        assert OptionStrategy.LONG_CALL.value == "long_call"
        assert OptionStrategy.IRON_CONDOR.value == "iron_condor"
        assert OptionStrategy.EQUITY_LONG.value == "equity_long"
        
    def test_risk_parameters_dataclass(self):
        """Test RiskParameters dataclass."""
        rp = RiskParameters()
        assert rp.max_position_pct == 0.04
        assert rp.stop_loss_atr_multiple == 2.0
        
        custom_rp = RiskParameters(max_position_pct=0.05, kelly_fraction=0.30)
        assert custom_rp.max_position_pct == 0.05
        assert custom_rp.kelly_fraction == 0.30
        
    def test_timeframe_config_dataclass(self):
        """Test TimeframeConfig dataclass."""
        tc = TimeframeConfig(
            min_dte=7,
            max_dte=30,
            stop_loss_pct=0.03,
            take_profit_pct=0.09,
            max_hold_hours=120,
            position_size_mult=1.0,
        )
        assert tc.min_dte == 7
        assert tc.max_hold_hours == 120
        
    def test_market_context_dataclass(self, sample_market_context):
        """Test MarketContext dataclass."""
        mc = sample_market_context
        assert mc.symbol == "AAPL"
        assert mc.spot_price == 175.0
        assert mc.iv_environment == IVEnvironment.MEDIUM


# =============================================================================
# TEST CLASS: Multi-Timeframe Signal Aggregation
# =============================================================================

class TestMultiTimeframeSignals:
    """Test multi-timeframe signal aggregation."""
    
    def test_aggregate_signals_basic(self, elite_agent, sample_pipeline_result):
        """Test basic signal aggregation."""
        result = elite_agent._aggregate_multi_timeframe_signals(sample_pipeline_result)
        
        assert "signals" in result
        assert "direction" in result
        assert "alignment" in result
        assert "confidence" in result
        
    def test_aggregate_signals_has_all_timeframes(self, elite_agent, sample_pipeline_result):
        """Test that all timeframes are represented."""
        result = elite_agent._aggregate_multi_timeframe_signals(sample_pipeline_result)
        
        signals = result["signals"]
        assert "scalp" in signals
        assert "intraday" in signals
        assert "swing" in signals
        assert "position" in signals
        
    def test_aggregate_signals_weights(self, elite_agent, sample_pipeline_result):
        """Test that signal weights are correct."""
        result = elite_agent._aggregate_multi_timeframe_signals(sample_pipeline_result)
        
        signals = result["signals"]
        total_weight = sum(s["weight"] for s in signals.values())
        assert abs(total_weight - 1.0) < 0.01  # Weights should sum to ~1.0
        
    def test_aggregate_signals_alignment(self, elite_agent, sample_pipeline_result):
        """Test alignment calculation."""
        result = elite_agent._aggregate_multi_timeframe_signals(sample_pipeline_result)
        
        alignment = result["alignment"]
        assert 0.0 <= alignment <= 1.0
        
    def test_aggregate_signals_direction_scores(self, elite_agent, sample_pipeline_result):
        """Test direction scores are present."""
        result = elite_agent._aggregate_multi_timeframe_signals(sample_pipeline_result)
        
        scores = result["direction_scores"]
        assert "long" in scores
        assert "short" in scores
        assert "neutral" in scores
        
    def test_aggregate_signals_with_no_data(self, elite_agent):
        """Test aggregation with minimal data."""
        minimal_result = PipelineResult(
            timestamp=datetime.utcnow(),
            symbol="TEST",
            consensus={"direction": "neutral", "confidence": 0.3},
        )
        
        result = elite_agent._aggregate_multi_timeframe_signals(minimal_result)
        assert result["direction"] in ["long", "short", "neutral"]


# =============================================================================
# TEST CLASS: Strategy Selection
# =============================================================================

class TestStrategySelection:
    """Test strategy selection logic."""
    
    def test_select_strategy_bullish_high_iv(self, elite_agent, sample_market_context):
        """Test strategy selection for bullish signal in high IV."""
        sample_market_context.iv_environment = IVEnvironment.HIGH
        
        strategy = elite_agent._select_strategy(
            direction="long",
            context=sample_market_context,
            confidence=0.65,
            timeframe=Timeframe.SWING,
        )
        
        assert strategy == OptionStrategy.BULL_PUT_SPREAD
        
    def test_select_strategy_bullish_low_iv(self, elite_agent, sample_market_context):
        """Test strategy selection for bullish signal in low IV."""
        sample_market_context.iv_environment = IVEnvironment.LOW
        
        strategy = elite_agent._select_strategy(
            direction="long",
            context=sample_market_context,
            confidence=0.65,
            timeframe=Timeframe.SWING,
        )
        
        assert strategy in [OptionStrategy.LONG_CALL, OptionStrategy.BULL_CALL_SPREAD]
        
    def test_select_strategy_bearish_high_iv(self, elite_agent, sample_market_context):
        """Test strategy selection for bearish signal in high IV."""
        sample_market_context.iv_environment = IVEnvironment.HIGH
        
        strategy = elite_agent._select_strategy(
            direction="short",
            context=sample_market_context,
            confidence=0.65,
            timeframe=Timeframe.SWING,
        )
        
        assert strategy == OptionStrategy.BEAR_CALL_SPREAD
        
    def test_select_strategy_neutral_high_iv(self, elite_agent, sample_market_context):
        """Test strategy selection for neutral signal in high IV."""
        sample_market_context.iv_environment = IVEnvironment.HIGH
        
        strategy = elite_agent._select_strategy(
            direction="neutral",
            context=sample_market_context,
            confidence=0.65,
            timeframe=Timeframe.SWING,
        )
        
        assert strategy in [
            OptionStrategy.IRON_CONDOR,
            OptionStrategy.IRON_BUTTERFLY,
            OptionStrategy.SHORT_STRANGLE,
        ]
        
    def test_select_strategy_high_confidence(self, elite_agent, sample_market_context):
        """Test strategy selection with very high confidence."""
        sample_market_context.iv_environment = IVEnvironment.LOW
        
        strategy = elite_agent._select_strategy(
            direction="long",
            context=sample_market_context,
            confidence=0.90,  # Very high confidence
            timeframe=Timeframe.SWING,
        )
        
        # High confidence should prefer outright options
        assert strategy == OptionStrategy.LONG_CALL
        
    def test_select_timeframe_high_volatility(self, elite_agent, sample_pipeline_result):
        """Test timeframe selection with high volatility."""
        sample_pipeline_result.elasticity_snapshot.volatility = 0.50
        
        timeframe = elite_agent._select_timeframe(sample_pipeline_result, confidence=0.5)
        
        assert timeframe in [Timeframe.SCALP, Timeframe.INTRADAY]
        
    def test_select_timeframe_low_volatility(self, elite_agent, sample_pipeline_result):
        """Test timeframe selection with low volatility."""
        sample_pipeline_result.elasticity_snapshot.volatility = 0.10
        
        timeframe = elite_agent._select_timeframe(sample_pipeline_result, confidence=0.85)
        
        assert timeframe in [Timeframe.SWING, Timeframe.POSITION]


# =============================================================================
# TEST CLASS: Dynamic Profit Thresholds
# =============================================================================

class TestDynamicProfitThresholds:
    """Test dynamic profit threshold calculation."""
    
    def test_calculate_thresholds_credit_strategy(self, elite_agent, sample_market_context):
        """Test thresholds for credit strategy."""
        thresholds = elite_agent._calculate_dynamic_profit_thresholds(
            strategy=OptionStrategy.BULL_PUT_SPREAD,
            context=sample_market_context,
            timeframe=Timeframe.SWING,
            confidence=0.65,
            target_dte=21,
        )
        
        assert thresholds.target_profit_pct > 0
        assert thresholds.stop_loss_pct > 0
        assert thresholds.early_profit_pct > 0
        
    def test_calculate_thresholds_debit_strategy(self, elite_agent, sample_market_context):
        """Test thresholds for debit strategy."""
        thresholds = elite_agent._calculate_dynamic_profit_thresholds(
            strategy=OptionStrategy.LONG_CALL,
            context=sample_market_context,
            timeframe=Timeframe.SWING,
            confidence=0.65,
            target_dte=21,
        )
        
        # Debit strategies should have higher targets
        assert thresholds.target_profit_pct >= 0.5  # At least 50%
        
    def test_calculate_thresholds_high_iv_adjustment(self, elite_agent, sample_market_context):
        """Test that high IV adjusts thresholds appropriately."""
        sample_market_context.iv_environment = IVEnvironment.HIGH
        
        thresholds = elite_agent._calculate_dynamic_profit_thresholds(
            strategy=OptionStrategy.IRON_CONDOR,
            context=sample_market_context,
            timeframe=Timeframe.SWING,
            confidence=0.65,
            target_dte=21,
        )
        
        # High IV should have faster profit taking (lower multiplier)
        assert thresholds.trailing_distance_pct > 0
        
    def test_calculate_thresholds_dte_acceleration(self, elite_agent, sample_market_context):
        """Test DTE acceleration factor."""
        # Short DTE should have higher acceleration
        thresholds_short = elite_agent._calculate_dynamic_profit_thresholds(
            strategy=OptionStrategy.BULL_PUT_SPREAD,
            context=sample_market_context,
            timeframe=Timeframe.INTRADAY,
            confidence=0.65,
            target_dte=5,  # Short DTE
        )
        
        thresholds_long = elite_agent._calculate_dynamic_profit_thresholds(
            strategy=OptionStrategy.BULL_PUT_SPREAD,
            context=sample_market_context,
            timeframe=Timeframe.POSITION,
            confidence=0.65,
            target_dte=45,  # Long DTE
        )
        
        assert thresholds_short.dte_acceleration_factor > thresholds_long.dte_acceleration_factor
        
    def test_get_strategy_category(self, elite_agent):
        """Test strategy categorization."""
        assert elite_agent._get_strategy_category(OptionStrategy.BULL_PUT_SPREAD) == "credit"
        assert elite_agent._get_strategy_category(OptionStrategy.LONG_CALL) == "debit"
        assert elite_agent._get_strategy_category(OptionStrategy.LONG_STRADDLE) == "neutral"
        assert elite_agent._get_strategy_category(OptionStrategy.EQUITY_LONG) == "equity"
        
    def test_get_dte_acceleration(self, elite_agent):
        """Test DTE acceleration factors."""
        assert elite_agent._get_dte_acceleration(5) == 2.0   # Critical zone
        assert elite_agent._get_dte_acceleration(10) == 1.5  # Elevated
        assert elite_agent._get_dte_acceleration(20) == 1.0  # Normal
        assert elite_agent._get_dte_acceleration(45) == 0.8  # Relaxed
        assert elite_agent._get_dte_acceleration(75) == 0.6  # Very relaxed
        
    def test_calculate_scale_out_levels(self, elite_agent):
        """Test scale-out level calculation."""
        # Credit strategy
        levels = elite_agent._calculate_scale_out_levels(
            strategy_category="credit",
            target_profit_pct=0.50,
            iv_environment=IVEnvironment.HIGH,
        )
        
        assert len(levels) > 0
        assert all(isinstance(l, tuple) for l in levels)
        assert all(len(l) == 2 for l in levels)


# =============================================================================
# TEST CLASS: Market Context Building
# =============================================================================

class TestMarketContextBuilding:
    """Test market context building."""
    
    def test_build_market_context(self, elite_agent, sample_pipeline_result):
        """Test building market context from pipeline result."""
        # Mock get_spot_price
        elite_agent._get_spot_price = Mock(return_value=175.0)
        
        context = elite_agent._build_market_context(sample_pipeline_result)
        
        assert context is not None
        assert context.symbol == "AAPL"
        assert context.spot_price == 175.0
        assert context.iv_environment in [IVEnvironment.HIGH, IVEnvironment.MEDIUM, IVEnvironment.LOW]
        
    def test_build_market_context_no_price(self, elite_agent, sample_pipeline_result):
        """Test building market context with no price."""
        elite_agent._get_spot_price = Mock(return_value=None)
        
        context = elite_agent._build_market_context(sample_pipeline_result)
        
        assert context is None
        
    def test_classify_regime_high_volatility(self, elite_agent, sample_hedge_snapshot, sample_elasticity_snapshot):
        """Test regime classification for high volatility."""
        sample_elasticity_snapshot.volatility = 0.50
        
        regime = elite_agent._classify_regime(
            hedge=sample_hedge_snapshot,
            elasticity=sample_elasticity_snapshot,
            sentiment=None,
        )
        
        assert regime == MarketRegime.HIGH_VOLATILITY
        
    def test_classify_regime_low_volatility(self, elite_agent, sample_hedge_snapshot, sample_elasticity_snapshot):
        """Test regime classification for low volatility."""
        sample_elasticity_snapshot.volatility = 0.08
        
        regime = elite_agent._classify_regime(
            hedge=sample_hedge_snapshot,
            elasticity=sample_elasticity_snapshot,
            sentiment=None,
        )
        
        assert regime == MarketRegime.LOW_VOLATILITY
        
    def test_classify_regime_trending(self, elite_agent, sample_hedge_snapshot, sample_elasticity_snapshot):
        """Test regime classification for trending market."""
        sample_elasticity_snapshot.trend_strength = 0.8
        sample_elasticity_snapshot.volatility = 0.25
        sample_hedge_snapshot.energy_asymmetry = 0.5
        
        regime = elite_agent._classify_regime(
            hedge=sample_hedge_snapshot,
            elasticity=sample_elasticity_snapshot,
            sentiment=None,
        )
        
        assert regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR, MarketRegime.BREAKOUT]
        
    def test_estimate_iv_rank(self, elite_agent):
        """Test IV rank estimation."""
        rank_low = elite_agent._estimate_iv_rank(0.15)
        rank_mid = elite_agent._estimate_iv_rank(0.30)
        rank_high = elite_agent._estimate_iv_rank(0.45)
        
        assert rank_low < rank_mid < rank_high
        assert 0 <= rank_low <= 100
        assert 0 <= rank_high <= 100


# =============================================================================
# TEST CLASS: Position Sizing
# =============================================================================

class TestPositionSizing:
    """Test position sizing logic."""
    
    def test_calculate_position_size_basic(self, elite_agent, sample_market_context):
        """Test basic position sizing."""
        quantity, position_value, risk_amount = elite_agent._calculate_position_size(
            context=sample_market_context,
            confidence=0.65,
            strategy=OptionStrategy.BULL_CALL_SPREAD,
            timeframe=Timeframe.SWING,
            risk_per_share=5.0,
        )
        
        assert quantity >= 0
        assert position_value >= 0
        assert risk_amount >= 0
        
    def test_calculate_position_size_respects_max(self, elite_agent, sample_market_context):
        """Test position sizing respects maximum."""
        elite_agent.risk_params.max_position_pct = 0.02  # 2%
        
        quantity, position_value, risk_amount = elite_agent._calculate_position_size(
            context=sample_market_context,
            confidence=0.95,  # Very high confidence
            strategy=OptionStrategy.LONG_CALL,
            timeframe=Timeframe.SWING,
            risk_per_share=5.0,
        )
        
        position_pct = position_value / elite_agent.portfolio_value
        assert position_pct <= 0.02 * 1.5  # Allow some tolerance for timeframe multiplier
        
    def test_calculate_position_size_portfolio_heat_limit(self, elite_agent, sample_market_context):
        """Test position sizing respects portfolio heat."""
        elite_agent.portfolio_heat = 0.18  # Already at 18%
        elite_agent.risk_params.max_portfolio_heat = 0.20  # 20% max
        
        quantity, position_value, risk_amount = elite_agent._calculate_position_size(
            context=sample_market_context,
            confidence=0.65,
            strategy=OptionStrategy.EQUITY_LONG,
            timeframe=Timeframe.SWING,
            risk_per_share=5.0,
        )
        
        # Should be limited by remaining heat (2%)
        position_pct = position_value / elite_agent.portfolio_value
        assert position_pct <= 0.03  # 2% remaining + tolerance
        
    def test_calculate_position_size_zero_heat_available(self, elite_agent, sample_market_context):
        """Test position sizing with no heat available."""
        elite_agent.portfolio_heat = 0.20  # At max
        
        quantity, position_value, risk_amount = elite_agent._calculate_position_size(
            context=sample_market_context,
            confidence=0.65,
            strategy=OptionStrategy.LONG_CALL,
            timeframe=Timeframe.SWING,
            risk_per_share=5.0,
        )
        
        assert quantity == 0


# =============================================================================
# TEST CLASS: Trade Idea Generation
# =============================================================================

class TestTradeIdeaGeneration:
    """Test trade idea generation."""
    
    def test_generate_ideas_basic(self, elite_agent, sample_pipeline_result):
        """Test basic idea generation."""
        elite_agent._get_spot_price = Mock(return_value=175.0)
        
        ideas = elite_agent.generate_ideas(
            pipeline_result=sample_pipeline_result,
            timestamp=datetime.utcnow(),
        )
        
        # Should generate ideas or empty list
        assert isinstance(ideas, list)
        
    def test_generate_ideas_no_consensus(self, elite_agent):
        """Test idea generation with no consensus."""
        result = PipelineResult(
            timestamp=datetime.utcnow(),
            symbol="TEST",
            consensus=None,
        )
        
        ideas = elite_agent.generate_ideas(result, datetime.utcnow())
        
        assert ideas == []
        
    def test_generate_ideas_low_confidence(self, elite_agent, sample_pipeline_result):
        """Test idea generation with low confidence."""
        sample_pipeline_result.consensus["confidence"] = 0.10  # Below threshold
        elite_agent._get_spot_price = Mock(return_value=175.0)
        
        ideas = elite_agent.generate_ideas(
            pipeline_result=sample_pipeline_result,
            timestamp=datetime.utcnow(),
        )
        
        assert ideas == []
        
    def test_generate_ideas_neutral_direction(self, elite_agent, sample_pipeline_result):
        """Test idea generation with neutral direction."""
        sample_pipeline_result.consensus["direction"] = "neutral"
        elite_agent._get_spot_price = Mock(return_value=175.0)
        
        ideas = elite_agent.generate_ideas(
            pipeline_result=sample_pipeline_result,
            timestamp=datetime.utcnow(),
        )
        
        assert ideas == []


# =============================================================================
# TEST CLASS: Trade Proposal Validation
# =============================================================================

class TestTradeProposalValidation:
    """Test trade proposal validation."""
    
    def test_validate_proposal_valid(self, elite_agent):
        """Test validation of valid proposal."""
        proposal = TradeProposal(
            symbol="AAPL",
            strategy=OptionStrategy.BULL_CALL_SPREAD,
            direction=DirectionEnum.LONG,
            confidence=0.65,
            entry_price=175.0,
            quantity=5,
            stop_loss=170.0,
            take_profit=185.0,
            trailing_stop_config={},
            max_loss=500.0,
            max_profit=750.0,
            risk_reward_ratio=1.5,
            position_value=2500.0,
            position_pct=0.025,
            risk_amount=500.0,
        )
        
        assert elite_agent._validate_proposal(proposal) is True
        
    def test_validate_proposal_low_rr(self, elite_agent):
        """Test validation rejects low R:R."""
        proposal = TradeProposal(
            symbol="AAPL",
            strategy=OptionStrategy.EQUITY_LONG,  # Non-credit strategy
            direction=DirectionEnum.LONG,
            confidence=0.65,
            entry_price=175.0,
            quantity=5,
            stop_loss=170.0,
            take_profit=176.0,
            trailing_stop_config={},
            max_loss=500.0,
            max_profit=100.0,
            risk_reward_ratio=0.2,  # Very low
            position_value=2500.0,
            position_pct=0.025,
            risk_amount=500.0,
        )
        
        assert elite_agent._validate_proposal(proposal) is False
        
    def test_validate_proposal_position_too_large(self, elite_agent):
        """Test validation rejects oversized position."""
        proposal = TradeProposal(
            symbol="AAPL",
            strategy=OptionStrategy.LONG_CALL,
            direction=DirectionEnum.LONG,
            confidence=0.65,
            entry_price=175.0,
            quantity=100,
            stop_loss=170.0,
            take_profit=190.0,
            trailing_stop_config={},
            max_loss=5000.0,
            max_profit=7500.0,
            risk_reward_ratio=1.5,
            position_value=17500.0,
            position_pct=0.10,  # 10% - exceeds 4% max
            risk_amount=5000.0,
        )
        
        assert elite_agent._validate_proposal(proposal) is False
        
    def test_validate_proposal_portfolio_heat_exceeded(self, elite_agent):
        """Test validation rejects when portfolio heat exceeded."""
        elite_agent.portfolio_heat = 0.18
        
        proposal = TradeProposal(
            symbol="AAPL",
            strategy=OptionStrategy.BULL_CALL_SPREAD,
            direction=DirectionEnum.LONG,
            confidence=0.65,
            entry_price=175.0,
            quantity=5,
            stop_loss=170.0,
            take_profit=185.0,
            trailing_stop_config={},
            max_loss=500.0,
            max_profit=750.0,
            risk_reward_ratio=1.5,
            position_value=2500.0,
            position_pct=0.035,  # Would push total to 21.5%
            risk_amount=500.0,
        )
        
        assert elite_agent._validate_proposal(proposal) is False
        
    def test_validate_proposal_existing_position(self, elite_agent):
        """Test validation rejects duplicate position."""
        elite_agent.current_positions.append("AAPL")
        
        proposal = TradeProposal(
            symbol="AAPL",
            strategy=OptionStrategy.BULL_CALL_SPREAD,
            direction=DirectionEnum.LONG,
            confidence=0.65,
            entry_price=175.0,
            quantity=5,
            stop_loss=170.0,
            take_profit=185.0,
            trailing_stop_config={},
            max_loss=500.0,
            max_profit=750.0,
            risk_reward_ratio=1.5,
            position_value=2500.0,
            position_pct=0.025,
            risk_amount=500.0,
        )
        
        assert elite_agent._validate_proposal(proposal) is False


# =============================================================================
# TEST CLASS: Options Order Building
# =============================================================================

class TestOptionsOrderBuilding:
    """Test options order construction."""
    
    def test_build_options_order_long_call(self, elite_agent, sample_market_context):
        """Test building long call order."""
        order = elite_agent._build_options_order(
            symbol="AAPL",
            strategy=OptionStrategy.LONG_CALL,
            context=sample_market_context,
            quantity=5,
            timeframe=Timeframe.SWING,
            confidence=0.65,
        )
        
        assert order is not None
        assert len(order.legs) == 1
        assert order.legs[0].type == "call"
        assert order.legs[0].side == "buy"
        
    def test_build_options_order_long_put(self, elite_agent, sample_market_context):
        """Test building long put order."""
        order = elite_agent._build_options_order(
            symbol="AAPL",
            strategy=OptionStrategy.LONG_PUT,
            context=sample_market_context,
            quantity=5,
            timeframe=Timeframe.SWING,
            confidence=0.65,
        )
        
        assert order is not None
        assert len(order.legs) == 1
        assert order.legs[0].type == "put"
        
    def test_build_options_order_bull_call_spread(self, elite_agent, sample_market_context):
        """Test building bull call spread order."""
        order = elite_agent._build_options_order(
            symbol="AAPL",
            strategy=OptionStrategy.BULL_CALL_SPREAD,
            context=sample_market_context,
            quantity=5,
            timeframe=Timeframe.SWING,
            confidence=0.65,
        )
        
        assert order is not None
        assert len(order.legs) == 2
        
    def test_build_options_order_iron_condor(self, elite_agent, sample_market_context):
        """Test building iron condor order."""
        order = elite_agent._build_options_order(
            symbol="AAPL",
            strategy=OptionStrategy.IRON_CONDOR,
            context=sample_market_context,
            quantity=5,
            timeframe=Timeframe.SWING,
            confidence=0.65,
        )
        
        assert order is not None
        assert len(order.legs) == 4
        
    def test_format_occ_symbol(self, elite_agent):
        """Test OCC symbol formatting."""
        expiration = datetime(2024, 3, 15)
        symbol = elite_agent._format_occ_symbol(
            symbol="AAPL",
            expiration=expiration,
            option_type="C",
            strike=175.0,
        )
        
        assert "AAPL" in symbol
        assert "240315" in symbol  # YYMMDD
        assert "C" in symbol


# =============================================================================
# TEST CLASS: Proposal to Trade Idea Conversion
# =============================================================================

class TestProposalConversion:
    """Test conversion from TradeProposal to TradeIdea."""
    
    def test_proposal_to_trade_idea(self, elite_agent):
        """Test conversion from proposal to trade idea."""
        proposal = TradeProposal(
            symbol="AAPL",
            strategy=OptionStrategy.BULL_CALL_SPREAD,
            direction=DirectionEnum.LONG,
            confidence=0.65,
            entry_price=175.0,
            quantity=5,
            stop_loss=170.0,
            take_profit=185.0,
            trailing_stop_config={},
            max_loss=500.0,
            max_profit=750.0,
            risk_reward_ratio=1.5,
            position_value=2500.0,
            position_pct=0.025,
            risk_amount=500.0,
            reasoning="Test reasoning",
        )
        
        idea = elite_agent._proposal_to_trade_idea(proposal)
        
        assert isinstance(idea, TradeIdea)
        assert idea.symbol == "AAPL"
        assert idea.direction == DirectionEnum.LONG
        assert idea.confidence == 0.65
        assert idea.entry_price == 175.0
        assert idea.stop_loss == 170.0
        assert idea.take_profit == 185.0
        
    def test_proposal_strategy_type_mapping(self, elite_agent):
        """Test strategy type mapping in conversion."""
        # Equity strategy
        equity_proposal = TradeProposal(
            symbol="AAPL",
            strategy=OptionStrategy.EQUITY_LONG,
            direction=DirectionEnum.LONG,
            confidence=0.65,
            entry_price=175.0,
            quantity=100,
            stop_loss=170.0,
            take_profit=185.0,
            trailing_stop_config={},
            max_loss=500.0,
            max_profit=1000.0,
            risk_reward_ratio=2.0,
            position_value=17500.0,
            position_pct=0.025,
            risk_amount=500.0,
        )
        
        idea = elite_agent._proposal_to_trade_idea(equity_proposal)
        assert idea.strategy_type == StrategyType.DIRECTIONAL
        
        # Spread strategy
        spread_proposal = TradeProposal(
            symbol="AAPL",
            strategy=OptionStrategy.BULL_PUT_SPREAD,
            direction=DirectionEnum.LONG,
            confidence=0.65,
            entry_price=175.0,
            quantity=5,
            stop_loss=170.0,
            take_profit=185.0,
            trailing_stop_config={},
            max_loss=500.0,
            max_profit=750.0,
            risk_reward_ratio=1.5,
            position_value=2500.0,
            position_pct=0.025,
            risk_amount=500.0,
        )
        
        idea = elite_agent._proposal_to_trade_idea(spread_proposal)
        assert idea.strategy_type == StrategyType.OPTIONS_SPREAD


# =============================================================================
# TEST CLASS: Integration Tests
# =============================================================================

class TestEliteAgentIntegration:
    """Integration tests for EliteTradeAgent."""
    
    def test_full_idea_generation_flow(self, elite_agent, sample_pipeline_result):
        """Test complete flow from pipeline result to trade idea."""
        elite_agent._get_spot_price = Mock(return_value=175.0)
        sample_pipeline_result.consensus["confidence"] = 0.70
        
        ideas = elite_agent.generate_ideas(
            pipeline_result=sample_pipeline_result,
            timestamp=datetime.utcnow(),
        )
        
        # Should produce ideas or be filtered out legitimately
        assert isinstance(ideas, list)
        
    def test_agent_state_tracking(self, elite_agent):
        """Test that agent tracks state correctly."""
        assert elite_agent.portfolio_heat == 0.0
        assert len(elite_agent.current_positions) == 0
        
        # Add a position
        elite_agent.current_positions.append("AAPL")
        assert len(elite_agent.current_positions) == 1
        
    def test_multiple_ideas_generation(self, elite_agent):
        """Test generating ideas for multiple symbols."""
        elite_agent._get_spot_price = Mock(return_value=100.0)
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        all_ideas = []
        
        for symbol in symbols:
            result = PipelineResult(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                hedge_snapshot=HedgeSnapshot(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    movement_energy=60.0,
                    energy_asymmetry=0.3,
                    data={},
                ),
                elasticity_snapshot=ElasticitySnapshot(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    volatility=0.25,
                    trend_strength=0.5,
                ),
                consensus={
                    "direction": "long",
                    "confidence": 0.60,
                },
            )
            
            ideas = elite_agent.generate_ideas(result, datetime.utcnow())
            all_ideas.extend(ideas)
        
        # Should produce some ideas or be legitimately filtered
        assert isinstance(all_ideas, list)


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_portfolio_value(self):
        """Test handling of zero portfolio value."""
        agent = EliteTradeAgent(config={"min_confidence": 0.25})
        agent.portfolio_value = 0
        
        # Should not crash
        context = MarketContext(
            symbol="TEST",
            spot_price=100.0,
            iv_rank=50.0,
            iv_percentile=50.0,
            historical_vol=0.25,
            implied_vol=0.25,
            vol_skew=0.0,
            put_call_ratio=1.0,
            regime=MarketRegime.RANGE_BOUND,
            iv_environment=IVEnvironment.MEDIUM,
            atr=2.0,
            atr_pct=0.02,
            trend_strength=0.5,
            momentum=0.0,
            support_level=95.0,
            resistance_level=105.0,
        )
        
        quantity, value, risk = agent._calculate_position_size(
            context=context,
            confidence=0.65,
            strategy=OptionStrategy.EQUITY_LONG,
            timeframe=Timeframe.SWING,
            risk_per_share=5.0,
        )
        
        assert quantity == 0
        
    def test_extreme_volatility(self, elite_agent, sample_pipeline_result):
        """Test handling of extreme volatility."""
        sample_pipeline_result.elasticity_snapshot.volatility = 2.0  # 200% vol
        
        timeframe = elite_agent._select_timeframe(sample_pipeline_result, 0.65)
        
        # Should still work
        assert timeframe in Timeframe.__members__.values()
        
    def test_negative_confidence(self, elite_agent, sample_pipeline_result):
        """Test handling of negative confidence."""
        sample_pipeline_result.consensus["confidence"] = -0.5
        elite_agent._get_spot_price = Mock(return_value=175.0)
        
        ideas = elite_agent.generate_ideas(
            sample_pipeline_result,
            datetime.utcnow(),
        )
        
        # Should filter out
        assert ideas == []
        
    def test_empty_consensus(self, elite_agent):
        """Test handling of empty consensus."""
        result = PipelineResult(
            timestamp=datetime.utcnow(),
            symbol="TEST",
            consensus={},
        )
        
        elite_agent._get_spot_price = Mock(return_value=175.0)
        
        ideas = elite_agent.generate_ideas(result, datetime.utcnow())
        
        # Should handle gracefully
        assert ideas == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
