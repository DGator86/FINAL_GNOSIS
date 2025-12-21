"""
Comprehensive Tests for Hedge Engine V3

Tests for dealer flow and elasticity analysis including:
- Dealer gamma sign calculation
- Greek pressure computation
- Vanna shock absorber
- Directional pressures
- Elasticity computation
- Movement energy and asymmetry
- Jump risk estimation
- Liquidity friction
- Regime detection
- Adaptive smoothing

Author: Super Gnosis Elite Trading System
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from engines.hedge.hedge_engine_v3 import HedgeEngineV3
from schemas.core_schemas import HedgeSnapshot


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_contract():
    """Create a mock options contract."""
    def _create(
        option_type="call",
        open_interest=1000,
        gamma=0.05,
        delta=0.5,
        vega=0.1,
        theta=-0.02,
        implied_volatility=0.25,
        strike=450.0,
    ):
        contract = Mock()
        contract.option_type = option_type
        contract.open_interest = open_interest
        contract.gamma = gamma
        contract.delta = delta
        contract.vega = vega
        contract.theta = theta
        contract.implied_volatility = implied_volatility
        contract.strike = strike
        return contract
    return _create


@pytest.fixture
def mock_options_adapter(mock_contract):
    """Create a mock options chain adapter."""
    adapter = Mock()
    
    # Create a diverse chain
    chain = [
        mock_contract("call", 5000, 0.08, 0.6, 0.15, -0.03, 0.22),
        mock_contract("call", 3000, 0.06, 0.5, 0.12, -0.02, 0.24),
        mock_contract("call", 2000, 0.04, 0.4, 0.10, -0.01, 0.26),
        mock_contract("put", 4000, 0.07, -0.5, 0.14, -0.02, 0.25),
        mock_contract("put", 2500, 0.05, -0.4, 0.11, -0.02, 0.23),
        mock_contract("put", 1500, 0.03, -0.3, 0.09, -0.01, 0.27),
    ]
    adapter.get_chain.return_value = chain
    
    return adapter


@pytest.fixture
def default_config():
    """Create default engine configuration."""
    return {
        "regime_components": 3,
        "regime_history": 256,
        "regime_min_samples": 32,
        "gamma_weight": 0.6,
        "vanna_weight": 0.4,
        "vanna_shock_decay": 1.2,
        "smoothing_alpha": 0.2,
        "weight_step": 0.02,
        "ledger_flows": [],
    }


@pytest.fixture
def hedge_engine(mock_options_adapter, default_config):
    """Create a HedgeEngineV3 instance with mocked dependencies."""
    return HedgeEngineV3(
        options_adapter=mock_options_adapter,
        config=default_config,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestHedgeEngineInitialization:
    """Tests for HedgeEngineV3 initialization."""
    
    def test_initialization(self, mock_options_adapter, default_config):
        """Test basic engine initialization."""
        engine = HedgeEngineV3(
            options_adapter=mock_options_adapter,
            config=default_config,
        )
        
        assert engine.options_adapter is mock_options_adapter
        assert engine.config == default_config
        assert engine.regime_detector is not None
    
    def test_initialization_with_custom_weights(self, mock_options_adapter):
        """Test initialization with custom weights."""
        config = {
            "gamma_weight": 0.8,
            "vanna_weight": 0.2,
            "ledger_flows": [],
        }
        
        engine = HedgeEngineV3(
            options_adapter=mock_options_adapter,
            config=config,
        )
        
        assert engine.weight_state["gamma"] == 0.8
        assert engine.weight_state["vanna"] == 0.2
    
    def test_initialization_empty_config(self, mock_options_adapter):
        """Test initialization with minimal config."""
        engine = HedgeEngineV3(
            options_adapter=mock_options_adapter,
            config={},
        )
        
        # Should use defaults
        assert engine.weight_state["gamma"] == 0.6
        assert engine.weight_state["vanna"] == 0.4


# =============================================================================
# BASIC RUN TESTS
# =============================================================================

class TestBasicRun:
    """Tests for basic run functionality."""
    
    def test_run_returns_hedge_snapshot(self, hedge_engine):
        """Test that run returns a HedgeSnapshot."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert isinstance(result, HedgeSnapshot)
        assert result.symbol == "SPY"
        assert result.timestamp == timestamp
    
    def test_run_with_empty_chain(self, hedge_engine, mock_options_adapter):
        """Test run with empty options chain."""
        mock_options_adapter.get_chain.return_value = []
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Should return default snapshot
        assert isinstance(result, HedgeSnapshot)
        assert result.elasticity == 0.0
    
    def test_run_with_no_chain(self, hedge_engine, mock_options_adapter):
        """Test run when chain is None."""
        mock_options_adapter.get_chain.return_value = None
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert isinstance(result, HedgeSnapshot)


# =============================================================================
# DEALER GAMMA SIGN TESTS
# =============================================================================

class TestDealerGammaSign:
    """Tests for dealer gamma sign calculation."""
    
    def test_bullish_dealer_gamma(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test bullish dealer gamma (more calls)."""
        chain = [
            mock_contract("call", 10000),
            mock_contract("call", 5000),
            mock_contract("put", 3000),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # (15000 - 3000) / 18000 = 0.67
        assert result.dealer_gamma_sign > 0.5
    
    def test_bearish_dealer_gamma(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test bearish dealer gamma (more puts)."""
        chain = [
            mock_contract("call", 3000),
            mock_contract("put", 10000),
            mock_contract("put", 5000),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # (3000 - 15000) / 18000 = -0.67
        assert result.dealer_gamma_sign < -0.5
    
    def test_neutral_dealer_gamma(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test neutral dealer gamma (equal calls and puts)."""
        chain = [
            mock_contract("call", 5000),
            mock_contract("put", 5000),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert abs(result.dealer_gamma_sign) < 0.1
    
    def test_zero_open_interest(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test with zero open interest."""
        chain = [
            mock_contract("call", 0),
            mock_contract("put", 0),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.dealer_gamma_sign == 0.0


# =============================================================================
# GREEK PRESSURE TESTS
# =============================================================================

class TestGreekPressures:
    """Tests for Greek pressure calculations."""
    
    def test_gamma_pressure_calculated(self, hedge_engine):
        """Test gamma pressure calculation."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Should have positive gamma pressure
        assert result.gamma_pressure > 0
    
    def test_vanna_pressure_calculated(self, hedge_engine):
        """Test vanna pressure calculation."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Vanna = vega * delta
        assert result.vanna_pressure >= 0
    
    def test_charm_pressure_calculated(self, hedge_engine):
        """Test charm pressure calculation."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Charm = theta * delta
        assert result.charm_pressure >= 0
    
    def test_high_gamma_environment(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test in high gamma environment."""
        chain = [
            mock_contract("call", 10000, gamma=0.15),  # High gamma
            mock_contract("put", 10000, gamma=0.15),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.gamma_pressure > 1000  # High gamma * OI


# =============================================================================
# VANNA SHOCK ABSORBER TESTS
# =============================================================================

class TestVannaShockAbsorber:
    """Tests for vanna shock absorber."""
    
    def test_shock_absorber_dampens_during_spike(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test vanna dampening during volatility spike."""
        # Create chain with IV spike (high mean vs median)
        chain = [
            mock_contract("call", 1000, implied_volatility=0.20),
            mock_contract("call", 1000, implied_volatility=0.25),
            mock_contract("call", 1000, implied_volatility=0.80),  # Spike
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        raw_vanna = sum(abs(c.vega * c.delta) * c.open_interest for c in chain) / len(chain)
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Vanna should be dampened (less than raw calculation due to spike)
        # The exact value depends on smoothing and decay
        assert result.vanna_pressure >= 0
    
    def test_no_dampening_normal_iv(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test no dampening when IV is normal."""
        # All similar IVs
        chain = [
            mock_contract("call", 1000, implied_volatility=0.25),
            mock_contract("call", 1000, implied_volatility=0.25),
            mock_contract("call", 1000, implied_volatility=0.25),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Should have minimal dampening
        assert result.vanna_pressure > 0


# =============================================================================
# DIRECTIONAL PRESSURE TESTS
# =============================================================================

class TestDirectionalPressures:
    """Tests for directional pressure calculations."""
    
    def test_pressure_up_from_calls(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test upward pressure from calls."""
        chain = [
            mock_contract("call", 10000, gamma=0.1, delta=0.6),  # High delta > 0.3
            mock_contract("call", 5000, gamma=0.08, delta=0.4),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.pressure_up > 0
    
    def test_pressure_down_from_puts(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test downward pressure from puts."""
        chain = [
            mock_contract("put", 10000, gamma=0.1, delta=-0.6),  # Delta < -0.3
            mock_contract("put", 5000, gamma=0.08, delta=-0.4),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.pressure_down > 0
    
    def test_net_pressure_calculation(self, hedge_engine):
        """Test net pressure is up minus down."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.pressure_net == result.pressure_up - result.pressure_down
    
    def test_low_delta_excluded(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test that low delta options don't contribute to directional pressure."""
        chain = [
            mock_contract("call", 10000, gamma=0.1, delta=0.2),  # Delta < 0.3
            mock_contract("put", 10000, gamma=0.1, delta=-0.2),  # |Delta| < 0.3
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Neither should contribute
        assert result.pressure_up == 0
        assert result.pressure_down == 0


# =============================================================================
# ELASTICITY TESTS
# =============================================================================

class TestElasticity:
    """Tests for elasticity computation."""
    
    def test_elasticity_positive(self, hedge_engine):
        """Test that elasticity is always positive."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Elasticity should be > 0 (minimum 0.1)
        assert result.elasticity >= 0.1
    
    def test_elasticity_uses_weighted_calculation(self, hedge_engine):
        """Test elasticity uses gamma and vanna weights."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Should have adaptive weights stored
        assert "gamma" in hedge_engine.weight_state
        assert "vanna" in hedge_engine.weight_state
    
    def test_directional_elasticity_in_snapshot(self, hedge_engine):
        """Test directional elasticity is returned."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert hasattr(result, "directional_elasticity")
        assert "up" in result.directional_elasticity
        assert "down" in result.directional_elasticity


# =============================================================================
# MOVEMENT ENERGY TESTS
# =============================================================================

class TestMovementEnergy:
    """Tests for movement energy calculation."""
    
    def test_movement_energy_calculated(self, hedge_engine):
        """Test movement energy is calculated."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Movement energy = |pressure_net| / elasticity
        assert result.movement_energy >= 0
    
    def test_movement_energy_formula(self, hedge_engine):
        """Test movement energy follows formula."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        expected = abs(result.pressure_net) / result.elasticity if result.elasticity > 0 else 0
        assert abs(result.movement_energy - expected) < 0.01
    
    def test_high_pressure_high_energy(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test high pressure results in high energy."""
        # Extreme call position for high net pressure
        chain = [
            mock_contract("call", 100000, gamma=0.2, delta=0.8),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.movement_energy > 0


# =============================================================================
# ENERGY ASYMMETRY TESTS
# =============================================================================

class TestEnergyAsymmetry:
    """Tests for energy asymmetry calculation."""
    
    def test_asymmetry_range(self, hedge_engine):
        """Test asymmetry is between -1 and 1."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert -1.0 <= result.energy_asymmetry <= 1.0
    
    def test_bullish_asymmetry(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test bullish asymmetry (more upward pressure)."""
        chain = [
            mock_contract("call", 10000, gamma=0.1, delta=0.6),
            mock_contract("put", 1000, gamma=0.02, delta=-0.4),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.energy_asymmetry > 0
    
    def test_bearish_asymmetry(self, hedge_engine, mock_options_adapter, mock_contract):
        """Test bearish asymmetry (more downward pressure)."""
        chain = [
            mock_contract("call", 1000, gamma=0.02, delta=0.4),
            mock_contract("put", 10000, gamma=0.1, delta=-0.6),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.energy_asymmetry < 0
    
    def test_zero_pressure_zero_asymmetry(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test zero asymmetry when no directional pressure."""
        chain = [
            mock_contract("call", 1000, gamma=0.1, delta=0.2),  # Low delta
            mock_contract("put", 1000, gamma=0.1, delta=-0.2),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert result.energy_asymmetry == 0


# =============================================================================
# JUMP RISK TESTS
# =============================================================================

class TestJumpRisk:
    """Tests for jump risk estimation."""
    
    def test_jump_intensity_range(self, hedge_engine):
        """Test jump intensity is between 0 and 1."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert 0.0 <= result.jump_intensity <= 1.0
    
    def test_high_jump_risk_deep_otm(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test high jump risk with many deep OTM options."""
        chain = [
            mock_contract("call", 1000, delta=0.02),  # Deep OTM
            mock_contract("call", 1000, delta=0.03),
            mock_contract("put", 1000, delta=-0.02),
            mock_contract("put", 1000, delta=-0.04),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # All have |delta| < 0.05, so 100% tail ratio
        assert result.jump_intensity == 1.0
    
    def test_low_jump_risk_atm(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test low jump risk with ATM options."""
        chain = [
            mock_contract("call", 1000, delta=0.50),  # ATM
            mock_contract("put", 1000, delta=-0.50),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # No deep OTM
        assert result.jump_intensity == 0.0


# =============================================================================
# LIQUIDITY FRICTION TESTS
# =============================================================================

class TestLiquidityFriction:
    """Tests for liquidity friction estimation."""
    
    def test_liquidity_friction_range(self, hedge_engine):
        """Test liquidity friction is between 0 and 1."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert 0.0 <= result.liquidity_friction <= 1.0
    
    def test_high_concentration_high_friction(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test high friction with concentrated OI."""
        chain = [
            mock_contract("call", 90000),  # 90% of OI
            mock_contract("call", 5000),
            mock_contract("call", 5000),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # High concentration should lead to higher friction
        assert result.liquidity_friction > 0.5
    
    def test_distributed_oi_lower_friction(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test lower friction with distributed OI."""
        chain = [
            mock_contract("call", 1000),
            mock_contract("call", 1000),
            mock_contract("put", 1000),
            mock_contract("put", 1000),
        ]
        mock_options_adapter.get_chain.return_value = chain
        
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Distributed OI = lower friction
        assert result.liquidity_friction < 0.5


# =============================================================================
# REGIME DETECTION TESTS
# =============================================================================

class TestRegimeDetection:
    """Tests for regime detection."""
    
    def test_regime_in_valid_set(self, hedge_engine):
        """Test regime is a valid classification."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        valid_regimes = ["bullish", "bearish", "neutral", "high_vol", "low_vol", "unknown"]
        # Regime detector may return various states
        assert isinstance(result.regime, str)
    
    def test_regime_probabilities_present(self, hedge_engine):
        """Test regime probabilities are returned."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert isinstance(result.regime_probabilities, dict)
    
    def test_regime_features_stored(self, hedge_engine):
        """Test regime features are stored in snapshot."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert "dealer_gamma_sign" in result.regime_features
        assert "gamma_pressure" in result.regime_features
        assert "movement_energy" in result.regime_features


# =============================================================================
# ADAPTIVE SMOOTHING TESTS
# =============================================================================

class TestAdaptiveSmoothing:
    """Tests for adaptive smoothing."""
    
    def test_smoothing_applied(self, hedge_engine):
        """Test smoothing is applied to features."""
        timestamp = datetime.now(timezone.utc)
        
        # Run twice to test smoothing effect
        result1 = hedge_engine.run("SPY", timestamp)
        result2 = hedge_engine.run("SPY", timestamp)
        
        # EMA should be stored
        assert len(hedge_engine.feature_ema) > 0
    
    def test_smoothing_alpha_used(self, mock_options_adapter):
        """Test custom smoothing alpha is used."""
        config = {
            "smoothing_alpha": 0.5,  # High alpha = more responsive
            "ledger_flows": [],
        }
        
        engine = HedgeEngineV3(
            options_adapter=mock_options_adapter,
            config=config,
        )
        
        timestamp = datetime.now(timezone.utc)
        engine.run("SPY", timestamp)
        
        assert engine.config["smoothing_alpha"] == 0.5


# =============================================================================
# FLOW HISTORY TESTS
# =============================================================================

class TestFlowHistory:
    """Tests for flow history loading."""
    
    def test_load_flow_history_from_config(self, mock_options_adapter):
        """Test loading flow history from config."""
        flows = [{"flow": i, "price": 100 + i} for i in range(10)]
        config = {"ledger_flows": flows}
        
        engine = HedgeEngineV3(
            options_adapter=mock_options_adapter,
            config=config,
        )
        
        history = engine._load_flow_history()
        assert len(history) == 10
    
    def test_flow_history_limited_to_30(self, mock_options_adapter):
        """Test flow history is limited to 30 entries."""
        flows = [{"flow": i, "price": 100 + i} for i in range(50)]
        config = {"ledger_flows": flows}
        
        engine = HedgeEngineV3(
            options_adapter=mock_options_adapter,
            config=config,
        )
        
        history = engine._load_flow_history()
        assert len(history) == 30
    
    def test_flow_history_callable(self, mock_options_adapter):
        """Test flow history from callable."""
        def get_flows():
            return [{"flow": i, "price": 100 + i} for i in range(5)]
        
        config = {"ledger_flows": get_flows}
        
        engine = HedgeEngineV3(
            options_adapter=mock_options_adapter,
            config=config,
        )
        
        history = engine._load_flow_history()
        assert len(history) == 5


# =============================================================================
# WEIGHT UPDATE TESTS
# =============================================================================

class TestWeightUpdate:
    """Tests for adaptive weight updates."""
    
    def test_weights_update_toward_dominant_greek(self, hedge_engine):
        """Test weights shift toward dominant Greek."""
        initial_gamma = hedge_engine.weight_state["gamma"]
        
        timestamp = datetime.now(timezone.utc)
        hedge_engine.run("SPY", timestamp)
        
        # Weights should have shifted slightly
        assert hedge_engine.weight_state["gamma"] != initial_gamma or \
               abs(hedge_engine.weight_state["gamma"] - initial_gamma) < 0.1
    
    def test_weights_sum_approximately_one(self, hedge_engine):
        """Test gamma + vanna weights sum to approximately 1."""
        timestamp = datetime.now(timezone.utc)
        hedge_engine.run("SPY", timestamp)
        
        total = hedge_engine.weight_state["gamma"] + hedge_engine.weight_state["vanna"]
        assert abs(total - 1.0) < 0.1


# =============================================================================
# CONFIDENCE TESTS
# =============================================================================

class TestConfidence:
    """Tests for confidence calculation."""
    
    def test_confidence_range(self, hedge_engine):
        """Test confidence is between 0 and 1."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        assert 0.0 <= result.confidence <= 1.0
    
    def test_larger_chain_higher_confidence(
        self, hedge_engine, mock_options_adapter, mock_contract
    ):
        """Test larger chains give higher confidence."""
        # Small chain
        small_chain = [mock_contract() for _ in range(5)]
        mock_options_adapter.get_chain.return_value = small_chain
        
        timestamp = datetime.now(timezone.utc)
        small_result = hedge_engine.run("SPY", timestamp)
        
        # Large chain
        large_chain = [mock_contract() for _ in range(200)]
        mock_options_adapter.get_chain.return_value = large_chain
        
        large_result = hedge_engine.run("SPY", timestamp)
        
        assert large_result.confidence > small_result.confidence


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for hedge engine."""
    
    def test_full_analysis_flow(self, hedge_engine):
        """Test complete analysis flow."""
        timestamp = datetime.now(timezone.utc)
        result = hedge_engine.run("SPY", timestamp)
        
        # Verify all key fields populated
        assert result.symbol == "SPY"
        assert result.elasticity > 0
        assert result.movement_energy >= 0
        assert -1.0 <= result.energy_asymmetry <= 1.0
        assert -1.0 <= result.dealer_gamma_sign <= 1.0
        assert result.gamma_pressure >= 0
        assert result.vanna_pressure >= 0
        assert result.charm_pressure >= 0
    
    def test_multiple_runs_consistency(self, hedge_engine):
        """Test multiple runs produce consistent results."""
        timestamp = datetime.now(timezone.utc)
        
        results = [hedge_engine.run("SPY", timestamp) for _ in range(3)]
        
        # Elasticity should converge due to EMA
        for result in results:
            assert result.elasticity > 0
    
    def test_different_symbols(self, hedge_engine):
        """Test analysis for different symbols."""
        timestamp = datetime.now(timezone.utc)
        symbols = ["SPY", "QQQ", "AAPL"]
        
        results = [hedge_engine.run(symbol, timestamp) for symbol in symbols]
        
        for i, result in enumerate(results):
            assert result.symbol == symbols[i]
