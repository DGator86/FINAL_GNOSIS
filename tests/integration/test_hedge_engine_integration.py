"""Integration tests for Hedge Engine v3."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from engines.hedge.hedge_engine_v3 import HedgeEngineV3
from schemas.core_schemas import HedgeSnapshot


class TestHedgeEngineIntegration:
    """Integration tests for HedgeEngineV3 with realistic scenarios."""

    @pytest.fixture
    def mock_options_adapter(self):
        """Create a mock options adapter with realistic chain data."""
        adapter = Mock()
        return adapter

    @pytest.fixture
    def hedge_engine(self, mock_options_adapter):
        """Create hedge engine instance."""
        config = {
            "enabled": True,
            "lookback_days": 30,
            "min_dte": 7,
            "max_dte": 60,
        }
        return HedgeEngineV3(mock_options_adapter, config)

    def test_elasticity_calculation_with_realistic_chain(self, hedge_engine, mock_options_adapter):
        """Test elasticity calculation with realistic options chain."""
        # Create realistic options chain (SPY at $450)
        chain = [
            # ATM calls - high gamma
            Mock(option_type="call", strike=450, delta=0.50, gamma=0.05, vega=0.15, theta=-0.02, open_interest=10000),
            Mock(option_type="call", strike=455, delta=0.35, gamma=0.04, vega=0.12, theta=-0.015, open_interest=8000),

            # ATM puts - high gamma
            Mock(option_type="put", strike=450, delta=-0.50, gamma=0.05, vega=0.15, theta=-0.02, open_interest=12000),
            Mock(option_type="put", strike=445, delta=-0.35, gamma=0.04, vega=0.12, theta=-0.015, open_interest=9000),

            # OTM calls - lower gamma
            Mock(option_type="call", strike=460, delta=0.20, gamma=0.02, vega=0.08, theta=-0.01, open_interest=5000),
            Mock(option_type="call", strike=465, delta=0.10, gamma=0.01, vega=0.05, theta=-0.005, open_interest=3000),

            # OTM puts - lower gamma
            Mock(option_type="put", strike=440, delta=-0.20, gamma=0.02, vega=0.08, theta=-0.01, open_interest=6000),
            Mock(option_type="put", strike=435, delta=-0.10, gamma=0.01, vega=0.05, theta=-0.005, open_interest=4000),
        ]

        mock_options_adapter.get_chain.return_value = chain

        # Run engine
        timestamp = datetime.now(timezone.utc)
        snapshot = hedge_engine.run("SPY", timestamp)

        # Verify snapshot structure
        assert isinstance(snapshot, HedgeSnapshot)
        assert snapshot.symbol == "SPY"
        assert snapshot.timestamp == timestamp

        # Verify elasticity calculations
        assert snapshot.elasticity > 0.0, "Elasticity should be positive"
        assert snapshot.gamma_pressure > 0.0, "Gamma pressure should be calculated"
        assert snapshot.vanna_pressure > 0.0, "Vanna pressure should be calculated"

        # Verify pressure calculations
        assert snapshot.pressure_up > 0.0, "Call pressure should be positive"
        assert snapshot.pressure_down > 0.0, "Put pressure should be positive"

        # Verify confidence based on chain size
        assert 0.0 <= snapshot.confidence <= 1.0
        assert snapshot.confidence == min(1.0, len(chain) / 100.0)

    def test_dealer_gamma_sign_calculation(self, hedge_engine, mock_options_adapter):
        """Test dealer gamma sign calculation for different scenarios."""
        # Scenario 1: Heavy call OI (dealer short gamma)
        chain_heavy_calls = [
            Mock(option_type="call", strike=450, delta=0.50, gamma=0.05, vega=0.15, theta=-0.02, open_interest=20000),
            Mock(option_type="put", strike=450, delta=-0.50, gamma=0.05, vega=0.15, theta=-0.02, open_interest=5000),
        ]

        mock_options_adapter.get_chain.return_value = chain_heavy_calls
        snapshot = hedge_engine.run("SPY", datetime.now(timezone.utc))

        # Dealer gamma sign should be positive (call OI > put OI)
        assert snapshot.dealer_gamma_sign > 0.0

        # Scenario 2: Heavy put OI (dealer long gamma)
        chain_heavy_puts = [
            Mock(option_type="call", strike=450, delta=0.50, gamma=0.05, vega=0.15, theta=-0.02, open_interest=5000),
            Mock(option_type="put", strike=450, delta=-0.50, gamma=0.05, vega=0.15, theta=-0.02, open_interest=20000),
        ]

        mock_options_adapter.get_chain.return_value = chain_heavy_puts
        snapshot = hedge_engine.run("SPY", datetime.now(timezone.utc))

        # Dealer gamma sign should be negative (put OI > call OI)
        assert snapshot.dealer_gamma_sign < 0.0

    def test_regime_classification(self, hedge_engine, mock_options_adapter):
        """Test regime classification logic."""
        timestamp = datetime.now(timezone.utc)

        # Test short_squeeze regime (high gamma, positive dealer sign)
        chain = [Mock(option_type="call", strike=450, delta=0.50, gamma=0.08, vega=0.15, theta=-0.02, open_interest=20000)]
        mock_options_adapter.get_chain.return_value = chain
        snapshot = hedge_engine.run("SPY", timestamp)
        assert snapshot.regime == "short_squeeze"

        # Test long_compression regime (high gamma, negative dealer sign)
        chain = [Mock(option_type="put", strike=450, delta=-0.50, gamma=0.08, vega=0.15, theta=-0.02, open_interest=20000)]
        mock_options_adapter.get_chain.return_value = chain
        snapshot = hedge_engine.run("SPY", timestamp)
        assert snapshot.regime == "long_compression"

    def test_movement_energy_asymmetry(self, hedge_engine, mock_options_adapter):
        """Test movement energy and asymmetry calculations."""
        # Create asymmetric chain (more call pressure)
        chain = [
            Mock(option_type="call", strike=450, delta=0.60, gamma=0.05, vega=0.15, theta=-0.02, open_interest=15000),
            Mock(option_type="put", strike=450, delta=-0.40, gamma=0.05, vega=0.15, theta=-0.02, open_interest=5000),
        ]

        mock_options_adapter.get_chain.return_value = chain
        snapshot = hedge_engine.run("SPY", datetime.now(timezone.utc))

        # Verify energy calculations
        assert snapshot.movement_energy >= 0.0
        assert -1.0 <= snapshot.energy_asymmetry <= 1.0

        # With more call pressure, asymmetry should be positive
        assert snapshot.energy_asymmetry > 0.0

    def test_empty_chain_handling(self, hedge_engine, mock_options_adapter):
        """Test engine handles empty options chain gracefully."""
        mock_options_adapter.get_chain.return_value = []

        snapshot = hedge_engine.run("SPY", datetime.now(timezone.utc))

        # Should return valid snapshot with default values
        assert isinstance(snapshot, HedgeSnapshot)
        assert snapshot.elasticity == 0.0
        assert snapshot.gamma_pressure == 0.0
        assert snapshot.confidence == 0.5

    def test_zero_division_protection(self, hedge_engine, mock_options_adapter):
        """Test protection against zero division errors."""
        # Chain with zero gamma
        chain = [
            Mock(option_type="call", strike=450, delta=0.50, gamma=0.0, vega=0.0, theta=0.0, open_interest=100),
        ]

        mock_options_adapter.get_chain.return_value = chain

        # Should not raise ZeroDivisionError
        snapshot = hedge_engine.run("SPY", datetime.now(timezone.utc))
        assert isinstance(snapshot, HedgeSnapshot)
