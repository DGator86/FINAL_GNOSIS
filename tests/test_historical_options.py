"""
Tests for Historical Options Data Integration

Tests coverage:
1. Polygon.io adapter functionality
2. Synthetic options generator
3. Historical options manager
4. Options backtest engine integration
"""

import pytest
from datetime import datetime, date, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# SYNTHETIC OPTIONS GENERATOR TESTS
# ============================================================================

class TestSyntheticOptionsGenerator:
    """Test synthetic options data generator."""
    
    def test_generator_initialization(self):
        """Test generator can be initialized."""
        from backtesting.synthetic_options_data import SyntheticOptionsGenerator
        
        generator = SyntheticOptionsGenerator(
            risk_free_rate=0.05,
            base_iv=0.25,
            seed=42,
        )
        
        assert generator is not None
        assert generator.risk_free_rate == 0.05
    
    def test_strike_generation(self):
        """Test strike price generation."""
        from backtesting.synthetic_options_data import SyntheticOptionsGenerator
        
        generator = SyntheticOptionsGenerator(seed=42)
        
        # Test for different price levels
        strikes_low = generator.generate_strikes(50.0, num_strikes=10)
        strikes_med = generator.generate_strikes(150.0, num_strikes=10)
        strikes_high = generator.generate_strikes(500.0, num_strikes=10)
        
        assert len(strikes_low) > 0
        assert len(strikes_med) > 0
        assert len(strikes_high) > 0
        
        # Strikes should be centered around spot
        assert min(strikes_low) < 50.0 < max(strikes_low)
        assert min(strikes_med) < 150.0 < max(strikes_med)
        assert min(strikes_high) < 500.0 < max(strikes_high)
    
    def test_expiration_generation(self):
        """Test expiration date generation."""
        from backtesting.synthetic_options_data import SyntheticOptionsGenerator
        
        generator = SyntheticOptionsGenerator(seed=42)
        
        current_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        expirations = generator.generate_expirations(
            current_date,
            num_weekly=4,
            num_monthly=3,
        )
        
        assert len(expirations) > 0
        
        # All expirations should be after current date
        for exp in expirations:
            assert exp > current_date
    
    def test_options_chain_generation(self):
        """Test full options chain generation."""
        from backtesting.synthetic_options_data import SyntheticOptionsGenerator
        
        generator = SyntheticOptionsGenerator(seed=42)
        
        chain = generator.generate_options_chain(
            underlying="SPY",
            spot=450.0,
            current_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            market_regime="neutral",
        )
        
        assert chain is not None
        assert chain.underlying == "SPY"
        assert chain.spot_price == 450.0
        assert len(chain.calls) > 0
        assert len(chain.puts) > 0
        assert chain.max_pain > 0
        assert chain.call_wall > chain.spot_price  # Call wall above spot
        assert chain.put_wall < chain.spot_price  # Put wall below spot
    
    def test_black_scholes_pricing(self):
        """Test Black-Scholes calculations."""
        from backtesting.synthetic_options_data import BlackScholes
        
        # ATM call
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        
        call_price = BlackScholes.call_price(S, K, T, r, sigma)
        put_price = BlackScholes.put_price(S, K, T, r, sigma)
        
        assert call_price > 0
        assert put_price > 0
        
        # Put-call parity check (approximately)
        parity_diff = abs(call_price - put_price - S + K * np.exp(-r * T))
        assert parity_diff < 0.1
        
        # Greeks
        delta_call = BlackScholes.delta(S, K, T, r, sigma, "call")
        delta_put = BlackScholes.delta(S, K, T, r, sigma, "put")
        gamma = BlackScholes.gamma(S, K, T, r, sigma)
        theta_call = BlackScholes.theta(S, K, T, r, sigma, "call")
        vega = BlackScholes.vega(S, K, T, r, sigma)
        
        assert 0 < delta_call < 1
        assert -1 < delta_put < 0
        assert gamma > 0
        assert theta_call < 0  # Time decay is negative
        assert vega > 0
    
    def test_iv_surface_generator(self):
        """Test IV surface generation."""
        from backtesting.synthetic_options_data import IVSurfaceGenerator
        
        iv_gen = IVSurfaceGenerator(base_iv=0.25)
        
        spot = 100.0
        
        # ATM options
        iv_atm_call = iv_gen.get_iv(spot, 100, 30, "call")
        iv_atm_put = iv_gen.get_iv(spot, 100, 30, "put")
        
        # OTM options
        iv_otm_call = iv_gen.get_iv(spot, 110, 30, "call")
        iv_otm_put = iv_gen.get_iv(spot, 90, 30, "put")
        
        assert 0.05 < iv_atm_call < 2.0
        assert 0.05 < iv_atm_put < 2.0
        assert 0.05 < iv_otm_call < 2.0
        assert 0.05 < iv_otm_put < 2.0
        
        # Put skew: OTM puts should have higher IV
        assert iv_otm_put > iv_atm_put


# ============================================================================
# POLYGON OPTIONS ADAPTER TESTS
# ============================================================================

class TestPolygonOptionsAdapter:
    """Test Polygon.io adapter."""
    
    def test_adapter_initialization_no_key(self):
        """Test adapter initialization without API key."""
        from engines.inputs.polygon_options_adapter import PolygonOptionsAdapter, PolygonConfig
        
        # Clear env var
        old_key = os.environ.pop("POLYGON_API_KEY", None)
        
        try:
            config = PolygonConfig(api_key="")
            adapter = PolygonOptionsAdapter(config=config)
            
            # Should initialize but log warning
            assert adapter is not None
        finally:
            if old_key:
                os.environ["POLYGON_API_KEY"] = old_key
    
    def test_config_from_env(self):
        """Test config loading from environment."""
        from engines.inputs.polygon_options_adapter import PolygonConfig
        
        # Set test key
        os.environ["POLYGON_API_KEY"] = "test_key_123"
        
        try:
            config = PolygonConfig.from_env()
            assert config.api_key == "test_key_123"
        finally:
            del os.environ["POLYGON_API_KEY"]
    
    def test_rate_limiting(self):
        """Test rate limiting behavior."""
        from engines.inputs.polygon_options_adapter import PolygonOptionsAdapter, PolygonConfig
        
        config = PolygonConfig(api_key="test", rate_limit_delay=0.1)
        adapter = PolygonOptionsAdapter(config=config)
        
        import time
        
        # Make two rate limited calls
        start = time.time()
        adapter._rate_limit()
        adapter._rate_limit()
        elapsed = time.time() - start
        
        # Should have waited at least one delay period
        assert elapsed >= 0.1
        
        adapter.close()


# ============================================================================
# HISTORICAL OPTIONS MANAGER TESTS
# ============================================================================

class TestHistoricalOptionsManager:
    """Test historical options manager."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        from backtesting.historical_options_manager import (
            HistoricalOptionsManager,
            HistoricalOptionsConfig,
        )
        
        config = HistoricalOptionsConfig(
            providers=["synthetic"],
            cache_enabled=False,
        )
        
        manager = HistoricalOptionsManager(config)
        
        assert manager is not None
        assert "synthetic" in manager.get_available_providers()
    
    def test_synthetic_provider(self):
        """Test synthetic provider returns data."""
        from backtesting.historical_options_manager import (
            HistoricalOptionsManager,
            HistoricalOptionsConfig,
        )
        
        config = HistoricalOptionsConfig(
            providers=["synthetic"],
            cache_enabled=False,
        )
        
        manager = HistoricalOptionsManager(config)
        
        chain = manager.get_chain(
            underlying="SPY",
            as_of_date=date(2023, 6, 15),
            spot_price=440.0,
        )
        
        assert chain is not None
        assert chain.underlying == "SPY"
        assert chain.spot_price == 440.0
        assert len(chain.contracts) > 0
        assert chain.data_source == "synthetic"
    
    def test_chain_enrichment(self):
        """Test chain metric enrichment."""
        from backtesting.historical_options_manager import (
            HistoricalOptionsManager,
            HistoricalOptionsConfig,
        )
        
        config = HistoricalOptionsConfig(
            providers=["synthetic"],
            cache_enabled=False,
        )
        
        manager = HistoricalOptionsManager(config)
        
        chain = manager.get_chain(
            underlying="SPY",
            as_of_date=date(2023, 6, 15),
            spot_price=440.0,
        )
        
        # Check enriched metrics
        assert chain.max_pain > 0
        assert chain.call_wall > 0
        assert chain.put_wall > 0
        assert chain.dealer_positioning in ("long_gamma", "short_gamma", "neutral")
    
    def test_caching(self):
        """Test caching functionality."""
        from backtesting.historical_options_manager import (
            HistoricalOptionsManager,
            HistoricalOptionsConfig,
        )
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HistoricalOptionsConfig(
                providers=["synthetic"],
                cache_enabled=True,
                cache_dir=tmpdir,
            )
            
            manager = HistoricalOptionsManager(config)
            
            # First call
            chain1 = manager.get_chain("SPY", date(2023, 6, 15), spot_price=440.0)
            
            # Second call (should hit cache)
            chain2 = manager.get_chain("SPY", date(2023, 6, 15), spot_price=440.0)
            
            assert chain1 is not None
            assert chain2 is not None
            
            # Clear cache
            manager.clear_cache()


# ============================================================================
# OPTIONS BACKTEST ENGINE TESTS
# ============================================================================

class TestOptionsBacktestEngine:
    """Test options backtest engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        from backtesting.options_backtest_engine import (
            OptionsBacktestEngine,
            OptionsBacktestConfig,
        )
        
        config = OptionsBacktestConfig(
            symbols=["SPY"],
            start_date="2023-06-01",
            end_date="2023-06-30",
            initial_capital=100000,
            use_real_options_data=False,  # Use synthetic
        )
        
        engine = OptionsBacktestEngine(config)
        
        assert engine is not None
        assert engine.options_manager is not None
    
    def test_find_target_option(self):
        """Test target option selection."""
        from backtesting.options_backtest_engine import (
            OptionsBacktestEngine,
            OptionsBacktestConfig,
        )
        from backtesting.historical_options_manager import HistoricalOptionsManager
        
        config = OptionsBacktestConfig(
            symbols=["SPY"],
            start_date="2023-06-01",
            end_date="2023-06-30",
            initial_capital=100000,
            use_real_options_data=False,
        )
        
        engine = OptionsBacktestEngine(config)
        
        # Get a chain
        chain = engine.options_manager.get_chain("SPY", date(2023, 6, 15), spot_price=440.0)
        
        # Find 40 delta call
        call = engine._find_target_option(chain, "call", 0.40)
        
        # Find 40 delta put
        put = engine._find_target_option(chain, "put", -0.40)
        
        if call:
            assert call.option_type == "call"
            # Delta should be reasonably close to target
            if call.delta != 0:
                assert abs(abs(call.delta) - 0.40) < 0.30
        
        if put:
            assert put.option_type == "put"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full options workflow."""
    
    def test_synthetic_chain_to_backtest(self):
        """Test synthetic data flows through backtest."""
        from backtesting.historical_options_manager import (
            HistoricalOptionsManager,
            HistoricalOptionsConfig,
        )
        
        config = HistoricalOptionsConfig(
            providers=["synthetic"],
            cache_enabled=False,
        )
        
        manager = HistoricalOptionsManager(config)
        
        # Get chain
        chain = manager.get_chain("SPY", date(2023, 6, 15), spot_price=440.0)
        
        # Verify structure
        assert chain.underlying == "SPY"
        assert len(chain.calls) > 0
        assert len(chain.puts) > 0
        
        # Verify contracts have required fields
        for contract in chain.contracts[:5]:
            assert contract.symbol
            assert contract.strike > 0
            assert contract.expiration
            assert contract.option_type in ("call", "put")
            assert contract.mid >= 0
    
    def test_credential_functions(self):
        """Test credential helper functions."""
        from config.credentials import (
            get_polygon_api_key,
            polygon_api_available,
            get_tradier_api_key,
            tradier_api_available,
        )
        
        # These should not raise
        key = get_polygon_api_key()
        available = polygon_api_available()
        
        tradier_key = get_tradier_api_key()
        tradier_available = tradier_api_available()
        
        # Functions should return consistent results
        if key:
            assert available == True
        else:
            assert available == False


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
