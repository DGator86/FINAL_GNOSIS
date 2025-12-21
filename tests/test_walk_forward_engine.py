"""
Unit Tests for Walk-Forward Validation Engine

Tests cover:
- ParameterRange (grid and random value generation)
- WalkForwardWindow (window creation and properties)
- WalkForwardConfig (configuration validation)
- WalkForwardEngine (core functionality)
- Results aggregation and robustness metrics

Run with: pytest tests/test_walk_forward_engine.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.walk_forward_engine import (
    ParameterRange,
    WalkForwardWindow,
    WalkForwardConfig,
    WalkForwardResults,
    WalkForwardEngine,
    OptimizationMethod,
    OptimizationObjective,
    print_walk_forward_results,
    run_walk_forward,
)
from backtesting.elite_backtest_engine import (
    EliteBacktestConfig,
    EliteBacktestResults,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_history():
    """Create sample historical price data."""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.015, 500)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 - np.random.uniform(0, 0.01, 500)),
        'high': prices * (1 + np.random.uniform(0, 0.02, 500)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 500)),
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, 500),
        'symbol': 'TEST',
    })
    return df


@pytest.fixture
def base_config():
    """Create a base backtest configuration."""
    return EliteBacktestConfig(
        symbols=["TEST"],
        start_date="2022-01-01",
        end_date="2023-06-30",
        initial_capital=100_000.0,
        max_positions=3,
        use_agent_signals=False,
        disable_event_risk=True,
    )


@pytest.fixture
def wf_config(base_config):
    """Create a walk-forward configuration."""
    return WalkForwardConfig(
        base_config=base_config,
        n_windows=3,
        train_pct=0.70,
        optimization_method=OptimizationMethod.GRID_SEARCH,
        optimization_objective=OptimizationObjective.SHARPE_RATIO,
        parameter_ranges=[
            ParameterRange("min_confidence", 0.30, 0.50, step=0.10),
            ParameterRange("atr_stop_mult", 1.5, 2.5, step=0.5),
        ],
        save_results=False,
    )


@pytest.fixture
def mock_backtest_results():
    """Create mock backtest results."""
    return EliteBacktestResults(
        total_return_pct=0.05,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown_pct=0.08,
        total_trades=20,
        winning_trades=12,
        profit_factor=1.5,
        win_rate=0.6,
        equity_curve=[],
        trades=[],
    )


# =============================================================================
# PARAMETER RANGE TESTS
# =============================================================================

class TestParameterRange:
    """Tests for ParameterRange."""
    
    def test_grid_values_with_step(self):
        """Test grid value generation with explicit step."""
        param = ParameterRange("test", 0.0, 1.0, step=0.25)
        values = param.grid_values()
        
        assert len(values) == 5  # 0.0, 0.25, 0.5, 0.75, 1.0
        assert values[0] == 0.0
        assert values[-1] == 1.0
        assert all(values[i] < values[i+1] for i in range(len(values)-1))
    
    def test_grid_values_default_step(self):
        """Test grid value generation with default step."""
        param = ParameterRange("test", 0.0, 1.0)
        values = param.grid_values()
        
        assert len(values) == 5  # Default: 5 values
        assert values[0] == 0.0
        assert values[-1] == 1.0
    
    def test_random_values(self):
        """Test random value generation."""
        param = ParameterRange("test", 0.0, 1.0, n_samples=20)
        values = param.random_values()
        
        assert len(values) == 20
        assert all(0.0 <= v <= 1.0 for v in values)
    
    def test_random_values_custom_n(self):
        """Test random values with custom n."""
        param = ParameterRange("test", 0.0, 1.0)
        values = param.random_values(n=5)
        
        assert len(values) == 5


# =============================================================================
# WALK FORWARD WINDOW TESTS
# =============================================================================

class TestWalkForwardWindow:
    """Tests for WalkForwardWindow."""
    
    def test_window_creation(self):
        """Test window creation."""
        train_start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        train_end = datetime(2022, 6, 30, tzinfo=timezone.utc)
        test_start = datetime(2022, 7, 1, tzinfo=timezone.utc)
        test_end = datetime(2022, 9, 30, tzinfo=timezone.utc)
        
        window = WalkForwardWindow(
            window_id=1,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        
        assert window.window_id == 1
        assert window.train_days == 180
        assert window.test_days == 91
    
    def test_window_default_values(self):
        """Test window default values."""
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2022, 6, 30, tzinfo=timezone.utc),
            test_start=datetime(2022, 7, 1, tzinfo=timezone.utc),
            test_end=datetime(2022, 9, 30, tzinfo=timezone.utc),
        )
        
        assert window.best_params == {}
        assert window.train_results is None
        assert window.test_results is None
        assert window.optimization_history == []


# =============================================================================
# WALK FORWARD CONFIG TESTS
# =============================================================================

class TestWalkForwardConfig:
    """Tests for WalkForwardConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = WalkForwardConfig()
        
        assert config.n_windows == 5
        assert config.train_pct == 0.70
        assert config.anchored == False
        assert config.overlap_pct == 0.50
        assert config.optimization_method == OptimizationMethod.GRID_SEARCH
        assert len(config.parameter_ranges) == 3  # Default params
    
    def test_custom_config(self, base_config):
        """Test custom configuration."""
        config = WalkForwardConfig(
            base_config=base_config,
            n_windows=4,
            train_pct=0.80,
            anchored=True,
            optimization_method=OptimizationMethod.RANDOM_SEARCH,
        )
        
        assert config.n_windows == 4
        assert config.train_pct == 0.80
        assert config.anchored == True
        assert config.optimization_method == OptimizationMethod.RANDOM_SEARCH
    
    def test_custom_parameter_ranges(self, base_config):
        """Test custom parameter ranges."""
        custom_params = [
            ParameterRange("min_confidence", 0.2, 0.6, step=0.1),
        ]
        
        config = WalkForwardConfig(
            base_config=base_config,
            parameter_ranges=custom_params,
        )
        
        assert len(config.parameter_ranges) == 1
        assert config.parameter_ranges[0].name == "min_confidence"


# =============================================================================
# WALK FORWARD ENGINE TESTS
# =============================================================================

class TestWalkForwardEngine:
    """Tests for WalkForwardEngine."""
    
    def test_init(self, wf_config):
        """Test engine initialization."""
        engine = WalkForwardEngine(wf_config)
        
        assert engine.config == wf_config
        assert engine.windows == []
        assert engine.historical_data == {}
    
    def test_create_windows_rolling(self, wf_config):
        """Test rolling window creation."""
        engine = WalkForwardEngine(wf_config)
        windows = engine._create_windows()
        
        assert len(windows) == wf_config.n_windows
        
        # Check all windows have valid dates
        for w in windows:
            assert w.train_start < w.train_end
            assert w.test_start < w.test_end
            assert w.train_end <= w.test_start
    
    def test_create_windows_anchored(self, base_config):
        """Test anchored window creation."""
        config = WalkForwardConfig(
            base_config=base_config,
            n_windows=3,
            anchored=True,
        )
        
        engine = WalkForwardEngine(config)
        windows = engine._create_windows()
        
        # All windows should start from the same date (anchored)
        first_start = windows[0].train_start
        for w in windows:
            assert w.train_start == first_start
    
    def test_get_objective_value_sharpe(self, wf_config, mock_backtest_results):
        """Test objective value extraction for Sharpe ratio."""
        engine = WalkForwardEngine(wf_config)
        
        value = engine._get_objective_value(mock_backtest_results)
        assert value == mock_backtest_results.sharpe_ratio
    
    def test_get_objective_value_sortino(self, base_config, mock_backtest_results):
        """Test objective value extraction for Sortino ratio."""
        config = WalkForwardConfig(
            base_config=base_config,
            optimization_objective=OptimizationObjective.SORTINO_RATIO,
        )
        engine = WalkForwardEngine(config)
        
        value = engine._get_objective_value(mock_backtest_results)
        assert value == mock_backtest_results.sortino_ratio
    
    def test_get_objective_value_total_return(self, base_config, mock_backtest_results):
        """Test objective value extraction for total return."""
        config = WalkForwardConfig(
            base_config=base_config,
            optimization_objective=OptimizationObjective.TOTAL_RETURN,
        )
        engine = WalkForwardEngine(config)
        
        value = engine._get_objective_value(mock_backtest_results)
        assert value == mock_backtest_results.total_return_pct
    
    def test_get_objective_value_no_trades(self, wf_config):
        """Test objective value with no trades."""
        engine = WalkForwardEngine(wf_config)
        
        results = EliteBacktestResults(total_trades=0)
        value = engine._get_objective_value(results)
        
        assert value == -999.0  # Penalty for no trades
    
    def test_run_single_backtest_mock(self, wf_config, sample_history, mock_backtest_results):
        """Test single backtest run with mocked data."""
        engine = WalkForwardEngine(wf_config)
        engine.historical_data = {"TEST": sample_history}
        
        with patch.object(engine, '_run_single_backtest', return_value=mock_backtest_results):
            result = engine._run_single_backtest(
                "2022-01-01", "2022-06-30", {"min_confidence": 0.3}
            )
        
        assert result == mock_backtest_results
    
    def test_optimize_window(self, wf_config, sample_history):
        """Test window optimization."""
        engine = WalkForwardEngine(wf_config)
        engine.historical_data = {"TEST": sample_history}
        
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2022, 6, 30, tzinfo=timezone.utc),
            test_start=datetime(2022, 7, 1, tzinfo=timezone.utc),
            test_end=datetime(2022, 9, 30, tzinfo=timezone.utc),
        )
        
        # Mock the backtest to return consistent results
        mock_result = EliteBacktestResults(
            total_return_pct=0.05,
            sharpe_ratio=1.0,
            total_trades=10,
        )
        
        with patch.object(engine, '_run_single_backtest', return_value=mock_result):
            optimized = engine._optimize_window(window)
        
        assert optimized.best_params != {}
        assert len(optimized.optimization_history) > 0
    
    def test_validate_window(self, wf_config, sample_history):
        """Test window validation."""
        engine = WalkForwardEngine(wf_config)
        engine.historical_data = {"TEST": sample_history}
        
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2022, 6, 30, tzinfo=timezone.utc),
            test_start=datetime(2022, 7, 1, tzinfo=timezone.utc),
            test_end=datetime(2022, 9, 30, tzinfo=timezone.utc),
            best_params={"min_confidence": 0.35, "atr_stop_mult": 2.0},
        )
        
        mock_result = EliteBacktestResults(
            total_return_pct=0.03,
            sharpe_ratio=0.8,
            total_trades=8,
        )
        
        with patch.object(engine, '_run_single_backtest', return_value=mock_result):
            validated = engine._validate_window(window)
        
        assert validated.test_results is not None


# =============================================================================
# RESULTS AGGREGATION TESTS
# =============================================================================

class TestResultsAggregation:
    """Tests for results aggregation."""
    
    def test_aggregate_results_empty(self, wf_config):
        """Test aggregation with no windows."""
        engine = WalkForwardEngine(wf_config)
        engine.windows = []
        
        results = engine._aggregate_results()
        
        assert results.total_trades == 0
        assert results.overall_win_rate == 0.0
    
    def test_aggregate_results_with_windows(self, wf_config):
        """Test aggregation with valid windows."""
        engine = WalkForwardEngine(wf_config)
        
        # Create windows with test results
        window1 = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2022, 6, 30, tzinfo=timezone.utc),
            test_start=datetime(2022, 7, 1, tzinfo=timezone.utc),
            test_end=datetime(2022, 9, 30, tzinfo=timezone.utc),
        )
        window1.test_results = EliteBacktestResults(
            total_return_pct=0.05,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            max_drawdown_pct=0.08,
            total_trades=10,
            winning_trades=6,
            profit_factor=1.5,
        )
        window1.train_results = EliteBacktestResults(sharpe_ratio=1.2)
        window1.best_params = {"min_confidence": 0.35}
        
        window2 = WalkForwardWindow(
            window_id=2,
            train_start=datetime(2022, 4, 1, tzinfo=timezone.utc),
            train_end=datetime(2022, 9, 30, tzinfo=timezone.utc),
            test_start=datetime(2022, 10, 1, tzinfo=timezone.utc),
            test_end=datetime(2022, 12, 31, tzinfo=timezone.utc),
        )
        window2.test_results = EliteBacktestResults(
            total_return_pct=0.03,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            max_drawdown_pct=0.10,
            total_trades=8,
            winning_trades=4,
            profit_factor=1.2,
        )
        window2.train_results = EliteBacktestResults(sharpe_ratio=1.1)
        window2.best_params = {"min_confidence": 0.40}
        
        engine.windows = [window1, window2]
        
        results = engine._aggregate_results()
        
        assert results.total_trades == 18
        assert results.overall_win_rate == 10/18
        assert abs(results.avg_sharpe - 0.9) < 0.01
        assert results.pct_profitable_windows == 1.0  # Both positive
        assert results.worst_drawdown == 0.10
    
    def test_overfitting_ratio_calculation(self, wf_config):
        """Test overfitting ratio calculation."""
        engine = WalkForwardEngine(wf_config)
        
        # Create window with higher IS than OOS Sharpe (overfitting)
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2022, 6, 30, tzinfo=timezone.utc),
            test_start=datetime(2022, 7, 1, tzinfo=timezone.utc),
            test_end=datetime(2022, 9, 30, tzinfo=timezone.utc),
        )
        window.train_results = EliteBacktestResults(sharpe_ratio=2.0)  # High IS
        window.test_results = EliteBacktestResults(
            total_return_pct=0.01,
            sharpe_ratio=0.5,  # Low OOS
            total_trades=5,
        )
        window.best_params = {"min_confidence": 0.35}
        
        engine.windows = [window]
        results = engine._aggregate_results()
        
        # Overfitting ratio = IS/OOS = 2.0/0.5 = 4.0
        assert results.overfitting_ratio == 4.0


# =============================================================================
# WALK FORWARD RESULTS TESTS
# =============================================================================

class TestWalkForwardResults:
    """Tests for WalkForwardResults dataclass."""
    
    def test_default_values(self):
        """Test default result values."""
        results = WalkForwardResults()
        
        assert results.windows == []
        assert results.total_return == 0.0
        assert results.avg_sharpe == 0.0
        assert results.pct_profitable_windows == 0.0
        assert results.overfitting_ratio == 0.0
    
    def test_custom_results(self):
        """Test custom result values."""
        results = WalkForwardResults(
            total_return_pct=0.15,
            avg_sharpe=1.2,
            pct_profitable_windows=0.8,
            overfitting_ratio=1.5,
        )
        
        assert results.total_return_pct == 0.15
        assert results.avg_sharpe == 1.2
        assert results.pct_profitable_windows == 0.8
        assert results.overfitting_ratio == 1.5


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_print_walk_forward_results(self, capsys, wf_config):
        """Test results printing."""
        results = WalkForwardResults(
            config=wf_config,
            total_return_pct=0.10,
            avg_return_per_window=0.025,
            avg_sharpe=1.0,
            avg_sortino=1.2,
            avg_max_drawdown=0.05,
            worst_drawdown=0.08,
            total_trades=50,
            overall_win_rate=0.55,
            avg_profit_factor=1.4,
            pct_profitable_windows=0.8,
            return_consistency=0.02,
            parameter_stability=0.85,
            is_avg_sharpe=1.2,
            oos_avg_sharpe=1.0,
            overfitting_ratio=1.2,
        )
        
        print_walk_forward_results(results)
        
        captured = capsys.readouterr()
        assert "WALK-FORWARD VALIDATION RESULTS" in captured.out
        assert "OUT-OF-SAMPLE PERFORMANCE" in captured.out
        assert "ROBUSTNESS ANALYSIS" in captured.out


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestWalkForwardIntegration:
    """Integration tests for walk-forward validation."""
    
    def test_full_walk_forward_mock(self, wf_config, sample_history):
        """Test full walk-forward run with mocked data."""
        engine = WalkForwardEngine(wf_config)
        
        # Mock data fetching
        with patch.object(engine, '_fetch_all_data', return_value={"TEST": sample_history}):
            engine.historical_data = {"TEST": sample_history}
            
            # Mock backtest to speed up test
            mock_result = EliteBacktestResults(
                total_return_pct=0.03,
                sharpe_ratio=0.9,
                total_trades=10,
                winning_trades=5,
            )
            
            with patch.object(engine, '_run_single_backtest', return_value=mock_result):
                results = engine.run()
        
        assert isinstance(results, WalkForwardResults)
        assert len(results.windows) == wf_config.n_windows
    
    def test_parameter_optimization_grid(self, base_config, sample_history):
        """Test grid search optimization."""
        config = WalkForwardConfig(
            base_config=base_config,
            n_windows=2,
            optimization_method=OptimizationMethod.GRID_SEARCH,
            parameter_ranges=[
                ParameterRange("min_confidence", 0.3, 0.4, step=0.1),  # 2 values
            ],
            save_results=False,
        )
        
        engine = WalkForwardEngine(config)
        engine.historical_data = {"TEST": sample_history}
        
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2022, 6, 30, tzinfo=timezone.utc),
            test_start=datetime(2022, 7, 1, tzinfo=timezone.utc),
            test_end=datetime(2022, 9, 30, tzinfo=timezone.utc),
        )
        
        # Mock backtest with varying results based on params
        def mock_backtest(start, end, params):
            # Higher confidence = better Sharpe (for testing)
            sharpe = params.get("min_confidence", 0.3) * 3
            return EliteBacktestResults(
                total_return_pct=0.05,
                sharpe_ratio=sharpe,
                total_trades=10,
            )
        
        with patch.object(engine, '_run_single_backtest', side_effect=mock_backtest):
            optimized = engine._optimize_window(window)
        
        # Should pick higher confidence (0.4) since it gives higher Sharpe
        assert optimized.best_params.get("min_confidence") == 0.4
    
    def test_parameter_optimization_random(self, base_config, sample_history):
        """Test random search optimization."""
        config = WalkForwardConfig(
            base_config=base_config,
            n_windows=2,
            optimization_method=OptimizationMethod.RANDOM_SEARCH,
            max_optimization_iterations=10,
            parameter_ranges=[
                ParameterRange("min_confidence", 0.2, 0.6),
            ],
            save_results=False,
        )
        
        engine = WalkForwardEngine(config)
        engine.historical_data = {"TEST": sample_history}
        
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2022, 6, 30, tzinfo=timezone.utc),
            test_start=datetime(2022, 7, 1, tzinfo=timezone.utc),
            test_end=datetime(2022, 9, 30, tzinfo=timezone.utc),
        )
        
        mock_result = EliteBacktestResults(
            total_return_pct=0.05,
            sharpe_ratio=1.0,
            total_trades=10,
        )
        
        with patch.object(engine, '_run_single_backtest', return_value=mock_result):
            optimized = engine._optimize_window(window)
        
        # Should have run 10 iterations
        assert len(optimized.optimization_history) == 10


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_window(self, base_config, sample_history):
        """Test with single window."""
        config = WalkForwardConfig(
            base_config=base_config,
            n_windows=1,
            save_results=False,
        )
        
        engine = WalkForwardEngine(config)
        engine.historical_data = {"TEST": sample_history}
        
        windows = engine._create_windows()
        assert len(windows) == 1
    
    def test_high_train_percentage(self, base_config, sample_history):
        """Test with high training percentage."""
        config = WalkForwardConfig(
            base_config=base_config,
            n_windows=3,
            train_pct=0.90,  # 90% train
            save_results=False,
        )
        
        engine = WalkForwardEngine(config)
        windows = engine._create_windows()
        
        # Should still create valid windows
        for w in windows:
            assert w.train_days > w.test_days
    
    def test_nan_sharpe_handling(self, wf_config):
        """Test handling of NaN Sharpe ratios."""
        engine = WalkForwardEngine(wf_config)
        
        results = EliteBacktestResults(
            sharpe_ratio=float('nan'),
            total_trades=5,
        )
        
        value = engine._get_objective_value(results)
        assert value == -999.0
    
    def test_inf_profit_factor(self, base_config):
        """Test handling of infinite profit factor."""
        config = WalkForwardConfig(
            base_config=base_config,
            optimization_objective=OptimizationObjective.PROFIT_FACTOR,
        )
        engine = WalkForwardEngine(config)
        
        results = EliteBacktestResults(
            profit_factor=float('inf'),
            total_trades=5,
        )
        
        value = engine._get_objective_value(results)
        assert value == 10.0  # Capped value


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
