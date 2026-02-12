"""
Tests for Strategy Parameter Optimizer

Comprehensive tests for parameter optimization including:
- Parameter definitions and validation
- Optimization methods (grid, random)
- Walk-forward integration
- Result analysis
- Recommendations generation

Author: Super Gnosis Elite Trading System
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from backtesting.strategy_optimizer import (
    StrategyOptimizer,
    StrategyOptimizationResults,
    OptimizationResult,
    ParameterCategory,
    OptimizableParameter,
    optimize_strategy,
    print_optimization_results,
)
from backtesting.walk_forward_engine import (
    ParameterRange,
    OptimizationMethod,
    OptimizationObjective,
    WalkForwardResults,
    WalkForwardWindow,
)
from backtesting.elite_backtest_engine import (
    EliteBacktestConfig,
    EliteBacktestResults,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def optimizer():
    """Create a basic optimizer instance."""
    return StrategyOptimizer(
        symbols=["SPY"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        use_walk_forward=False,  # Faster for tests
    )


@pytest.fixture
def optimizer_with_wf():
    """Create optimizer with walk-forward enabled."""
    return StrategyOptimizer(
        symbols=["SPY"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        use_walk_forward=True,
        n_windows=3,
    )


@pytest.fixture
def mock_backtest_results():
    """Create mock backtest results."""
    return EliteBacktestResults(
        initial_capital=100000.0,
        final_capital=110000.0,
        total_return=10000.0,
        total_return_pct=0.10,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown_pct=0.05,
        win_rate=0.55,
        profit_factor=1.8,
        total_trades=50,
    )


@pytest.fixture
def sample_optimization_results():
    """Create sample optimization results."""
    results = StrategyOptimizationResults(
        optimization_objective=OptimizationObjective.SHARPE_RATIO,
        parameters_optimized=["min_confidence", "atr_stop_mult"],
        total_combinations=25,
        best_parameters={"min_confidence": 0.35, "atr_stop_mult": 2.5},
        best_objective_value=1.5,
    )
    
    # Add some results history
    for i in range(25):
        results.all_results.append(OptimizationResult(
            parameters={"min_confidence": 0.25 + i * 0.02, "atr_stop_mult": 2.0 + (i % 5) * 0.25},
            objective_value=1.0 + np.random.random() * 0.5,
            total_return=0.08 + np.random.random() * 0.04,
            sharpe_ratio=1.2 + np.random.random() * 0.3,
            sortino_ratio=1.5 + np.random.random() * 0.5,
            max_drawdown=0.05 + np.random.random() * 0.03,
            win_rate=0.5 + np.random.random() * 0.1,
            profit_factor=1.5 + np.random.random() * 0.5,
            total_trades=40 + int(np.random.random() * 20),
        ))
    
    return results


# =============================================================================
# PARAMETER DEFINITION TESTS
# =============================================================================

class TestOptimizableParameter:
    """Tests for OptimizableParameter dataclass."""
    
    def test_parameter_creation(self):
        """Test creating an optimizable parameter."""
        param = OptimizableParameter(
            name="test_param",
            category=ParameterCategory.POSITION_SIZING,
            min_value=0.1,
            max_value=0.5,
            default_value=0.25,
            step=0.05,
            description="Test parameter",
        )
        
        assert param.name == "test_param"
        assert param.min_value == 0.1
        assert param.max_value == 0.5
        assert param.default_value == 0.25
    
    def test_parameter_to_range(self):
        """Test converting parameter to ParameterRange."""
        param = OptimizableParameter(
            name="test",
            category=ParameterCategory.RISK_MANAGEMENT,
            min_value=1.0,
            max_value=3.0,
            default_value=2.0,
            step=0.5,
        )
        
        range_obj = param.to_range()
        
        assert isinstance(range_obj, ParameterRange)
        assert range_obj.name == "test"
        assert range_obj.min_value == 1.0
        assert range_obj.max_value == 3.0
    
    def test_parameter_validation(self):
        """Test parameter value validation."""
        param = OptimizableParameter(
            name="test",
            category=ParameterCategory.POSITION_SIZING,
            min_value=0.1,
            max_value=0.5,
            default_value=0.25,
        )
        
        # Value within range
        assert param.validate(0.3) == 0.3
        
        # Value below min
        assert param.validate(0.05) == 0.1
        
        # Value above max
        assert param.validate(0.6) == 0.5
    
    def test_parameter_integer_validation(self):
        """Test integer parameter validation."""
        param = OptimizableParameter(
            name="dte",
            category=ParameterCategory.DTE_SETTINGS,
            min_value=7,
            max_value=45,
            default_value=14,
            integer_only=True,
        )
        
        # Should round to integer
        assert param.validate(14.7) == 15
        assert param.validate(14.2) == 14


# =============================================================================
# OPTIMIZER INITIALIZATION TESTS
# =============================================================================

class TestOptimizerInitialization:
    """Tests for StrategyOptimizer initialization."""
    
    def test_default_initialization(self):
        """Test default optimizer creation."""
        optimizer = StrategyOptimizer()
        
        assert optimizer.symbols == ["SPY"]
        assert optimizer.optimization_objective == OptimizationObjective.SHARPE_RATIO
        assert optimizer.use_walk_forward is True
    
    def test_custom_initialization(self, optimizer):
        """Test optimizer with custom parameters."""
        assert optimizer.symbols == ["SPY"]
        assert optimizer.start_date == "2023-01-01"
        assert optimizer.end_date == "2023-12-31"
        assert optimizer.use_walk_forward is False
    
    def test_all_parameters_loaded(self, optimizer):
        """Test that all parameter categories are loaded."""
        params = optimizer.all_parameters
        
        # Check each category has parameters
        categories_found = set()
        for param in params.values():
            categories_found.add(param.category)
        
        assert ParameterCategory.POSITION_SIZING in categories_found
        assert ParameterCategory.RISK_MANAGEMENT in categories_found
        assert ParameterCategory.SIGNAL_THRESHOLDS in categories_found
    
    def test_get_parameter_info(self, optimizer):
        """Test getting parameter info."""
        info = optimizer.get_parameter_info()
        
        assert len(info) > 0
        assert "kelly_fraction" in info
        assert "min_confidence" in info
        assert "atr_stop_mult" in info
    
    def test_get_parameter_info_by_category(self, optimizer):
        """Test getting parameter info filtered by category."""
        info = optimizer.get_parameter_info(ParameterCategory.POSITION_SIZING)
        
        for name, param_info in info.items():
            assert param_info["category"] == "position_sizing"


# =============================================================================
# PARAMETER SELECTION TESTS
# =============================================================================

class TestParameterSelection:
    """Tests for parameter selection logic."""
    
    def test_select_by_name(self, optimizer):
        """Test selecting parameters by name."""
        params = optimizer._get_parameters_to_optimize(
            parameters=["kelly_fraction", "min_confidence"],
            categories=None,
        )
        
        assert len(params) == 2
        assert "kelly_fraction" in params
        assert "min_confidence" in params
    
    def test_select_by_category(self, optimizer):
        """Test selecting parameters by category."""
        params = optimizer._get_parameters_to_optimize(
            parameters=None,
            categories=[ParameterCategory.POSITION_SIZING],
        )
        
        for param in params.values():
            assert param.category == ParameterCategory.POSITION_SIZING
    
    def test_select_multiple_categories(self, optimizer):
        """Test selecting from multiple categories."""
        params = optimizer._get_parameters_to_optimize(
            parameters=None,
            categories=[
                ParameterCategory.POSITION_SIZING,
                ParameterCategory.RISK_MANAGEMENT,
            ],
        )
        
        categories = {p.category for p in params.values()}
        assert ParameterCategory.POSITION_SIZING in categories
        assert ParameterCategory.RISK_MANAGEMENT in categories
    
    def test_default_selection(self, optimizer):
        """Test default parameter selection."""
        params = optimizer._get_parameters_to_optimize(
            parameters=None,
            categories=None,
        )
        
        # Should return default set
        assert len(params) > 0
        assert "min_confidence" in params


# =============================================================================
# OPTIMIZATION METHOD TESTS
# =============================================================================

class TestOptimizationMethods:
    """Tests for optimization methods."""
    
    @patch.object(StrategyOptimizer, '_evaluate_parameters')
    def test_grid_search_generates_combinations(self, mock_eval, optimizer):
        """Test that grid search generates all combinations."""
        mock_eval.return_value = (1.5, {
            "total_return": 0.10,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": 0.05,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "total_trades": 50,
        })
        
        param_ranges = [
            ParameterRange("param1", 0.1, 0.3, step=0.1),  # 3 values
            ParameterRange("param2", 1.0, 2.0, step=0.5),  # 3 values
        ]
        
        results = optimizer._optimize_direct(
            param_ranges=param_ranges,
            method=OptimizationMethod.GRID_SEARCH,
            max_iterations=100,
        )
        
        # 3 x 3 = 9 combinations
        assert mock_eval.call_count == 9
        assert results.total_combinations == 9
    
    @patch.object(StrategyOptimizer, '_evaluate_parameters')
    def test_random_search_iterations(self, mock_eval, optimizer):
        """Test that random search runs specified iterations."""
        mock_eval.return_value = (1.5, {
            "total_return": 0.10,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": 0.05,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "total_trades": 50,
        })
        
        param_ranges = [
            ParameterRange("param1", 0.1, 0.5),
            ParameterRange("param2", 1.0, 3.0),
        ]
        
        max_iter = 20
        results = optimizer._optimize_direct(
            param_ranges=param_ranges,
            method=OptimizationMethod.RANDOM_SEARCH,
            max_iterations=max_iter,
        )
        
        assert mock_eval.call_count == max_iter
        assert results.total_combinations == max_iter
    
    @patch.object(StrategyOptimizer, '_evaluate_parameters')
    def test_best_parameters_tracked(self, mock_eval, optimizer):
        """Test that best parameters are correctly tracked."""
        # Return increasing scores
        call_count = [0]
        def mock_return(*args):
            call_count[0] += 1
            return (call_count[0], {
                "total_return": 0.10,
                "sharpe_ratio": call_count[0],
                "sortino_ratio": 2.0,
                "max_drawdown": 0.05,
                "win_rate": 0.55,
                "profit_factor": 1.8,
                "total_trades": 50,
            })
        
        mock_eval.side_effect = mock_return
        
        param_ranges = [
            ParameterRange("param1", 0.1, 0.2, step=0.1),
        ]
        
        results = optimizer._optimize_direct(
            param_ranges=param_ranges,
            method=OptimizationMethod.GRID_SEARCH,
            max_iterations=100,
        )
        
        # Last combination should have best score
        assert results.best_objective_value == call_count[0]


# =============================================================================
# RESULT ANALYSIS TESTS
# =============================================================================

class TestResultAnalysis:
    """Tests for optimization result analysis."""
    
    def test_parameter_sensitivity_calculation(self, optimizer, sample_optimization_results):
        """Test parameter sensitivity analysis."""
        params = {
            "min_confidence": optimizer.all_parameters["min_confidence"],
            "atr_stop_mult": optimizer.all_parameters["atr_stop_mult"],
        }
        
        optimizer._analyze_results(sample_optimization_results, params)
        
        # Should have sensitivity for each parameter
        assert "min_confidence" in sample_optimization_results.parameter_sensitivity
        assert "atr_stop_mult" in sample_optimization_results.parameter_sensitivity
    
    def test_optimal_ranges_calculated(self, optimizer, sample_optimization_results):
        """Test optimal range calculation."""
        params = {
            "min_confidence": optimizer.all_parameters["min_confidence"],
            "atr_stop_mult": optimizer.all_parameters["atr_stop_mult"],
        }
        
        optimizer._analyze_results(sample_optimization_results, params)
        
        # Should have optimal ranges
        assert "min_confidence" in sample_optimization_results.optimal_ranges
        assert len(sample_optimization_results.optimal_ranges["min_confidence"]) == 2


# =============================================================================
# RECOMMENDATIONS TESTS
# =============================================================================

class TestRecommendations:
    """Tests for recommendation generation."""
    
    def test_boundary_warning(self, optimizer):
        """Test warning when optimal value is at boundary."""
        results = StrategyOptimizationResults(
            best_parameters={"min_confidence": 0.21},  # Near min of 0.20
        )
        
        params = {"min_confidence": optimizer.all_parameters["min_confidence"]}
        
        optimizer._generate_recommendations(results, params)
        
        # Should warn about being near minimum
        has_boundary_warning = any("minimum" in rec.lower() for rec in results.recommendations)
        assert has_boundary_warning
    
    def test_low_sharpe_warning(self, optimizer):
        """Test warning for low Sharpe ratio."""
        results = StrategyOptimizationResults(
            optimization_objective=OptimizationObjective.SHARPE_RATIO,
            best_objective_value=0.3,
        )
        
        optimizer._generate_recommendations(results, {})
        
        # Should warn about low Sharpe
        has_sharpe_warning = any("sharpe" in warn.lower() for warn in results.warnings)
        assert has_sharpe_warning
    
    def test_overfitting_warning(self, optimizer):
        """Test warning for overfitting in walk-forward results."""
        wf_results = WalkForwardResults()
        wf_results.overfitting_ratio = 2.5
        wf_results.is_avg_sharpe = 2.0
        wf_results.oos_avg_sharpe = 0.8
        wf_results.pct_profitable_windows = 0.6
        wf_results.parameter_stability = 0.7
        
        results = StrategyOptimizationResults(
            walk_forward_results=wf_results,
        )
        
        optimizer._generate_recommendations(results, {})
        
        # Should warn about overfitting
        has_overfit_warning = any("overfit" in warn.lower() for warn in results.warnings)
        assert has_overfit_warning


# =============================================================================
# CONVENIENCE METHOD TESTS
# =============================================================================

class TestConvenienceMethods:
    """Tests for convenience methods."""
    
    @patch.object(StrategyOptimizer, 'optimize_parameters')
    def test_optimize_position_sizing(self, mock_opt, optimizer):
        """Test position sizing optimization convenience method."""
        optimizer.optimize_position_sizing()
        
        mock_opt.assert_called_once()
        call_args = mock_opt.call_args
        assert ParameterCategory.POSITION_SIZING in call_args.kwargs.get('categories', [])
    
    @patch.object(StrategyOptimizer, 'optimize_parameters')
    def test_optimize_risk_management(self, mock_opt, optimizer):
        """Test risk management optimization convenience method."""
        optimizer.optimize_risk_management()
        
        mock_opt.assert_called_once()
        call_args = mock_opt.call_args
        assert ParameterCategory.RISK_MANAGEMENT in call_args.kwargs.get('categories', [])
    
    @patch.object(StrategyOptimizer, 'optimize_parameters')
    def test_optimize_signal_thresholds(self, mock_opt, optimizer):
        """Test signal thresholds optimization convenience method."""
        optimizer.optimize_signal_thresholds()
        
        mock_opt.assert_called_once()
        call_args = mock_opt.call_args
        assert ParameterCategory.SIGNAL_THRESHOLDS in call_args.kwargs.get('categories', [])


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the optimizer."""
    
    @patch.object(StrategyOptimizer, '_evaluate_parameters')
    def test_full_optimization_flow(self, mock_eval, optimizer):
        """Test complete optimization flow."""
        mock_eval.return_value = (1.5, {
            "total_return": 0.10,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": 0.05,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "total_trades": 50,
        })
        
        results = optimizer.optimize_parameters(
            parameters=["min_confidence"],
            method=OptimizationMethod.GRID_SEARCH,
            save_results=False,
        )
        
        assert results.best_parameters is not None
        assert "min_confidence" in results.best_parameters
        assert results.best_objective_value > 0
        assert len(results.all_results) > 0
    
    @patch.object(StrategyOptimizer, '_evaluate_parameters')
    def test_optimization_with_analysis(self, mock_eval, optimizer):
        """Test optimization includes analysis."""
        mock_eval.return_value = (1.5, {
            "total_return": 0.10,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": 0.05,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "total_trades": 50,
        })
        
        results = optimizer.optimize_parameters(
            parameters=["min_confidence", "atr_stop_mult"],
            method=OptimizationMethod.GRID_SEARCH,
            save_results=False,
        )
        
        # Should have sensitivity analysis
        assert len(results.parameter_sensitivity) > 0
        
        # Should have recommendations
        assert isinstance(results.recommendations, list)
        assert isinstance(results.warnings, list)


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunction:
    """Tests for the optimize_strategy factory function."""
    
    @patch.object(StrategyOptimizer, 'optimize_parameters')
    def test_optimize_strategy_function(self, mock_opt):
        """Test optimize_strategy convenience function."""
        mock_opt.return_value = StrategyOptimizationResults()
        
        optimize_strategy(
            symbols=["SPY"],
            parameters=["min_confidence"],
            objective="sharpe_ratio",
            method="grid_search",
            use_walk_forward=False,
        )
        
        mock_opt.assert_called_once()


# =============================================================================
# OUTPUT TESTS
# =============================================================================

class TestOutput:
    """Tests for output formatting."""
    
    def test_print_results(self, sample_optimization_results, capsys):
        """Test result printing."""
        sample_optimization_results.recommendations = ["Test recommendation"]
        sample_optimization_results.warnings = ["Test warning"]
        sample_optimization_results.parameter_sensitivity = {"min_confidence": 0.5}
        sample_optimization_results.optimal_ranges = {"min_confidence": (0.25, 0.45)}
        
        print_optimization_results(sample_optimization_results)
        
        captured = capsys.readouterr()
        assert "STRATEGY PARAMETER OPTIMIZATION" in captured.out
        assert "BEST PARAMETERS" in captured.out
        assert "min_confidence" in captured.out


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_results(self, optimizer):
        """Test handling of empty results."""
        results = StrategyOptimizationResults()
        
        optimizer._analyze_results(results, {})
        optimizer._generate_recommendations(results, {})
        
        # Should not raise
        assert results.parameter_sensitivity == {}
    
    def test_unknown_parameter(self, optimizer):
        """Test handling of unknown parameter."""
        params = optimizer._get_parameters_to_optimize(
            parameters=["unknown_param"],
            categories=None,
        )
        
        # Should return default set, not including unknown
        assert "unknown_param" not in params
    
    def test_single_result(self, optimizer):
        """Test analysis with single result."""
        results = StrategyOptimizationResults()
        results.all_results = [OptimizationResult(
            parameters={"min_confidence": 0.3},
            objective_value=1.5,
            total_return=0.10,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.05,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=50,
        )]
        
        params = {"min_confidence": optimizer.all_parameters["min_confidence"]}
        
        # Should not raise
        optimizer._analyze_results(results, params)


# =============================================================================
# PARAMETER RANGE TESTS
# =============================================================================

class TestParameterRanges:
    """Tests for parameter range validation."""
    
    def test_position_sizing_ranges(self, optimizer):
        """Test position sizing parameter ranges are valid."""
        params = optimizer.POSITION_SIZING_PARAMS
        
        assert params["kelly_fraction"].min_value >= 0
        assert params["kelly_fraction"].max_value <= 1
        assert params["max_position_pct"].max_value <= 0.25  # Reasonable max
    
    def test_risk_management_ranges(self, optimizer):
        """Test risk management parameter ranges are valid."""
        params = optimizer.RISK_MANAGEMENT_PARAMS
        
        assert params["atr_stop_mult"].min_value >= 0.5
        assert params["atr_target_mult"].min_value > params["atr_stop_mult"].min_value
    
    def test_signal_threshold_ranges(self, optimizer):
        """Test signal threshold parameter ranges are valid."""
        params = optimizer.SIGNAL_THRESHOLD_PARAMS
        
        assert params["min_confidence"].min_value >= 0
        assert params["min_confidence"].max_value <= 1
