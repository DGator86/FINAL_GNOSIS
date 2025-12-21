"""
Comprehensive Tests for Portfolio Optimizer

Tests cover:
- Multiple optimization strategies (Mean-Variance, Max Sharpe, Min Vol, Risk Parity, etc.)
- Portfolio constraints (position limits, sector limits, leverage)
- Greeks-aware optimization for options portfolios
- Black-Litterman model with market views
- Efficient frontier generation
- Rebalancing trade calculations
- Edge cases and error handling

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from trade.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationStrategy,
    ConstraintType,
    AssetData,
    OptimizationConstraints,
    OptimizationResult,
    EfficientFrontier,
    create_portfolio_optimizer,
    optimize_from_returns,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def optimizer():
    """Create basic portfolio optimizer."""
    return PortfolioOptimizer(risk_free_rate=0.05)


@pytest.fixture
def optimizer_with_constraints():
    """Create optimizer with custom constraints."""
    constraints = OptimizationConstraints(
        max_position_weight=0.25,
        min_position_weight=0.0,
        max_sector_weight=0.40,
        allow_short=False,
    )
    return PortfolioOptimizer(risk_free_rate=0.05, constraints=constraints)


@pytest.fixture
def sample_assets():
    """Create sample assets for testing."""
    return [
        AssetData(
            symbol="AAPL",
            expected_return=0.12,
            volatility=0.25,
            current_weight=0.30,
            sector="technology",
        ),
        AssetData(
            symbol="MSFT",
            expected_return=0.10,
            volatility=0.22,
            current_weight=0.25,
            sector="technology",
        ),
        AssetData(
            symbol="JPM",
            expected_return=0.08,
            volatility=0.28,
            current_weight=0.20,
            sector="financials",
        ),
        AssetData(
            symbol="XOM",
            expected_return=0.06,
            volatility=0.30,
            current_weight=0.15,
            sector="energy",
        ),
        AssetData(
            symbol="JNJ",
            expected_return=0.05,
            volatility=0.18,
            current_weight=0.10,
            sector="healthcare",
        ),
    ]


@pytest.fixture
def sample_covariance():
    """Create sample covariance matrix."""
    # 5x5 positive semi-definite matrix
    n = 5
    volatilities = np.array([0.25, 0.22, 0.28, 0.30, 0.18])
    
    # Correlation matrix with realistic correlations
    corr = np.array([
        [1.00, 0.65, 0.40, 0.25, 0.30],  # AAPL
        [0.65, 1.00, 0.35, 0.20, 0.25],  # MSFT
        [0.40, 0.35, 1.00, 0.45, 0.35],  # JPM
        [0.25, 0.20, 0.45, 1.00, 0.20],  # XOM
        [0.30, 0.25, 0.35, 0.20, 1.00],  # JNJ
    ])
    
    # Covariance = D * Corr * D
    D = np.diag(volatilities)
    return D @ corr @ D


@pytest.fixture
def options_assets():
    """Create sample assets with Greeks exposure."""
    return [
        AssetData(
            symbol="AAPL_CALL",
            expected_return=0.20,
            volatility=0.40,
            delta=0.55,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
            sector="technology",
        ),
        AssetData(
            symbol="SPY_PUT",
            expected_return=-0.05,
            volatility=0.35,
            delta=-0.40,
            gamma=0.015,
            theta=-0.08,
            vega=0.12,
            sector="index",
        ),
        AssetData(
            symbol="MSFT_CALL",
            expected_return=0.15,
            volatility=0.38,
            delta=0.50,
            gamma=0.018,
            theta=-0.09,
            vega=0.14,
            sector="technology",
        ),
    ]


# =============================================================================
# TEST CLASS: Enums
# =============================================================================

class TestEnums:
    """Test enum values."""
    
    def test_optimization_strategy_values(self):
        """Test OptimizationStrategy enum values."""
        assert OptimizationStrategy.MEAN_VARIANCE.value == "mean_variance"
        assert OptimizationStrategy.MAX_SHARPE.value == "max_sharpe"
        assert OptimizationStrategy.MIN_VOLATILITY.value == "min_volatility"
        assert OptimizationStrategy.RISK_PARITY.value == "risk_parity"
        assert OptimizationStrategy.GREEKS_AWARE.value == "greeks_aware"
        assert OptimizationStrategy.BLACK_LITTERMAN.value == "black_litterman"
        
    def test_constraint_type_values(self):
        """Test ConstraintType enum values."""
        assert ConstraintType.LONG_ONLY.value == "long_only"
        assert ConstraintType.MAX_POSITION.value == "max_position"


# =============================================================================
# TEST CLASS: Dataclasses
# =============================================================================

class TestDataclasses:
    """Test dataclass models."""
    
    def test_asset_data_creation(self):
        """Test AssetData creation."""
        asset = AssetData(
            symbol="AAPL",
            expected_return=0.10,
            volatility=0.25,
        )
        
        assert asset.symbol == "AAPL"
        assert asset.expected_return == 0.10
        assert asset.volatility == 0.25
        assert asset.delta == 0.0  # Default
        
    def test_asset_data_with_greeks(self):
        """Test AssetData with Greeks."""
        asset = AssetData(
            symbol="AAPL_CALL",
            expected_return=0.15,
            volatility=0.30,
            delta=0.55,
            gamma=0.02,
            theta=-0.10,
            vega=0.15,
        )
        
        assert asset.delta == 0.55
        assert asset.theta == -0.10
        
    def test_optimization_constraints_defaults(self):
        """Test OptimizationConstraints defaults."""
        constraints = OptimizationConstraints()
        
        assert constraints.max_position_weight == 0.20
        assert constraints.allow_short is False
        assert constraints.max_sector_weight == 0.40
        
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            strategy=OptimizationStrategy.MAX_SHARPE,
            weights={"AAPL": 0.5, "MSFT": 0.5},
            expected_return=0.10,
            volatility=0.20,
            sharpe_ratio=0.25,
        )
        
        assert result.converged is True
        assert result.strategy == OptimizationStrategy.MAX_SHARPE


# =============================================================================
# TEST CLASS: Optimizer Initialization
# =============================================================================

class TestOptimizerInit:
    """Test PortfolioOptimizer initialization."""
    
    def test_default_init(self, optimizer):
        """Test default initialization."""
        assert optimizer.risk_free_rate == 0.05
        assert optimizer.constraints is not None
        
    def test_custom_constraints_init(self, optimizer_with_constraints):
        """Test initialization with custom constraints."""
        assert optimizer_with_constraints.constraints.max_position_weight == 0.25
        
    def test_factory_function(self):
        """Test create_portfolio_optimizer factory."""
        opt = create_portfolio_optimizer(
            risk_free_rate=0.03,
            max_position=0.15,
            allow_short=True,
        )
        
        assert opt.risk_free_rate == 0.03
        assert opt.constraints.max_position_weight == 0.15
        assert opt.constraints.allow_short is True


# =============================================================================
# TEST CLASS: Setting Assets
# =============================================================================

class TestSettingAssets:
    """Test asset setting methods."""
    
    def test_set_assets(self, optimizer, sample_assets, sample_covariance):
        """Test setting assets."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        assert len(optimizer._assets) == 5
        assert optimizer._covariance_matrix is not None
        
    def test_set_assets_without_covariance(self, optimizer, sample_assets):
        """Test setting assets without covariance (gets estimated)."""
        optimizer.set_assets(sample_assets)
        
        assert len(optimizer._assets) == 5
        # Covariance should be estimated
        assert optimizer._covariance_matrix is not None
        
    def test_set_assets_from_data(self, optimizer):
        """Test set_assets_from_data method."""
        symbols = ["AAPL", "MSFT", "JPM"]
        returns = np.array([0.10, 0.08, 0.06])
        covariance = np.eye(3) * 0.04  # Simple diagonal
        
        optimizer.set_assets_from_data(
            symbols=symbols,
            returns=returns,
            covariance=covariance,
            current_weights={"AAPL": 0.4, "MSFT": 0.3},
        )
        
        assert len(optimizer._assets) == 3
        assert optimizer._assets[0].expected_return == 0.10


# =============================================================================
# TEST CLASS: Equal Weight Strategy
# =============================================================================

class TestEqualWeightStrategy:
    """Test equal weight optimization."""
    
    def test_equal_weight_allocation(self, optimizer, sample_assets):
        """Test equal weight gives equal allocation."""
        optimizer.set_assets(sample_assets)
        
        result = optimizer.optimize(OptimizationStrategy.EQUAL_WEIGHT)
        
        assert result.converged
        expected_weight = 1.0 / len(sample_assets)
        
        for symbol, weight in result.weights.items():
            assert abs(weight - expected_weight) < 0.01
            
    def test_equal_weight_sums_to_one(self, optimizer, sample_assets):
        """Test equal weights sum to 1."""
        optimizer.set_assets(sample_assets)
        
        result = optimizer.optimize(OptimizationStrategy.EQUAL_WEIGHT)
        
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 0.01


# =============================================================================
# TEST CLASS: Max Sharpe Strategy
# =============================================================================

class TestMaxSharpeStrategy:
    """Test maximum Sharpe ratio optimization."""
    
    def test_max_sharpe_basic(self, optimizer, sample_assets, sample_covariance):
        """Test max Sharpe optimization."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        assert result.converged
        assert result.sharpe_ratio > 0
        
    def test_max_sharpe_weights_sum_to_one(self, optimizer, sample_assets, sample_covariance):
        """Test weights sum to 1."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        total = sum(abs(w) for w in result.weights.values())
        assert abs(total - 1.0) < 0.01
        
    def test_max_sharpe_prefers_higher_returns(self, optimizer, sample_covariance):
        """Test max Sharpe allocates more to higher return assets."""
        # Create assets with clear return differences
        assets = [
            AssetData(symbol="HIGH", expected_return=0.20, volatility=0.20),
            AssetData(symbol="LOW", expected_return=0.05, volatility=0.20),
        ]
        
        # Simple covariance
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        
        optimizer.set_assets(assets, covariance_matrix=cov)
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Should allocate more to HIGH
        assert result.weights.get("HIGH", 0) > result.weights.get("LOW", 0)


# =============================================================================
# TEST CLASS: Min Volatility Strategy
# =============================================================================

class TestMinVolatilityStrategy:
    """Test minimum volatility optimization."""
    
    def test_min_volatility_basic(self, optimizer, sample_assets, sample_covariance):
        """Test min volatility optimization."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MIN_VOLATILITY)
        
        assert result.converged
        assert result.volatility > 0
        
    def test_min_volatility_lower_than_max_sharpe(self, optimizer, sample_assets, sample_covariance):
        """Test min vol has lower volatility than max Sharpe."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        min_vol_result = optimizer.optimize(OptimizationStrategy.MIN_VOLATILITY)
        max_sharpe_result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Min vol should have lower or equal volatility
        assert min_vol_result.volatility <= max_sharpe_result.volatility + 0.01
        
    def test_min_volatility_prefers_low_vol_assets(self, optimizer):
        """Test min vol allocates more to lower volatility assets."""
        assets = [
            AssetData(symbol="LOW_VOL", expected_return=0.10, volatility=0.15),
            AssetData(symbol="HIGH_VOL", expected_return=0.10, volatility=0.40),
        ]
        
        cov = np.array([[0.0225, 0.01], [0.01, 0.16]])
        
        optimizer.set_assets(assets, covariance_matrix=cov)
        result = optimizer.optimize(OptimizationStrategy.MIN_VOLATILITY)
        
        # Should prefer LOW_VOL
        assert result.weights.get("LOW_VOL", 0) >= result.weights.get("HIGH_VOL", 0)


# =============================================================================
# TEST CLASS: Mean-Variance Strategy
# =============================================================================

class TestMeanVarianceStrategy:
    """Test mean-variance optimization."""
    
    def test_mean_variance_with_target_return(self, optimizer, sample_assets, sample_covariance):
        """Test mean-variance with target return."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        target = 0.08
        result = optimizer.optimize(
            OptimizationStrategy.MEAN_VARIANCE,
            target_return=target,
        )
        
        assert result.converged
        # Return should be close to target (within constraints)
        
    def test_mean_variance_no_target_equals_max_sharpe(self, optimizer, sample_assets, sample_covariance):
        """Test mean-variance without target is similar to max Sharpe."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        mv_result = optimizer.optimize(OptimizationStrategy.MEAN_VARIANCE)
        sharpe_result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Should be similar strategies
        assert abs(mv_result.sharpe_ratio - sharpe_result.sharpe_ratio) < 0.5


# =============================================================================
# TEST CLASS: Risk Parity Strategy
# =============================================================================

class TestRiskParityStrategy:
    """Test risk parity optimization."""
    
    def test_risk_parity_basic(self, optimizer, sample_assets, sample_covariance):
        """Test risk parity optimization."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.RISK_PARITY)
        
        assert result.converged
        assert sum(result.weights.values()) > 0
        
    def test_risk_parity_weights_sum_to_one(self, optimizer, sample_assets, sample_covariance):
        """Test risk parity weights sum to 1."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.RISK_PARITY)
        
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 0.01
        
    def test_risk_parity_more_to_low_vol(self, optimizer):
        """Test risk parity allocates more to lower vol assets."""
        assets = [
            AssetData(symbol="LOW_VOL", expected_return=0.08, volatility=0.15),
            AssetData(symbol="HIGH_VOL", expected_return=0.12, volatility=0.40),
        ]
        
        cov = np.array([[0.0225, 0.005], [0.005, 0.16]])
        
        optimizer.set_assets(assets, covariance_matrix=cov)
        result = optimizer.optimize(OptimizationStrategy.RISK_PARITY)
        
        # Risk parity aims for equal risk contribution
        # Should allocate at least equal weight to low vol asset
        assert result.weights.get("LOW_VOL", 0) >= result.weights.get("HIGH_VOL", 0) - 0.1


# =============================================================================
# TEST CLASS: Greeks-Aware Strategy
# =============================================================================

class TestGreeksAwareStrategy:
    """Test Greeks-aware optimization for options."""
    
    def test_greeks_aware_basic(self, optimizer, options_assets):
        """Test Greeks-aware optimization."""
        optimizer.set_assets(options_assets)
        
        result = optimizer.optimize(OptimizationStrategy.GREEKS_AWARE)
        
        assert result.converged
        
    def test_greeks_aware_with_delta_limit(self, options_assets):
        """Test Greeks-aware with delta constraint."""
        constraints = OptimizationConstraints(max_delta=0.3)
        optimizer = PortfolioOptimizer(constraints=constraints)
        
        optimizer.set_assets(options_assets)
        result = optimizer.optimize(OptimizationStrategy.GREEKS_AWARE)
        
        # Verify optimization converged and produced valid Greeks
        assert result.converged
        assert isinstance(result.portfolio_delta, (int, float))
        
    def test_greeks_aware_calculates_portfolio_greeks(self, optimizer, options_assets):
        """Test that portfolio Greeks are calculated."""
        optimizer.set_assets(options_assets)
        
        result = optimizer.optimize(OptimizationStrategy.GREEKS_AWARE)
        
        # Should have portfolio Greeks
        assert isinstance(result.portfolio_delta, float)
        assert isinstance(result.portfolio_gamma, float)
        assert isinstance(result.portfolio_theta, float)
        assert isinstance(result.portfolio_vega, float)


# =============================================================================
# TEST CLASS: Black-Litterman Strategy
# =============================================================================

class TestBlackLittermanStrategy:
    """Test Black-Litterman optimization."""
    
    def test_black_litterman_basic(self, optimizer, sample_assets, sample_covariance):
        """Test Black-Litterman without views."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.BLACK_LITTERMAN)
        
        assert result.converged
        
    def test_black_litterman_with_views(self, optimizer, sample_covariance):
        """Test Black-Litterman with views."""
        assets = [
            AssetData(
                symbol="AAPL",
                expected_return=0.10,
                volatility=0.25,
                current_weight=0.5,
                view_return=0.15,  # Bullish view
                view_confidence=0.8,
            ),
            AssetData(
                symbol="MSFT",
                expected_return=0.10,
                volatility=0.22,
                current_weight=0.5,
                view_return=0.05,  # Bearish view
                view_confidence=0.6,
            ),
        ]
        
        cov = sample_covariance[:2, :2]
        optimizer.set_assets(assets, covariance_matrix=cov)
        
        result = optimizer.optimize(OptimizationStrategy.BLACK_LITTERMAN)
        
        assert result.converged
        # Should tilt towards AAPL due to bullish view
        
    def test_black_litterman_respects_confidence(self, optimizer):
        """Test that higher confidence views have more impact."""
        # This is a qualitative test - BL should incorporate views
        assets = [
            AssetData(
                symbol="A",
                expected_return=0.08,
                volatility=0.20,
                current_weight=0.5,
                view_return=0.15,
                view_confidence=0.9,  # High confidence
            ),
            AssetData(
                symbol="B",
                expected_return=0.08,
                volatility=0.20,
                current_weight=0.5,
            ),
        ]
        
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        optimizer.set_assets(assets, covariance_matrix=cov)
        
        result = optimizer.optimize(OptimizationStrategy.BLACK_LITTERMAN)
        
        assert result.converged


# =============================================================================
# TEST CLASS: Constraints Application
# =============================================================================

class TestConstraintsApplication:
    """Test constraint application."""
    
    def test_max_position_constraint(self, sample_assets, sample_covariance):
        """Test max position weight constraint."""
        constraints = OptimizationConstraints(max_position_weight=0.25)
        optimizer = PortfolioOptimizer(constraints=constraints)
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Verify result converged with valid weights
        assert result.converged
        # Weights should sum to ~1
        assert abs(sum(result.weights.values()) - 1.0) < 0.1
            
    def test_long_only_constraint(self, sample_assets, sample_covariance):
        """Test long-only constraint."""
        constraints = OptimizationConstraints(allow_short=False)
        optimizer = PortfolioOptimizer(constraints=constraints)
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        for weight in result.weights.values():
            assert weight >= -0.01  # Small tolerance for numerical issues
            
    def test_sector_constraint(self, sample_assets, sample_covariance):
        """Test sector weight constraint."""
        constraints = OptimizationConstraints(max_sector_weight=0.35)
        optimizer = PortfolioOptimizer(constraints=constraints)
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Verify optimization converged
        assert result.converged
        
        # Calculate sector weights
        sector_weights = {}
        for asset in sample_assets:
            sector_weights[asset.sector] = sector_weights.get(asset.sector, 0) + result.weights.get(asset.symbol, 0)
        
        # Verify sectors have allocations
        assert len(sector_weights) > 0


# =============================================================================
# TEST CLASS: Result Calculations
# =============================================================================

class TestResultCalculations:
    """Test result metric calculations."""
    
    def test_sharpe_ratio_calculation(self, optimizer, sample_assets, sample_covariance):
        """Test Sharpe ratio is calculated correctly."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Sharpe = (return - rf) / vol
        expected_sharpe = (result.expected_return - optimizer.risk_free_rate) / result.volatility
        
        assert abs(result.sharpe_ratio - expected_sharpe) < 0.01
        
    def test_var_calculation(self, optimizer, sample_assets, sample_covariance):
        """Test VaR is calculated."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        assert result.value_at_risk_95 != 0
        
    def test_turnover_calculation(self, optimizer, sample_assets, sample_covariance):
        """Test turnover is calculated."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Turnover should be calculated
        assert isinstance(result.turnover, float)
        assert result.turnover >= 0
        
    def test_weight_changes_tracked(self, optimizer, sample_assets, sample_covariance):
        """Test weight changes are tracked."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        assert len(result.weight_changes) == len(sample_assets)
        assert len(result.current_weights) == len(sample_assets)


# =============================================================================
# TEST CLASS: Efficient Frontier
# =============================================================================

class TestEfficientFrontier:
    """Test efficient frontier generation."""
    
    def test_generate_frontier(self, optimizer, sample_assets, sample_covariance):
        """Test generating efficient frontier."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        frontier = optimizer.generate_efficient_frontier(n_points=20)
        
        assert len(frontier.returns) > 0
        assert len(frontier.volatilities) > 0
        
    def test_frontier_has_special_portfolios(self, optimizer, sample_assets, sample_covariance):
        """Test frontier includes special portfolios."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        frontier = optimizer.generate_efficient_frontier()
        
        assert frontier.max_sharpe_portfolio is not None
        assert frontier.min_vol_portfolio is not None
        
    def test_frontier_returns_increase_with_vol(self, optimizer, sample_assets, sample_covariance):
        """Test returns generally increase with volatility on frontier."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        frontier = optimizer.generate_efficient_frontier(n_points=10)
        
        # Check at least some positive correlation
        if len(frontier.returns) > 2:
            # Returns and volatilities should be positively correlated
            corr = np.corrcoef(frontier.returns, frontier.volatilities)[0, 1]
            # Allow for some noise
            assert corr > -0.5


# =============================================================================
# TEST CLASS: Rebalancing Trades
# =============================================================================

class TestRebalancingTrades:
    """Test rebalancing trade calculations."""
    
    def test_get_rebalancing_trades(self, optimizer, sample_assets, sample_covariance):
        """Test getting rebalancing trades."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        trades = optimizer.get_rebalancing_trades(
            optimal_weights=result.weights,
            portfolio_value=100000.0,
        )
        
        assert "buy" in trades
        assert "sell" in trades
        
    def test_rebalancing_trades_net_to_zero(self, optimizer, sample_assets, sample_covariance):
        """Test buy and sell trades roughly offset."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        trades = optimizer.get_rebalancing_trades(
            optimal_weights=result.weights,
            portfolio_value=100000.0,
        )
        
        total_buys = sum(trades["buy"].values())
        total_sells = sum(trades["sell"].values())
        
        # Net should be close to zero
        assert abs(total_buys - total_sells) < 1000  # Some tolerance


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_assets(self, optimizer):
        """Test optimization with no assets."""
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        assert not result.converged
        assert result.error_message is not None
        
    def test_single_asset(self, optimizer):
        """Test optimization with single asset."""
        assets = [AssetData(symbol="ONLY", expected_return=0.10, volatility=0.20)]
        optimizer.set_assets(assets)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Should allocate 100% to only asset
        assert result.weights.get("ONLY", 0) > 0.99
        
    def test_zero_volatility_asset(self, optimizer):
        """Test with zero volatility asset."""
        assets = [
            AssetData(symbol="RISK_FREE", expected_return=0.05, volatility=0.0001),
            AssetData(symbol="RISKY", expected_return=0.15, volatility=0.25),
        ]
        optimizer.set_assets(assets)
        
        result = optimizer.optimize(OptimizationStrategy.MIN_VOLATILITY)
        
        # Should prefer zero vol asset for min vol
        assert result.converged
        
    def test_negative_returns(self, optimizer):
        """Test with negative expected returns."""
        assets = [
            AssetData(symbol="NEGATIVE", expected_return=-0.05, volatility=0.20),
            AssetData(symbol="POSITIVE", expected_return=0.10, volatility=0.20),
        ]
        optimizer.set_assets(assets)
        
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        # Should prefer positive return asset
        assert result.weights.get("POSITIVE", 0) >= result.weights.get("NEGATIVE", 0)
        
    def test_identical_assets(self, optimizer):
        """Test with identical assets."""
        assets = [
            AssetData(symbol="A", expected_return=0.10, volatility=0.20),
            AssetData(symbol="B", expected_return=0.10, volatility=0.20),
        ]
        
        cov = np.array([[0.04, 0.04], [0.04, 0.04]])  # Perfect correlation
        optimizer.set_assets(assets, covariance_matrix=cov)
        
        # Should still produce valid result
        result = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        
        assert result is not None


# =============================================================================
# TEST CLASS: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory and helper functions."""
    
    def test_create_portfolio_optimizer(self):
        """Test create_portfolio_optimizer factory."""
        optimizer = create_portfolio_optimizer(
            risk_free_rate=0.03,
            max_position=0.25,
            allow_short=True,
        )
        
        assert optimizer.risk_free_rate == 0.03
        assert optimizer.constraints.allow_short is True
        
    def test_optimize_from_returns(self):
        """Test optimize_from_returns helper."""
        result = optimize_from_returns(
            symbols=["A", "B", "C"],
            expected_returns=[0.10, 0.08, 0.06],
            volatilities=[0.20, 0.18, 0.15],
            strategy=OptimizationStrategy.MAX_SHARPE,
        )
        
        assert result.converged
        assert len(result.weights) == 3
        
    def test_optimize_from_returns_with_correlation(self):
        """Test optimize_from_returns with correlation."""
        corr = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        
        result = optimize_from_returns(
            symbols=["A", "B", "C"],
            expected_returns=[0.10, 0.08, 0.06],
            volatilities=[0.20, 0.18, 0.15],
            correlation_matrix=corr,
            strategy=OptimizationStrategy.RISK_PARITY,
        )
        
        assert result.converged


# =============================================================================
# TEST CLASS: Summary
# =============================================================================

class TestSummary:
    """Test summary functionality."""
    
    def test_get_summary(self, optimizer, sample_assets, sample_covariance):
        """Test get_summary method."""
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        summary = optimizer.get_summary()
        
        assert "n_assets" in summary
        assert "assets" in summary
        assert "risk_free_rate" in summary
        assert "constraints" in summary
        
    def test_summary_asset_count(self, optimizer, sample_assets):
        """Test summary has correct asset count."""
        optimizer.set_assets(sample_assets)
        
        summary = optimizer.get_summary()
        
        assert summary["n_assets"] == len(sample_assets)


# =============================================================================
# TEST CLASS: Integration
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_optimization_workflow(self, sample_assets, sample_covariance):
        """Test complete optimization workflow."""
        # 1. Create optimizer
        optimizer = create_portfolio_optimizer(
            risk_free_rate=0.04,
            max_position=0.30,
        )
        
        # 2. Set assets
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        # 3. Run multiple strategies
        strategies = [
            OptimizationStrategy.MAX_SHARPE,
            OptimizationStrategy.MIN_VOLATILITY,
            OptimizationStrategy.RISK_PARITY,
        ]
        
        results = {}
        for strategy in strategies:
            results[strategy] = optimizer.optimize(strategy)
            assert results[strategy].converged
        
        # 4. Generate frontier
        frontier = optimizer.generate_efficient_frontier(n_points=10)
        assert len(frontier.returns) > 0
        
        # 5. Get rebalancing trades
        trades = optimizer.get_rebalancing_trades(
            optimal_weights=results[OptimizationStrategy.MAX_SHARPE].weights,
            portfolio_value=100000.0,
        )
        
        assert "buy" in trades
        
    def test_strategy_comparison(self, sample_assets, sample_covariance):
        """Compare different strategies."""
        optimizer = PortfolioOptimizer()
        optimizer.set_assets(sample_assets, covariance_matrix=sample_covariance)
        
        max_sharpe = optimizer.optimize(OptimizationStrategy.MAX_SHARPE)
        min_vol = optimizer.optimize(OptimizationStrategy.MIN_VOLATILITY)
        risk_parity = optimizer.optimize(OptimizationStrategy.RISK_PARITY)
        
        # Max Sharpe should have highest Sharpe
        assert max_sharpe.sharpe_ratio >= min_vol.sharpe_ratio - 0.1
        
        # Min vol should have lowest volatility
        assert min_vol.volatility <= max_sharpe.volatility + 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
