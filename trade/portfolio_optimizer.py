"""
Portfolio Optimizer - Institutional Grade Portfolio Optimization

Implements multiple portfolio optimization strategies:
- Mean-Variance (Markowitz) Optimization
- Risk Parity Optimization
- Maximum Sharpe Ratio
- Minimum Volatility
- Greeks-Aware Optimization
- Black-Litterman Model Integration

OPTIMIZATION STRATEGIES:
├── Mean-Variance: Classic Markowitz efficient frontier
├── Risk Parity: Equal risk contribution per asset
├── Max Sharpe: Maximize risk-adjusted returns
├── Min Volatility: Minimize portfolio variance
├── Greeks-Aware: Optimize considering options Greeks exposure
└── Black-Litterman: Incorporate market views with equilibrium

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from loguru import logger

# Suppress scipy warnings during optimization
warnings.filterwarnings('ignore', category=RuntimeWarning)


class OptimizationStrategy(str, Enum):
    """Portfolio optimization strategies."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    GREEKS_AWARE = "greeks_aware"
    BLACK_LITTERMAN = "black_litterman"
    EQUAL_WEIGHT = "equal_weight"


class ConstraintType(str, Enum):
    """Types of portfolio constraints."""
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"
    MAX_POSITION = "max_position"
    SECTOR_LIMIT = "sector_limit"
    TURNOVER = "turnover"


@dataclass
class AssetData:
    """Data for a single asset."""
    symbol: str
    expected_return: float  # Annual expected return
    volatility: float  # Annual volatility (standard deviation)
    current_weight: float = 0.0  # Current portfolio weight
    
    # Optional: Options Greeks exposure
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Optional: Sector classification
    sector: str = "other"
    
    # Optional: Views for Black-Litterman
    view_return: Optional[float] = None
    view_confidence: float = 0.5


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    # Position limits
    max_position_weight: float = 0.20  # Max 20% per position
    min_position_weight: float = 0.0  # Min weight (0 for long-only)
    
    # Sector limits
    max_sector_weight: float = 0.40  # Max 40% per sector
    
    # Portfolio-level limits
    max_turnover: float = 1.0  # Max 100% turnover
    target_volatility: Optional[float] = None  # Target portfolio vol
    
    # Greeks limits (for options portfolios)
    max_delta: Optional[float] = None
    max_gamma: Optional[float] = None
    max_vega: Optional[float] = None
    min_theta: Optional[float] = None
    
    # Leverage
    max_leverage: float = 1.0  # Max gross exposure
    allow_short: bool = False


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Strategy used
    strategy: OptimizationStrategy = OptimizationStrategy.MEAN_VARIANCE
    
    # Optimal weights
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Portfolio metrics
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk metrics
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    max_drawdown: float = 0.0
    
    # Greeks (for options portfolios)
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_theta: float = 0.0
    portfolio_vega: float = 0.0
    
    # Optimization metadata
    converged: bool = True
    iterations: int = 0
    error_message: Optional[str] = None
    
    # Comparison to current
    current_weights: Dict[str, float] = field(default_factory=dict)
    weight_changes: Dict[str, float] = field(default_factory=dict)
    turnover: float = 0.0


@dataclass
class EfficientFrontier:
    """Efficient frontier data points."""
    returns: List[float] = field(default_factory=list)
    volatilities: List[float] = field(default_factory=list)
    sharpe_ratios: List[float] = field(default_factory=list)
    weights_list: List[Dict[str, float]] = field(default_factory=list)
    
    # Special portfolios on the frontier
    max_sharpe_portfolio: Optional[OptimizationResult] = None
    min_vol_portfolio: Optional[OptimizationResult] = None
    tangent_portfolio: Optional[OptimizationResult] = None


class PortfolioOptimizer:
    """
    Institutional-grade portfolio optimizer.
    
    Features:
    - Multiple optimization strategies
    - Comprehensive constraints
    - Greeks-aware optimization for options
    - Efficient frontier generation
    - Black-Litterman views integration
    """
    
    # Default sector mappings
    SECTOR_MAP: Dict[str, str] = {
        "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
        "NVDA": "technology", "AMD": "technology", "META": "technology",
        "JPM": "financials", "BAC": "financials", "GS": "financials",
        "XOM": "energy", "CVX": "energy", "COP": "energy",
        "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare",
        "TSLA": "consumer", "NKE": "consumer", "AMZN": "consumer",
        "SPY": "index", "QQQ": "index", "IWM": "index",
    }
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        constraints: Optional[OptimizationConstraints] = None,
    ):
        """Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            constraints: Optimization constraints
        """
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or OptimizationConstraints()
        
        # Cache
        self._assets: List[AssetData] = []
        self._covariance_matrix: Optional[np.ndarray] = None
        self._correlation_matrix: Optional[np.ndarray] = None
        
        logger.info(
            f"PortfolioOptimizer initialized | "
            f"risk_free_rate={risk_free_rate:.2%} | "
            f"max_position={self.constraints.max_position_weight:.0%}"
        )
    
    def set_assets(
        self,
        assets: List[AssetData],
        covariance_matrix: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Set assets for optimization.
        
        Args:
            assets: List of AssetData objects
            covariance_matrix: Optional pre-computed covariance matrix
            correlation_matrix: Optional correlation matrix (for generating cov)
        """
        self._assets = assets
        self._covariance_matrix = covariance_matrix
        self._correlation_matrix = correlation_matrix
        
        # If no covariance matrix provided, estimate from volatilities
        if covariance_matrix is None:
            self._estimate_covariance_matrix()
        
        logger.info(f"Set {len(assets)} assets for optimization")
    
    def set_assets_from_data(
        self,
        symbols: List[str],
        returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set assets from return/covariance data.
        
        Args:
            symbols: List of asset symbols
            returns: Expected returns array
            covariance: Covariance matrix
            current_weights: Current portfolio weights
        """
        current_weights = current_weights or {}
        
        assets = []
        for i, symbol in enumerate(symbols):
            vol = np.sqrt(covariance[i, i]) if i < len(covariance) else 0.0
            assets.append(AssetData(
                symbol=symbol,
                expected_return=returns[i] if i < len(returns) else 0.0,
                volatility=vol,
                current_weight=current_weights.get(symbol, 0.0),
                sector=self.SECTOR_MAP.get(symbol, "other"),
            ))
        
        self._assets = assets
        self._covariance_matrix = covariance
    
    def _estimate_covariance_matrix(self) -> None:
        """Estimate covariance matrix from asset volatilities."""
        n = len(self._assets)
        if n == 0:
            return
        
        vols = np.array([a.volatility for a in self._assets])
        
        if self._correlation_matrix is not None:
            # Covariance = D * Correlation * D where D is diagonal volatility
            D = np.diag(vols)
            self._covariance_matrix = D @ self._correlation_matrix @ D
        else:
            # Assume some correlation structure (0.3 average correlation)
            avg_corr = 0.3
            corr = np.full((n, n), avg_corr)
            np.fill_diagonal(corr, 1.0)
            D = np.diag(vols)
            self._covariance_matrix = D @ corr @ D
    
    def optimize(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.MAX_SHARPE,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
    ) -> OptimizationResult:
        """Run portfolio optimization.
        
        Args:
            strategy: Optimization strategy to use
            target_return: Target return (for mean-variance with return constraint)
            target_volatility: Target volatility
            
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        if not self._assets:
            return OptimizationResult(
                strategy=strategy,
                converged=False,
                error_message="No assets provided",
            )
        
        n = len(self._assets)
        
        try:
            if strategy == OptimizationStrategy.EQUAL_WEIGHT:
                weights = self._equal_weight()
            elif strategy == OptimizationStrategy.MAX_SHARPE:
                weights = self._max_sharpe()
            elif strategy == OptimizationStrategy.MIN_VOLATILITY:
                weights = self._min_volatility()
            elif strategy == OptimizationStrategy.MEAN_VARIANCE:
                weights = self._mean_variance(target_return)
            elif strategy == OptimizationStrategy.RISK_PARITY:
                weights = self._risk_parity()
            elif strategy == OptimizationStrategy.GREEKS_AWARE:
                weights = self._greeks_aware()
            elif strategy == OptimizationStrategy.BLACK_LITTERMAN:
                weights = self._black_litterman()
            else:
                weights = self._equal_weight()
            
            # Apply constraints
            weights = self._apply_constraints(weights)
            
            # Calculate portfolio metrics
            result = self._calculate_result(weights, strategy)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                strategy=strategy,
                converged=False,
                error_message=str(e),
            )
    
    def _equal_weight(self) -> np.ndarray:
        """Equal weight allocation."""
        n = len(self._assets)
        return np.ones(n) / n
    
    def _max_sharpe(self) -> np.ndarray:
        """Maximize Sharpe ratio using analytical solution."""
        n = len(self._assets)
        
        if self._covariance_matrix is None:
            return self._equal_weight()
        
        returns = np.array([a.expected_return for a in self._assets])
        excess_returns = returns - self.risk_free_rate
        
        try:
            # Analytical solution: w* = Sigma^-1 * (r - rf) / (1' * Sigma^-1 * (r - rf))
            inv_cov = np.linalg.inv(self._covariance_matrix)
            weights = inv_cov @ excess_returns
            
            # Handle negative weights based on constraints
            if not self.constraints.allow_short:
                weights = np.maximum(weights, 0)
            
            # Normalize to sum to 1
            weights = weights / np.sum(np.abs(weights))
            
            return weights
            
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, falling back to equal weight")
            return self._equal_weight()
    
    def _min_volatility(self) -> np.ndarray:
        """Minimize portfolio volatility using analytical solution."""
        n = len(self._assets)
        
        if self._covariance_matrix is None:
            return self._equal_weight()
        
        try:
            # Analytical solution: w* = Sigma^-1 * 1 / (1' * Sigma^-1 * 1)
            inv_cov = np.linalg.inv(self._covariance_matrix)
            ones = np.ones(n)
            weights = inv_cov @ ones / (ones @ inv_cov @ ones)
            
            # Handle negative weights
            if not self.constraints.allow_short:
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
            
            return weights
            
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, falling back to equal weight")
            return self._equal_weight()
    
    def _mean_variance(self, target_return: Optional[float] = None) -> np.ndarray:
        """Mean-variance optimization with optional return target."""
        if target_return is None:
            # Default to max Sharpe
            return self._max_sharpe()
        
        n = len(self._assets)
        returns = np.array([a.expected_return for a in self._assets])
        
        if self._covariance_matrix is None:
            return self._equal_weight()
        
        try:
            # Constrained optimization using Lagrange multipliers
            inv_cov = np.linalg.inv(self._covariance_matrix)
            ones = np.ones(n)
            
            A = ones @ inv_cov @ ones
            B = ones @ inv_cov @ returns
            C = returns @ inv_cov @ returns
            D = A * C - B * B
            
            if abs(D) < 1e-10:
                return self._max_sharpe()
            
            # Lambda coefficients
            lambda1 = (C - B * target_return) / D
            lambda2 = (A * target_return - B) / D
            
            weights = inv_cov @ (lambda1 * ones + lambda2 * returns)
            
            # Handle negative weights
            if not self.constraints.allow_short:
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
            
            return weights
            
        except np.linalg.LinAlgError:
            return self._max_sharpe()
    
    def _risk_parity(self) -> np.ndarray:
        """Risk parity allocation - equal risk contribution."""
        n = len(self._assets)
        
        if self._covariance_matrix is None:
            return self._equal_weight()
        
        # Iterative approach to risk parity
        weights = np.ones(n) / n
        
        for iteration in range(100):
            # Calculate marginal risk contributions
            portfolio_vol = np.sqrt(weights @ self._covariance_matrix @ weights)
            if portfolio_vol < 1e-10:
                break
            
            marginal_contrib = self._covariance_matrix @ weights / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Target: equal risk contribution
            target_contrib = portfolio_vol / n
            
            # Update weights
            for i in range(n):
                if marginal_contrib[i] > 0:
                    weights[i] = target_contrib / marginal_contrib[i]
            
            # Normalize
            weights = weights / np.sum(weights)
            
            # Check convergence
            actual_contrib = weights * marginal_contrib
            if np.max(np.abs(actual_contrib - target_contrib)) < 1e-6:
                break
        
        return weights
    
    def _greeks_aware(self) -> np.ndarray:
        """Optimize considering Greeks exposure limits."""
        n = len(self._assets)
        
        # Start with max Sharpe
        weights = self._max_sharpe()
        
        # Adjust for Greeks constraints
        constraints = self.constraints
        
        for iteration in range(50):
            # Calculate portfolio Greeks
            delta = sum(w * a.delta for w, a in zip(weights, self._assets))
            gamma = sum(w * a.gamma for w, a in zip(weights, self._assets))
            vega = sum(w * a.vega for w, a in zip(weights, self._assets))
            theta = sum(w * a.theta for w, a in zip(weights, self._assets))
            
            adjustments = np.zeros(n)
            
            # Adjust for delta limit
            if constraints.max_delta and abs(delta) > constraints.max_delta:
                scale = constraints.max_delta / abs(delta)
                for i, a in enumerate(self._assets):
                    if a.delta * delta > 0:  # Same direction
                        adjustments[i] -= 0.1 * (1 - scale)
            
            # Adjust for gamma limit
            if constraints.max_gamma and abs(gamma) > constraints.max_gamma:
                scale = constraints.max_gamma / abs(gamma)
                for i, a in enumerate(self._assets):
                    if abs(a.gamma) > 0:
                        adjustments[i] -= 0.1 * (1 - scale) * (abs(a.gamma) / max(abs(a.gamma) for a in self._assets))
            
            # Adjust for vega limit
            if constraints.max_vega and abs(vega) > constraints.max_vega:
                scale = constraints.max_vega / abs(vega)
                for i, a in enumerate(self._assets):
                    if a.vega * vega > 0:
                        adjustments[i] -= 0.1 * (1 - scale)
            
            # Apply adjustments
            weights = weights + adjustments
            weights = np.maximum(weights, 0) if not constraints.allow_short else weights
            if np.sum(np.abs(weights)) > 0:
                weights = weights / np.sum(np.abs(weights))
            
            if np.max(np.abs(adjustments)) < 1e-6:
                break
        
        return weights
    
    def _black_litterman(self) -> np.ndarray:
        """Black-Litterman model with market views."""
        n = len(self._assets)
        
        # Equilibrium returns (from market weights or equal weight)
        market_weights = np.array([a.current_weight if a.current_weight > 0 else 1/n 
                                   for a in self._assets])
        market_weights = market_weights / np.sum(market_weights)
        
        if self._covariance_matrix is None:
            return self._max_sharpe()
        
        # Risk aversion parameter
        risk_aversion = 2.5
        
        # Implied equilibrium returns
        pi = risk_aversion * self._covariance_matrix @ market_weights
        
        # Views matrix
        views = []
        view_returns = []
        view_confidences = []
        
        for i, asset in enumerate(self._assets):
            if asset.view_return is not None:
                view_vec = np.zeros(n)
                view_vec[i] = 1.0
                views.append(view_vec)
                view_returns.append(asset.view_return)
                view_confidences.append(asset.view_confidence)
        
        if not views:
            # No views, return equilibrium weights
            return market_weights
        
        P = np.array(views)
        Q = np.array(view_returns)
        
        # Uncertainty in views (diagonal matrix)
        tau = 0.05  # Scaling factor
        omega = np.diag([tau * self._covariance_matrix[i, i] / (c + 0.01) 
                        for i, c in enumerate(view_confidences) if i < len(view_confidences)])
        
        # Black-Litterman formula
        try:
            M1 = np.linalg.inv(tau * self._covariance_matrix)
            M2 = P.T @ np.linalg.inv(omega) @ P
            
            posterior_returns = np.linalg.inv(M1 + M2) @ (M1 @ pi + P.T @ np.linalg.inv(omega) @ Q)
            
            # Optimize with posterior returns
            inv_cov = np.linalg.inv(self._covariance_matrix)
            weights = inv_cov @ posterior_returns
            
            if not self.constraints.allow_short:
                weights = np.maximum(weights, 0)
            
            weights = weights / np.sum(np.abs(weights))
            
            return weights
            
        except np.linalg.LinAlgError:
            return market_weights
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply portfolio constraints to weights."""
        constraints = self.constraints
        n = len(self._assets)
        
        # Max position constraint
        weights = np.clip(weights, -constraints.max_position_weight, constraints.max_position_weight)
        
        # Min position constraint (for long-only)
        if not constraints.allow_short:
            weights = np.maximum(weights, constraints.min_position_weight)
        
        # Sector constraints
        sectors = {}
        for i, asset in enumerate(self._assets):
            if asset.sector not in sectors:
                sectors[asset.sector] = []
            sectors[asset.sector].append(i)
        
        for sector, indices in sectors.items():
            sector_weight = sum(weights[i] for i in indices)
            if sector_weight > constraints.max_sector_weight:
                scale = constraints.max_sector_weight / sector_weight
                for i in indices:
                    weights[i] *= scale
        
        # Leverage constraint
        gross_exposure = np.sum(np.abs(weights))
        if gross_exposure > constraints.max_leverage:
            weights = weights * constraints.max_leverage / gross_exposure
        
        # Normalize to sum to 1 (or max_leverage for leveraged portfolios)
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
        
        return weights
    
    def _calculate_result(
        self,
        weights: np.ndarray,
        strategy: OptimizationStrategy,
    ) -> OptimizationResult:
        """Calculate portfolio metrics from weights."""
        n = len(self._assets)
        
        # Convert to dict
        weight_dict = {self._assets[i].symbol: float(weights[i]) for i in range(n)}
        
        # Current weights
        current_dict = {a.symbol: a.current_weight for a in self._assets}
        
        # Weight changes
        changes = {s: weight_dict[s] - current_dict.get(s, 0) for s in weight_dict}
        
        # Turnover
        turnover = sum(abs(v) for v in changes.values()) / 2
        
        # Expected return
        returns = np.array([a.expected_return for a in self._assets])
        expected_return = float(weights @ returns)
        
        # Volatility
        if self._covariance_matrix is not None:
            volatility = float(np.sqrt(weights @ self._covariance_matrix @ weights))
        else:
            vols = np.array([a.volatility for a in self._assets])
            volatility = float(np.sqrt(np.sum((weights * vols) ** 2)))
        
        # Sharpe ratio
        sharpe = (expected_return - self.risk_free_rate) / max(volatility, 0.001)
        
        # VaR (parametric, 95%)
        var_95 = -(expected_return - 1.645 * volatility)
        
        # CVaR (parametric approximation)
        cvar_95 = -(expected_return - 2.063 * volatility)
        
        # Portfolio Greeks
        delta = sum(weights[i] * self._assets[i].delta for i in range(n))
        gamma = sum(weights[i] * self._assets[i].gamma for i in range(n))
        theta = sum(weights[i] * self._assets[i].theta for i in range(n))
        vega = sum(weights[i] * self._assets[i].vega for i in range(n))
        
        return OptimizationResult(
            strategy=strategy,
            weights=weight_dict,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            portfolio_delta=delta,
            portfolio_gamma=gamma,
            portfolio_theta=theta,
            portfolio_vega=vega,
            current_weights=current_dict,
            weight_changes=changes,
            turnover=turnover,
            converged=True,
        )
    
    def generate_efficient_frontier(
        self,
        n_points: int = 50,
    ) -> EfficientFrontier:
        """Generate the efficient frontier.
        
        Args:
            n_points: Number of points on the frontier
            
        Returns:
            EfficientFrontier with portfolios along the frontier
        """
        if not self._assets:
            return EfficientFrontier()
        
        returns = np.array([a.expected_return for a in self._assets])
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier = EfficientFrontier()
        
        for target in target_returns:
            result = self.optimize(
                strategy=OptimizationStrategy.MEAN_VARIANCE,
                target_return=target,
            )
            
            if result.converged:
                frontier.returns.append(result.expected_return)
                frontier.volatilities.append(result.volatility)
                frontier.sharpe_ratios.append(result.sharpe_ratio)
                frontier.weights_list.append(result.weights)
        
        # Find special portfolios
        frontier.max_sharpe_portfolio = self.optimize(OptimizationStrategy.MAX_SHARPE)
        frontier.min_vol_portfolio = self.optimize(OptimizationStrategy.MIN_VOLATILITY)
        
        return frontier
    
    def get_rebalancing_trades(
        self,
        optimal_weights: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, Dict[str, float]]:
        """Get trades needed to rebalance to optimal weights.
        
        Args:
            optimal_weights: Target weights
            portfolio_value: Total portfolio value
            
        Returns:
            Dict with buy/sell trades
        """
        trades = {"buy": {}, "sell": {}}
        
        for asset in self._assets:
            symbol = asset.symbol
            current_value = asset.current_weight * portfolio_value
            target_value = optimal_weights.get(symbol, 0) * portfolio_value
            
            diff = target_value - current_value
            
            if diff > 0:
                trades["buy"][symbol] = abs(diff)
            elif diff < 0:
                trades["sell"][symbol] = abs(diff)
        
        return trades
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimizer summary."""
        return {
            "n_assets": len(self._assets),
            "assets": [a.symbol for a in self._assets],
            "risk_free_rate": self.risk_free_rate,
            "constraints": {
                "max_position": self.constraints.max_position_weight,
                "max_sector": self.constraints.max_sector_weight,
                "allow_short": self.constraints.allow_short,
            },
            "has_covariance": self._covariance_matrix is not None,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_portfolio_optimizer(
    risk_free_rate: float = 0.05,
    max_position: float = 0.20,
    allow_short: bool = False,
) -> PortfolioOptimizer:
    """Create a PortfolioOptimizer with common settings.
    
    Args:
        risk_free_rate: Annual risk-free rate
        max_position: Maximum weight per position
        allow_short: Allow short positions
        
    Returns:
        Configured PortfolioOptimizer
    """
    constraints = OptimizationConstraints(
        max_position_weight=max_position,
        allow_short=allow_short,
    )
    
    return PortfolioOptimizer(
        risk_free_rate=risk_free_rate,
        constraints=constraints,
    )


def optimize_from_returns(
    symbols: List[str],
    expected_returns: List[float],
    volatilities: List[float],
    correlation_matrix: Optional[np.ndarray] = None,
    strategy: OptimizationStrategy = OptimizationStrategy.MAX_SHARPE,
) -> OptimizationResult:
    """Quick optimization from return/volatility data.
    
    Args:
        symbols: Asset symbols
        expected_returns: Expected annual returns
        volatilities: Annual volatilities
        correlation_matrix: Correlation matrix (optional)
        strategy: Optimization strategy
        
    Returns:
        OptimizationResult
    """
    optimizer = create_portfolio_optimizer()
    
    assets = [
        AssetData(
            symbol=sym,
            expected_return=ret,
            volatility=vol,
        )
        for sym, ret, vol in zip(symbols, expected_returns, volatilities)
    ]
    
    optimizer.set_assets(assets, correlation_matrix=correlation_matrix)
    
    return optimizer.optimize(strategy=strategy)
