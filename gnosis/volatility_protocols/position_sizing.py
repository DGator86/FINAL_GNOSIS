"""
Position Sizing and Greek Limits
=================================

Risk-based position sizing algorithms and portfolio Greek management.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class StrategyRiskProfile(Enum):
    """Risk profile categories"""
    DEFINED_RISK = "defined_risk"  # Spreads
    UNDEFINED_RISK = "undefined_risk"  # Short straddles, strangles
    LONG_PREMIUM = "long_premium"  # Long calls, puts
    TIME_SPREAD = "time_spread"  # Calendars, diagonals
    COVERED = "covered"  # Wheel strategies


@dataclass
class GreekLimits:
    """Portfolio Greek limits (per $100k account)"""

    # Warning levels
    delta_warning: float = 25.0
    gamma_warning: float = 3.0
    vega_warning: float = 40.0
    theta_warning: float = 75.0  # $ per day

    # Danger levels (immediate action required)
    delta_danger: float = 40.0
    gamma_danger: float = 5.0
    vega_danger: float = 60.0
    theta_danger: float = 125.0  # $ per day

    def scale_to_account(self, account_value: float) -> 'GreekLimits':
        """Scale limits to actual account size"""
        scale_factor = account_value / 100000.0

        return GreekLimits(
            delta_warning=self.delta_warning * scale_factor,
            gamma_warning=self.gamma_warning * scale_factor,
            vega_warning=self.vega_warning * scale_factor,
            theta_warning=self.theta_warning * scale_factor,
            delta_danger=self.delta_danger * scale_factor,
            gamma_danger=self.gamma_danger * scale_factor,
            vega_danger=self.vega_danger * scale_factor,
            theta_danger=self.theta_danger * scale_factor,
        )

    def check_greek(
        self,
        greek_name: str,
        current_value: float,
    ) -> Tuple[str, str]:
        """
        Check if Greek is within limits

        Returns:
            (status, message) where status in ['ok', 'warning', 'danger']
        """
        limits = {
            'delta': (self.delta_warning, self.delta_danger),
            'gamma': (self.gamma_warning, self.gamma_danger),
            'vega': (self.vega_warning, self.vega_danger),
            'theta': (self.theta_warning, self.theta_danger),
        }

        warning, danger = limits.get(greek_name.lower(), (0, 0))
        abs_value = abs(current_value)

        if abs_value >= danger:
            return 'danger', f'{greek_name} {current_value:.2f} exceeds danger limit ±{danger:.2f}'
        elif abs_value >= warning:
            return 'warning', f'{greek_name} {current_value:.2f} exceeds warning limit ±{warning:.2f}'
        else:
            return 'ok', f'{greek_name} within limits'


@dataclass
class PositionSizingInput:
    """Input parameters for position sizing"""

    # Account
    account_value: float
    account_risk_budget_pct: float  # 2-5% per trade

    # Position
    max_loss_per_contract: float
    strategy_risk_profile: StrategyRiskProfile

    # Edge & Confidence
    edge_confidence: float  # 0.5-1.0 based on signal strength
    regime_stability: float  # 0.5-1.0 based on regime transition risk

    # Portfolio exposure (current)
    current_notional_exposure_pct: float = 0.0  # % of account already deployed

    # Greeks (projected for this position per contract)
    projected_delta_per_contract: float = 0.0
    projected_gamma_per_contract: float = 0.0
    projected_vega_per_contract: float = 0.0
    projected_theta_per_contract: float = 0.0


@dataclass
class PositionSizingResult:
    """Position sizing calculation result"""

    # Final sizing
    contracts: int
    total_risk: float
    risk_pct_of_account: float

    # Limits applied
    risk_budget_limit: bool = False
    notional_limit: bool = False
    greek_limit: bool = False
    position_count_limit: bool = False

    # Breakdown
    raw_contracts: int = 0  # Before limits
    limiting_factor: str = ""

    # Greeks (total position)
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0


# Position size limits by strategy type
POSITION_LIMITS = {
    StrategyRiskProfile.DEFINED_RISK: {
        'max_risk_pct': 0.05,  # 5% per trade
        'max_notional_pct': 0.25,  # 25% total
        'max_positions': 5,
    },
    StrategyRiskProfile.UNDEFINED_RISK: {
        'max_risk_pct': 0.02,  # 2% per trade
        'max_notional_pct': 0.10,  # 10% total
        'max_positions': 3,
    },
    StrategyRiskProfile.LONG_PREMIUM: {
        'max_risk_pct': 0.03,  # 3% per trade
        'max_notional_pct': 0.15,  # 15% total
        'max_positions': 4,
    },
    StrategyRiskProfile.TIME_SPREAD: {
        'max_risk_pct': 0.04,  # 4% per trade
        'max_notional_pct': 0.20,  # 20% total
        'max_positions': 3,
    },
    StrategyRiskProfile.COVERED: {
        'max_risk_pct': 0.05,  # 5% per trade
        'max_notional_pct': 0.30,  # 30% total
        'max_positions': 6,
    },
}


class PositionSizer:
    """
    Risk-based position sizing calculator
    """

    def __init__(self, greek_limits: Optional[GreekLimits] = None):
        self.greek_limits = greek_limits or GreekLimits()

    def calculate(
        self,
        inputs: PositionSizingInput,
        current_portfolio_greeks: Optional[Dict[str, float]] = None,
    ) -> PositionSizingResult:
        """
        Calculate optimal position size

        Formula:
        Position Size = (Account Risk Budget × Edge Confidence × Regime Stability) / Max Loss Per Contract

        Args:
            inputs: PositionSizingInput object
            current_portfolio_greeks: Current portfolio Greeks (optional)

        Returns:
            PositionSizingResult
        """
        # Get strategy limits
        limits = POSITION_LIMITS[inputs.strategy_risk_profile]

        # ========================================
        # 1. Calculate base position size
        # ========================================
        risk_budget = inputs.account_value * inputs.account_risk_budget_pct
        adjusted_risk_budget = risk_budget * inputs.edge_confidence * inputs.regime_stability

        if inputs.max_loss_per_contract == 0:
            raw_contracts = 0
        else:
            raw_contracts = int(adjusted_risk_budget / inputs.max_loss_per_contract)

        # Ensure at least 1 contract if any risk budget available
        raw_contracts = max(1, raw_contracts) if adjusted_risk_budget > 0 else 0

        # ========================================
        # 2. Apply position limits
        # ========================================
        final_contracts = raw_contracts
        limiting_factor = "none"

        # Risk percentage limit
        max_risk_allowed = inputs.account_value * limits['max_risk_pct']
        max_contracts_by_risk = int(max_risk_allowed / inputs.max_loss_per_contract) if inputs.max_loss_per_contract > 0 else 0

        if final_contracts > max_contracts_by_risk:
            final_contracts = max_contracts_by_risk
            limiting_factor = f"risk_limit_{limits['max_risk_pct']*100:.0f}%"

        # Notional exposure limit
        # This would require position notional value (strike × contracts × multiplier)
        # Simplified here - would need actual implementation

        # ========================================
        # 3. Check Greek limits
        # ========================================
        if current_portfolio_greeks is not None:
            scaled_limits = self.greek_limits.scale_to_account(inputs.account_value)

            # Calculate total Greeks with new position
            new_delta = current_portfolio_greeks.get('delta', 0) + (inputs.projected_delta_per_contract * final_contracts)
            new_gamma = current_portfolio_greeks.get('gamma', 0) + (inputs.projected_gamma_per_contract * final_contracts)
            new_vega = current_portfolio_greeks.get('vega', 0) + (inputs.projected_vega_per_contract * final_contracts)
            new_theta = current_portfolio_greeks.get('theta', 0) + (inputs.projected_theta_per_contract * final_contracts)

            # Check if any Greek would exceed danger limit
            greeks_to_check = [
                ('delta', new_delta, scaled_limits.delta_danger),
                ('gamma', new_gamma, scaled_limits.gamma_danger),
                ('vega', new_vega, scaled_limits.vega_danger),
                ('theta', new_theta, scaled_limits.theta_danger),
            ]

            for greek_name, value, danger_limit in greeks_to_check:
                if abs(value) > danger_limit:
                    # Reduce position size to stay within limit
                    current_value = current_portfolio_greeks.get(greek_name, 0)
                    per_contract = getattr(inputs, f'projected_{greek_name}_per_contract')

                    if per_contract != 0:
                        available_greek_budget = danger_limit - abs(current_value)
                        max_contracts_by_greek = int(available_greek_budget / abs(per_contract))
                        max_contracts_by_greek = max(0, max_contracts_by_greek)

                        if max_contracts_by_greek < final_contracts:
                            final_contracts = max_contracts_by_greek
                            limiting_factor = f"{greek_name}_limit"

        # ========================================
        # 4. Calculate final metrics
        # ========================================
        total_risk = final_contracts * inputs.max_loss_per_contract
        risk_pct = (total_risk / inputs.account_value) * 100 if inputs.account_value > 0 else 0

        result = PositionSizingResult(
            contracts=final_contracts,
            total_risk=total_risk,
            risk_pct_of_account=risk_pct,
            raw_contracts=raw_contracts,
            limiting_factor=limiting_factor,
            total_delta=final_contracts * inputs.projected_delta_per_contract,
            total_gamma=final_contracts * inputs.projected_gamma_per_contract,
            total_vega=final_contracts * inputs.projected_vega_per_contract,
            total_theta=final_contracts * inputs.projected_theta_per_contract,
        )

        # Set limit flags
        if limiting_factor.startswith('risk_limit'):
            result.risk_budget_limit = True
        elif limiting_factor in ['delta_limit', 'gamma_limit', 'vega_limit', 'theta_limit']:
            result.greek_limit = True

        return result


def calculate_position_size(
    account_value: float,
    max_loss_per_contract: float,
    strategy_risk_profile: StrategyRiskProfile,
    edge_confidence: float = 0.8,
    regime_stability: float = 1.0,
    risk_pct: float = 0.02,
) -> int:
    """
    Simple position size calculator (convenience function)

    Args:
        account_value: Total account value
        max_loss_per_contract: Maximum loss per contract
        strategy_risk_profile: Strategy risk profile
        edge_confidence: Edge confidence (0.5-1.0)
        regime_stability: Regime stability (0.5-1.0)
        risk_pct: Risk percentage (0.01-0.05)

    Returns:
        Number of contracts
    """
    inputs = PositionSizingInput(
        account_value=account_value,
        account_risk_budget_pct=risk_pct,
        max_loss_per_contract=max_loss_per_contract,
        strategy_risk_profile=strategy_risk_profile,
        edge_confidence=edge_confidence,
        regime_stability=regime_stability,
    )

    sizer = PositionSizer()
    result = sizer.calculate(inputs)

    return result.contracts


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate Kelly Criterion fraction for position sizing

    Formula: f = (p × b - q) / b
    Where:
        p = probability of win
        q = probability of loss (1-p)
        b = win/loss ratio

    Args:
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade size
        avg_loss: Average losing trade size (positive number)

    Returns:
        Kelly fraction (typically use 1/4 to 1/2 of full Kelly)
    """
    if avg_loss == 0 or win_rate >= 1.0 or win_rate <= 0:
        return 0.0

    p = win_rate
    q = 1 - win_rate
    b = avg_win / avg_loss

    kelly_full = (p * b - q) / b

    # Return 1/2 Kelly (more conservative)
    return max(0.0, min(0.25, kelly_full * 0.5))


def calculate_max_loss_defined_risk(
    spread_width: float,
    credit_received: float,
) -> float:
    """
    Calculate maximum loss for defined risk spread

    Max Loss = Spread Width - Credit Received

    Args:
        spread_width: Width of the spread (in dollars)
        credit_received: Credit received (for credit spreads) or debit paid (negative)

    Returns:
        Maximum loss in dollars
    """
    return (spread_width - credit_received) * 100  # × 100 for contract multiplier


def calculate_max_loss_debit_spread(
    debit_paid: float,
) -> float:
    """
    Calculate maximum loss for debit spread

    Max Loss = Debit Paid

    Args:
        debit_paid: Debit paid for the spread

    Returns:
        Maximum loss in dollars
    """
    return debit_paid * 100  # × 100 for contract multiplier


def calculate_buying_power_reduction(
    strategy_type: str,
    short_strike: float,
    long_strike: Optional[float] = None,
    contracts: int = 1,
    underlying_price: float = 100.0,
) -> float:
    """
    Estimate buying power reduction (BPR) for strategy

    Args:
        strategy_type: 'short_put', 'short_call', 'credit_spread', 'iron_condor', etc.
        short_strike: Short strike price
        long_strike: Long strike price (if defined risk)
        contracts: Number of contracts
        underlying_price: Current underlying price

    Returns:
        Estimated BPR in dollars
    """
    # Simplified BPR calculation (actual varies by broker)

    if strategy_type in ['credit_spread', 'debit_spread']:
        # Defined risk: BPR = spread width
        if long_strike is not None:
            bpr = abs(short_strike - long_strike) * 100 * contracts
        else:
            bpr = 0
    elif strategy_type == 'short_put':
        # Short put: ~20% of underlying value
        bpr = short_strike * 0.20 * 100 * contracts
    elif strategy_type == 'short_call':
        # Short call: ~20% of underlying value
        bpr = underlying_price * 0.20 * 100 * contracts
    elif strategy_type == 'iron_condor':
        # Iron condor: width of wider spread
        if long_strike is not None:
            bpr = abs(short_strike - long_strike) * 100 * contracts
        else:
            bpr = 0
    else:
        # Default conservative estimate
        bpr = underlying_price * 0.20 * 100 * contracts

    return bpr


@dataclass
class PortfolioGreeks:
    """Portfolio-level Greek tracking"""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    account_value: float = 100000.0

    def add_position(
        self,
        contracts: int,
        delta_per: float,
        gamma_per: float,
        vega_per: float,
        theta_per: float,
    ) -> 'PortfolioGreeks':
        """Add a new position to portfolio Greeks"""
        return PortfolioGreeks(
            delta=self.delta + (contracts * delta_per),
            gamma=self.gamma + (contracts * gamma_per),
            vega=self.vega + (contracts * vega_per),
            theta=self.theta + (contracts * theta_per),
            account_value=self.account_value,
        )

    def check_limits(self, limits: GreekLimits) -> Dict[str, Tuple[str, str]]:
        """Check all Greeks against limits"""
        scaled_limits = limits.scale_to_account(self.account_value)

        return {
            'delta': scaled_limits.check_greek('delta', self.delta),
            'gamma': scaled_limits.check_greek('gamma', self.gamma),
            'vega': scaled_limits.check_greek('vega', self.vega),
            'theta': scaled_limits.check_greek('theta', self.theta),
        }

    def get_summary(self) -> Dict[str, any]:
        """Get portfolio Greek summary"""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'theta_per_day': self.theta,
            'account_value': self.account_value,
        }
