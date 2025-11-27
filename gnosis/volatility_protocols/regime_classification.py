"""
Regime Classification Module
=============================

Market regime detection and transition risk analysis (R1-R5).

Regimes:
--------
R1: VIX <15, Contango >10%, VVIX <90 (Low Vol Environment)
R2: VIX 15-20, Contango 5-10%, VVIX 90-110 (Normal Vol)
R3: VIX 20-30, Contango <5%, VVIX 110-130 (High Vol)
R4: VIX 30-40, Flat/Backwardation, VVIX >130 (Elevated Vol)
R5: VIX >40, Backwardation, VVIX >150 (Crisis Mode)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np


class Regime(Enum):
    """Market regime enumeration"""
    R1 = "Low Vol Environment"
    R2 = "Normal Vol"
    R3 = "High Vol"
    R4 = "Elevated Vol"
    R5 = "Crisis Mode"

    @property
    def numeric_value(self) -> int:
        """Get numeric regime value (1-5)"""
        return int(self.name[1])

    @property
    def is_low_vol(self) -> bool:
        return self in [Regime.R1, Regime.R2]

    @property
    def is_high_vol(self) -> bool:
        return self in [Regime.R3, Regime.R4, Regime.R5]

    @property
    def is_crisis(self) -> bool:
        return self == Regime.R5


@dataclass
class RegimeMetrics:
    """Complete regime classification metrics"""
    regime: Regime
    vix_level: float
    term_structure: float  # Percentage contango/backwardation
    vvix_level: float  # Vol-of-vol
    stability_days: int  # Days in current regime
    transition_risk: float  # 0-100 percentage
    timestamp: datetime

    @property
    def is_stable(self) -> bool:
        """Check if regime meets minimum stability requirement"""
        min_days = {
            Regime.R1: 3,
            Regime.R2: 2,
            Regime.R3: 1,
            Regime.R4: 0,  # Immediate action
            Regime.R5: 0,  # Crisis mode
        }
        return self.stability_days >= min_days[self.regime]

    @property
    def requires_immediate_action(self) -> bool:
        """Check if regime requires immediate position management"""
        return self.regime in [Regime.R4, Regime.R5]

    @property
    def allows_short_vol(self) -> bool:
        """Check if regime allows short volatility strategies"""
        return self.regime in [Regime.R1, Regime.R2, Regime.R3]

    @property
    def favors_long_vol(self) -> bool:
        """Check if regime favors long volatility strategies"""
        return self.regime in [Regime.R1] or self.transition_risk > 75


@dataclass
class RegimeTransition:
    """Regime transition event"""
    from_regime: Regime
    to_regime: Regime
    timestamp: datetime
    trigger: str  # What caused the transition

    @property
    def is_escalation(self) -> bool:
        """Check if transition is an escalation (increasing volatility)"""
        return self.to_regime.numeric_value > self.from_regime.numeric_value

    @property
    def is_deescalation(self) -> bool:
        """Check if transition is a de-escalation (decreasing volatility)"""
        return self.to_regime.numeric_value < self.from_regime.numeric_value

    @property
    def severity(self) -> int:
        """Calculate transition severity (0-4)"""
        return abs(self.to_regime.numeric_value - self.from_regime.numeric_value)


class RegimeClassifier:
    """
    Market regime classifier with history tracking
    """

    def __init__(self):
        self.history: List[RegimeMetrics] = []
        self.transitions: List[RegimeTransition] = []

    def classify(
        self,
        vix_level: float,
        term_structure: float,
        vvix_level: float,
    ) -> RegimeMetrics:
        """
        Classify current market regime

        Args:
            vix_level: Current VIX value
            term_structure: Term structure percentage (positive = contango)
            vvix_level: VVIX (volatility of volatility)

        Returns:
            RegimeMetrics object
        """
        # Determine regime based on VIX, term structure, and VVIX
        if vix_level < 15 and term_structure > 10 and vvix_level < 90:
            regime = Regime.R1
            regime_midpoint = 12.5
            regime_width = 15.0
        elif 15 <= vix_level < 20 and 5 <= term_structure <= 10 and 90 <= vvix_level < 110:
            regime = Regime.R2
            regime_midpoint = 17.5
            regime_width = 5.0
        elif 20 <= vix_level < 30 and term_structure < 5 and 110 <= vvix_level < 130:
            regime = Regime.R3
            regime_midpoint = 25.0
            regime_width = 10.0
        elif 30 <= vix_level < 40 and term_structure <= 0 and vvix_level >= 130:
            regime = Regime.R4
            regime_midpoint = 35.0
            regime_width = 10.0
        else:  # VIX > 40
            regime = Regime.R5
            regime_midpoint = 50.0
            regime_width = 20.0

        # Calculate stability (days in current regime)
        stability_days = self._calculate_stability_days(regime)

        # Calculate transition risk
        transition_risk = self._calculate_transition_risk(
            vix_level, regime_midpoint, regime_width
        )

        metrics = RegimeMetrics(
            regime=regime,
            vix_level=vix_level,
            term_structure=term_structure,
            vvix_level=vvix_level,
            stability_days=stability_days,
            transition_risk=transition_risk,
            timestamp=datetime.now(),
        )

        # Track transition if regime changed
        if self.history:
            last_regime = self.history[-1].regime
            if last_regime != regime:
                transition = RegimeTransition(
                    from_regime=last_regime,
                    to_regime=regime,
                    timestamp=datetime.now(),
                    trigger=f"VIX {vix_level:.2f}",
                )
                self.transitions.append(transition)

        # Add to history
        self.history.append(metrics)

        return metrics

    def _calculate_stability_days(self, current_regime: Regime) -> int:
        """Calculate number of consecutive days in current regime"""
        if not self.history:
            return 0

        days = 0
        for metrics in reversed(self.history):
            if metrics.regime == current_regime:
                days += 1
            else:
                break

        return days

    def _calculate_transition_risk(
        self,
        vix_level: float,
        regime_midpoint: float,
        regime_width: float,
    ) -> float:
        """
        Calculate transition risk percentage

        Formula: |VIX - Regime Midpoint| / Regime Width * 100
        """
        if regime_width == 0:
            return 0.0

        transition_risk = (abs(vix_level - regime_midpoint) / regime_width) * 100

        return min(100.0, transition_risk)

    def get_forced_exit_signal(
        self,
        current_regime: RegimeMetrics,
    ) -> Optional[Dict[str, any]]:
        """
        Check if current regime requires forced position exits

        Returns:
            Exit signal dict or None
        """
        if not self.transitions:
            return None

        last_transition = self.transitions[-1]

        # R1→R2 (VIX >15): Review within 1 hour
        if (
            last_transition.from_regime == Regime.R1 and
            last_transition.to_regime == Regime.R2
        ):
            return {
                'urgency': 'review',
                'timeframe': '1 hour',
                'action': 'Consider short vol reduction',
                'reason': 'R1→R2 transition detected',
            }

        # R2→R3 (VIX >20): Action within 2 hours
        if (
            last_transition.from_regime == Regime.R2 and
            last_transition.to_regime == Regime.R3
        ):
            return {
                'urgency': 'action',
                'timeframe': '2 hours',
                'action': 'Reduce short vol by 50%',
                'reason': 'R2→R3 transition detected',
            }

        # R3→R4 (VIX >30): IMMEDIATE
        if (
            last_transition.from_regime == Regime.R3 and
            last_transition.to_regime == Regime.R4
        ):
            return {
                'urgency': 'immediate',
                'timeframe': 'now',
                'action': 'Close ALL short vol positions',
                'reason': 'R3→R4 transition - VIX >30',
            }

        # R4→R5 or any→R5 (Backwardation): EMERGENCY
        if last_transition.to_regime == Regime.R5:
            return {
                'urgency': 'emergency',
                'timeframe': 'now',
                'action': 'DEFCON 1: Close everything, shift to long vol',
                'reason': 'R5 Crisis Mode - Backwardation detected',
            }

        return None

    def get_regime_summary(self) -> Dict[str, any]:
        """Get comprehensive regime summary"""
        if not self.history:
            return {'status': 'No regime data'}

        current = self.history[-1]

        summary = {
            'current_regime': current.regime.value,
            'vix_level': current.vix_level,
            'term_structure': current.term_structure,
            'vvix_level': current.vvix_level,
            'stability_days': current.stability_days,
            'transition_risk': current.transition_risk,
            'is_stable': current.is_stable,
            'allows_short_vol': current.allows_short_vol,
            'favors_long_vol': current.favors_long_vol,
        }

        # Add exit signal if present
        exit_signal = self.get_forced_exit_signal(current)
        if exit_signal:
            summary['exit_signal'] = exit_signal

        # Add recent transitions
        if self.transitions:
            summary['recent_transitions'] = [
                {
                    'from': t.from_regime.value,
                    'to': t.to_regime.value,
                    'timestamp': t.timestamp.isoformat(),
                    'severity': t.severity,
                }
                for t in self.transitions[-5:]  # Last 5 transitions
            ]

        return summary


def classify_regime(
    vix_level: float,
    term_structure: float,
    vvix_level: float,
) -> Regime:
    """
    Simple regime classification (without history tracking)

    Args:
        vix_level: Current VIX value
        term_structure: Term structure percentage
        vvix_level: VVIX value

    Returns:
        Regime enum
    """
    if vix_level < 15 and term_structure > 10 and vvix_level < 90:
        return Regime.R1
    elif 15 <= vix_level < 20 and 5 <= term_structure <= 10 and 90 <= vvix_level < 110:
        return Regime.R2
    elif 20 <= vix_level < 30 and term_structure < 5 and 110 <= vvix_level < 130:
        return Regime.R3
    elif 30 <= vix_level < 40 and term_structure <= 0 and vvix_level >= 130:
        return Regime.R4
    else:
        return Regime.R5


def calculate_transition_risk(
    vix_level: float,
    current_regime: Regime,
) -> float:
    """
    Calculate transition risk for current VIX level

    Args:
        vix_level: Current VIX value
        current_regime: Current regime

    Returns:
        Transition risk percentage (0-100)
    """
    regime_params = {
        Regime.R1: {'midpoint': 12.5, 'width': 15.0},
        Regime.R2: {'midpoint': 17.5, 'width': 5.0},
        Regime.R3: {'midpoint': 25.0, 'width': 10.0},
        Regime.R4: {'midpoint': 35.0, 'width': 10.0},
        Regime.R5: {'midpoint': 50.0, 'width': 20.0},
    }

    params = regime_params[current_regime]
    transition_risk = (abs(vix_level - params['midpoint']) / params['width']) * 100

    return min(100.0, transition_risk)


# Regime-based strategy recommendations
REGIME_STRATEGY_MAP = {
    Regime.R1: {
        'preferred': ['short_strangle', 'short_straddle', 'iron_condor', 'jade_lizard'],
        'avoid': ['long_straddle', 'long_strangle'],
        'max_position_risk': 0.04,  # 4% per trade
        'description': 'Premium selling environment',
    },
    Regime.R2: {
        'preferred': ['iron_condor', 'credit_spreads', 'calendars', 'covered_calls'],
        'avoid': ['naked_options'],
        'max_position_risk': 0.03,  # 3% per trade
        'description': 'Balanced environment',
    },
    Regime.R3: {
        'preferred': ['credit_spreads', 'iron_condor', 'put_backspread'],
        'avoid': ['short_straddle', 'short_strangle'],
        'max_position_risk': 0.02,  # 2% per trade
        'description': 'Elevated vol - reduce short exposure',
    },
    Regime.R4: {
        'preferred': ['long_calls', 'long_puts', 'debit_spreads'],
        'avoid': ['short_vol_strategies'],
        'max_position_risk': 0.015,  # 1.5% per trade
        'description': 'High vol - directional only',
    },
    Regime.R5: {
        'preferred': ['long_vol', 'cash'],
        'avoid': ['everything_else'],
        'max_position_risk': 0.01,  # 1% per trade
        'description': 'Crisis mode - preservation',
    },
}
