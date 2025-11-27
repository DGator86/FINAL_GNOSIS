"""
Entry Validation Framework
===========================

Complete pre-entry validation protocol with mandatory checkboxes.
NO ENTRY allowed unless ALL conditions are met.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from .edge_detection import (
    VolEdgeScore,
    SkewAnalysis,
    TermStructure,
    calculate_spread_quality,
    meets_liquidity_standards,
    ENTRY_THRESHOLDS,
)
from .regime_classification import RegimeMetrics, Regime


class StrategyCategory(Enum):
    """Strategy categories for validation"""
    SHORT_VOL = "short_vol"
    LONG_VOL = "long_vol"
    NEUTRAL_VOL = "neutral_vol"
    DIRECTIONAL = "directional"
    TIME_SPREAD = "time_spread"


@dataclass
class EventCalendar:
    """Upcoming events that may impact entry"""
    earnings_date: Optional[datetime] = None
    fomc_date: Optional[datetime] = None
    cpi_date: Optional[datetime] = None
    opex_date: Optional[datetime] = None
    ex_dividend_date: Optional[datetime] = None

    def check_blackout(self, entry_date: datetime) -> Tuple[bool, Optional[str]]:
        """
        Check if entry date falls in event blackout window

        Returns:
            (is_blackout, reason)
        """
        # Earnings: 3 days before to 1 day after
        if self.earnings_date:
            days_to_earnings = (self.earnings_date - entry_date).days
            if -1 <= days_to_earnings <= 3:
                return True, f"Earnings in {days_to_earnings} days (blackout period)"

        # FOMC: 2 days before to 1 day after
        if self.fomc_date:
            days_to_fomc = (self.fomc_date - entry_date).days
            if -1 <= days_to_fomc <= 2:
                return True, f"FOMC in {days_to_fomc} days (blackout period)"

        # CPI/NFP: 1 day before to same day
        if self.cpi_date:
            days_to_cpi = (self.cpi_date - entry_date).days
            if 0 <= days_to_cpi <= 1:
                return True, f"CPI in {days_to_cpi} days (blackout period)"

        # OpEx: Final 2 days
        if self.opex_date:
            days_to_opex = (self.opex_date - entry_date).days
            if 0 <= days_to_opex <= 2:
                return True, f"OpEx in {days_to_opex} days (blackout period)"

        # Ex-Dividend: 4 days before
        if self.ex_dividend_date:
            days_to_ex_div = (self.ex_dividend_date - entry_date).days
            if 0 <= days_to_ex_div <= 4:
                return True, f"Ex-Dividend in {days_to_ex_div} days (blackout period)"

        return False, None


@dataclass
class DirectionalSignal:
    """Directional confirmation for directional strategies"""
    trend_score: float  # -3.0 to +3.0
    price: float
    ma_20: float
    ma_50: float
    atr_20: float

    @property
    def is_bullish(self) -> bool:
        return self.trend_score > 1.0

    @property
    def is_bearish(self) -> bool:
        return self.trend_score < -1.0

    @property
    def is_neutral(self) -> bool:
        return -0.5 <= self.trend_score <= 0.5


@dataclass
class EntryConditions:
    """Complete entry validation conditions"""

    # Edge Analysis
    vol_edge: VolEdgeScore
    skew: Optional[SkewAnalysis] = None
    term_structure: Optional[TermStructure] = None

    # Regime
    regime: RegimeMetrics = None

    # Liquidity
    spread_quality: float = 0.0
    open_interest: int = 0
    daily_volume: int = 0
    asset_type: str = 'etf'

    # Directional (if applicable)
    directional_signal: Optional[DirectionalSignal] = None

    # Events
    event_calendar: EventCalendar = field(default_factory=EventCalendar)

    # Strategy Details
    strategy_category: StrategyCategory = StrategyCategory.NEUTRAL_VOL
    strategy_name: str = ""
    max_loss: Optional[float] = None
    position_size: int = 0

    # Greeks (projected)
    projected_delta: float = 0.0
    projected_gamma: float = 0.0
    projected_vega: float = 0.0
    projected_theta: float = 0.0

    # Exit Plan
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    time_exit_dte: Optional[int] = None
    regime_exit_trigger: Optional[str] = None

    # Risk Management
    account_risk_available: float = 0.0  # Available risk budget


@dataclass
class ValidationResult:
    """Result of entry validation"""
    is_valid: bool
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str]
    validation_timestamp: datetime
    conditions: EntryConditions

    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary = []
        summary.append(f"ENTRY VALIDATION: {'✓ APPROVED' if self.is_valid else '✗ REJECTED'}")
        summary.append(f"Timestamp: {self.validation_timestamp}")
        summary.append(f"\nPassed: {len(self.passed_checks)}/{len(self.passed_checks) + len(self.failed_checks)}")

        if self.failed_checks:
            summary.append("\n❌ FAILED CHECKS:")
            for check in self.failed_checks:
                summary.append(f"  • {check}")

        if self.warnings:
            summary.append("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                summary.append(f"  • {warning}")

        return "\n".join(summary)


class EntryValidator:
    """
    Complete entry validation system

    Validates ALL entry conditions before allowing trade execution.
    """

    def __init__(self):
        self.validation_history: List[ValidationResult] = []

    def validate(self, conditions: EntryConditions) -> ValidationResult:
        """
        Execute complete entry validation protocol

        Returns:
            ValidationResult with pass/fail for each check
        """
        passed = []
        failed = []
        warnings = []

        # ========================================
        # 1. VOL EDGE VALIDATION
        # ========================================
        strategy_cat = conditions.strategy_category.value

        # Check if vol edge exists
        if conditions.vol_edge is None:
            failed.append("Vol Edge not calculated")
        else:
            # Check threshold
            if conditions.vol_edge.meets_threshold(strategy_cat):
                passed.append(f"Vol Edge {conditions.vol_edge.vol_edge:.2f}% meets threshold")
            else:
                failed.append(
                    f"Vol Edge {conditions.vol_edge.vol_edge:.2f}% below minimum for {strategy_cat}"
                )

            # Check optimal range
            if conditions.vol_edge.in_optimal_range(strategy_cat):
                passed.append(f"Vol Edge in optimal range")
            else:
                warnings.append(f"Vol Edge outside optimal range (still tradeable)")

        # ========================================
        # 2. IV RANK VALIDATION
        # ========================================
        if conditions.vol_edge:
            iv_rank = conditions.vol_edge.iv_rank

            # Check IV Rank requirements by strategy
            if strategy_cat == 'short_vol':
                if iv_rank >= 70.0:
                    passed.append(f"IV Rank {iv_rank:.1f}% ≥ 70% for short vol")
                else:
                    failed.append(f"IV Rank {iv_rank:.1f}% below 70% minimum for short vol")

            elif strategy_cat == 'long_vol':
                if iv_rank <= 30.0:
                    passed.append(f"IV Rank {iv_rank:.1f}% ≤ 30% for long vol")
                else:
                    failed.append(f"IV Rank {iv_rank:.1f}% above 30% maximum for long vol")

            elif strategy_cat == 'neutral_vol':
                if 40.0 <= iv_rank <= 80.0:
                    passed.append(f"IV Rank {iv_rank:.1f}% in 40-80% range")
                else:
                    warnings.append(f"IV Rank {iv_rank:.1f}% outside 40-80% optimal range")

        # ========================================
        # 3. REGIME VALIDATION
        # ========================================
        if conditions.regime is None:
            failed.append("Regime not classified")
        else:
            # Check stability
            if conditions.regime.is_stable:
                passed.append(
                    f"Regime {conditions.regime.regime.value} stable for "
                    f"{conditions.regime.stability_days} days"
                )
            else:
                failed.append(
                    f"Regime {conditions.regime.regime.value} not stable "
                    f"({conditions.regime.stability_days} days)"
                )

            # Check transition risk
            if conditions.regime.transition_risk > 75:
                warnings.append(
                    f"High transition risk {conditions.regime.transition_risk:.1f}% - "
                    f"consider 50% position size reduction"
                )

        # ========================================
        # 4. SKEW/TERM STRUCTURE (if applicable)
        # ========================================
        if conditions.strategy_category == StrategyCategory.TIME_SPREAD:
            if conditions.term_structure is None:
                failed.append("Term structure required for calendar/diagonal strategies")
            elif conditions.term_structure.calendar_favorable():
                passed.append(
                    f"Term structure favorable: "
                    f"{conditions.term_structure.term_premium:.2f}% contango"
                )
            else:
                failed.append("Term structure not favorable for calendar spread")

        # Check skew for asymmetric strategies
        if conditions.skew is not None:
            if 'jade' in conditions.strategy_name.lower():
                if conditions.skew.jade_lizard_favorable():
                    passed.append(f"Put skew {conditions.skew.put_skew:.2f}% > 8% (favorable)")
                else:
                    warnings.append(f"Put skew {conditions.skew.put_skew:.2f}% below 8%")

        # ========================================
        # 5. DIRECTIONAL ALIGNMENT (if applicable)
        # ========================================
        if conditions.strategy_category == StrategyCategory.DIRECTIONAL:
            if conditions.directional_signal is None:
                failed.append("Directional signal required for directional strategy")
            else:
                signal = conditions.directional_signal

                if 'bull' in conditions.strategy_name.lower() or 'call' in conditions.strategy_name.lower():
                    if signal.is_bullish:
                        passed.append(f"Bullish signal confirmed (trend score {signal.trend_score:.2f})")
                    else:
                        failed.append(f"Bullish strategy but trend score {signal.trend_score:.2f} < 1.0")

                elif 'bear' in conditions.strategy_name.lower() or 'put' in conditions.strategy_name.lower():
                    if signal.is_bearish:
                        passed.append(f"Bearish signal confirmed (trend score {signal.trend_score:.2f})")
                    else:
                        failed.append(f"Bearish strategy but trend score {signal.trend_score:.2f} > -1.0")

        # ========================================
        # 6. EVENT CALENDAR CHECK
        # ========================================
        is_blackout, blackout_reason = conditions.event_calendar.check_blackout(datetime.now())

        if is_blackout:
            # Hard fail unless long vol strategy
            if conditions.strategy_category != StrategyCategory.LONG_VOL:
                failed.append(f"Event blackout: {blackout_reason}")
            else:
                warnings.append(f"Event blackout: {blackout_reason} (allowed for long vol)")
        else:
            passed.append("No event blackouts detected")

        # ========================================
        # 7. LIQUIDITY VALIDATION
        # ========================================
        meets_liquidity, liquidity_reason = meets_liquidity_standards(
            spread_quality=conditions.spread_quality,
            open_interest=conditions.open_interest,
            daily_volume=conditions.daily_volume,
            asset_type=conditions.asset_type,
        )

        if meets_liquidity:
            passed.append(f"Liquidity standards met: {liquidity_reason}")
        else:
            failed.append(f"Liquidity failed: {liquidity_reason}")

        # ========================================
        # 8. POSITION SIZE VALIDATION
        # ========================================
        if conditions.position_size <= 0:
            failed.append("Position size not calculated")
        else:
            passed.append(f"Position size calculated: {conditions.position_size} contracts")

        # ========================================
        # 9. MAX LOSS DEFINED
        # ========================================
        if conditions.max_loss is None:
            failed.append("Maximum loss not defined")
        else:
            passed.append(f"Max loss defined: ${conditions.max_loss:,.2f}")

        # ========================================
        # 10. EXIT PLAN DOCUMENTED
        # ========================================
        exit_plan_complete = all([
            conditions.profit_target is not None,
            conditions.stop_loss is not None,
            conditions.time_exit_dte is not None,
            conditions.regime_exit_trigger is not None,
        ])

        if exit_plan_complete:
            passed.append("Complete exit plan documented")
        else:
            missing = []
            if conditions.profit_target is None:
                missing.append("profit target")
            if conditions.stop_loss is None:
                missing.append("stop loss")
            if conditions.time_exit_dte is None:
                missing.append("time exit")
            if conditions.regime_exit_trigger is None:
                missing.append("regime exit")

            failed.append(f"Exit plan incomplete: missing {', '.join(missing)}")

        # ========================================
        # 11. RISK BUDGET AVAILABLE
        # ========================================
        if conditions.account_risk_available <= 0:
            failed.append("No risk budget available (overallocated)")
        elif conditions.max_loss and conditions.max_loss > conditions.account_risk_available:
            failed.append(
                f"Max loss ${conditions.max_loss:,.2f} exceeds available "
                f"risk budget ${conditions.account_risk_available:,.2f}"
            )
        else:
            passed.append("Risk budget available")

        # ========================================
        # 12. GREEKS PROJECTED AND ACCEPTABLE
        # ========================================
        # This is a soft check - just verify they were projected
        if conditions.projected_delta == 0 and conditions.projected_theta == 0:
            warnings.append("Greeks not projected (should be calculated)")
        else:
            passed.append("Greeks projected")

        # ========================================
        # FINAL RESULT
        # ========================================
        is_valid = len(failed) == 0

        result = ValidationResult(
            is_valid=is_valid,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            validation_timestamp=datetime.now(),
            conditions=conditions,
        )

        # Store in history
        self.validation_history.append(result)

        return result

    def get_validation_stats(self) -> Dict[str, any]:
        """Get statistics on validation history"""
        if not self.validation_history:
            return {'total': 0, 'approved': 0, 'rejected': 0}

        total = len(self.validation_history)
        approved = sum(1 for v in self.validation_history if v.is_valid)
        rejected = total - approved

        return {
            'total': total,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': (approved / total) * 100 if total > 0 else 0,
        }


def validate_entry(conditions: EntryConditions) -> ValidationResult:
    """
    Convenience function for one-off validation

    Args:
        conditions: EntryConditions object

    Returns:
        ValidationResult
    """
    validator = EntryValidator()
    return validator.validate(conditions)
