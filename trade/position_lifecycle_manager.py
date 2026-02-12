"""
Position Lifecycle Manager - Trade Exit, Roll, and Adjustment Logic

Manages the complete lifecycle of options and equity positions including:
- Exit conditions (profit target, stop loss, time-based)
- Rolling strategies (same strike roll, diagonal roll, etc.)
- Position adjustments (adding to winners, defending losers)
- Assignment/exercise risk management

LIFECYCLE STAGES:
├── Open: Position just entered
├── Active: Normal monitoring
├── Profit Zone: Near/at profit target
├── Risk Zone: Near/at stop loss
├── Expiration Zone: DTE-based management
├── Roll Candidate: Meeting roll criteria
└── Close: Exit triggered

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class PositionStage(str, Enum):
    """Current stage in position lifecycle."""
    OPEN = "open"
    ACTIVE = "active"
    PROFIT_ZONE = "profit_zone"
    RISK_ZONE = "risk_zone"
    EXPIRATION_ZONE = "expiration_zone"
    ROLL_CANDIDATE = "roll_candidate"
    ADJUSTMENT_NEEDED = "adjustment_needed"
    CLOSE = "close"


class ExitReason(str, Enum):
    """Reason for position exit."""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    DTE_EXIT = "dte_exit"
    ASSIGNMENT_RISK = "assignment_risk"
    GAMMA_RISK = "gamma_risk"
    MANUAL = "manual"
    ROLL = "roll"
    CIRCUIT_BREAKER = "circuit_breaker"


class RollType(str, Enum):
    """Types of roll strategies."""
    SAME_STRIKE_FORWARD = "same_strike_forward"  # Roll to next expiration, same strike
    DIAGONAL_UP = "diagonal_up"    # Roll out and up in strike
    DIAGONAL_DOWN = "diagonal_down"  # Roll out and down in strike
    INVERTED_ROLL = "inverted_roll"  # Roll for credit, accepting worse strike
    CALENDAR_SPREAD = "calendar"   # Convert to calendar spread
    NONE = "none"


class AdjustmentType(str, Enum):
    """Types of position adjustments."""
    ADD_TO_WINNER = "add_to_winner"
    SCALE_OUT_PARTIAL = "scale_out"
    HEDGE_WITH_SPREAD = "hedge_spread"
    ROLL_STRIKE = "roll_strike"
    CONVERT_TO_SPREAD = "convert_spread"
    CLOSE_PARTIAL = "close_partial"


@dataclass
class PositionMetrics:
    """Current metrics for a position."""
    symbol: str
    underlying: str
    entry_price: float
    current_price: float
    quantity: int
    
    # P&L metrics
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    
    # Time metrics
    entry_time: Optional[datetime] = None
    days_held: int = 0
    
    # Options-specific
    is_option: bool = False
    option_type: str = ""  # "call" or "put"
    strike: float = 0.0
    expiration: Optional[datetime] = None
    dte: int = 0
    
    # Greeks (if available)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    
    # Underlying metrics
    underlying_price: float = 0.0
    moneyness: float = 0.0  # (underlying - strike) / strike for calls
    
    # Risk metrics
    max_gain: float = 0.0
    max_loss: float = 0.0
    
    # Multi-leg
    is_spread: bool = False
    spread_type: str = ""


@dataclass
class LifecycleDecision:
    """Decision from lifecycle analysis."""
    stage: PositionStage
    action: str  # "hold", "close", "roll", "adjust", "scale_out"
    urgency: str  # "immediate", "today", "soon", "monitor"
    
    # Exit details (if closing)
    exit_reason: Optional[ExitReason] = None
    exit_price_target: float = 0.0
    
    # Roll details (if rolling)
    roll_type: Optional[RollType] = None
    roll_to_expiration: Optional[datetime] = None
    roll_to_strike: float = 0.0
    roll_credit_target: float = 0.0
    
    # Adjustment details
    adjustment_type: Optional[AdjustmentType] = None
    
    # Reasoning
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Scale-out details
    scale_out_quantity: int = 0
    scale_out_pct: float = 0.0


@dataclass
class LifecycleConfig:
    """Configuration for position lifecycle management."""
    # Profit taking
    profit_target_pct: float = 50.0      # Default 50% profit target
    early_profit_pct: float = 25.0       # Begin scale-out at 25%
    max_profit_pct: float = 75.0         # Strong scale-out signal
    
    # Stop loss
    stop_loss_pct: float = 100.0         # Default 100% loss (1:1 R:R)
    mental_stop_pct: float = 75.0        # Tighten focus at 75% loss
    
    # Time-based
    max_hold_days: int = 45              # Max days to hold
    dte_warning: int = 14                # Warning at 14 DTE
    dte_exit: int = 7                    # Exit at 7 DTE (avoid gamma)
    dte_roll_target: int = 21            # Target DTE when rolling
    
    # Roll criteria
    min_credit_for_roll: float = 0.0     # Min credit to accept roll
    max_loss_for_roll_pct: float = 200.0  # Don't roll if losing > 200%
    
    # Adjustment criteria
    min_profit_to_add: float = 20.0      # Min profit % to add to position
    max_position_size: int = 10          # Max contracts per position
    
    # Assignment risk
    itm_warning_pct: float = 2.0         # Warn if 2% ITM
    itm_critical_pct: float = 5.0        # Critical if 5% ITM
    
    # Credit strategies
    credit_profit_target_pct: float = 50.0  # Take profit at 50% of credit
    credit_roll_trigger_pct: float = -100.0  # Roll if at 100% of credit
    
    # Debit strategies
    debit_profit_target_pct: float = 100.0  # Target 100% gain
    debit_stop_loss_pct: float = 50.0       # Stop at 50% loss


class PositionLifecycleManager:
    """
    Manages the complete lifecycle of trading positions.
    
    Features:
    - Stage classification based on P&L and time
    - Dynamic exit recommendations
    - Roll strategy selection
    - Adjustment recommendations
    - Assignment/exercise risk monitoring
    """
    
    # Scale-out schedule (profit_pct -> scale_out_pct)
    SCALE_OUT_SCHEDULE = [
        (25.0, 0.25),   # At 25% profit, scale out 25%
        (50.0, 0.35),   # At 50% profit, scale out 35%
        (75.0, 0.25),   # At 75% profit, scale out 25%
        (100.0, 0.15),  # At 100% profit, scale out remaining 15%
    ]
    
    # DTE-based profit acceleration
    DTE_PROFIT_ACCELERATION = {
        7: 0.5,    # At 7 DTE, take 50% of target profit
        14: 0.65,  # At 14 DTE, take 65% of target profit
        21: 0.80,  # At 21 DTE, take 80% of target profit
    }
    
    def __init__(self, config: Optional[LifecycleConfig] = None):
        """Initialize the lifecycle manager.
        
        Args:
            config: Configuration for lifecycle rules
        """
        self.config = config or LifecycleConfig()
        
        # Track positions being managed
        self.positions: Dict[str, PositionMetrics] = {}
        self.decisions_history: List[LifecycleDecision] = []
        
        logger.info(
            f"PositionLifecycleManager initialized | "
            f"profit_target={self.config.profit_target_pct}% | "
            f"stop_loss={self.config.stop_loss_pct}% | "
            f"dte_exit={self.config.dte_exit}"
        )
    
    def analyze_position(self, position: PositionMetrics) -> LifecycleDecision:
        """Analyze a position and determine lifecycle decision.
        
        Args:
            position: Current position metrics
            
        Returns:
            LifecycleDecision with recommended action
        """
        decision = LifecycleDecision(
            stage=PositionStage.ACTIVE,
            action="hold",
            urgency="monitor",
        )
        
        reasons = []
        warnings = []
        
        # Calculate key metrics
        pnl_pct = position.unrealized_pnl_pct
        dte = position.dte
        is_credit = position.entry_price < 0  # Credit received
        
        # Use appropriate config based on strategy type
        if is_credit:
            profit_target = self.config.credit_profit_target_pct
            stop_loss = abs(self.config.credit_roll_trigger_pct)
        else:
            profit_target = self.config.debit_profit_target_pct
            stop_loss = self.config.debit_stop_loss_pct
        
        # =====================================
        # Stage 1: Check for immediate exits
        # =====================================
        
        # Stop loss check
        if pnl_pct <= -stop_loss:
            decision.stage = PositionStage.CLOSE
            decision.action = "close"
            decision.urgency = "immediate"
            decision.exit_reason = ExitReason.STOP_LOSS
            reasons.append(f"Stop loss hit: {pnl_pct:.1f}% loss (limit: -{stop_loss:.0f}%)")
            
            # Check if roll is possible instead
            if position.is_option and pnl_pct > -self.config.max_loss_for_roll_pct:
                roll_type = self._determine_roll_type(position)
                if roll_type != RollType.NONE:
                    decision.action = "roll"
                    decision.roll_type = roll_type
                    decision.exit_reason = ExitReason.ROLL
                    reasons.append(f"Consider rolling instead of closing")
            
            decision.reasons = reasons
            decision.warnings = warnings
            return decision
        
        # Max profit check
        if pnl_pct >= profit_target:
            decision.stage = PositionStage.PROFIT_ZONE
            decision.action = "close"
            decision.urgency = "today"
            decision.exit_reason = ExitReason.PROFIT_TARGET
            reasons.append(f"Profit target reached: {pnl_pct:.1f}% (target: {profit_target:.0f}%)")
            decision.reasons = reasons
            return decision
        
        # =====================================
        # Stage 2: DTE-based management
        # =====================================
        
        if position.is_option and dte is not None:
            # Critical DTE - must exit
            if dte <= self.config.dte_exit:
                decision.stage = PositionStage.EXPIRATION_ZONE
                
                # Check if profitable enough to close
                dte_adjusted_target = profit_target * self.DTE_PROFIT_ACCELERATION.get(7, 0.5)
                
                if pnl_pct >= dte_adjusted_target:
                    decision.action = "close"
                    decision.urgency = "immediate"
                    decision.exit_reason = ExitReason.DTE_EXIT
                    reasons.append(
                        f"DTE exit: {dte} days remaining, profit {pnl_pct:.1f}% "
                        f"exceeds DTE-adjusted target {dte_adjusted_target:.1f}%"
                    )
                else:
                    # Roll or close
                    roll_type = self._determine_roll_type(position)
                    if roll_type != RollType.NONE and pnl_pct > -50:
                        decision.action = "roll"
                        decision.roll_type = roll_type
                        decision.roll_to_expiration = datetime.utcnow() + timedelta(
                            days=self.config.dte_roll_target
                        )
                        reasons.append(f"Roll recommended: DTE={dte}, targeting {self.config.dte_roll_target} DTE")
                    else:
                        decision.action = "close"
                        decision.urgency = "today"
                        decision.exit_reason = ExitReason.GAMMA_RISK
                        reasons.append(f"Close to avoid gamma risk: DTE={dte}")
                
                decision.reasons = reasons
                return decision
            
            # Warning DTE
            if dte <= self.config.dte_warning:
                decision.stage = PositionStage.EXPIRATION_ZONE
                warnings.append(f"⚠️ Approaching expiration: {dte} DTE")
                
                # Adjust profit target based on DTE
                dte_factor = self.DTE_PROFIT_ACCELERATION.get(
                    min(dte, max(self.DTE_PROFIT_ACCELERATION.keys())), 
                    0.8
                )
                dte_adjusted_target = profit_target * dte_factor
                
                if pnl_pct >= dte_adjusted_target:
                    decision.action = "close"
                    decision.urgency = "today"
                    decision.exit_reason = ExitReason.PROFIT_TARGET
                    reasons.append(
                        f"DTE-adjusted profit target: {pnl_pct:.1f}% >= {dte_adjusted_target:.1f}%"
                    )
        
        # =====================================
        # Stage 3: Assignment risk (ITM options)
        # =====================================
        
        if position.is_option and position.quantity < 0:  # Short options
            itm_pct = abs(position.moneyness) * 100
            
            if position.option_type == "call" and position.underlying_price > position.strike:
                if itm_pct >= self.config.itm_critical_pct:
                    decision.stage = PositionStage.RISK_ZONE
                    warnings.append(
                        f"🚨 ASSIGNMENT RISK: Short call {itm_pct:.1f}% ITM"
                    )
                    if dte and dte <= 7:
                        decision.action = "close"
                        decision.urgency = "immediate"
                        decision.exit_reason = ExitReason.ASSIGNMENT_RISK
                        reasons.append("Close to avoid assignment")
                        
            elif position.option_type == "put" and position.underlying_price < position.strike:
                if itm_pct >= self.config.itm_critical_pct:
                    decision.stage = PositionStage.RISK_ZONE
                    warnings.append(
                        f"🚨 ASSIGNMENT RISK: Short put {itm_pct:.1f}% ITM"
                    )
        
        # =====================================
        # Stage 4: Scale-out opportunities
        # =====================================
        
        if pnl_pct >= self.config.early_profit_pct:
            decision.stage = PositionStage.PROFIT_ZONE
            
            # Determine scale-out quantity
            for threshold, scale_pct in self.SCALE_OUT_SCHEDULE:
                if pnl_pct >= threshold:
                    decision.scale_out_pct = scale_pct
                    decision.scale_out_quantity = max(
                        1, 
                        int(abs(position.quantity) * scale_pct)
                    )
            
            if decision.scale_out_quantity > 0:
                decision.action = "scale_out"
                decision.urgency = "today"
                reasons.append(
                    f"Scale-out opportunity: {pnl_pct:.1f}% profit, "
                    f"close {decision.scale_out_quantity} of {abs(position.quantity)}"
                )
        
        # =====================================
        # Stage 5: Risk zone management
        # =====================================
        
        if pnl_pct <= -self.config.mental_stop_pct:
            decision.stage = PositionStage.RISK_ZONE
            warnings.append(f"⚠️ Approaching stop loss: {pnl_pct:.1f}%")
            decision.urgency = "soon"
            
            # Check for adjustment opportunities
            adjustment = self._recommend_adjustment(position)
            if adjustment:
                decision.adjustment_type = adjustment
                reasons.append(f"Consider adjustment: {adjustment.value}")
        
        # =====================================
        # Stage 6: Time-based management
        # =====================================
        
        if position.days_held >= self.config.max_hold_days:
            decision.stage = PositionStage.CLOSE
            decision.action = "close"
            decision.urgency = "today"
            decision.exit_reason = ExitReason.TIME_STOP
            reasons.append(
                f"Max hold time exceeded: {position.days_held} days "
                f"(max: {self.config.max_hold_days})"
            )
        
        # =====================================
        # Stage 7: Roll candidate check
        # =====================================
        
        if position.is_option and dte and dte <= self.config.dte_warning:
            if pnl_pct > -50 and pnl_pct < profit_target * 0.5:
                # Not at target but not deeply losing - roll candidate
                roll_type = self._determine_roll_type(position)
                if roll_type != RollType.NONE:
                    decision.stage = PositionStage.ROLL_CANDIDATE
                    decision.roll_type = roll_type
                    reasons.append(
                        f"Roll candidate: DTE={dte}, P&L={pnl_pct:.1f}%"
                    )
        
        decision.reasons = reasons
        decision.warnings = warnings
        
        return decision
    
    def _determine_roll_type(self, position: PositionMetrics) -> RollType:
        """Determine the best roll type for a position.
        
        Args:
            position: Current position metrics
            
        Returns:
            Recommended roll type
        """
        if not position.is_option:
            return RollType.NONE
        
        pnl_pct = position.unrealized_pnl_pct
        is_short = position.quantity < 0
        
        # Short options (credit strategies)
        if is_short:
            if pnl_pct > -50:
                # Small loss or profit - roll to same strike
                return RollType.SAME_STRIKE_FORWARD
            elif pnl_pct > -100:
                # Moderate loss - roll diagonally for better strike
                if position.option_type == "put":
                    return RollType.DIAGONAL_DOWN  # Roll to lower strike put
                else:
                    return RollType.DIAGONAL_UP    # Roll to higher strike call
            else:
                # Large loss - inverted roll to collect more credit
                return RollType.INVERTED_ROLL
        
        # Long options (debit strategies)
        else:
            if pnl_pct < -25:
                # Losing position - consider calendar
                return RollType.CALENDAR_SPREAD
            else:
                # Diagonal to lock in some profit
                return RollType.SAME_STRIKE_FORWARD
        
        return RollType.NONE
    
    def _recommend_adjustment(self, position: PositionMetrics) -> Optional[AdjustmentType]:
        """Recommend adjustment for a position.
        
        Args:
            position: Current position metrics
            
        Returns:
            Recommended adjustment type or None
        """
        pnl_pct = position.unrealized_pnl_pct
        
        # Winning position
        if pnl_pct >= self.config.min_profit_to_add:
            if abs(position.quantity) < self.config.max_position_size:
                return AdjustmentType.ADD_TO_WINNER
            return AdjustmentType.SCALE_OUT_PARTIAL
        
        # Losing position
        if pnl_pct <= -self.config.mental_stop_pct:
            if position.is_option and not position.is_spread:
                # Convert naked to spread to define risk
                return AdjustmentType.CONVERT_TO_SPREAD
            elif position.is_spread:
                # Roll the tested leg
                return AdjustmentType.ROLL_STRIKE
        
        return None
    
    def get_roll_details(
        self,
        position: PositionMetrics,
        roll_type: RollType,
        target_dte: int = 21,
    ) -> Dict[str, Any]:
        """Get detailed roll execution parameters.
        
        Args:
            position: Current position
            roll_type: Type of roll to execute
            target_dte: Target DTE for new position
            
        Returns:
            Dict with roll execution details
        """
        details = {
            "original_position": {
                "symbol": position.symbol,
                "strike": position.strike,
                "expiration": position.expiration.isoformat() if position.expiration else None,
                "quantity": position.quantity,
                "current_price": position.current_price,
            },
            "roll_type": roll_type.value,
            "target_dte": target_dte,
            "new_expiration": (
                datetime.utcnow() + timedelta(days=target_dte)
            ).date().isoformat(),
        }
        
        # Calculate new strike based on roll type
        if roll_type == RollType.SAME_STRIKE_FORWARD:
            details["new_strike"] = position.strike
            details["description"] = "Roll to same strike, further expiration"
            
        elif roll_type == RollType.DIAGONAL_UP:
            # Move strike up ~5%
            details["new_strike"] = round(position.strike * 1.05, 2)
            details["description"] = "Roll to higher strike (improve cost basis)"
            
        elif roll_type == RollType.DIAGONAL_DOWN:
            # Move strike down ~5%
            details["new_strike"] = round(position.strike * 0.95, 2)
            details["description"] = "Roll to lower strike (improve cost basis)"
            
        elif roll_type == RollType.INVERTED_ROLL:
            # Aggressive strike move ~10%
            if position.option_type == "put":
                details["new_strike"] = round(position.strike * 0.90, 2)
            else:
                details["new_strike"] = round(position.strike * 1.10, 2)
            details["description"] = "Inverted roll for additional credit"
            
        elif roll_type == RollType.CALENDAR_SPREAD:
            details["new_strike"] = position.strike
            details["description"] = "Convert to calendar spread"
            details["keep_original"] = True
        
        # Estimate credit/debit
        # This is simplified - real implementation would price options
        days_extension = target_dte - position.dte if position.dte else target_dte
        
        if position.quantity < 0:  # Short option
            # Rolling short options typically collects additional credit
            estimated_credit = position.current_price * 0.3 * (days_extension / 30)
            details["estimated_credit"] = round(estimated_credit, 2)
        else:  # Long option
            # Rolling long options costs additional debit
            estimated_debit = position.current_price * 0.4 * (days_extension / 30)
            details["estimated_debit"] = round(estimated_debit, 2)
        
        return details
    
    def get_close_order(
        self,
        position: PositionMetrics,
        order_type: str = "limit",
        aggression: str = "normal",
    ) -> Dict[str, Any]:
        """Generate close order details.
        
        Args:
            position: Position to close
            order_type: "limit" or "market"
            aggression: "passive", "normal", or "aggressive"
            
        Returns:
            Dict with order details
        """
        # Base price
        price = position.current_price
        
        # Adjust for aggression
        if order_type == "limit":
            if aggression == "passive":
                # Try to get better fill
                adjustment = 0.02 if position.quantity > 0 else -0.02
            elif aggression == "aggressive":
                # Cross the spread
                adjustment = -0.03 if position.quantity > 0 else 0.03
            else:
                # Normal - mid price
                adjustment = 0
            
            price = round(price * (1 + adjustment), 2)
        
        return {
            "symbol": position.symbol,
            "action": "sell" if position.quantity > 0 else "buy",
            "quantity": abs(position.quantity),
            "order_type": order_type,
            "limit_price": price if order_type == "limit" else None,
            "time_in_force": "day",
        }
    
    def batch_analyze(
        self,
        positions: List[PositionMetrics],
    ) -> List[Tuple[PositionMetrics, LifecycleDecision]]:
        """Analyze multiple positions and return decisions.
        
        Args:
            positions: List of positions to analyze
            
        Returns:
            List of (position, decision) tuples, sorted by urgency
        """
        results = []
        
        for position in positions:
            decision = self.analyze_position(position)
            results.append((position, decision))
        
        # Sort by urgency
        urgency_order = {"immediate": 0, "today": 1, "soon": 2, "monitor": 3}
        results.sort(
            key=lambda x: (
                urgency_order.get(x[1].urgency, 4),
                -abs(x[0].unrealized_pnl_pct)  # Higher P&L impact first
            )
        )
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of lifecycle management config."""
        return {
            "profit_target_pct": self.config.profit_target_pct,
            "stop_loss_pct": self.config.stop_loss_pct,
            "dte_warning": self.config.dte_warning,
            "dte_exit": self.config.dte_exit,
            "scale_out_schedule": self.SCALE_OUT_SCHEDULE,
            "max_hold_days": self.config.max_hold_days,
        }


# Factory function
def create_lifecycle_manager(
    profit_target_pct: float = 50.0,
    stop_loss_pct: float = 100.0,
    dte_exit: int = 7,
) -> PositionLifecycleManager:
    """Create a PositionLifecycleManager with custom config.
    
    Args:
        profit_target_pct: Default profit target percentage
        stop_loss_pct: Default stop loss percentage
        dte_exit: Days to expiration for forced exit
        
    Returns:
        Configured PositionLifecycleManager
    """
    config = LifecycleConfig(
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        dte_exit=dte_exit,
    )
    return PositionLifecycleManager(config=config)
